# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from helpers.llk_params import DestAccumulation, DestSync, GoldenType, format_dict

from ..fpu_node import FpuNode
from ..pack_node import PackNode
from ..pipeline_plan import PlannedBlock, PlannedNode
from ..sfpu_node import SfpuNode
from .state import (
    DestBank,
    GoldenState,
    Inputs,
    OperandTiles,
    OutputLayout,
    finalize_output,
    tile_dimensions,
)


class GoldenExecutor:
    def __init__(self, operation, config, golden_type: GoldenType):
        if config.perf_run_type is not None:
            raise ValueError(
                f"golden() needs a functional run, got perf_run_type={config.perf_run_type}"
            )
        self.operation = operation
        self.config = config
        self.golden_type = golden_type
        self.pipeline = operation.math
        self.pack_nodes = [
            node for node in self.pipeline.pack_nodes if isinstance(node, PackNode)
        ]
        self.output_format = self.pack_nodes[0].output.data_format
        self.buffers = {id(node.output): {} for node in self.pack_nodes}
        self.views = {}
        for node in self.pipeline.math_nodes:
            if not isinstance(node, FpuNode):
                continue
            for slot, operand in (("a", node.src_a), ("b", node.src_b)):
                if operand is not None and (node, slot) not in self.views:
                    source = (
                        operand.raw_data
                        if golden_type == GoldenType.L1_GOLDEN
                        else operand.master_golden
                    )
                    self.views[(node, slot)] = OperandTiles(operand, source)

    def _output_layout(self, node: PackNode) -> OutputLayout:
        if node.packer.output_layout != OutputLayout.ROW_MAJOR:
            return node.packer.output_layout
        for math_node in self.pipeline.math_nodes:
            if (
                isinstance(math_node, FpuNode)
                and math_node.unpacker is not None
                and math_node.unpacker.output_layout != OutputLayout.ROW_MAJOR
            ):
                return math_node.unpacker.output_layout
        return OutputLayout.ROW_MAJOR

    def _dest_size(self, planned: PlannedBlock) -> int:
        if not self.pipeline.custom_op:
            return planned.region.block_tiles
        faces = 32 if self.operation.dest_sync == DestSync.Half else 64
        if self.config.dest_acc == DestAccumulation.Yes:
            faces //= 2
        return faces // self.operation.tile_shape.total_num_faces()

    def _run_node(self, planned: PlannedNode, bank, state: GoldenState) -> None:
        state.dest.block_tiles_x = planned.block.block_cols
        state.dest.block_tiles_y = planned.block.block_rows
        for call in planned.loop.calls(bank):
            planned.unit.golden_fn(
                call, state, planned.node, self.operation, self.config
            )

    def _run_math(self, planned: PlannedBlock, bank, state: GoldenState) -> None:
        for node in self.pipeline.math_nodes:
            self.config.sentinel.configure_golden(
                self.config, self.operation, node, output_format=self.output_format
            )
            if isinstance(node, SfpuNode):
                self._run_node(planned.plan(node, "sfpu"), bank, state)
                continue
            math = planned.plan(node, "math")
            state.begin_fpu(
                Inputs(
                    self.views.get((node, "a")),
                    self.views.get((node, "b")),
                    math.block.block_cols,
                    math.block.block_rows,
                )
            )
            if node.unpacker is not None:
                self._run_node(planned.plan(node, "unpack"), bank, state)
            self._run_node(math, bank, state)

    def _run_pack(self, planned: PlannedBlock, bank, state: GoldenState) -> None:
        for node in self.pipeline.pack_nodes:
            if isinstance(node, SfpuNode):
                self._run_node(planned.plan(node, "sfpu"), bank, state)
                continue
            self.config.sentinel.configure_golden(
                self.config,
                self.operation,
                output_format=node.output.data_format,
                set_math_format=False,
            )
            state.output = self.buffers[id(node.output)]
            self._run_node(planned.plan(node, "pack"), bank, state)

    def run(self, blocks) -> None:
        operation, config = self.operation, self.config
        config.sentinel.configure_golden(
            config, operation, output_format=self.output_format
        )
        dest_dtype = format_dict[config.sentinel.golden_math_format]
        relu_configs = {}
        for planned in blocks:
            for bank in planned.bank.bank_assignments():
                state = GoldenState(
                    DestBank(
                        self._dest_size(planned),
                        tile_dimensions(operation.tile_shape),
                        operation.tile_shape.total_num_faces(),
                        dest_dtype,
                        planned.region.block_tiles_x,
                        planned.region.block_tiles_y,
                    ),
                    relu_configs,
                )
                self._run_math(planned, bank, state)
                self._run_pack(planned, bank, state)

        for node in self.pack_nodes:
            result = finalize_output(
                self._output_layout(node), self.buffers[id(node.output)], node.output
            )
            if self.golden_type == GoldenType.L1_GOLDEN:
                node.output.l1_golden = result
            else:
                node.output._master_golden = result
