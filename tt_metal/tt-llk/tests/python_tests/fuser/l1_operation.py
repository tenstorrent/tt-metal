# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from helpers.llk_params import (
    DestSync,
    GoldenType,
    ReduceDimension,
    Tilize,
)
from helpers.tile_shape import TileShape

from .arch_common import fpu_common, pack_common, unpack_common
from .base_fpu import Fpu
from .base_sfpu import Sfpu
from .fpu_node import FpuNode
from .golden.executor import GoldenExecutor
from .pack_node import PackNode
from .pipeline_plan import PlannedBlock
from .sfpu_node import SfpuNode

if TYPE_CHECKING:
    from .fuser_config import GlobalConfig


@dataclass
class L1Operation:
    math_nodes: List[Union[FpuNode, SfpuNode]]
    pack_nodes: List[Union[PackNode, SfpuNode]]
    max_output_dimensions: Tuple[int, int]
    tile_shape: TileShape
    stage_id: int = field(default=0, init=False)
    needs_pack_sync: bool = field(default=False, init=False)
    has_pack_consumer: bool = field(default=False, init=False)
    dest_sync: DestSync = DestSync.Half
    bh_tilize: Tilize = Tilize.No

    def __post_init__(self):
        nodes = self.math_nodes + self.pack_nodes
        self.block_tiles_x = max(node.block_tiles_x for node in nodes)
        self.block_tiles_y = max(node.block_tiles_y for node in nodes)

    @property
    def reduce_dim(self) -> Optional[ReduceDimension]:
        for node in self.math_nodes:
            if isinstance(node, FpuNode) and hasattr(node.fpu, "reduce_dim"):
                return node.fpu.reduce_dim
        return None

    @property
    def custom_op(self) -> bool:
        return any(
            node.index_spec is not None for node in self.math_nodes + self.pack_nodes
        )

    def _get_pack_nodes(self) -> List[PackNode]:
        return [pn for pn in self.pack_nodes if isinstance(pn, PackNode)]

    def get_math_units(self) -> List[Union[Fpu, Sfpu]]:
        math_units = []
        for node in self.math_nodes:
            if isinstance(node, FpuNode):
                math_units.append(node.fpu)
            elif isinstance(node, SfpuNode):
                math_units.append(node.sfpu)
        return math_units

    @property
    def first_unpack(self):
        return next(
            (
                node
                for node in self.math_nodes
                if isinstance(node, FpuNode) and node.src_a is not None
            ),
            None,
        )

    @property
    def first_math(self):
        return self.first_unpack or self.math_nodes[0]

    @staticmethod
    def _math_reconfig(config, old, new):
        if config.skip_math_init or old == new:
            return ""
        return fpu_common.configure_math(config.dest_acc.cpp_enum_value, old, new)

    def _batch_loop(self, blocks, body_fn, init_fn=None, uninit_fn=None) -> str:
        code = ""
        for planned in blocks:
            body = planned.bank.emit_banks(
                lambda constants: body_fn(planned, constants)
            )
            if not body:
                continue
            if init_fn is not None:
                code += init_fn(planned)
            code += body
            if uninit_fn is not None:
                code += uninit_fn(planned)
        return code

    def _zone(self, config: "GlobalConfig", name: str, body: str) -> str:
        if not config.profiler_enabled:
            return body
        code = "{\n"
        code += f'ZONE_SCOPED("{name}")\n'
        code += body
        code += "PROFILER_SYNC();\n"
        code += "}\n"
        return code

    def _zone_loop(self, config: "GlobalConfig", name: str, body: str) -> str:
        if not config.profiler_enabled:
            return body
        code = "{\n"
        code += f'ZONE_SCOPED("{name}")\n'
        code += f"for(int loop = 0; loop < {config.loop_factor}; loop++)\n"
        code += "{\n"
        code += body
        code += "}\n"
        code += "PROFILER_SYNC();\n"
        code += "}\n"
        return code

    def unpack(
        self, config: "GlobalConfig", blocks: List[PlannedBlock], previous=None
    ) -> str:
        unpack_ops = [
            cu
            for cu in self.math_nodes
            if isinstance(cu, FpuNode) and cu.unpacker is not None
        ]
        hoist = len(unpack_ops) == 1
        first = self.first_unpack

        init_code = ""
        init_code += unpack_common.dvalid_init(config=config, operation=self)
        if first is not None:
            init_code += first.configure_unpack(config, previous)
        init_code += unpack_common.unpack_to_dest_sync_init(config, self)
        if hoist and not unpack_ops[0].unpacker.per_block_init:
            init_code += unpack_ops[0].unpack_init(self, config, None)
        code = self._zone(config, "INIT", init_code)

        code += unpack_common.sync_with_packer(config, self)

        init_fn = None
        uninit_fn = None
        if hoist and unpack_ops[0].unpacker.per_block_init:
            init_fn = lambda planned: unpack_ops[0].unpack_init(
                self, config, planned.block_for(unpack_ops[0])
            )
            uninit_fn = lambda planned: unpack_ops[0].unpack_uninit(
                self, config, planned.block_for(unpack_ops[0])
            )

        def batch_body(planned: PlannedBlock, constants):
            body = unpack_common.unpack_to_dest_wait_for_dest_available(config)
            after_commit = ""
            previous = first
            for cu in self.math_nodes:
                if not isinstance(cu, FpuNode):
                    continue
                block = planned.block_for(cu)
                node_body = ""
                if cu.src_a is not None:
                    node_body += cu.configure_unpack(config, previous)
                    previous = cu
                if not hoist:
                    node_body += cu.unpack_init(self, config, block)
                if cu.unpacker is not None:
                    node_body += planned.plan(cu, "unpack").emit_calls(
                        constants,
                        partial(cu.unpack_call, self, config, block),
                    )
                if not hoist:
                    node_body += cu.unpack_uninit(self, config, block)
                if (
                    config.uses_unpack_to_dest_sync
                    and cu.unpacker is not None
                    and cu.unpacker.emits_dummy_srcb_valid
                ):
                    after_commit += node_body
                else:
                    body += node_body
            if first is not None:
                body += first.configure_unpack(config, previous)
            body += unpack_common.unpack_to_dest_section_done(config, self)
            return body + after_commit

        code += self._zone_loop(
            config,
            "TILE_LOOP",
            self._batch_loop(blocks, batch_body, init_fn, uninit_fn),
        )

        uninit_code = ""
        if hoist and not unpack_ops[0].unpacker.per_block_init:
            uninit_code += unpack_ops[0].unpack_uninit(self, config, None)
        code += self._zone(config, "UNINIT", uninit_code)

        return code

    def do_math(
        self, config: "GlobalConfig", blocks: List[PlannedBlock], previous=None
    ) -> str:
        code = f"// Operation {self.stage_id}: Math Setup\n"
        fpu_ops = [cu for cu in self.math_nodes if isinstance(cu, FpuNode)]
        hoist = len(fpu_ops) == 1
        first = self.first_math
        init_code = (
            fpu_common.hw_configure_math(
                config.dest_acc.cpp_enum_value, first.alu_formats(config)
            )
            if previous is None
            else self._math_reconfig(
                config, previous.alu_formats(config), first.alu_formats(config)
            )
        )
        init_code += fpu_common.math_pack_sync_init(config, self)
        init_code += fpu_common.math_dest_remap_config(
            any(pn.packer.requires_dest_remap for pn in self._get_pack_nodes())
        )
        if hoist and not fpu_ops[0].fpu.per_block_init:
            init_code += fpu_ops[0].fpu_init(self, config, None)
        code += self._zone(config, "INIT", init_code)

        init_fn = None
        uninit_fn = None
        if hoist and fpu_ops[0].fpu.per_block_init:
            init_fn = lambda planned: fpu_ops[0].fpu_init(
                self, config, planned.block_for(fpu_ops[0])
            )
            uninit_fn = lambda planned: fpu_ops[0].fpu_uninit(
                self, config, planned.block_for(fpu_ops[0])
            )

        def batch_body(planned: PlannedBlock, constants):
            body = fpu_common.math_wait_for_dest(config, self)
            alu_formats = first.alu_formats(config)
            for cu in self.math_nodes:
                block = planned.block_for(cu)
                if isinstance(cu, FpuNode) or not config.dest_acc.value:
                    formats = cu.alu_formats(config)
                    body += self._math_reconfig(config, alu_formats, formats)
                    alu_formats = formats
                if isinstance(cu, FpuNode):
                    if not hoist:
                        body += cu.fpu_init(self, config, block)
                    body += planned.plan(cu, "math").emit_calls(
                        constants,
                        partial(cu.fpu_call, self, config, block),
                    )
                    if not hoist:
                        body += cu.fpu_uninit(self, config, block)
                elif isinstance(cu, SfpuNode):
                    body += cu.sfpu_init(self, config, block)
                    body += planned.plan(cu, "sfpu").emit_calls(
                        constants,
                        partial(cu.sfpu_call, self, config, block),
                    )
                    body += cu.sfpu_uninit(self, config, block)
            body += self._math_reconfig(config, alu_formats, first.alu_formats(config))
            body += fpu_common.math_dest_section_done(config, self)
            return body

        code += self._zone_loop(
            config,
            "TILE_LOOP",
            self._batch_loop(blocks, batch_body, init_fn, uninit_fn),
        )

        uninit_code = ""
        if hoist and not fpu_ops[0].fpu.per_block_init:
            uninit_code += fpu_ops[0].fpu_uninit(self, config, None)
        code += self._zone(config, "UNINIT", uninit_code)

        return code

    def pack(
        self, config: "GlobalConfig", blocks: List[PlannedBlock], previous=None
    ) -> str:
        code = f"// Operation {self.stage_id}: Packer\n"
        pack_only = self._get_pack_nodes()
        hoist = len(pack_only) == 1 and len(self.pack_nodes) == 1
        first = pack_only[0]
        init_code = first.configure_pack(config, self, previous)
        init_code += pack_common.pack_reduce_mask_config(self)
        init_code += pack_common.pack_dest_init(config, self, first)
        if hoist and not first.packer.per_block_init:
            init_code += first.init(self, config, None)
        code += self._zone(config, "INIT", init_code)

        init_fn = None
        uninit_fn = None
        if hoist and first.packer.per_block_init:
            init_fn = lambda planned: first.init(self, config, planned.block_for(first))
            uninit_fn = lambda planned: first.uninit(self, config)

        def batch_body(planned: PlannedBlock, constants):
            body = pack_common.packer_wait_for_math(config, self)
            previous = first
            prev_was_pack = False
            for pack_node in self.pack_nodes:
                block = planned.block_for(pack_node)
                if isinstance(pack_node, SfpuNode):
                    if prev_was_pack:
                        body += "TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::PACK);\n"
                    body += pack_node.sfpu_init(self, config, block)
                    body += planned.plan(pack_node, "sfpu").emit_calls(
                        constants,
                        partial(pack_node.sfpu_call, self, config, block),
                    )
                    body += pack_node.sfpu_uninit(self, config, block)
                    prev_was_pack = False
                elif isinstance(pack_node, PackNode):
                    body += pack_node.configure_pack(config, self, previous)
                    previous = pack_node
                    if not hoist:
                        body += pack_node.init(self, config, block)
                    body += planned.plan(pack_node, "pack").emit_calls(
                        constants,
                        partial(pack_node.pack_call, self, config, block),
                    )
                    if not hoist:
                        body += pack_node.uninit(self, config)
                    prev_was_pack = True
            body += first.configure_pack(config, self, previous)
            body += pack_common.packer_dest_section_done(config, self)
            return body

        code += self._zone_loop(
            config,
            "TILE_LOOP",
            self._batch_loop(blocks, batch_body, init_fn, uninit_fn),
        )

        uninit_code = pack_common.packer_sync_with_unpacker(config, self)
        if hoist and not first.packer.per_block_init:
            uninit_code += first.uninit(self, config)
        uninit_code += pack_common.pack_reduce_mask_clear(self)
        code += self._zone(config, "UNINIT", uninit_code)

        return code

    def golden(self, config: "GlobalConfig", blocks: List[PlannedBlock]):
        for golden_type in (GoldenType.L1_GOLDEN, GoldenType.MASTER_GOLDEN):
            GoldenExecutor(self, config, golden_type).run(blocks)

    def __str__(self):
        result = (
            f"\n{'=' * 60}\n" f"Operation {self.stage_id}\n" f"{'=' * 60}\n" "  Math:"
        )
        for node in self.math_nodes:
            result += f"\n    {node}"
        result += "\n  Pack:"
        for node in self.pack_nodes:
            result += f"\n    {node.output if isinstance(node, PackNode) else node}"
        result += f"\n  Dest Sync: {self.dest_sync}\n"
        return result
