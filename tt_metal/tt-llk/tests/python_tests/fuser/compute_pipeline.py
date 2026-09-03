# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

import torch

if TYPE_CHECKING:
    from .l1_operation import L1Operation
    from .fuser_config import GlobalConfig

from helpers.llk_params import GoldenType, L1Accumulation, format_dict

from .arch_common import fpu_common, pack_common, unpack_common
from .base_fpu import Fpu
from .base_sfpu import Sfpu
from .base_unpacker import Unpacker
from .block_data import BlockData
from .fpu_node import FpuNode
from .golden_state import (
    DestBank,
    Inputs,
    OperandTiles,
    OutputTiles,
    SourceRegisters,
    tile_dimensions,
)
from .indexing import (
    INDEX_NAMES,
    BlockRegion,
    LoopPlan,
    SlotIndex,
    bind_indices,
    block_regions,
    default_plan,
)
from .pack_node import PackNode
from .sfpu_node import SfpuNode


@dataclass
class PlannedBlock:
    region: BlockRegion
    block: BlockData
    bank: LoopPlan
    plans: Dict[Tuple[int, str], LoopPlan]

    def plan(self, node, role: str) -> LoopPlan:
        return self.plans[(id(node), role)]


class ComputePipeline:
    math_nodes: List[Union[FpuNode, SfpuNode]]
    pack_nodes: List[Union[PackNode, SfpuNode]]

    def __init__(
        self,
        math_nodes: List[Union[FpuNode, SfpuNode]],
        pack_nodes: List[Union[PackNode, SfpuNode]],
    ):
        self.math_nodes = math_nodes
        self.pack_nodes = pack_nodes

    def _get_pack_nodes(self) -> List[PackNode]:
        return [pn for pn in self.pack_nodes if isinstance(pn, PackNode)]

    def get_unpackers(self) -> List["Unpacker"]:
        unpackers: List["Unpacker"] = []

        for operation in self.math_nodes:
            if isinstance(operation, FpuNode) and operation.unpacker is not None:
                unpackers.append(operation.unpacker)

        return unpackers

    def get_math_units(self) -> List[Union["Fpu", "Sfpu"]]:
        math_units = []

        for operation in self.math_nodes:
            if isinstance(operation, FpuNode):
                math_units.append(operation.fpu)
            elif isinstance(operation, SfpuNode):
                math_units.append(operation.sfpu)

        return math_units

    def _plan_node(self, node, role, region, granularity, row_tiles):
        if role in ("unpack", "math"):
            slots = ["in0", "dest"] if node.src_b is None else ["in0", "in1", "dest"]
        elif role == "pack":
            slots = ["dest", "out"]
        elif node.sfpu.input_count == 2:
            slots = ["src0", "src1", "dest"]
        else:
            slots = ["dest"]

        plan = default_plan(region, granularity, slots, row_tiles)
        overrides = {}

        if role == "sfpu":
            overrides["dest"] = SlotIndex(
                base=getattr(node.sfpu, "dst_index_out", None)
                or getattr(node.sfpu, "dest_idx", 0)
            )
            if node.sfpu.input_count == 2:
                overrides["src0"] = SlotIndex(base=node.sfpu.dst_index_in0)
                overrides["src1"] = SlotIndex(base=node.sfpu.dst_index_in1)
        if role in ("unpack", "math") and node.src_b is not None:
            if node.broadcast_tile is not None:
                overrides["in1"] = SlotIndex(base=node.broadcast_tile)
        if role == "math" and getattr(node, "reduce_to_tile", False):
            overrides["dest"] = SlotIndex()
        if role == "pack" and node.pack_l1_accumulation == L1Accumulation.Yes:
            multipliers = {
                var: value
                for var, value in plan.slots["out"].multipliers.items()
                if var.startswith("tile_")
            }
            overrides["out"] = SlotIndex(multipliers=multipliers)

        if overrides:
            plan = replace(plan, slots={**plan.slots, **overrides})
        return plan

    def _planned(
        self, operation: "L1Operation", config: "GlobalConfig"
    ) -> List[PlannedBlock]:
        tile_count_x = (
            operation.max_output_dimensions[1] // operation.tile_shape.total_col_dim()
        )
        tile_count_y = (
            operation.max_output_dimensions[0] // operation.tile_shape.total_row_dim()
        )
        full_x_limit = tile_count_x // operation.block_tiles_x * operation.block_tiles_x
        full_y_limit = tile_count_y // operation.block_tiles_y * operation.block_tiles_y

        row_tiles = dict.fromkeys(("in0", "in1", "out"), tile_count_x)

        planned = []
        for region in block_regions(
            tile_count_x, tile_count_y, operation.block_tiles_x, operation.block_tiles_y
        ):
            block = BlockData(
                block_x=region.x.var if region.x.looped else region.x.origin,
                block_y=region.y.var if region.y.looped else region.y.origin,
                block_tiles_x=region.block_tiles_x,
                block_tiles_y=region.block_tiles_y,
                tile_count_x=tile_count_x,
                tile_count_y=tile_count_y,
                full_x_limit=full_x_limit,
                full_y_limit=full_y_limit,
                tile_id_global="0",
                tile_id_block="0",
            )
            plans: Dict[Tuple[int, str], LoopPlan] = {}
            for node in self.math_nodes:
                if isinstance(node, SfpuNode):
                    plans[(id(node), "sfpu")] = self._plan_node(
                        node, "sfpu", region, node.sfpu.granularity, row_tiles
                    )
                    continue
                if node.unpacker is not None:
                    plans[(id(node), "unpack")] = self._plan_node(
                        node, "unpack", region, node.unpacker.granularity, row_tiles
                    )
                plans[(id(node), "math")] = self._plan_node(
                    node, "math", region, node.fpu.granularity, row_tiles
                )
            for node in self.pack_nodes:
                if isinstance(node, SfpuNode):
                    plans[(id(node), "sfpu")] = self._plan_node(
                        node, "sfpu", region, node.sfpu.granularity, row_tiles
                    )
                    continue
                plans[(id(node), "pack")] = self._plan_node(
                    node, "pack", region, node.packer.granularity, row_tiles
                )
            planned.append(
                PlannedBlock(
                    region=region,
                    block=block,
                    bank=LoopPlan(bank_levels=region.bank_levels),
                    plans=plans,
                )
            )
        return planned

    @staticmethod
    def _emit_calls(planned: PlannedBlock, node, role, bank_constants, emit) -> str:
        names = INDEX_NAMES.get(role, {})

        def render(call):
            declarations, bound = bind_indices(call, names)
            body = emit(bound)
            if not body:
                return ""
            if not declarations:
                return body
            return f"{{\n{declarations}{body}}}\n"

        return planned.plan(node, role).emit_calls(render, bank_constants)

    def _all_same_operand_formats(self, ops: List[FpuNode]) -> bool:
        def signature(op: FpuNode):
            return (
                op.src_a.data_format if op.src_a is not None else None,
                op.src_b.data_format if op.src_b is not None else None,
            )

        return len({signature(op) for op in ops}) <= 1

    def _batch_loop(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        body_fn,
        init_fn=None,
        uninit_fn=None,
    ) -> str:
        code = ""
        for planned in self._planned(operation, config):
            body = planned.bank.emit_banks(
                lambda constants: body_fn(planned, constants)
            )
            if not body:
                continue
            if init_fn is not None:
                code += init_fn(planned.block)
            code += body
            if uninit_fn is not None:
                code += uninit_fn(planned.block)
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

    def unpack_body(self, operation: "L1Operation", config: "GlobalConfig") -> str:
        unpack_ops = [
            cu
            for cu in self.math_nodes
            if isinstance(cu, FpuNode) and cu.unpacker is not None
        ]
        hoist = len(unpack_ops) == 1
        hoist_reconfig = hoist or self._all_same_operand_formats(unpack_ops)

        init_code = ""
        init_code += unpack_common.dvalid_init(config=config, operation=operation)
        init_code += config.sentinel.hw_configure_unpack(config, operation)
        if hoist_reconfig and unpack_ops and not config.skip_unpack_init:
            init_code += config.sentinel.configure_unpack(
                config, operation, unpack_ops[0]
            )
        if hoist and not unpack_ops[0].unpacker.per_block_init:
            init_code += unpack_ops[0].unpack_init(operation, config, None)
        code = self._zone(config, "INIT", init_code)

        code += unpack_common.sync_with_packer(config, operation)

        init_fn = None
        uninit_fn = None
        if hoist and unpack_ops[0].unpacker.per_block_init:
            init_fn = lambda block: unpack_ops[0].unpack_init(operation, config, block)
            uninit_fn = lambda block: unpack_ops[0].unpack_uninit(
                operation, config, block
            )

        def batch_body(planned: PlannedBlock, constants):
            block = planned.block
            body = ""
            for cu in self.math_nodes:
                if not isinstance(cu, FpuNode):
                    continue
                if (
                    not hoist_reconfig
                    and cu.unpacker is not None
                    and not config.skip_unpack_init
                ):
                    body += config.sentinel.configure_unpack(config, operation, cu)
                if not hoist:
                    body += cu.unpack_init(operation, config, block)
                if cu.unpacker is not None:
                    body += self._emit_calls(
                        planned,
                        cu,
                        "unpack",
                        constants,
                        lambda call, cu=cu: cu.unpack_call(
                            operation, config, block, call
                        ),
                    )
                if not hoist:
                    body += cu.unpack_uninit(operation, config, block)
            return body

        code += self._zone_loop(
            config,
            "TILE_LOOP",
            self._batch_loop(operation, config, batch_body, init_fn, uninit_fn),
        )

        uninit_code = ""
        if hoist and not unpack_ops[0].unpacker.per_block_init:
            uninit_code += unpack_ops[0].unpack_uninit(operation, config, None)
        code += self._zone(config, "INIT", uninit_code)

        return code

    def math_body(self, operation: "L1Operation", config: "GlobalConfig") -> str:
        code = f"// Operation {operation.stage_id}: Math Setup\n"
        fpu_ops = [cu for cu in self.math_nodes if isinstance(cu, FpuNode)]
        hoist = len(fpu_ops) == 1
        hoist_reconfig = hoist or self._all_same_operand_formats(fpu_ops)

        init_code = config.sentinel.hw_configure_math(config, operation)
        init_code += fpu_common.math_pack_sync_init(config, operation)
        init_code += fpu_common.math_dest_remap_config(
            any(pn.packer.requires_dest_remap for pn in self._get_pack_nodes())
        )
        if hoist_reconfig and fpu_ops and not config.skip_math_init:
            init_code += config.sentinel.configure_math(config, operation, fpu_ops[0])
        if hoist and not fpu_ops[0].fpu.per_block_init:
            init_code += fpu_ops[0].fpu_init(operation, config, None)
        code += self._zone(config, "INIT", init_code)

        init_fn = None
        uninit_fn = None
        if hoist and fpu_ops[0].fpu.per_block_init:
            init_fn = lambda block: fpu_ops[0].fpu_init(operation, config, block)
            uninit_fn = lambda block: fpu_ops[0].fpu_uninit(operation, config, block)

        def batch_body(planned: PlannedBlock, constants):
            block = planned.block
            body = fpu_common.math_wait_for_dest(config, operation)
            for cu in self.math_nodes:
                if isinstance(cu, FpuNode):
                    if not hoist_reconfig and not config.skip_math_init:
                        body += config.sentinel.configure_math(config, operation, cu)
                    if not hoist:
                        body += cu.fpu_init(operation, config, block)
                    body += self._emit_calls(
                        planned,
                        cu,
                        "math",
                        constants,
                        lambda call, cu=cu: cu.fpu_call(operation, config, block, call),
                    )
                    if not hoist:
                        body += cu.fpu_uninit(operation, config, block)
                elif isinstance(cu, SfpuNode):
                    body += cu.sfpu_init(operation, config, block)
                    body += self._emit_calls(
                        planned,
                        cu,
                        "sfpu",
                        constants,
                        lambda call, cu=cu: cu.sfpu_call(
                            operation, config, block, call
                        ),
                    )
                    body += cu.sfpu_uninit(operation, config, block)
            body += fpu_common.math_dest_section_done(config, operation)
            return body

        code += self._zone_loop(
            config,
            "TILE_LOOP",
            self._batch_loop(operation, config, batch_body, init_fn, uninit_fn),
        )

        uninit_code = ""
        if hoist and not fpu_ops[0].fpu.per_block_init:
            uninit_code += fpu_ops[0].fpu_uninit(operation, config, None)
        code += self._zone(config, "INIT", uninit_code)

        return code

    def _all_same_pack_formats(self) -> bool:
        pack_only = self._get_pack_nodes()
        if len(pack_only) <= 1:
            return True
        first_fmt = pack_only[0].output.data_format
        return all(pn.output.data_format == first_fmt for pn in pack_only[1:])

    def pack_body(self, operation: "L1Operation", config: "GlobalConfig") -> str:
        code = f"// Operation {operation.stage_id}: Packer\n"
        pack_only = self._get_pack_nodes()
        hoist = len(pack_only) == 1 and len(self.pack_nodes) == 1
        hoist_reconfig = hoist or self._all_same_pack_formats()

        init_code = config.sentinel.hw_configure_pack(config, operation, pack_only)
        if hoist_reconfig and pack_only:
            init_code += config.sentinel.configure_pack(config, operation, pack_only[0])
        init_code += pack_common.pack_reduce_mask_config(operation)
        init_code += pack_common.pack_dest_init(config, operation, pack_only[0])
        if hoist and not pack_only[0].packer.per_block_init:
            init_code += pack_only[0].init(operation, config, None)
        code += self._zone(config, "INIT", init_code)

        init_fn = None
        uninit_fn = None
        if hoist and pack_only[0].packer.per_block_init:
            init_fn = lambda block: pack_only[0].init(operation, config, block)
            uninit_fn = lambda block: pack_only[0].uninit(operation, config)

        def batch_body(planned: PlannedBlock, constants):
            block = planned.block
            body = pack_common.packer_wait_for_math(config, operation)
            if not hoist_reconfig:
                config.sentinel.reset_pack_formats()
            prev_was_pack = False
            for pack_node in self.pack_nodes:
                if isinstance(pack_node, SfpuNode):
                    if prev_was_pack:
                        body += "TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::PACK);\n"
                    body += pack_node.sfpu_init(operation, config, block)
                    body += self._emit_calls(
                        planned,
                        pack_node,
                        "sfpu",
                        constants,
                        lambda call, node=pack_node: node.sfpu_call(
                            operation, config, block, call
                        ),
                    )
                    body += pack_node.sfpu_uninit(operation, config, block)
                    prev_was_pack = False
                elif isinstance(pack_node, PackNode):
                    if not hoist_reconfig:
                        body += config.sentinel.configure_pack(
                            config, operation, pack_node
                        )
                    if not hoist:
                        body += pack_node.init(operation, config, block)
                    body += self._emit_calls(
                        planned,
                        pack_node,
                        "pack",
                        constants,
                        lambda call, node=pack_node: node.pack_call(
                            operation, config, block, call
                        ),
                    )
                    if not hoist:
                        body += pack_node.uninit(operation, config)
                    prev_was_pack = True
            body += pack_common.packer_dest_section_done(config, operation)
            return body

        code += self._zone_loop(
            config,
            "TILE_LOOP",
            self._batch_loop(operation, config, batch_body, init_fn, uninit_fn),
        )

        uninit_code = pack_common.packer_sync_with_unpacker(config, operation)
        if hoist and not pack_only[0].packer.per_block_init:
            uninit_code += pack_only[0].uninit(operation, config)
        uninit_code += pack_common.pack_reduce_mask_clear(operation)
        code += self._zone(config, "INIT", uninit_code)

        return code

    def _supports_per_call(self) -> bool:
        for node in self.math_nodes:
            if isinstance(node, SfpuNode):
                if not node.sfpu.supports_per_call(node):
                    return False
                continue
            if node.unpacker is not None and not node.unpacker.supports_per_call(node):
                return False
            if not node.fpu.supports_per_call(node):
                return False
        for node in self.pack_nodes:
            if isinstance(node, SfpuNode):
                if not node.sfpu.supports_per_call(node):
                    return False
            elif not node.packer.supports_per_call(node):
                return False
        return True

    @staticmethod
    def _golden_source(operand, golden_type):
        if operand is None:
            return None
        if golden_type == GoldenType.L1_GOLDEN:
            return operand.raw_data
        return operand.master_golden

    @staticmethod
    def _store_golden(operand, result, golden_type):
        if golden_type == GoldenType.L1_GOLDEN:
            operand.l1_golden = result
        else:
            operand._master_golden = result

    def _per_call_golden(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        golden_type: GoldenType,
    ):
        tile_dims = tile_dimensions(operation.tile_shape)
        pack_nodes = self._get_pack_nodes()
        outputs = {id(node): OutputTiles(node.output) for node in pack_nodes}
        config.sentinel.configure_golden(
            config, operation, output_format=pack_nodes[0].output.data_format
        )
        dest_dtype = format_dict[config.sentinel.golden_math_format]
        views = {}
        for node in self.math_nodes:
            if isinstance(node, SfpuNode):
                continue
            for slot, operand in (("a", node.src_a), ("b", node.src_b)):
                if operand is None or (id(node), slot) in views:
                    continue
                views[(id(node), slot)] = OperandTiles(
                    operand, self._golden_source(operand, golden_type)
                )

        for planned in self._planned(operation, config):
            for bank in planned.bank.bank_assignments():
                dest = DestBank(planned.region.block_tiles, tile_dims, dest_dtype)
                for node in self.math_nodes:
                    config.sentinel.configure_golden(config, operation, node)
                    if isinstance(node, SfpuNode):
                        for call in planned.plan(node, "sfpu").calls(bank):
                            node.sfpu.golden_call(call, dest, node, operation, config)
                        continue
                    srcs = SourceRegisters()
                    if node.unpacker is not None:
                        inputs = Inputs(
                            views.get((id(node), "a")), views.get((id(node), "b"))
                        )
                        for call in planned.plan(node, "unpack").calls(bank):
                            node.unpacker.golden_call(
                                call, inputs, srcs, node, operation, config
                            )
                    for call in planned.plan(node, "math").calls(bank):
                        node.fpu.golden_call(call, srcs, dest, node, operation, config)
                for node in self.pack_nodes:
                    if isinstance(node, SfpuNode):
                        for call in planned.plan(node, "sfpu").calls(bank):
                            node.sfpu.golden_call(call, dest, node, operation, config)
                        continue
                    config.sentinel.configure_golden(
                        config, operation, output_format=node.output.data_format
                    )
                    for call in planned.plan(node, "pack").calls(bank):
                        node.packer.golden_call(
                            call, dest, outputs[id(node)], node, operation, config
                        )

        for node in pack_nodes:
            self._store_golden(node.output, outputs[id(node)].finish(), golden_type)

    def golden(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        golden_type: GoldenType,
    ):
        if self._supports_per_call():
            return self._per_call_golden(operation, config, golden_type)

        first_fpu = next(
            (
                op
                for op in self.math_nodes
                if isinstance(op, FpuNode) and op.src_a is not None
            ),
            None,
        )
        if first_fpu is not None:
            tensor_a = torch.zeros(first_fpu.src_a.dimensions)
            tensor_b = torch.zeros(
                first_fpu.src_b.dimensions
                if first_fpu.src_b is not None
                else first_fpu.src_a.dimensions
            )
        else:
            tensor_a = torch.zeros(operation.max_output_dimensions)
            tensor_b = torch.zeros(operation.max_output_dimensions)
        tensor_dst = torch.zeros(operation.max_output_dimensions)
        for op in self.math_nodes:
            config.sentinel.configure_golden(config, operation, op)
            if isinstance(op, FpuNode) and op.src_a is not None:
                input_tensor_a = (
                    op.src_a.raw_data
                    if golden_type == GoldenType.L1_GOLDEN
                    else op.src_a.master_golden
                )
                input_tensor_b = (
                    (
                        op.src_b.raw_data
                        if golden_type == GoldenType.L1_GOLDEN
                        else op.src_b.master_golden
                    )
                    if op.src_b is not None
                    else None
                )
            else:
                input_tensor_a = None
                input_tensor_b = None
            tensor_a, tensor_b, tensor_dst = op.golden(
                input_tensor_a,
                input_tensor_b,
                tensor_a,
                tensor_b,
                tensor_dst,
                operation,
                config,
            )

        for pack_node in self.pack_nodes:
            if isinstance(pack_node, SfpuNode):
                tensor_a, tensor_b, tensor_dst = pack_node.golden(
                    None, None, tensor_a, tensor_b, tensor_dst, operation, config
                )
                continue

            config.sentinel.configure_golden(
                config, operation, output_format=pack_node.output.data_format
            )

            dimensions = pack_node.output.dimensions
            cropped = tensor_dst.reshape(operation.max_output_dimensions)[
                : dimensions[0], : dimensions[1]
            ]
            result = pack_node.golden(cropped, operation, config)

            if golden_type == GoldenType.L1_GOLDEN:
                pack_node.output.l1_golden = result
            else:
                pack_node.output._master_golden = result

    def __str__(self):
        result = "Math:"
        for op in self.math_nodes:
            result += "\n    "
            result += op.__str__()
        result += "\n  Pack:"
        for pn in self.pack_nodes:
            result += "\n    "
            if isinstance(pn, PackNode):
                result += pn.output.__str__()
            else:
                result += str(pn)
        return result
