# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

if TYPE_CHECKING:
    from .l1_operation import L1Operation
    from .fuser_config import GlobalConfig

from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    GoldenType,
    L1Accumulation,
    format_dict,
)

from .arch_common import fpu_common, pack_common, unpack_common
from .base_fpu import Fpu
from .base_sfpu import Sfpu
from .base_unpacker import Unpacker
from .block_data import BlockData
from .fpu_node import FpuNode
from .golden.state import (
    DestBank,
    GoldenState,
    Inputs,
    OperandTiles,
    OutputLayout,
    finalize_output,
    tile_dimensions,
)
from .indexing import (
    BANK_VAR,
    DEST_SLOTS,
    INDEX_NAMES,
    BlockRegion,
    InvocationGranularity,
    Level,
    LoopPlan,
    SlotIndex,
    _is_template,
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

        self.custom_op = any(
            getattr(node, "loop_spec", None) is not None
            for node in math_nodes + pack_nodes
        )

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

    @staticmethod
    def _apply_loop_spec(plan, loop_spec):
        if loop_spec is None:
            return plan
        slot_overrides = loop_spec.slot_overrides()
        if not slot_overrides:
            return plan

        slots = dict(plan.slots)
        origins = dict(plan.origins)
        declared = {lv.var for lv in plan.bank_levels + plan.call_levels}
        template_lens = [
            len(v)
            for s, v in slot_overrides.items()
            if s in DEST_SLOTS and isinstance(v, list)
        ]
        blocks_per_bank = template_lens[0] if template_lens else 1
        for slot, value in slot_overrides.items():
            if slot not in slots:
                continue
            index = slots[slot]
            if isinstance(value, list):
                base_var = f"{slot}_base"
                origins[base_var] = tuple(value)
                slots[slot] = replace(
                    index, multipliers={**index.multipliers, base_var: 1}
                )
            elif isinstance(value, int):
                slots[slot] = replace(index, base=index.base + value)
            else:
                unknown = sorted(set(value.multipliers) - declared)
                if unknown:
                    raise ValueError(
                        f"loop slot '{slot}' references undeclared loop vars: {unknown}"
                    )
                slots[slot] = replace(
                    index,
                    base=index.base + value.base,
                    multipliers={**index.multipliers, **value.multipliers},
                )
        return replace(
            plan, slots=slots, origins=origins, blocks_per_bank=blocks_per_bank
        )

    def _plan_node(self, node, role, region, granularity, row_tiles):
        if role in ("unpack", "math"):
            slots = ["in0", "dest"] if node.src_b is None else ["in0", "in1", "dest"]
        elif role == "pack":
            slots = ["dest", "out"]
        elif node.sfpu.input_count == 2:
            slots = ["src0", "src1", "dest"]
        else:
            slots = ["dest"]

        nx = node.block_tiles_x or region.block_tiles_x
        ny = node.block_tiles_y or region.block_tiles_y
        if region.block_tiles_x % nx == 0 and region.block_tiles_y % ny == 0:
            plan = default_plan(region, granularity, slots, row_tiles, nx, ny)
        else:
            plan = default_plan(region, granularity, slots, row_tiles)
        overrides = {}

        def shift(slot, base):
            return replace(plan.slots[slot], base=plan.slots[slot].base + base)

        if role == "sfpu":
            overrides["dest"] = shift(
                "dest",
                getattr(node.sfpu, "dst_index_out", None)
                or getattr(node.sfpu, "dest_idx", 0),
            )
            if node.sfpu.input_count == 2:
                overrides["src0"] = shift("src0", node.sfpu.dst_index_in0)
                overrides["src1"] = shift("src1", node.sfpu.dst_index_in1)
        if role == "pack" and node.pack_l1_accumulation == L1Accumulation.Yes:
            multipliers = {
                var: value
                for var, value in plan.slots["out"].multipliers.items()
                if var.startswith("tile_")
            }
            overrides["out"] = SlotIndex(multipliers=multipliers)

        if overrides:
            plan = replace(plan, slots={**plan.slots, **overrides})

        return self._apply_loop_spec(plan, node.loop_spec)

    @staticmethod
    def _dest_capacity(operation, config) -> int:
        faces = 32 if operation.dest_sync == DestSync.Half else 64
        if config.dest_acc == DestAccumulation.Yes:
            faces //= 2
        return faces // operation.tile_shape.total_num_faces()

    def _planned(
        self, operation: "L1Operation", config: "GlobalConfig"
    ) -> List[PlannedBlock]:
        if self.custom_op:
            # A custom op addresses inputs/outputs through its index arrays, not
            # the output grid, so it is a single dest-bank region; chunking (not
            # block_regions) drives the bank iterations.
            tile_count_x = operation.block_tiles_x
            tile_count_y = operation.block_tiles_y
        else:
            tile_count_x = (
                operation.max_output_dimensions[1]
                // operation.tile_shape.total_col_dim()
            )
            tile_count_y = (
                operation.max_output_dimensions[0]
                // operation.tile_shape.total_row_dim()
            )
        row_tiles = dict.fromkeys(("in0", "in1", "out"), tile_count_x)

        planned = []
        for region in block_regions(
            tile_count_x, tile_count_y, operation.block_tiles_x, operation.block_tiles_y
        ):
            block = BlockData(
                block_origin_x=region.x.var if region.x.looped else region.x.origin,
                block_origin_y=region.y.var if region.y.looped else region.y.origin,
                block_cols=region.block_tiles_x,
                block_rows=region.block_tiles_y,
            )
            plans: Dict[Tuple[int, str], LoopPlan] = {}

            def add_plan(node, role, unit):
                if unit.granularity == InvocationGranularity.NONE:
                    raise ValueError(f"{type(unit).__name__} has no granularity set")
                plans[(id(node), role)] = self._plan_node(
                    node, role, region, unit.granularity, row_tiles
                )

            for node in self.math_nodes:
                if isinstance(node, SfpuNode):
                    add_plan(node, "sfpu", node.sfpu)
                else:
                    if node.unpacker is not None:
                        if node.unpacker.granularity != node.fpu.granularity:
                            raise ValueError(
                                "unpacker and fpu granularity must match, got "
                                f"{node.unpacker.granularity} and {node.fpu.granularity}"
                            )
                        add_plan(node, "unpack", node.unpacker)
                    add_plan(node, "math", node.fpu)
            for node in self.pack_nodes:
                if isinstance(node, SfpuNode):
                    add_plan(node, "sfpu", node.sfpu)
                else:
                    add_plan(node, "pack", node.packer)
            bank_levels = region.bank_levels
            if self.custom_op:
                bank_levels = (Level(BANK_VAR, self._num_banks(plans), 1),)
            planned.append(
                PlannedBlock(
                    region=region,
                    block=block,
                    bank=LoopPlan(bank_levels=bank_levels),
                    plans=plans,
                )
            )
            self._check_dest_reads(planned[-1])
        return planned

    def _dest_tiles(self, planned, node, role, slots) -> set:
        plan = planned.plan(node, role)
        block = role == "math" and not plan.call_levels and not plan.fanout_levels
        bx = (node.block_tiles_x or 1) if block else 1
        by = (node.block_tiles_y or 1) if block else 1
        offsets = [tx + ty * bx for ty in range(by) for tx in range(bx)]
        tiles = set()
        for bank in planned.bank.bank_assignments():
            for call in plan.calls(bank):
                for tile in (call,) + call.tiles:
                    for slot in slots:
                        value = getattr(tile, slot)
                        if value is not None:
                            tiles.update(value + off for off in offsets)
        return tiles

    def _check_dest_reads(self, planned: "PlannedBlock") -> None:
        valid: set = set()

        def check(node, role, reads):
            missing = self._dest_tiles(planned, node, role, reads) - valid
            if missing:
                raise ValueError(
                    f"{type(node).__name__} reads dest tiles {sorted(missing)} "
                    "that no earlier node writes"
                )

        def sfpu_reads(node):
            return ("src0", "src1") if node.sfpu.input_count == 2 else ("dest",)

        for node in self.math_nodes:
            if isinstance(node, SfpuNode):
                check(node, "sfpu", sfpu_reads(node))
                valid |= self._dest_tiles(planned, node, "sfpu", ("dest",))
            else:
                valid |= self._dest_tiles(planned, node, "math", ("dest",))
        for node in self.pack_nodes:
            if isinstance(node, SfpuNode):
                check(node, "sfpu", sfpu_reads(node))
                valid |= self._dest_tiles(planned, node, "sfpu", ("dest",))
            else:
                check(node, "pack", ("dest",))

    @staticmethod
    def _num_banks(plans) -> int:
        counts = set()
        for plan in plans.values():
            walked = [v for k, v in plan.origins.items() if not _is_template(k)]
            if walked:
                counts.add(-(-len(walked[0]) // plan.blocks_per_bank))
        if not counts:
            return 1
        if len(counts) != 1:
            raise ValueError(
                f"math and pack nodes disagree on dest-bank count: {sorted(counts)}"
            )
        return counts.pop()

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

    def _output_layout(self, node: PackNode) -> OutputLayout:
        if node.packer.output_layout != OutputLayout.ROW_MAJOR:
            return node.packer.output_layout
        for math_node in self.math_nodes:
            if (
                isinstance(math_node, FpuNode)
                and math_node.unpacker is not None
                and math_node.unpacker.output_layout != OutputLayout.ROW_MAJOR
            ):
                return math_node.unpacker.output_layout
        return OutputLayout.ROW_MAJOR

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

    def golden(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        golden_type: GoldenType,
    ):
        if config.perf_run_type is not None:
            raise ValueError(
                f"golden() needs a functional run, got perf_run_type={config.perf_run_type}"
            )
        tile_dims = tile_dimensions(operation.tile_shape)
        pack_nodes = self._get_pack_nodes()
        layouts = {id(node): self._output_layout(node) for node in pack_nodes}
        buffers = {id(node): {} for node in pack_nodes}
        relu_configs = {}
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

        dest_tiles = self._dest_capacity(operation, config) if self.custom_op else None
        for planned in self._planned(operation, config):
            for bank in planned.bank.bank_assignments():
                size = (
                    dest_tiles if dest_tiles is not None else planned.region.block_tiles
                )
                state = GoldenState(
                    DestBank(
                        size,
                        tile_dims,
                        operation.tile_shape.total_num_faces(),
                        dest_dtype,
                        planned.region.block_tiles_x,
                        planned.region.block_tiles_y,
                    ),
                    relu_configs,
                )

                def run(node, role, golden_fn):
                    for call in planned.plan(node, role).calls(bank):
                        golden_fn(call, state, node, operation, config)

                for node in self.math_nodes:
                    config.sentinel.configure_golden(
                        config,
                        operation,
                        node,
                        output_format=pack_nodes[0].output.data_format,
                    )
                    if isinstance(node, SfpuNode):
                        run(node, "sfpu", node.sfpu.golden_fn)
                        continue
                    state.begin_fpu(
                        Inputs(
                            views.get((id(node), "a")),
                            views.get((id(node), "b")),
                            planned.region.block_tiles_x,
                            planned.region.block_tiles_y,
                        )
                    )
                    if node.unpacker is not None:
                        run(node, "unpack", node.unpacker.golden_fn)
                    run(node, "math", node.fpu.golden_fn)
                for node in self.pack_nodes:
                    if isinstance(node, SfpuNode):
                        run(node, "sfpu", node.sfpu.golden_fn)
                        continue
                    config.sentinel.configure_golden(
                        config,
                        operation,
                        output_format=node.output.data_format,
                        set_math_format=False,
                    )
                    state.output = buffers[id(node)]
                    run(node, "pack", node.packer.golden_fn)

        for node in pack_nodes:
            result = finalize_output(layouts[id(node)], buffers[id(node)], node.output)
            self._store_golden(node.output, result, golden_type)

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
