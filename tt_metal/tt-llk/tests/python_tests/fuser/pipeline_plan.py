# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from itertools import chain
from typing import TYPE_CHECKING, List, Tuple, Union

from helpers.llk_params import L1Accumulation

from .base_fpu import Fpu
from .base_packer import Packer
from .base_sfpu import Sfpu
from .base_unpacker import Unpacker
from .block_data import BlockData
from .fpu_node import FpuNode
from .indexing import (
    BANK_VAR,
    DEST_SLOTS,
    INDEX_NAMES,
    BlockRegion,
    InvocationGranularity,
    Level,
    LoopPlan,
    SlotIndex,
    bind_indices,
    block_regions,
    default_plan,
)
from .pack_node import PackNode
from .sfpu_node import SfpuNode

if TYPE_CHECKING:
    from .l1_operation import L1Operation
    from .validator import LoopSchema

Node = Union[FpuNode, SfpuNode, PackNode]
Unit = Union[Fpu, Sfpu, Unpacker, Packer]


@dataclass(frozen=True)
class PlannedNode:
    node: Node
    role: str
    unit: Unit
    block: BlockData
    loop: LoopPlan

    def emit_calls(self, constants, emit) -> str:
        def render(call):
            declarations, bound = bind_indices(call, INDEX_NAMES.get(self.role, {}))
            body = emit(bound)
            if not body or not declarations:
                return body
            return f"{{\n{declarations}{body}}}\n"

        return self.loop.emit_calls(render, constants)

    def dest_tiles(self, banks, slots) -> set:
        is_block = (
            self.role == "math"
            and not self.loop.call_levels
            and not self.loop.fanout_levels
        )
        size = self.block.block_cols * self.block.block_rows if is_block else 1
        calls = chain.from_iterable(self.loop.calls(bank) for bank in banks)
        tiles = chain.from_iterable((call,) + call.tiles for call in calls)
        result = set()
        for tile in tiles:
            for slot in slots:
                value = getattr(tile, slot)
                if value is not None:
                    result.update(range(value, value + size))
        return result


@dataclass(frozen=True)
class PlannedBlock:
    region: BlockRegion
    bank: LoopPlan
    nodes: Tuple[PlannedNode, ...]

    def plan(self, node: Node, role: str) -> PlannedNode:
        return next(
            planned
            for planned in self.nodes
            if planned.node is node and planned.role == role
        )

    def block_for(self, node: Node) -> BlockData:
        cols, rows = self.region.node_shape(node.block_tiles_x, node.block_tiles_y)
        return BlockData(
            block_origin_x=(
                self.region.x.var if self.region.x.looped else self.region.x.origin
            ),
            block_origin_y=(
                self.region.y.var if self.region.y.looped else self.region.y.origin
            ),
            block_cols=cols,
            block_rows=rows,
        )

    def check_dest_reads(self) -> None:
        valid = set()
        banks = self.bank.bank_assignments()
        for planned in self.nodes:
            if planned.role == "unpack":
                continue
            reads = ()
            if planned.role == "pack":
                reads = ("dest",)
            elif planned.role == "sfpu":
                reads = ("src0", "src1") if planned.unit.input_count == 2 else ("dest",)
            missing = planned.dest_tiles(banks, reads) - valid
            if missing:
                raise ValueError(
                    f"{type(planned.node).__name__} reads dest tiles {sorted(missing)} "
                    "that no earlier node writes"
                )
            if planned.role != "pack":
                valid |= planned.dest_tiles(banks, ("dest",))


def apply_loop_spec(plan: LoopPlan, loop_spec: "LoopSchema") -> LoopPlan:
    if loop_spec is None:
        return plan
    slot_overrides = loop_spec.slot_overrides()
    if not slot_overrides:
        return plan

    slots = dict(plan.slots)
    origins = dict(plan.origins)
    declared = {level.var for level in plan.bank_levels + plan.call_levels}
    template_lengths = [
        len(value)
        for slot, value in slot_overrides.items()
        if slot in DEST_SLOTS and isinstance(value, list)
    ]
    for slot, value in slot_overrides.items():
        if slot not in slots:
            continue
        index = slots[slot]
        if isinstance(value, list):
            base_var = f"{slot}_base"
            origins[base_var] = tuple(value)
            slots[slot] = replace(index, multipliers={**index.multipliers, base_var: 1})
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
        plan,
        slots=slots,
        origins=origins,
        blocks_per_bank=template_lengths[0] if template_lengths else 1,
    )


def _node_units(math_nodes, pack_nodes):
    for node in math_nodes:
        if isinstance(node, SfpuNode):
            yield node, "sfpu", node.sfpu
            continue
        if node.unpacker is not None:
            if node.unpacker.granularity != node.fpu.granularity:
                raise ValueError(
                    "unpacker and fpu granularity must match, got "
                    f"{node.unpacker.granularity} and {node.fpu.granularity}"
                )
            yield node, "unpack", node.unpacker
        yield node, "math", node.fpu
    for node in pack_nodes:
        if isinstance(node, SfpuNode):
            yield node, "sfpu", node.sfpu
        else:
            yield node, "pack", node.packer


def _plan_node(
    operation, planned: PlannedBlock, node: Node, role: str, unit: Unit
) -> PlannedNode:
    if unit.granularity == InvocationGranularity.NONE:
        raise ValueError(f"{type(unit).__name__} has no granularity set")

    row_tiles = {}
    if role in ("unpack", "math"):
        row_tiles = {
            slot: operand.tile_count_x
            for slot, operand in (("in0", node.src_a), ("in1", node.src_b))
            if operand is not None
        }
        slots = [*row_tiles, "dest"]
        if unit.granularity == InvocationGranularity.BLOCK:
            row_tiles["in0"] = (
                operation.max_output_dimensions[1]
                // node.src_b.tile_shape.total_col_dim()
            )
    elif role == "pack":
        row_tiles["out"] = node.output.tile_count_x
        slots = ["dest", "out"]
    else:
        slots = ["src0", "src1", "dest"] if unit.input_count == 2 else ["dest"]

    plan = default_plan(
        planned.region,
        unit.granularity,
        slots,
        row_tiles,
        node.block_tiles_x,
        node.block_tiles_y,
    )
    indices = dict(plan.slots)
    if role == "sfpu":
        offsets = {
            "dest": getattr(unit, "dst_index_out", None) or getattr(unit, "dest_idx", 0)
        }
        if unit.input_count == 2:
            offsets.update(src0=unit.dst_index_in0, src1=unit.dst_index_in1)
        for slot, offset in offsets.items():
            indices[slot] = replace(indices[slot], base=indices[slot].base + offset)
    if role == "pack" and node.pack_l1_accumulation == L1Accumulation.Yes:
        indices["out"] = SlotIndex(
            multipliers={
                var: value
                for var, value in indices["out"].multipliers.items()
                if var.startswith("tile_")
            }
        )
    plan = apply_loop_spec(replace(plan, slots=indices), node.loop_spec)
    return PlannedNode(node, role, unit, planned.block_for(node), plan)


def _num_banks(nodes) -> int:
    counts = {
        planned.loop.num_banks
        for planned in nodes
        if planned.loop.num_banks is not None
    }
    if not counts:
        return 1
    if len(counts) != 1:
        raise ValueError(
            f"math and pack nodes disagree on dest-bank count: {sorted(counts)}"
        )
    return counts.pop()


def plan_pipeline(operation: "L1Operation") -> List[PlannedBlock]:
    pipeline = operation.math
    tile_count_x = (
        operation.max_output_dimensions[1] // operation.tile_shape.total_col_dim()
    )
    tile_count_y = (
        operation.max_output_dimensions[0] // operation.tile_shape.total_row_dim()
    )
    if pipeline.custom_op:
        # Custom index arrays drive bank iterations within a single block region.
        tile_count_x, tile_count_y = operation.block_tiles_x, operation.block_tiles_y

    units = list(_node_units(pipeline.math_nodes, pipeline.pack_nodes))
    result = []
    for region in block_regions(
        tile_count_x, tile_count_y, operation.block_tiles_x, operation.block_tiles_y
    ):
        bank = LoopPlan(bank_levels=region.bank_levels)
        planned = PlannedBlock(region, bank, ())
        nodes = tuple(
            _plan_node(operation, planned, node, role, unit)
            for node, role, unit in units
        )
        if pipeline.custom_op:
            bank = LoopPlan(bank_levels=(Level(BANK_VAR, _num_banks(nodes)),))
        planned = replace(planned, bank=bank, nodes=nodes)
        planned.check_dest_reads()
        result.append(planned)
    return result
