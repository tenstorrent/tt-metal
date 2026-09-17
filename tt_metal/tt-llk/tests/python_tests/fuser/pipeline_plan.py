# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field, replace
from itertools import chain
from typing import TYPE_CHECKING, List, Tuple, Union

from helpers.llk_params import DestSync, EltwiseBinaryReuseDestType, L1Accumulation

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


def dest_tile_capacity(tile_shape, dest_sync: DestSync, dest_acc: bool) -> int:
    faces = 32 if dest_sync == DestSync.Half else 64
    if dest_acc:
        faces //= 2
    return faces // tile_shape.total_num_faces()


def _check_tile_bounds(tiles, capacity: int, label: str) -> None:
    invalid = next((tile for tile in sorted(tiles) if not 0 <= tile < capacity), None)
    if invalid is not None:
        raise ValueError(f"{label} accesses tile {invalid} outside [0, {capacity})")


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
        banks = tuple(banks)
        return set().union(*(self._tiles(banks, slot) for slot in slots))

    def _tiles(self, banks, slot, operand=None) -> set:
        if slot not in self.loop.slots:
            return set()

        result = set()
        for call in chain.from_iterable(self.loop.calls(bank) for bank in banks):
            if self.unit.granularity != InvocationGranularity.BLOCK:
                result.update(getattr(tile, slot) for tile in (call,) + call.tiles)
                continue

            start = getattr(call, slot)
            rows, cols = self.block.block_rows, self.block.block_cols
            stride = operand.tile_count_x if operand is not None else cols
            if self.role == "unpack" and operand is not None:
                # Matmul reads complete K slices from each operand's tile origin.
                if slot == "in0":
                    cols = operand.tile_count_x
                else:
                    rows = operand.tile_count_y
            for row in range(rows):
                first = start + row * stride
                result.update(range(first, first + cols))
        return result

    def check_bounds(self, banks, dest_capacity: int) -> None:
        if self.role == "math" and not self.unit.supports_dest_offset:
            _check_tile_bounds(
                (call.dest for bank in banks for call in self.loop.calls(bank)),
                1,
                f"{type(self.unit).__name__} dest origin",
            )
        operands = {}
        if self.role == "unpack":
            operands = {"in0": self.node.src_a, "in1": self.node.src_b}
        elif self.role == "pack":
            operands = {"out": self.node.output}

        for slot in self.loop.slots:
            operand = operands.get(slot)
            if operand is not None:
                capacity, label = operand.tile_count, f"operand '{operand.name}'"
            elif slot in DEST_SLOTS and self.role != "unpack":
                capacity, label = dest_capacity, "(Dst)"
            else:
                continue
            _check_tile_bounds(
                self._tiles(banks, slot, operand),
                capacity,
                f"{type(self.unit).__name__} {slot} {label}",
            )


@dataclass(frozen=True)
class PlannedBlock:
    region: BlockRegion
    bank: LoopPlan
    nodes: Tuple[PlannedNode, ...]
    dest_sources: dict[Node, set[FpuNode]] = field(default_factory=dict, repr=False)

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

    def trace_dest_sources(self) -> dict[Node, set[FpuNode]]:
        def read(node, tiles, values):
            missing = tiles - values.keys()
            if missing:
                raise ValueError(
                    f"{type(node).__name__} reads dest tiles {sorted(missing)} "
                    "that no earlier node writes"
                )
            return set().union(*(values[tile] for tile in tiles))

        def trace_sfpu(planned, bank, values):
            sources = set()
            slots = ("src0", "src1") if planned.unit.input_count == 2 else ("dest",)
            for call in planned.loop.calls(bank):
                producers = read(
                    planned.node, {getattr(call, slot) for slot in slots}, values
                )
                sources.update(producers)
                values[call.dest] = producers
            return sources

        sources = {}
        for bank in self.bank.bank_assignments():
            values = {}
            for planned in self.nodes:
                if planned.role == "unpack":
                    continue
                node = planned.node
                if planned.role == "sfpu":
                    sources.setdefault(node, set()).update(
                        trace_sfpu(planned, bank, values)
                    )
                    continue

                tiles = planned.dest_tiles((bank,), ("dest",))
                if (
                    planned.role == "pack"
                    or node.src_a is None
                    or node.reuse_dest != EltwiseBinaryReuseDestType.NONE
                ):
                    sources.setdefault(node, set()).update(read(node, tiles, values))
                if planned.role == "math" and node.src_a is not None:
                    values.update({tile: {node} for tile in tiles})
        return sources


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


def _plan_node(planned: PlannedBlock, node: Node, role: str, unit: Unit) -> PlannedNode:
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
    if role in ("unpack", "math") and unit.granularity == InvocationGranularity.BLOCK:
        indices = dict(plan.slots)
        for slot, axis, stride in (
            ("in0", planned.region.y, node.src_a.tile_count_x),
            ("in1", planned.region.x, 1),
        ):
            indices[slot] = SlotIndex(
                base=axis.origin * stride,
                multipliers={axis.var: stride} if axis.looped else {},
            )
        plan = replace(plan, slots=indices)
    if role == "pack" and node.pack_l1_accumulation == L1Accumulation.Yes:
        indices = dict(plan.slots)
        indices["out"] = SlotIndex(
            multipliers={
                var: value
                for var, value in indices["out"].multipliers.items()
                if var.startswith("tile_")
            }
        )
        plan = replace(plan, slots=indices)
    plan = apply_loop_spec(plan, node.loop_spec)
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


def plan_pipeline(
    operation: "L1Operation", dest_acc: bool = False
) -> List[PlannedBlock]:
    dest_capacity = dest_tile_capacity(
        operation.tile_shape, operation.dest_sync, dest_acc
    )
    tile_count_x = (
        operation.max_output_dimensions[1] // operation.tile_shape.total_col_dim()
    )
    tile_count_y = (
        operation.max_output_dimensions[0] // operation.tile_shape.total_row_dim()
    )
    if operation.custom_op:
        # Custom index arrays drive bank iterations within a single block region.
        tile_count_x, tile_count_y = operation.block_tiles_x, operation.block_tiles_y

    units = list(_node_units(operation.math_nodes, operation.pack_nodes))
    result = []
    for region in block_regions(
        tile_count_x, tile_count_y, operation.block_tiles_x, operation.block_tiles_y
    ):
        bank = LoopPlan(bank_levels=region.bank_levels)
        planned = PlannedBlock(region, bank, ())
        nodes = tuple(
            _plan_node(planned, node, role, unit) for node, role, unit in units
        )
        if operation.custom_op:
            bank = LoopPlan(bank_levels=(Level(BANK_VAR, _num_banks(nodes)),))
        planned = replace(planned, bank=bank, nodes=nodes)
        banks = planned.bank.bank_assignments()
        for node in nodes:
            node.check_bounds(banks, dest_capacity)
        planned = replace(planned, dest_sources=planned.trace_dest_sources())
        result.append(planned)
    return result
