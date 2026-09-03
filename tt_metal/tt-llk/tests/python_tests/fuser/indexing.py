# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field, replace
from enum import Enum, auto
from itertools import product
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

DEST_SLOTS = ("src0", "src1", "dest")

BLOCK_X, BLOCK_Y = "block_x", "block_y"
TILE_X, TILE_Y = "tile_x", "tile_y"

Index = Optional[Union[int, str]]


class InvocationGranularity(Enum):
    NONE = auto()
    TILE = auto()
    ROW = auto()
    BLOCK = auto()


@dataclass(frozen=True)
class KernelInvocation:
    in0: Index = None
    in1: Index = None
    src0: Index = None
    src1: Index = None
    dest: Index = None
    out: Index = None


@dataclass(frozen=True)
class Level:
    var: str
    count: int
    step: int = 1

    @property
    def trivial(self) -> bool:
        return self.count <= 1


@dataclass(frozen=True)
class SlotIndex:
    base: int = 0
    multipliers: Mapping[str, int] = field(default_factory=dict)

    def value(self, assignment: Mapping[str, int]) -> int:
        return self.base + sum(
            multiplier * assignment[var] for var, multiplier in self.multipliers.items()
        )

    def cpp(self, constants: Mapping[str, int]) -> str:
        constant = self.base
        terms: List[str] = []
        for var, multiplier in self.multipliers.items():
            if var in constants:
                constant += multiplier * constants[var]
            elif multiplier == 1:
                terms.append(var)
            else:
                terms.append(f"{multiplier} * {var}")
        if constant or not terms:
            terms.append(str(constant))
        return " + ".join(terms)


@dataclass(frozen=True)
class LoopPlan:
    bank_levels: Tuple[Level, ...] = ()
    call_levels: Tuple[Level, ...] = ()
    slots: Mapping[str, SlotIndex] = field(default_factory=dict)

    def __post_init__(self) -> None:
        declared = {level.var for level in self.bank_levels + self.call_levels}
        for name, index in self.slots.items():
            unknown = sorted(set(index.multipliers) - declared)
            if unknown:
                raise ValueError(
                    f"slot '{name}' references undeclared loop vars {unknown}\n"
                    f"declared: {sorted(declared)}"
                )

    @staticmethod
    def _assignments(levels: Sequence[Level]) -> Iterator[Dict[str, int]]:
        for indices in product(*(range(level.count) for level in levels)):
            yield {
                level.var: index * level.step for level, index in zip(levels, indices)
            }

    def bank_assignments(self) -> List[Dict[str, int]]:
        return list(self._assignments(self.bank_levels))

    def calls(self, bank: Mapping[str, int]) -> List[KernelInvocation]:
        return [
            KernelInvocation(
                **{
                    slot: index.value({**bank, **call})
                    for slot, index in self.slots.items()
                }
            )
            for call in self._assignments(self.call_levels)
        ]

    def _emit(self, levels: Sequence[Level], render, constants: Dict[str, int]) -> str:
        if not levels:
            return render(constants)
        level, rest = levels[0], levels[1:]
        if level.trivial:
            return self._emit(rest, render, {**constants, level.var: 0})
        body = self._emit(rest, render, constants)
        if not body:
            return ""
        limit = level.count * level.step
        increment = (
            f"{level.var}++" if level.step == 1 else f"{level.var} += {level.step}"
        )
        return (
            f"for (std::uint32_t {level.var} = 0; {level.var} < {limit}; {increment}) {{\n"
            f"{body}}}\n"
        )

    def emit_banks(self, body_fn) -> str:
        return self._emit(self.bank_levels, body_fn, {})

    def emit_calls(
        self, render, bank_constants: Optional[Mapping[str, int]] = None
    ) -> str:
        def body(consts: Mapping[str, int]) -> str:
            return render(
                KernelInvocation(
                    **{slot: index.cpp(consts) for slot, index in self.slots.items()}
                )
            )

        return self._emit(self.call_levels, body, dict(bank_constants or {}))


@dataclass(frozen=True)
class Axis:
    var: str
    origin: int
    blocks: int
    block_tiles: int

    @property
    def looped(self) -> bool:
        return self.blocks > 1

    @property
    def level(self) -> Optional[Level]:
        if not self.looped:
            return None
        return Level(var=self.var, count=self.blocks, step=self.block_tiles)


@dataclass(frozen=True)
class BlockRegion:
    x: Axis
    y: Axis

    @property
    def block_tiles_x(self) -> int:
        return self.x.block_tiles

    @property
    def block_tiles_y(self) -> int:
        return self.y.block_tiles

    @property
    def block_tiles(self) -> int:
        return self.block_tiles_x * self.block_tiles_y

    @property
    def bank_levels(self) -> Tuple[Level, ...]:
        return tuple(
            level for level in (self.x.level, self.y.level) if level is not None
        )


def block_regions(
    tile_count_x: int,
    tile_count_y: int,
    block_tiles_x: int,
    block_tiles_y: int,
) -> List[BlockRegion]:

    def axes(var: str, tile_count: int, block_tiles: int) -> List[Axis]:
        full_blocks = tile_count // block_tiles
        remainder = tile_count % block_tiles
        result = []
        if full_blocks:
            result.append(
                Axis(var=var, origin=0, blocks=full_blocks, block_tiles=block_tiles)
            )
        if remainder:
            result.append(
                Axis(
                    var=var,
                    origin=full_blocks * block_tiles,
                    blocks=1,
                    block_tiles=remainder,
                )
            )
        return result

    return [
        BlockRegion(x=x, y=y)
        for x, y in product(
            axes(BLOCK_X, tile_count_x, block_tiles_x),
            axes(BLOCK_Y, tile_count_y, block_tiles_y),
        )
    ]


def default_plan(
    region: BlockRegion,
    granularity: InvocationGranularity,
    slots: Sequence[str],
    row_tiles: Mapping[str, int],
) -> LoopPlan:
    if granularity == InvocationGranularity.TILE:
        levels = (
            Level(var=TILE_X, count=region.block_tiles_x),
            Level(var=TILE_Y, count=region.block_tiles_y),
        )
    elif granularity == InvocationGranularity.ROW:
        levels = (Level(var=TILE_Y, count=region.block_tiles_y),)
    else:
        levels = ()
    call_vars = {level.var for level in levels}

    def index(width: int, banked: bool) -> SlotIndex:
        multipliers = {}
        if TILE_X in call_vars:
            multipliers[TILE_X] = 1
        if TILE_Y in call_vars:
            multipliers[TILE_Y] = width
        if not banked:
            return SlotIndex(multipliers=multipliers)
        if region.x.looped:
            multipliers[region.x.var] = 1
        if region.y.looped:
            multipliers[region.y.var] = width
        return SlotIndex(
            base=width * region.y.origin + region.x.origin,
            multipliers=multipliers,
        )

    return LoopPlan(
        bank_levels=region.bank_levels,
        call_levels=levels,
        slots={
            slot: index(
                region.block_tiles_x if slot in DEST_SLOTS else row_tiles[slot],
                banked=slot not in DEST_SLOTS,
            )
            for slot in slots
        },
    )


INDEX_NAMES = {
    "unpack": {"in0": "tile_id", "in1": "tile_id_b"},
    "math": {"dest": "tile_id"},
    "pack": {"dest": "dest_tile_id", "out": "l1_tile_id"},
}


def bind_indices(
    call: KernelInvocation, names: Mapping[str, str]
) -> Tuple[str, KernelInvocation]:
    declarations = ""
    replacements = {}
    for slot, name in names.items():
        expression = getattr(call, slot)
        if expression is None:
            continue
        text = str(expression)
        if text.isidentifier() or text.isdigit():
            continue
        declarations += f"[[maybe_unused]] const std::uint32_t {name} = {text};\n"
        replacements[slot] = name
    if not replacements:
        return "", call
    return declarations, replace(call, **replacements)
