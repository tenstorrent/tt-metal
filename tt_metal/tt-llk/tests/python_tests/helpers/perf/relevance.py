# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Opt-in TILE_LOOP relevance maps for PerfConfig.

A run type whose TILE_LOOP cannot observe a template still needs that constexpr
in the C++ header, so unused templates are pinned to canonical defaults rather
than omitted. Unused runtime / format fields are dropped from the execute cache
key only; L1 writes keep the original values unless SPEED_OF_LIGHT promotes
runtimes into the compile header.
"""

from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Any, Iterable

from ..llk_params import DestSync, MathFidelity, PerfRunType
from ..test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_SYNC,
    IN_TILE_DIMS,
    INPUT_DIMENSIONS,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    PARTIAL_FACE,
    PERF_RUN_TYPE,
    RELU_CONFIG,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
    RuntimeParameter,
    TemplateParameter,
)


@dataclass(frozen=True)
class RunTypeRelevance:
    """TILE_LOOP-observable fields for one PerfRunType.

    ``None`` on a slot means keep every value (no skip). An empty frozenset
    means keep none of that slot (pin / drop everything in it).
    """

    templates: frozenset[type] | None = None
    runtime_types: frozenset[type] | None = None
    runtime_fields: frozenset[str] | None = None
    format_fields: frozenset[str] | None = None
    dest_acc: bool = True


_UNPACK_FORMATS = frozenset(
    {"unpack_A_src", "unpack_B_src", "unpack_A_dst", "unpack_B_dst"}
)
_PACK_FORMATS = frozenset({"pack_src", "pack_dst"})
_ALL_FORMATS = (
    _UNPACK_FORMATS | _PACK_FORMATS | frozenset({"math", "sfpu_src", "sfpu_dst"})
)
_MATH_FORMATS = frozenset({"math"})
_IO_FORMATS = _UNPACK_FORMATS | _PACK_FORMATS

_ALL_RUN_TYPES = (
    PerfRunType.L1_TO_L1,
    PerfRunType.UNPACK_ISOLATE,
    PerfRunType.MATH_ISOLATE,
    PerfRunType.PACK_ISOLATE,
    PerfRunType.L1_CONGESTION,
)


class PerfRelevance:
    """Isolate-mode TILE_LOOP defaults. Subclasses add test-specific runtimes.

    L1_TO_L1 always keeps every field. Runtime slots default to ``None`` (keep
    whatever the test passed). Subclasses set frozensets to drop unused types.
    """

    run_types: tuple[PerfRunType, ...] = _ALL_RUN_TYPES

    unpack_templates: frozenset[type] | None = frozenset()
    math_templates: frozenset[type] | None = frozenset({MATH_FIDELITY, THROTTLE_LEVEL})
    pack_templates: frozenset[type] | None = frozenset({DEST_SYNC})
    cong_templates: frozenset[type] | None = frozenset({DEST_SYNC})

    unpack_runtimes: frozenset[type] | None = None
    math_runtimes: frozenset[type] | None = None
    pack_runtimes: frozenset[type] | None = None
    cong_runtimes: frozenset[type] | None = None

    unpack_runtime_fields: frozenset[str] | None = None
    math_runtime_fields: frozenset[str] | None = None
    pack_runtime_fields: frozenset[str] | None = None
    cong_runtime_fields: frozenset[str] | None = None

    unpack_formats: frozenset[str] | None = _UNPACK_FORMATS
    math_formats: frozenset[str] | None = _MATH_FORMATS
    pack_formats: frozenset[str] | None = _PACK_FORMATS
    cong_formats: frozenset[str] | None = _IO_FORMATS

    def as_map(self) -> dict[PerfRunType, RunTypeRelevance]:
        specs = {
            PerfRunType.L1_TO_L1: RunTypeRelevance(),
            PerfRunType.UNPACK_ISOLATE: RunTypeRelevance(
                templates=self.unpack_templates,
                runtime_types=self.unpack_runtimes,
                runtime_fields=self.unpack_runtime_fields,
                format_fields=self.unpack_formats,
            ),
            PerfRunType.MATH_ISOLATE: RunTypeRelevance(
                templates=self.math_templates,
                runtime_types=self.math_runtimes,
                runtime_fields=self.math_runtime_fields,
                format_fields=self.math_formats,
            ),
            PerfRunType.PACK_ISOLATE: RunTypeRelevance(
                templates=self.pack_templates,
                runtime_types=self.pack_runtimes,
                runtime_fields=self.pack_runtime_fields,
                format_fields=self.pack_formats,
            ),
            PerfRunType.L1_CONGESTION: RunTypeRelevance(
                templates=self.cong_templates,
                runtime_types=self.cong_runtimes,
                runtime_fields=self.cong_runtime_fields,
                format_fields=self.cong_formats,
            ),
        }
        return {run_type: specs[run_type] for run_type in self.run_types}

    def __getitem__(self, run_type: PerfRunType) -> RunTypeRelevance:
        return self.as_map()[run_type]


class MatmulRelevance(PerfRelevance):
    unpack_runtimes = frozenset(
        {UNPACK_TRANS_FACES, NUM_FACES, LOOP_FACTOR, CRK_TILE_DIMM}
    )
    math_runtimes = frozenset({UNPACK_TRANS_FACES, LOOP_FACTOR, CRK_TILE_DIMM})
    pack_runtimes = frozenset({LOOP_FACTOR, CRK_TILE_DIMM})
    cong_runtimes = frozenset(
        {UNPACK_TRANS_FACES, NUM_FACES, LOOP_FACTOR, CRK_TILE_DIMM}
    )
    pack_runtime_fields = frozenset({"loop_factor", "c_dimm", "r_dimm"})


class MathMatmulRelevance(MatmulRelevance):
    _EXTRA = frozenset(
        {NUM_BLOCKS, PARTIAL_FACE, IN_TILE_DIMS, UNPACK_TRANS_WITHIN_FACE}
    )
    unpack_runtimes = MatmulRelevance.unpack_runtimes | _EXTRA
    math_runtimes = MatmulRelevance.math_runtimes | _EXTRA
    pack_runtimes = MatmulRelevance.pack_runtimes | frozenset({NUM_BLOCKS})
    cong_runtimes = MatmulRelevance.cong_runtimes | _EXTRA
    pack_runtime_fields = MatmulRelevance.pack_runtime_fields | frozenset(
        {"num_blocks"}
    )


_PACK_BLOCK_RUNTIMES = frozenset(
    {NUM_BLOCKS, NUM_TILES_IN_BLOCK, LOOP_FACTOR, NUM_FACES}
)


class PackRelevance(PerfRelevance):
    math_templates = frozenset({DEST_SYNC})
    unpack_runtimes = _PACK_BLOCK_RUNTIMES
    math_runtimes = _PACK_BLOCK_RUNTIMES
    pack_runtimes = _PACK_BLOCK_RUNTIMES | frozenset({RELU_CONFIG})
    cong_runtimes = _PACK_BLOCK_RUNTIMES | frozenset({RELU_CONFIG})


class PackUntilizeRelevance(PerfRelevance):
    run_types = (
        PerfRunType.L1_TO_L1,
        PerfRunType.PACK_ISOLATE,
        PerfRunType.L1_CONGESTION,
    )
    pack_templates = None
    pack_runtimes = None


class UnpackTilizeRelevance(PerfRelevance):
    run_types = (
        PerfRunType.L1_TO_L1,
        PerfRunType.UNPACK_ISOLATE,
        PerfRunType.PACK_ISOLATE,
        PerfRunType.L1_CONGESTION,
    )
    _DIM_RUNTIMES = frozenset({INPUT_DIMENSIONS, TILE_COUNT, LOOP_FACTOR})
    unpack_runtimes = _DIM_RUNTIMES
    pack_templates = frozenset()
    # Keep INPUT_DIMENSIONS on PACK. SPEED_OF_LIGHT inlines runtimes, and
    # unpack_tilize_perf.cpp asserts FULL_RT_DIM * FULL_CT_DIM == TILE_CNT on
    # the unpack thread before PACK_ISOLATE returns. Dropping dims zeros them
    # via project_runtimes and trips that assert.
    pack_runtimes = _DIM_RUNTIMES
    cong_runtimes = _DIM_RUNTIMES


MATMUL_RELEVANCE = MatmulRelevance()
MATH_MATMUL_RELEVANCE = MathMatmulRelevance()
PACK_RELEVANCE = PackRelevance()
PACK_UNTILIZE_RELEVANCE = PackUntilizeRelevance()
UNPACK_TILIZE_RELEVANCE = UnpackTilizeRelevance()


def pin_template(param: TemplateParameter) -> TemplateParameter:
    """Canonical compile-time stand-in for a template that TILE_LOOP cannot see."""
    if isinstance(param, MATH_FIDELITY):
        return MATH_FIDELITY(MathFidelity.LoFi)
    if isinstance(param, DEST_SYNC):
        return DEST_SYNC(DestSync.Full)
    if isinstance(param, THROTTLE_LEVEL):
        return THROTTLE_LEVEL(0)
    return param


def project_templates(
    templates: Iterable[TemplateParameter],
    spec: RunTypeRelevance | None,
) -> list[TemplateParameter]:
    """Keep relevant templates; pin the rest. Always keep PERF_RUN_TYPE."""
    if spec is None or spec.templates is None:
        return list(templates)
    projected = []
    for param in templates:
        if isinstance(param, PERF_RUN_TYPE) or type(param) in spec.templates:
            projected.append(param)
        else:
            projected.append(pin_template(param))
    return projected


def _default_runtime(param: RuntimeParameter) -> RuntimeParameter:
    cls = type(param)
    try:
        return cls()
    except TypeError:
        return param


def _default_field_value(value: Any) -> Any:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return 1 if value else 0
    if isinstance(value, Enum):
        return value
    if hasattr(value, "value") and isinstance(getattr(value, "value"), int):
        return type(value)(1)
    return value


def project_runtimes(
    runtimes: Iterable[RuntimeParameter],
    spec: RunTypeRelevance | None,
) -> list[RuntimeParameter]:
    """Pin unused runtime types/fields. Used when SPEED_OF_LIGHT inlines runtimes."""
    if spec is None:
        return list(runtimes)
    projected = []
    for param in runtimes:
        if spec.runtime_types is not None and type(param) not in spec.runtime_types:
            projected.append(_default_runtime(param))
            continue
        if spec.runtime_fields is None:
            projected.append(param)
            continue
        updates = {
            f.name: _default_field_value(getattr(param, f.name))
            for f in fields(param)
            if f.name not in spec.runtime_fields
        }
        projected.append(replace(param, **updates) if updates else param)
    return projected


def _hashable(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool, type(None), Enum)):
        return value
    raw = getattr(value, "value", None)
    if not isinstance(value, Enum) and isinstance(raw, (int, float, str, bool)):
        return raw
    return repr(value)


def _dataclass_items(param: Any) -> list[tuple[str, Any]]:
    return [(f.name, _hashable(getattr(param, f.name))) for f in fields(param)]


def execute_key(
    *,
    test_name: str,
    run_type: PerfRunType,
    dest_acc: Any,
    templates: Iterable[TemplateParameter],
    runtimes: Iterable[RuntimeParameter],
    formats: Any,
    speed_of_light: bool,
    spec: RunTypeRelevance | None,
) -> tuple:
    """Hashable identity of one run-type measurement under ``spec``."""
    dest = dest_acc if (spec is None or spec.dest_acc) else None

    template_items: list[tuple[str, Any]] = []
    for param in templates:
        if isinstance(param, PERF_RUN_TYPE):
            continue
        if spec is None or spec.templates is None or type(param) in spec.templates:
            template_items.extend(_dataclass_items(param))

    runtime_items: list[tuple[str, Any]] = []
    for param in runtimes:
        if spec is not None and spec.runtime_types is not None:
            if type(param) not in spec.runtime_types:
                continue
        for name, value in _dataclass_items(param):
            if spec is not None and spec.runtime_fields is not None:
                if name not in spec.runtime_fields:
                    continue
            runtime_items.append((name, value))

    format_items: list[tuple[str, Any]] = []
    fmt = None
    if formats:
        fmt = formats[0] if isinstance(formats, list) else formats
    names = (
        spec.format_fields
        if spec is not None and spec.format_fields is not None
        else _ALL_FORMATS
    )
    if fmt is not None:
        for name in sorted(names):
            format_items.append((name, _hashable(getattr(fmt, name, None))))

    return (
        test_name,
        run_type,
        _hashable(dest),
        tuple(template_items),
        tuple(runtime_items),
        tuple(format_items),
        speed_of_light,
    )
