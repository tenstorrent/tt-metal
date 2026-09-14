# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Opt-in TILE_LOOP relevance maps for PerfConfig.

A run type whose TILE_LOOP cannot observe a template still needs that constexpr
in the C++ header, so unused templates are pinned to canonical defaults rather
than omitted. Unused runtime / format fields are dropped from the execute cache
key only; L1 writes keep the original values unless SPEED_OF_LIGHT promotes
runtimes into the compile header.
"""

import os
from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Any, Iterable

from ..llk_params import DestSync, MathFidelity, PerfRunType
from ..test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_INDEX,
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

KEEP_ALL = None  # keep every value in this slot
PIN_ALL = frozenset()  # keep none (pin templates / drop runtimes+formats)


@dataclass(frozen=True)
class RunTypeRelevance:
    """TILE_LOOP-observable fields for one PerfRunType.

    ``KEEP_ALL`` (None) keeps every value. ``PIN_ALL`` (empty frozenset) keeps
    none of that slot (pin templates / drop runtimes and formats).
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

_PACK_BLOCK_RUNTIMES = frozenset(
    {NUM_BLOCKS, NUM_TILES_IN_BLOCK, LOOP_FACTOR, NUM_FACES}
)

# pin_template only rewrites these. Any other template stays in the header, so
# execute_key must keep it even when spec.templates omits the type.
_PINNABLE_TEMPLATES = frozenset({MATH_FIDELITY, DEST_SYNC, THROTTLE_LEVEL})


def _runtime_fields(*types: type, drop: Iterable[str] = ()) -> frozenset[str]:
    names = {f.name for cls in types for f in fields(cls)}
    return frozenset(names - set(drop))


class PerfRelevance:
    """Isolate-mode TILE_LOOP defaults. Subclasses add test-specific runtimes.

    L1_TO_L1 always keeps every field. Runtime slots default to ``KEEP_ALL``.
    Subclasses set frozensets to drop unused types.
    """

    run_types: tuple[PerfRunType, ...] = _ALL_RUN_TYPES

    unpack_templates: frozenset[type] | None = PIN_ALL
    math_templates: frozenset[type] | None = frozenset({MATH_FIDELITY, THROTTLE_LEVEL})
    pack_templates: frozenset[type] | None = frozenset({DEST_SYNC})
    cong_templates: frozenset[type] | None = frozenset({DEST_SYNC})

    unpack_runtimes: frozenset[type] | None = KEEP_ALL
    math_runtimes: frozenset[type] | None = KEEP_ALL
    pack_runtimes: frozenset[type] | None = KEEP_ALL
    cong_runtimes: frozenset[type] | None = KEEP_ALL

    unpack_runtime_fields: frozenset[str] | None = KEEP_ALL
    math_runtime_fields: frozenset[str] | None = KEEP_ALL
    pack_runtime_fields: frozenset[str] | None = KEEP_ALL
    cong_runtime_fields: frozenset[str] | None = KEEP_ALL

    unpack_formats: frozenset[str] | None = _UNPACK_FORMATS
    math_formats: frozenset[str] | None = _MATH_FORMATS
    pack_formats: frozenset[str] | None = _PACK_FORMATS
    cong_formats: frozenset[str] | None = _IO_FORMATS

    def as_map(self) -> dict[PerfRunType, RunTypeRelevance]:
        cached = getattr(self, "_as_map", None)
        if cached is not None:
            return cached
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
        self._as_map = {run_type: specs[run_type] for run_type in self.run_types}
        return self._as_map

    def __getitem__(self, run_type: PerfRunType) -> RunTypeRelevance:
        return self.as_map()[run_type]


class MatmulRelevance(PerfRelevance):
    unpack_runtimes = frozenset(
        {UNPACK_TRANS_FACES, NUM_FACES, LOOP_FACTOR, CRK_TILE_DIMM}
    )
    math_runtimes = frozenset({UNPACK_TRANS_FACES, LOOP_FACTOR, CRK_TILE_DIMM})
    pack_runtimes = frozenset({LOOP_FACTOR, CRK_TILE_DIMM})
    cong_runtimes = unpack_runtimes
    pack_runtime_fields = _runtime_fields(*pack_runtimes, drop={"k_dimm"})


class MathMatmulRelevance(MatmulRelevance):
    _EXTRA = frozenset(
        {NUM_BLOCKS, PARTIAL_FACE, IN_TILE_DIMS, UNPACK_TRANS_WITHIN_FACE}
    )
    # PACK INIT uses faces / partial / tile rows; TILE_LOOP uses DST_INDEX.
    _PACK_EXTRA = frozenset(
        {NUM_BLOCKS, NUM_FACES, PARTIAL_FACE, IN_TILE_DIMS, DEST_INDEX}
    )
    unpack_runtimes = MatmulRelevance.unpack_runtimes | _EXTRA
    math_runtimes = MatmulRelevance.math_runtimes | _EXTRA | frozenset({DEST_INDEX})
    pack_runtimes = MatmulRelevance.pack_runtimes | _PACK_EXTRA
    cong_runtimes = MatmulRelevance.cong_runtimes | _EXTRA | frozenset({DEST_INDEX})
    pack_runtime_fields = _runtime_fields(*pack_runtimes, drop={"k_dimm"})


class PackRelevance(PerfRelevance):
    math_templates = frozenset({DEST_SYNC})
    unpack_runtimes = _PACK_BLOCK_RUNTIMES
    math_runtimes = _PACK_BLOCK_RUNTIMES | frozenset({DEST_INDEX})
    pack_runtimes = _PACK_BLOCK_RUNTIMES | frozenset({RELU_CONFIG, DEST_INDEX})
    cong_runtimes = _PACK_BLOCK_RUNTIMES | frozenset({RELU_CONFIG, DEST_INDEX})


class PackUntilizeRelevance(PerfRelevance):
    run_types = (
        PerfRunType.L1_TO_L1,
        PerfRunType.PACK_ISOLATE,
        PerfRunType.L1_CONGESTION,
    )
    pack_templates = KEEP_ALL
    pack_runtimes = KEEP_ALL
    # INPUT_DIMENSIONS is a template; default cong_templates is DEST_SYNC only
    # and would alias layouts that share tile_cnt (4x5 vs 5x4).
    cong_templates = KEEP_ALL


class UnpackTilizeRelevance(PerfRelevance):
    run_types = (
        PerfRunType.L1_TO_L1,
        PerfRunType.UNPACK_ISOLATE,
        PerfRunType.PACK_ISOLATE,
        PerfRunType.L1_CONGESTION,
    )
    _DIM_RUNTIMES = frozenset({INPUT_DIMENSIONS, TILE_COUNT, LOOP_FACTOR})
    unpack_runtimes = _DIM_RUNTIMES
    pack_templates = PIN_ALL
    # Unpack asserts FULL_RT_DIM * FULL_CT_DIM == TILE_CNT before PACK returns.
    # SPEED_OF_LIGHT inlines runtimes, so PACK must keep those values.
    pack_runtimes = _DIM_RUNTIMES
    cong_runtimes = _DIM_RUNTIMES
    # Blackhole tilize workaround keys off unpack_A_src, not only pack_src/pack_dst.
    pack_formats = _PACK_FORMATS | frozenset({"unpack_A_src"})


LLK_DISABLE_PERF_RELEVANCE = "LLK_DISABLE_PERF_RELEVANCE"


def maybe_relevance(
    relevance: dict[PerfRunType, RunTypeRelevance] | PerfRelevance | None,
) -> dict[PerfRunType, RunTypeRelevance] | PerfRelevance | None:
    """Return ``None`` when ``LLK_DISABLE_PERF_RELEVANCE=1``, else ``relevance``.

    Honored in ``PerfConfig.__init__``. Set the env var to run the same sweep
    without isolate reuse.
    """
    if os.environ.get(LLK_DISABLE_PERF_RELEVANCE) == "1":
        return None
    return relevance


MATMUL_RELEVANCE = MatmulRelevance()
MATH_MATMUL_RELEVANCE = MathMatmulRelevance()
PACK_RELEVANCE = PackRelevance()
PACK_UNTILIZE_RELEVANCE = PackUntilizeRelevance()
UNPACK_TILIZE_RELEVANCE = UnpackTilizeRelevance()


def pin_template(param: TemplateParameter) -> TemplateParameter:
    """Canonical compile-time stand-in for a template that TILE_LOOP cannot see."""
    if type(param) not in _PINNABLE_TEMPLATES:
        return param
    if isinstance(param, MATH_FIDELITY):
        return MATH_FIDELITY(MathFidelity.LoFi)
    if isinstance(param, DEST_SYNC):
        return DEST_SYNC(DestSync.Full)
    if isinstance(param, THROTTLE_LEVEL):
        return THROTTLE_LEVEL(0)
    return param


def _template_visible(param: TemplateParameter, spec: RunTypeRelevance | None) -> bool:
    if spec is None or spec.templates is None:
        return True
    return type(param) in spec.templates or type(param) not in _PINNABLE_TEMPLATES


def project_templates(
    templates: Iterable[TemplateParameter],
    spec: RunTypeRelevance | None,
) -> list[TemplateParameter]:
    """Keep relevant templates; pin the rest. Always keep PERF_RUN_TYPE."""
    projected = []
    for param in templates:
        if isinstance(param, PERF_RUN_TYPE) or _template_visible(param, spec):
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
    # Nonzero ints pin to 1. Dropped TILE_COUNT uses _default_runtime() (0).
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
        if _template_visible(param, spec):
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
