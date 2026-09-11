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
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_FACES,
    PERF_RUN_TYPE,
    THROTTLE_LEVEL,
    UNPACK_TRANS_FACES,
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

MATMUL_RELEVANCE: dict[PerfRunType, RunTypeRelevance] = {
    PerfRunType.L1_TO_L1: RunTypeRelevance(),
    PerfRunType.UNPACK_ISOLATE: RunTypeRelevance(
        templates=frozenset(),
        runtime_types=frozenset(
            {UNPACK_TRANS_FACES, NUM_FACES, LOOP_FACTOR, CRK_TILE_DIMM}
        ),
        format_fields=_UNPACK_FORMATS,
    ),
    PerfRunType.MATH_ISOLATE: RunTypeRelevance(
        templates=frozenset({MATH_FIDELITY, THROTTLE_LEVEL}),
        runtime_types=frozenset({UNPACK_TRANS_FACES, LOOP_FACTOR, CRK_TILE_DIMM}),
        format_fields=frozenset({"math"}),
    ),
    PerfRunType.PACK_ISOLATE: RunTypeRelevance(
        templates=frozenset({DEST_SYNC}),
        runtime_types=frozenset({LOOP_FACTOR, CRK_TILE_DIMM}),
        runtime_fields=frozenset({"loop_factor", "c_dimm", "r_dimm"}),
        format_fields=_PACK_FORMATS,
    ),
    PerfRunType.L1_CONGESTION: RunTypeRelevance(
        templates=frozenset({DEST_SYNC}),
        runtime_types=frozenset(
            {UNPACK_TRANS_FACES, NUM_FACES, LOOP_FACTOR, CRK_TILE_DIMM}
        ),
        format_fields=_UNPACK_FORMATS | _PACK_FORMATS,
    ),
}


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
