# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared Dest-flag parametrize helpers.

``dest_sync``, ``dest_acc``, and ``unpack_to_dest`` are independent axes.
Occupancy (input dimensions, dest index, tiles in Dest) is a function of
``dest_sync`` x ``dest_acc``. Tests should call these helpers instead of
inlining ``is_32_bit() and dest_acc`` or packing ``(dest_sync, dest_acc)``.
"""

from enum import Enum
from typing import Iterable, List, Optional, Sequence

from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.constraints import (
    _quasar_effective_sfpu_format,
    _quasar_fpu_source_format,
    distinct_dest_accumulation_modes,
    get_valid_dest_accumulation_modes,
    is_valid_quasar_fpu_path,
    is_valid_quasar_unpack_to_dest,
)
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, DestSync

DEST_SYNC_TILE_LIMITS = {
    DestSync.Half: 8,
    DestSync.Full: 16,
}


class UnpackPath(Enum):
    """Which Dest-write path a kernel uses, for ``unpack_to_dest_modes``."""

    FpuMath = "fpu_math"
    Sfpu = "sfpu"
    Int32Dest = "int32_dest"
    ForceTrue = "force_true"
    ForceFalse = "force_false"


def dest_sync_modes(*, is_perf: bool = False) -> List[DestSync]:
    """DestSync values for a sweep.

    Quasar perf stays ``SyncHalf``. Functional tests (every arch) and WH/BH
    perf sweep Half and Full.
    """
    if is_perf and get_chip_architecture() == ChipArchitecture.QUASAR:
        return [DestSync.Half]
    return [DestSync.Half, DestSync.Full]


def dest_acc_modes(
    formats,
    *,
    allowed: Optional[Iterable[DestAccumulation]] = None,
    distinct: bool = False,
) -> List[DestAccumulation]:
    """Legal Dest widths for ``formats``.

    ``allowed`` intersects with the hardware gate (for kernels that only
    support a subset). ``distinct`` drops WH/BH modes that ``TestConfig``
    would promote onto another requested mode.
    """
    modes = get_valid_dest_accumulation_modes(formats)
    if allowed is not None:
        allowed_set = set(allowed)
        modes = [mode for mode in modes if mode in allowed_set]
    if distinct:
        modes = distinct_dest_accumulation_modes(formats, modes)
    return modes


def dest_tile_capacity(dest_sync: DestSync, dest_acc) -> int:
    """Tiles that fit in Dest for this sync mode and Dest width."""
    dest_acc_yes = dest_acc is True or dest_acc == DestAccumulation.Yes
    return DEST_SYNC_TILE_LIMITS[dest_sync] // (2 if dest_acc_yes else 1)


def _dest_acc_enabled(dest_acc) -> bool:
    return dest_acc is True or dest_acc == DestAccumulation.Yes


def _input_format(formats) -> DataFormat:
    return formats.input_format if hasattr(formats, "input_format") else formats.input


def _output_format(formats) -> DataFormat:
    return (
        formats.output_format if hasattr(formats, "output_format") else formats.output
    )


def _quasar_dest_format_candidates(formats, dest_acc) -> Sequence[DataFormat]:
    input_format = _input_format(formats)
    output_format = _output_format(formats)
    if _dest_acc_enabled(dest_acc):
        return (DataFormat.Int32 if input_format.is_integer() else DataFormat.Float32,)
    candidates = []
    for candidate in (
        _quasar_effective_sfpu_format(input_format),
        _quasar_effective_sfpu_format(output_format),
    ):
        if not candidate.is_32_bit() and candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def _quasar_has_unpack_to_dest(formats, dest_acc) -> bool:
    input_format = _input_format(formats)
    for dest_format in _quasar_dest_format_candidates(formats, dest_acc):
        if is_valid_quasar_unpack_to_dest(input_format, dest_format, dest_acc):
            return True
    return False


def _quasar_has_fpu_path(formats, dest_acc) -> bool:
    input_format = _input_format(formats)
    fpu_source = _quasar_fpu_source_format(input_format)
    for dest_format in _quasar_dest_format_candidates(formats, dest_acc):
        if is_valid_quasar_fpu_path(input_format, fpu_source, dest_format, dest_acc):
            return True
    return False


def unpack_to_dest_modes(
    formats,
    dest_acc,
    *,
    path: UnpackPath,
) -> List[bool]:
    """Legal ``unpack_to_dest`` values for this kernel path.

    Returned as a list so the flag is always a parametrize axis, even when
    only one value is legal.
    """
    if path is UnpackPath.ForceTrue:
        return [True]
    if path is UnpackPath.ForceFalse or path is UnpackPath.FpuMath:
        return [False]
    if path is UnpackPath.Int32Dest:
        input_format = _input_format(formats)
        return (
            [True]
            if input_format.is_32_bit() and input_format.is_integer()
            else [False]
        )
    if path is not UnpackPath.Sfpu:
        raise ValueError(f"Unknown unpack path: {path}")

    if get_chip_architecture() != ChipArchitecture.QUASAR:
        return [_input_format(formats).is_32_bit() and _dest_acc_enabled(dest_acc)]

    modes = []
    if _quasar_has_unpack_to_dest(formats, dest_acc):
        modes.append(True)
    if _quasar_has_fpu_path(formats, dest_acc):
        modes.append(False)
    return modes or [True]
