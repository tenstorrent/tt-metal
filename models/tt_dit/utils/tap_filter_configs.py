# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Measured ``ttnn.conv1d`` configurations for the depthwise tap filter (``audio_ops.depthwise_tap_filter``), per
device class ``(arch, compute grid)``, like the tables in ``utils/matmul.py``, ``utils/conv3d.py`` and
``layers/conv2d.py``.

* ``_FORMULATIONS``: ``(C, K, stride) -> "direct" | chunk width``. Whether the full-C conv fits L1 depends on the
  activation block width ``C * K``, not on ``T``, so one row covers every clip length.
* ``_SLICES``: ``(channels_run, K, stride) -> (T_out_ref, num_slices_ref)``, the DRAM slice count the conv is fastest
  with at one length; other lengths scale it up proportionally (``derive_num_slices``) and pass it as an explicit
  ``slice_config``, so the slicer never searches.

The tables are hints: ``depthwise_tap_filter`` keeps its trial chain behind every lookup, so a missing or stale row
costs one warning and one retry. Rows are never borrowed across shapes (a fit does not interpolate). Regenerate with
``tests/models/minimax_h3/tools/sweep_tap_filter_configs.py``; ``tests/unit/test_audio_tap_path.py`` walks every row
on the device.
"""

from __future__ import annotations

import math
from typing import Any

import ttnn

# Trial order, widest first; the caller appends the shift-multiply-add fallback, which is never tabled.
TAP_FORMULATIONS: tuple[Any, ...] = ("direct", 128, 64, 32)

DeviceKey = tuple[str, int, int]  # (arch, grid_x, grid_y)
ShapeKey = tuple[int, int, int]  # (channels, K, stride)


def tap_device_key(mesh_device) -> DeviceKey:
    """``(arch, grid_x, grid_y)`` of the device class the tables are keyed on."""
    arch = str(mesh_device.arch()).rsplit(".", 1)[-1].lower()
    grid = mesh_device.compute_with_storage_grid_size()
    return (arch, int(grid.x), int(grid.y))


def applicable_formulations(C: int) -> list[Any]:
    """The formulations that can run at ``C``: "direct" always, a chunk width only if it divides ``C`` and is narrower."""
    return [f for f in TAP_FORMULATIONS if f == "direct" or (C % f == 0 and f < C)]


# Keep each block's provenance (date, host, l1_small_size, clip lengths) so a stale row can be traced.

_FORMULATIONS: dict[DeviceKey, dict[ShapeKey, Any]] = {
    # Blackhole Galaxy, swept 2026-09-10 on bh-glx-120-c03u02 (l1_small_size 65536) over the MiniMax-H3 audio
    # decoder's 5 s and 15 s shapes. Only the two widest K=7 filters chunk; the 128-chunk beats full C at 256 by 20-25 %.
    ("blackhole", 12, 10): {
        (512, 12, 2): "direct",
        (512, 7, 1): 128,
        (256, 12, 2): "direct",
        (256, 7, 1): 128,
        (128, 12, 2): "direct",
        (128, 7, 1): "direct",
        (64, 12, 2): "direct",
        (64, 7, 1): "direct",
        (32, 12, 2): "direct",
        (32, 7, 1): "direct",
        (16, 12, 2): "direct",
        (16, 7, 1): "direct",
        (8, 12, 2): "direct",
        (8, 7, 1): "direct",
    },
}
_SLICES: dict[DeviceKey, dict[ShapeKey, tuple[int, int]]] = {
    # Same sweep; rows only where the explicit count beat conv1d's auto-slicing by >= 10 % (the long, narrow filters;
    # for C >= 128 the two tied). Derived counts checked against the smallest fitting count at all swept lengths.
    ("blackhole", 12, 10): {
        (64, 12, 2): (60300, 8),
        (64, 7, 1): (60300, 5),
        (32, 12, 2): (120600, 8),
        (32, 7, 1): (120600, 5),
        (16, 7, 1): (241200, 2),
        (8, 12, 2): (482400, 4),
        (8, 7, 1): (482400, 4),
    },
}


def register_tap_configs(
    device_key: DeviceKey,
    *,
    formulations: dict[ShapeKey, Any] | None = None,
    slices: dict[ShapeKey, tuple[int, int]] | None = None,
) -> None:
    """Add rows for a device class (a model's own sweep, or a test forcing a row).

    ``formulations`` maps ``(C, K, stride)`` to ``"direct"`` or a chunk width from ``TAP_FORMULATIONS``;
    ``slices`` maps ``(channels_run, K, stride)`` to ``(T_out_ref, num_slices_ref)``.
    """
    if formulations:
        for key, value in formulations.items():
            if value not in TAP_FORMULATIONS:
                raise ValueError(f"formulation {value!r} for {key} is not one of {TAP_FORMULATIONS}")
        _FORMULATIONS.setdefault(device_key, {}).update(formulations)
    if slices:
        for key, (t_ref, n_ref) in slices.items():
            if t_ref <= 0 or n_ref <= 0:
                raise ValueError(
                    f"slice row {key}: T_out_ref and num_slices_ref must be positive, got {(t_ref, n_ref)}"
                )
        _SLICES.setdefault(device_key, {}).update(slices)


def clear_tap_configs(device_key: DeviceKey) -> None:
    """Drop a device class's rows (tests)."""
    _FORMULATIONS.pop(device_key, None)
    _SLICES.pop(device_key, None)


def tap_formulation(device_key: DeviceKey, C: int, K: int, stride: int) -> Any:
    """The tabled formulation for ``(C, K, stride)``, or None when the table has no row."""
    return _FORMULATIONS.get(device_key, {}).get((C, K, stride))


def derive_num_slices(T_out: int, T_out_ref: int, num_slices_ref: int) -> int:
    """Slice count for ``T_out`` scaled from the reference: rounded up, at least 1, at most one per output row."""
    if T_out <= 0 or T_out_ref <= 0 or num_slices_ref <= 0:
        raise ValueError(
            f"T_out, T_out_ref and num_slices_ref must be positive, got {(T_out, T_out_ref, num_slices_ref)}"
        )
    return max(1, min(T_out, math.ceil(num_slices_ref * T_out / T_out_ref)))


def slice_config_for(num_slices: int):
    """Explicit width slicing with ``num_slices`` (never ``L1_FULL``: that selects conv2d's separate L1 path,
    measured 2x slower than a single DRAM slice)."""
    return ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=max(1, int(num_slices)))


def tap_slice_config(device_key: DeviceKey, channels: int, K: int, stride: int, T_out: int):
    """Explicit slice config for a conv run over ``channels`` channels at output length ``T_out``, or None when
    the device class has no slice row for ``(channels, K, stride)`` (then conv1d's auto-slicer decides)."""
    row = _SLICES.get(device_key, {}).get((channels, K, stride))
    if row is None:
        return None
    return slice_config_for(derive_num_slices(T_out, *row))


def slice_signature(slice_config) -> tuple[str, int] | None:
    """Hashable identity of a slice config (``None`` for auto) -- part of the prepared-weight cache key, because
    conv1d prepares its weight for the parallelization it runs with."""
    if slice_config is None:
        return None
    slice_type = slice_config.slice_type
    name = str(slice_type)
    for member, label in (
        (ttnn.Conv2dDRAMSliceWidth, "DRAM_WIDTH"),
        (ttnn.Conv2dDRAMSliceHeight, "DRAM_HEIGHT"),
        (ttnn.Conv2dL1Full, "L1_FULL"),
    ):
        if slice_type == member:
            name = label
            break
    return (name, int(slice_config.num_slices))


def format_formulation_row(C: int, K: int, stride: int, formulation: Any) -> str:
    return f"({C}, {K}, {stride}): {formulation!r},"


def format_slice_row(channels: int, K: int, stride: int, T_out_ref: int, num_slices_ref: int) -> str:
    return f"({channels}, {K}, {stride}): ({T_out_ref}, {num_slices_ref}),"
