# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Measured ``ttnn.conv1d`` configurations for the depthwise tap filter (``audio_ops.depthwise_tap_filter``).

The tap filter is the anti-alias resampler of the MiniMax-H3 / LTX audio decoders: a depthwise ``K``-tap conv1d
over ``(B, T_pad, C)``. ``conv1d`` runs it through conv2d's DRAM slicer, and whether a formulation fits L1 depends
on the activation block width ``C * K`` (the ``K`` sticks of a depthwise conv are laid out contiguously), so at
large ``C`` the full-channel conv never fits however finely ``T`` is sliced and the channels have to be split
into independent chunks. Until this module existed the filter *discovered* the fitting formulation by trial: each
miss was a failed op with a ``TT_FATAL`` log (18 per MiniMax-H3 decoder build at 15 s).

This module records the answers, the way ``utils/matmul.py`` (matmul blockings), ``utils/conv3d.py`` (conv3d
blockings) and ``layers/conv2d.py`` (conv2d DRAM slice counts) record theirs -- one table per device class,
measured by ``models/tt_dit/tests/models/minimax_h3/tools/sweep_tap_filter_configs.py``:

* ``_FORMULATIONS``: ``(C, K, stride) -> "direct" | chunk width``. Which formulation to run; independent of ``T``.
* ``_SLICES``: ``(channels_run, K, stride) -> (T_out_ref, num_slices_ref)``. The slice count the conv is fastest
  with at one measured length. Per-slice L1 use scales with the slice's output length, so the count for any
  other length is ``ceil(num_slices_ref * T_out / T_out_ref)`` (``derive_num_slices``): more slices than
  needed is always safe, merely a little slower, so the derivation rounds up. Handing conv1d this count as an
  explicit ``slice_config`` skips the slicer's search entirely -- it never runs, so it can never fail.

The tables are hints, never the authority: ``depthwise_tap_filter`` keeps its trial chain (direct, then the
chunk widths, then the shift-multiply-add fallback) behind every lookup, so a missing or stale row costs one
warning and one retry, never a failed call. Do not borrow rows across shapes -- a *fit* does not interpolate
the way a performance choice does; ``matmul.py``'s nearest-neighbour lookup is deliberately absent here.

Rows are keyed by ``tap_device_key`` = (architecture, compute grid); L1 per core differs between devices, so
a device class with no table simply probes as before. Regenerate a table with the sweep tool whenever the
conv1d/slicing implementation changes, and let ``tests/unit/test_audio_tap_path.py`` (which walks every
row on the device) tell you when a row has gone stale.
"""

from __future__ import annotations

import math
from typing import Any

import ttnn

# Conv1d formulations of the filter, widest first (the trial order); "mac" (shift-multiply-add) is appended by
# the caller as the last resort and is never tabled -- it is what runs when nothing here fits.
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


# ---------------------------------------------------------------------------------------------- tables
# Filled from sweep_tap_filter_configs.py; see the module docstring for the key/value meanings. Keep the
# sweep's provenance comment (date, host, l1_small_size, clip lengths) with every block so a row's origin is
# knowable when it goes stale.

_FORMULATIONS: dict[DeviceKey, dict[ShapeKey, Any]] = {}
_SLICES: dict[DeviceKey, dict[ShapeKey, tuple[int, int]]] = {}


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
    """Slice count for ``T_out`` from a reference measured at ``T_out_ref``: proportional, rounded up, at least 1,
    at most one slice per output row."""
    if T_out <= 0:
        raise ValueError(f"T_out must be positive, got {T_out}")
    return max(1, min(T_out, math.ceil(num_slices_ref * T_out / T_out_ref)))


def slice_config_for(num_slices: int):
    """The explicit conv slice config for a count, along the sequence (conv1d's only sliceable dimension).

    One slice is passed as ``num_slices=1``, not as ``L1_FULL``: inside the slicer a provided count of 1 takes the
    same single-L1-op route the auto path converts a one-slice answer to, whereas an explicit ``L1_FULL`` selects
    conv2d's separate L1 execution path (measured 2x slower for the 512-channel, 128-chunk filter, 2026-09-09).
    """
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
