# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for TTML Python tests: bf16 error metrics and device-mesh setup."""

from __future__ import annotations

import contextlib
import math
import os
import pathlib
from collections.abc import Generator
from typing import Optional

import numpy as np
import pytest

import ttnn
import ttml

TP_AXIS_SIZE = 2  # devices on the 'tp' axis for tests that need a multi-device mesh

BF16_MANTISSA_BITS = 7
BF16_MIN_NORMAL = 2.0**-126  # bf16 goes subnormal below this and the spacing stops shrinking

_MGD_ENV = "TT_MESH_GRAPH_DESC_PATH"
_MGD_DIR = pathlib.Path(__file__).resolve().parents[3] / "configs" / "mgd"  # tt-train/configs/mgd
_BUNDLED_MGD: dict[tuple[str, tuple[int, ...]], str] = {
    ("blackhole", (1, 2)): "bh_galaxy_1_2_line_line.textproto",
    ("wormhole_b0", (1, 2)): "n300_1_2_line_line.textproto",
}


def bf16_spacing(x):
    """Gap between adjacent bf16 values where ``x`` lands, i.e. one bf16 ULP at that magnitude."""
    x = np.maximum(np.abs(np.asarray(x, np.float64)), BF16_MIN_NORMAL)
    mantissa, exponent = np.frexp(x)
    binade = np.where(mantissa >= 1.0 - 2.0**-9, exponent, exponent - 1)
    return 2.0 ** (binade - BF16_MANTISSA_BITS)


def ulp_error(got, expected) -> tuple[float, float]:
    """``|got - expected|`` in bf16 ULP.

    Returns ``(peak_ulp, p99_ulp)``: the max error in ULP at ``max |expected|``, then the p99 of
    the per-element errors, each in ULP at its own element.
    """
    err = np.abs(got - expected)
    scale = max(float(np.abs(expected).max()), 1e-30)
    return (
        float(err.max() / bf16_spacing(scale)),
        float(np.percentile(err / bf16_spacing(expected), 99)),
    )


def assert_within_ulp(got, expected, label: str, max_ulp: float, max_ulp_p99: float = np.inf) -> None:
    got, expected = np.asarray(got, np.float64), np.asarray(expected, np.float64)
    assert got.shape == expected.shape, f"{label}: shape {got.shape} != {expected.shape}"
    ulp, ulp_p99 = ulp_error(got, expected)
    detail = f"{label}: ulp={ulp:.2f} (limit {max_ulp}), ulp_p99={ulp_p99:.2f} (limit {max_ulp_p99})"
    assert ulp <= max_ulp, detail
    assert ulp_p99 <= max_ulp_p99, detail


def mesh_composer(concat_dims: dict[str, int] | None = None):
    """Composer concatenating each mesh axis onto a distinct tensor dim."""
    mesh = ttml.mesh()
    dims = list(range(len(mesh.shape)))
    for axis_name, dim in (concat_dims or {}).items():
        dims[mesh.axis_index(axis_name)] = dim
    assert len(set(dims)) == len(dims), f"composer needs distinct dims per axis, got {dims}"
    device = ttml.autograd.AutoContext.get_instance().get_device()
    return ttnn.create_mesh_composer(device, ttnn.MeshComposerConfig(dims))


def read_mesh_tensor(tensor, concat_dims: dict[str, int] | None = None) -> np.ndarray:
    return tensor.to_numpy(ttnn.DataType.FLOAT32, composer=mesh_composer(concat_dims)).astype(np.float64)


def _close_mesh_quietly() -> None:
    with contextlib.suppress(Exception):
        ttml.close_device_mesh()


def _chip_count() -> int:
    """Chips present, counted from ``/dev/tenstorrent`` so the cluster stays uninitialised."""
    devices = pathlib.Path("/dev/tenstorrent")
    return sum(1 for entry in devices.iterdir() if entry.name.isdigit()) if devices.exists() else 0


def _bundled_mgd(shape: tuple[int, ...]) -> Optional[str]:
    try:
        arch = ttnn.get_arch_name().lower()
    except Exception:  # noqa: BLE001
        return None
    name = _BUNDLED_MGD.get((arch, tuple(shape)))
    path = str(_MGD_DIR / name) if name else None
    return path if path and os.path.isfile(path) else None


@contextlib.contextmanager
def device_mesh(shape: tuple[int, ...], axis_names: tuple[str, ...], reason: str) -> Generator[ttml.Mesh]:
    """Open a named device mesh for the duration of the block, skipping if unavailable.

    Points ``TT_MESH_GRAPH_DESC_PATH`` at a bundled MGD when the caller has not set one, and
    restores the environment afterwards so tests cannot leak mesh config into each other.
    """
    required = math.prod(shape)
    if (present := _chip_count()) < required:
        pytest.skip(f"{reason}: {present} chip(s) present, need {required}")

    previous = os.environ.get(_MGD_ENV)
    if not previous and (bundled := _bundled_mgd(shape)):
        os.environ[_MGD_ENV] = bundled

    def restore_env() -> None:
        if previous is None:
            os.environ.pop(_MGD_ENV, None)
        else:
            os.environ[_MGD_ENV] = previous

    _close_mesh_quietly()
    try:
        ttml.open_device_mesh(ttml.Mesh(tuple(shape), axis_names))
    except Exception as e:  # noqa: BLE001
        restore_env()
        pytest.skip(
            f"{reason}: mesh would not open, so this file's coverage is LOST, not unavailable. "
            f"Usually another test in this process already took a device -- run this file alone. ({e})"
        )
    try:
        yield ttml.mesh()
    finally:
        _close_mesh_quietly()
        restore_env()
