# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for TTML Python tests: bf16 error metrics and mesh tensor reads."""

from __future__ import annotations

import numpy as np

import ttnn
import ttml

BF16_MANTISSA_BITS = 7
BF16_MIN_NORMAL = 2.0**-126  # bf16 goes subnormal below this and the spacing stops shrinking


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
    got, expected = np.asarray(got, np.float64), np.asarray(expected, np.float64)
    err = np.abs(got - expected)
    scale = float(np.abs(expected).max())
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
