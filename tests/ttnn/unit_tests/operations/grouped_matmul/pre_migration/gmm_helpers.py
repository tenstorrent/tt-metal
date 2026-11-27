# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Test-only helpers of the grouped_matmul pre-migration suite: device tensors, per-chip readback and the fp64
reference (y[g] = x[g] @ w[g] of the operands as they sit on the device, i.e. after their dtype's quantization)."""

from __future__ import annotations

import torch
import ttnn

PKG = "tests.ttnn.unit_tests.operations.grouped_matmul.pre_migration"

# Relative Frobenius error of the device result against the fp64 product of the device's own (quantized) operands.
# HiFi2 (the op's default fidelity): the golden bank spec's bound for its bfloat8_b / bfloat16 / float32 outputs
# ("bfp8 inputs and weights vs their exact product: the shipped path's own error"). HiFi4 with bfloat16 operands and a
# float32 output: exact products, fp32 accumulation; 1e-3 still leaves ~3 decades of headroom. A wrong group, a lost K
# block or a misplaced column tile costs >= 0.2.
REL_TOL_HIFI2 = 2e-2
REL_TOL_HIFI4_EXACT = 1e-3


def compute_config(mesh, fidelity=ttnn.MathFidelity.HiFi2, fp32=True):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=fp32
    )


def randn(seed, *shape, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * scale


def operands(seed, G, M, K, N):
    """x [1, G, M, K] and w [1, G, K, N] (w scaled by 1/sqrt(K), as the golden spec). Every group gets its own scale
    (1 + g / 4), so a swapped or repeated group is visible in the values, not only in the noise."""
    x = randn(seed, 1, G, M, K)
    w = randn(seed + 1, 1, G, K, N, scale=K**-0.5)
    gscale = (1.0 + torch.arange(G, dtype=torch.float32) / 4).reshape(1, G, 1, 1)
    return x * gscale, w


def to_dev(mesh, t, dtype, memory_config=None, layout=ttnn.TILE_LAYOUT, mapper=None):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper or ttnn.ReplicateTensorToMesh(mesh),
    )


def chips(t):
    return [ttnn.to_torch(d).double() for d in ttnn.get_device_tensors(t)]


def chip0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).double()


def addr(t):
    return ttnn.get_device_tensors(t)[0].buffer_address()


def entries(mesh):
    return mesh.num_program_cache_entries()


def truth_of(x_dev, w_dev):
    """fp64 reference from the device operands (chip 0)."""
    return chip0(x_dev) @ chip0(w_dev)


def rel_err(got, truth):
    return float((got - truth).norm() / truth.norm())


def assert_output_spec(y, shape, dtype, buffer_type=ttnn.BufferType.DRAM):
    assert list(y.shape) == list(shape), (list(y.shape), shape)
    assert list(y.padded_shape) == list(shape), (list(y.padded_shape), shape)
    assert y.dtype == dtype, (y.dtype, dtype)
    assert y.layout == ttnn.TILE_LAYOUT, y.layout
    mc = y.memory_config()
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, mc
    assert mc.buffer_type == buffer_type, mc
    assert y.storage_type() == ttnn.StorageType.DEVICE


def assert_close(y, truth, tol, label=""):
    """Every chip equal to chip 0 (replicated operands), chip 0 within ``tol`` of the fp64 truth. Returns chip 0."""
    got = chips(y)
    for i, c in enumerate(got[1:], start=1):
        assert torch.equal(
            c, got[0]
        ), f"{label}: chip {i} differs from chip 0 (max |d| {float((c - got[0]).abs().max())})"
    assert torch.isfinite(got[0]).all(), f"{label}: non-finite output"
    rel = rel_err(got[0], truth)
    assert rel <= tol, f"{label}: rel error {rel:.3e} > {tol:.1e}"
    return got[0]
