# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures and helpers of the fabric_reduce_scatter pre-migration suite.

The repository runs pytest with ``--import-mode=importlib``, so the test files cannot import a sibling helper module;
the helpers are handed to them through the ``rs`` fixture instead. Every helper is independent of the operation: it
uploads partials, reads chips back and computes the torch reference (never the operation's own output)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import ttnn


def upload(mesh_device, parts, dtype=ttnn.bfloat16, *, layout=ttnn.TILE_LAYOUT, memory_config=None):
    """``parts`` [rows, cols, R, W]: chip (r, c) gets ``parts[r, c]`` as its [1, 1, R, W] partial."""
    rows, cols = tuple(mesh_device.shape)
    assert tuple(parts.shape[:2]) == (rows, cols), (tuple(parts.shape), (rows, cols))
    t = ttnn.from_torch(
        parts,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, 1)),
    )
    return ttnn.reshape(t, (1, 1, parts.shape[2], parts.shape[3]))


def per_chip(t):
    """Every chip's [1, 1, R, W] tensor as a float32 [R, W] torch tensor, in row-major mesh order."""
    return [ttnn.to_torch(d).float().reshape(d.shape[-2], d.shape[-1]) for d in ttnn.get_device_tensors(t)]


def group_size(mesh_device, cluster_axis):
    rows, cols = tuple(mesh_device.shape)
    return rows * cols if cluster_axis is None else (rows, cols)[cluster_axis]


def block(full, pos, g, dim):
    """Block ``pos`` of ``g``: rows (dim 2) or columns (dim 3) of a 2D tensor."""
    dim = dim % 4
    n, w = full.shape
    if dim == 2:
        return full[pos * n // g : (pos + 1) * n // g]
    return full[:, pos * w // g : (pos + 1) * w // g]


def expected(parts, cluster_axis, dim, *, acc=torch.float64):
    """Per chip (row-major), the block the chip owns of the sum over its group, computed in ``acc``.

    cluster_axis 0: the group is the chip's column, it owns block r; 1: its row, block c; None: every chip, block
    r * cols + c."""
    rows, cols = parts.shape[0], parts.shape[1]
    p = parts.to(acc)
    out = []
    for d in range(rows * cols):
        r, c = divmod(d, cols)
        if cluster_axis == 0:
            full, pos, g = p[:, c].sum(0), r, rows
        elif cluster_axis == 1:
            full, pos, g = p[r, :].sum(0), c, cols
        else:
            full, pos, g = p.sum((0, 1)), d, rows * cols
        out.append(block(full, pos, g, dim))
    return out


def int_parts(rows, cols, n, w, seed, lo=-4, hi=4):
    """Integer-valued partials: every partial and every partial sum of up to 32 of them (|sum| <= 128) is exact in
    bf16 and fp32 (and at TF32 / 16-bit DEST), so the expected output is exact and any stale, misplaced or doubled
    block shows."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(lo, hi + 1, (rows, cols, n, w), generator=g).float()


def fp32_spread_parts(rows, cols, n, w, seed):
    """The golden exact-fp32 inputs (test_fabric_rs_exact_fp32._partials): randn x 2^U[-12, 12], full mantissas."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, cols, n, w, generator=g, dtype=torch.float64)
    e = torch.randint(-12, 13, (rows, cols, n, w), generator=g).double()
    return (x * torch.pow(2.0, e)).float()


def out_shape(in_shape, g, dim):
    _, _, n, w = in_shape
    return [1, 1, n // g, w] if dim % 4 == 2 else [1, 1, n, w // g]


def in_shape(g, dim, block_tiles):
    """The [R, W] partial whose per-chip block is ``block_tiles`` = (tile rows, tile cols) for a group of ``g``."""
    br, bc = block_tiles
    return (32 * br * g, 32 * bc) if dim % 4 == 2 else (32 * br, 32 * bc * g)


def assert_output_meta(out, want_shape, dtype):
    assert list(out.shape) == want_shape, (list(out.shape), want_shape)
    assert list(out.padded_shape) == want_shape, (list(out.padded_shape), want_shape)
    assert out.dtype == dtype, out.dtype
    assert out.layout == ttnn.TILE_LAYOUT, out.layout
    mc = out.memory_config()
    assert mc.buffer_type == ttnn.BufferType.DRAM, mc
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, mc
    assert len(ttnn.get_device_tensors(out)) == out.device().get_num_devices()


def assert_exact(got, want, what=""):
    """Chip by chip, bit-identical values."""
    assert len(got) == len(want)
    for d, (g, w) in enumerate(zip(got, want)):
        w = w.float()
        assert g.shape == w.shape, (what, d, g.shape, w.shape)
        n_bad = int((g != w).sum())
        assert n_bad == 0, f"{what} dev {d}: {n_bad} of {w.numel()} values differ (max abs {(g - w).abs().max():.4g})"


def assert_max_abs(got, want, tol, what=""):
    worst = 0.0
    for d, (g, w) in enumerate(zip(got, want)):
        assert g.shape == w.shape, (what, d, g.shape, w.shape)
        worst = max(worst, float((g.double() - w.double()).abs().max()))
    assert worst <= tol, f"{what}: max abs error {worst:.4g} > {tol}"
    return worst


def cache_entries(mesh_device):
    return mesh_device.num_program_cache_entries()


@pytest.fixture
def rs():
    return SimpleNamespace(
        upload=upload,
        per_chip=per_chip,
        group_size=group_size,
        block=block,
        expected=expected,
        int_parts=int_parts,
        fp32_spread_parts=fp32_spread_parts,
        out_shape=out_shape,
        in_shape=in_shape,
        assert_output_meta=assert_output_meta,
        assert_exact=assert_exact,
        assert_max_abs=assert_max_abs,
        cache_entries=cache_entries,
    )
