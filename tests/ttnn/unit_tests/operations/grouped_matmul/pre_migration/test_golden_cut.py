# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden coverage cut of grouped_matmul: the golden specs' cases, judged by their own references and tolerances.

Golden specs (run once by the driver after the port; not run here):
    ttnn/ttnn/bringup/grouped_matmul_ttnn/tests/unit/test_grouped_matmul.py       (interleaved path)
    ttnn/ttnn/bringup/grouped_matmul_ttnn/tests/unit/test_grouped_matmul_bank.py  (w_layout, bank-local weights)
Neither declares SUPPORTED / INPUTS tables: their axes are the columns of their CASES lists. Every value of every
column appears below at least once, with the smallest and the largest case of each spec (by work G*M*K*N):
    interleaved  G {1,2,3,8}  M {32,64,96,160}  K {256,512,1024,4096}  N {64,96,352,1024}  out {bf8,bf16}
                 grid {11x10, 4x3, 1x2}  block_w {None,4,8}  depth {2,3,4}       (4 of its 5 cases; the 5th is the
                 wo_a at block_w None, whose values the others cover)
    bank         G {4,8}  M {32,64,96,160,256}  K {256,512,1024,4096}  out {bf8,bf16,f32}  block_w {2,4}
                 depth {2,3,4}  rt {1,2}  cores_per_bank {10,11,12}  w_inflight {1,2,3}  x_ahead {0,1,2,3}
                 shape {band,row}  sender {start,anchor}                          (7 of its 8 cases)
References: interleaved: fp64 product of the device operands, error <= 1.05 x the stock batched ttnn.matmul's + 1e-6,
every chip equal, a cache hit on fresh buffers into a caller's output the same bytes. Bank: bit-identical to the
interleaved path at the same knobs on every chip, a repeat on a fresh x bit-identical, rel error < 2e-2 vs fp64.
Inputs generated exactly as the golden specs generate them (seeds, scales, bfloat8_b operands).
"""

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.grouped_matmul.pre_migration.gmm_helpers import chip0, chips, compute_config
from tests.ttnn.unit_tests.operations.grouped_matmul.pre_migration.test_acceptance import (  # noqa: F401
    grouped_matmul_op,
)
from ttnn.bringup.grouped_matmul_ttnn import bank


def _dev(t, mesh, dtype=ttnn.bfloat8_b):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


INTERLEAVED = [  # (G, M, K, N, output dtype, grid, block_w, depth): rows of the golden spec's CASES
    (1, 64, 256, 96, ttnn.bfloat16, (11, 10), None, 2),  # smallest
    (2, 32, 1024, 64, ttnn.bfloat16, (1, 2), None, 2),
    (3, 96, 512, 352, ttnn.bfloat8_b, (4, 3), 8, 3),
    (8, 160, 4096, 1024, ttnn.bfloat8_b, (11, 10), 4, 4),  # largest: the 8x4's wo_a
]


@pytest.mark.parametrize(
    "G, M, K, N, odt, grid, bw, depth",
    INTERLEAVED,
    ids=[f"G{c[0]}-M{c[1]}-K{c[2]}-N{c[3]}-bw{c[6]}-d{c[7]}-{c[5][0]}x{c[5][1]}" for c in INTERLEAVED],
)
def test_golden_interleaved(gmm_mesh, grouped_matmul_op, G, M, K, N, odt, grid, bw, depth):
    mesh = gmm_mesh
    g = torch.Generator().manual_seed(G * 7 + M)
    xt = torch.randn(1, G, M, K, generator=g)
    wt = torch.randn(1, G, K, N, generator=g) / K**0.5
    x, w = _dev(xt, mesh), _dev(wt, mesh)
    ckc = compute_config(mesh)
    kw = dict(dtype=odt, compute_kernel_config=ckc, max_grid=ttnn.CoreCoord(*grid), block_w=bw, depth=depth)
    y = grouped_matmul_op(x, w, **kw)
    assert list(y.shape) == [1, G, M, N] and y.dtype == odt and y.layout == ttnn.TILE_LAYOUT
    truth = chip0(x) @ chip0(w)
    stock = ttnn.matmul(x, w, dtype=odt, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    got = chips(y)
    rel = float((got[0] - truth).norm() / truth.norm())
    rel_ref = float((chip0(stock) - truth).norm() / truth.norm())
    assert rel <= rel_ref * 1.05 + 1e-6, (rel, rel_ref)
    for i, c in enumerate(got[1:], start=1):
        assert torch.equal(c, got[0]), f"chip {i} differs"
    x2, w2 = _dev(xt, mesh), _dev(wt, mesh)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, G, M, N]), odt, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
    y2 = grouped_matmul_op(x2, w2, output_tensor=out, **kw)
    assert y2 is out
    assert torch.equal(chip0(y2), got[0])


BANK = [  # (G, M, K, N, output dtype, block_w, depth, rt, cores_per_bank, w_inflight, x_ahead, shape, sender)
    (8, 32, 256, 1024, ttnn.bfloat8_b, 2, 2, 1, 12, 1, 0, "band", "start"),  # smallest
    (8, 32, 256, 1024, ttnn.bfloat8_b, 2, 3, 1, 12, 1, 2, "band", "anchor"),
    (8, 32, 256, 1024, ttnn.bfloat8_b, 2, 2, 1, 11, 2, 1, "row", "start"),
    (4, 64, 512, 1024, ttnn.bfloat16, 4, 3, 1, 12, 2, 1, "band", "start"),  # two banks a group
    (8, 96, 1024, 1024, ttnn.float32, 4, 4, 2, 10, 3, 3, "band", "start"),
    (8, 160, 4096, 1024, ttnn.bfloat8_b, 4, 4, 1, 12, 2, 2, "band", "start"),  # the 8x4's wo_a at 5120
    (8, 256, 4096, 1024, ttnn.bfloat8_b, 2, 2, 1, 12, 1, 1, "band", "start"),  # largest: the wo_a at 8192
]


@pytest.mark.parametrize(
    "G, M, K, N, odt, bw, depth, rt, cpb, wi, xa, shape, sender",
    BANK,
    ids=[
        f"G{c[0]}-M{c[1]}-K{c[2]}-N{c[3]}-bw{c[5]}-d{c[6]}-rt{c[7]}-c{c[8]}-wi{c[9]}-xa{c[10]}-{c[11]}-{c[12]}"
        for c in BANK
    ],
)
def test_golden_bank(gmm_mesh, grouped_matmul_op, G, M, K, N, odt, bw, depth, rt, cpb, wi, xa, shape, sender):
    mesh = gmm_mesh
    g = torch.Generator().manual_seed(G * 7 + M + K)
    xt = torch.randn(1, G, M, K, generator=g)
    wt = torch.randn(1, G, K, N, generator=g) / K**0.5
    grid = mesh.compute_with_storage_grid_size()
    lay = bank.plan(mesh, G, K, N, max_cores=110, cores_per_bank=cpb, shape=shape, sender=sender)
    assert lay is not None, (G, K, N, cpb, shape, sender)
    assert len(lay.slots) <= 110 and len({s.core for s in lay.slots}) == len(lay.slots)
    assert all(0 <= s.core[0] < grid.x and 0 <= s.core[1] < grid.y for s in lay.slots)
    x, w = _dev(xt, mesh), _dev(wt, mesh)
    wb = bank.upload(wt, lay, mesh)
    kw = dict(
        dtype=odt,
        compute_kernel_config=compute_config(mesh),
        max_grid=ttnn.CoreCoord(11, 10),
        block_w=bw,
        depth=depth,
        rt=rt,
    )
    today = grouped_matmul_op(x, w, **kw)
    y = grouped_matmul_op(x, wb, w_layout=lay, w_inflight=wi, x_ahead=xa, **kw)
    assert list(y.shape) == [1, G, M, N] and y.dtype == odt
    truth = chip0(x) @ chip0(w)
    got, ref = chips(y), chips(today)
    x2 = _dev(xt, mesh)
    y2 = grouped_matmul_op(x2, wb, w_layout=lay, w_inflight=wi, x_ahead=xa, **kw)
    rep = chips(y2)
    for i in range(len(got)):
        assert torch.equal(got[i], ref[i]), f"chip {i}: bank path differs from the interleaved path"
        assert torch.equal(rep[i], got[i]), f"chip {i}: bank path not repeatable"
        assert torch.equal(got[i], got[0]), f"chip {i} differs from chip 0"
    rel = float((got[0] - truth).norm() / truth.norm())
    assert rel < 2e-2, rel
    for t in (x, w, wb, today, y, x2, y2):
        ttnn.deallocate(t)
