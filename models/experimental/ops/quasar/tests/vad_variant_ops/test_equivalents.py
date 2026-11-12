# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Stock-ttnn equivalents of the fork's custom ops, at the shapes the model ran.

The model's fork adds ops that are not in
main, so they can't be called here. Their compile-time args were decoded from the cache
(fork-specific; interleaved tensor accessors are ``(cfg, page_size)``, cfg 2 = DRAM) and each
is exercised below through the stock op the reference port (``models/experimental/vadv2``)
uses for the same step of spatial cross-attention (SCA) / temporal self-attention (TSA):

Recovered invocations
---------------------
* Deformable attention (reader ct[12..22] = batch, queries, heads, keys, levels, points,
  100, 100, value pages, ..., total sticks; ``total = B*Q*H*L*P/8`` holds for all 5) [H]:
    TSA: B=2 (BEV queue), Q=10000, H=8, D=32, value 100x100, L=1, P=4 (``is_QHB``: mean over B)
    SCA: (B, Q) = (1, 3680), (1, 3072), (2, 1376), (2, 992); value 12x20 (240 keys), L=1, P=8.
  The SCA groups cover the 6 cameras (1+1+2+2) with hit counts padded to a multiple of 32.
  The fork samples with pixel coordinates and a 1/100 bilinear-weight lookup table; the
  stock op takes normalized [-1, 1] coordinates, so values match only to PCC.
* nonzero: input ``bev_mask.sum(-1)`` row (1, 1, 1, 10000) bf16, UInt32 index output [H].
* getitem: Int32 index on dim -2 — ``query[:, idx, :]`` gather from (1, 10000, 256) [M].
* setitem: index-tensor scatter back into ``slots`` (1, 10000, 256) [M]; plus an fp32
  slice assignment writing 16-wide rows into a 32-wide fp32 DRAM tensor [M rows / H widths].
* reshard: 16-wide bf16 rows (32-byte units) DRAM -> L1 shards over 20 cores [H width / L rows].

Quasar status: of these, only reshard / slice_write have Quasar ports in
``ttnn/cpp/ttnn/operations/experimental/quasar``; the others need ports before they pass there.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U

Q, E, H, D = U.NUM_QUERY, U.EMBED, U.NUM_HEADS, U.HEAD_DIM


# =============================================================================
# Deformable attention
# =============================================================================


def _reference_msda(value, grid, attn):
    """value (N, h, w, D), grid (N, Q*P, 1, 2) in [-1, 1], attn (N, Q, P) -> (N, Q, D).

    Same contract as tests/ttnn/unit_tests/operations/experimental/test_multi_scale_deformable_attn.py.
    """
    n, _, _, d = value.shape
    q, p = attn.shape[1], attn.shape[2]
    sampled = torch.nn.functional.grid_sample(
        value.permute(0, 3, 1, 2), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )  # (N, D, Q*P, 1)
    sampled = sampled.squeeze(-1).permute(0, 2, 1).reshape(n, q, p, d)
    return (sampled * attn.unsqueeze(-1)).sum(dim=2)


# fmt: off
# (batch, num_query, value_h, value_w, points) — per-head N = batch * heads
MSDA_CASES = [
    pytest.param(U.BEV_QUEUE, Q, U.BEV_H, U.BEV_W, U.TSA_POINTS, id="tsa-b2-q10000-100x100-p4"),
    pytest.param(1, 3680, U.FEAT_H, U.FEAT_W, U.SCA_POINTS, id="sca-b1-q3680-12x20-p8"),
    pytest.param(1, 3072, U.FEAT_H, U.FEAT_W, U.SCA_POINTS, id="sca-b1-q3072-12x20-p8"),
    pytest.param(2, 1376, U.FEAT_H, U.FEAT_W, U.SCA_POINTS, id="sca-b2-q1376-12x20-p8"),
    pytest.param(2, 992, U.FEAT_H, U.FEAT_W, U.SCA_POINTS, id="sca-b2-q992-12x20-p8"),
]
# fmt: on


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize("batch, num_query, value_h, value_w, points", MSDA_CASES)
def test_multi_scale_deformable_attn(ttnn_mesh_device, reset_seeds, batch, num_query, value_h, value_w, points):
    mesh = ttnn_mesh_device
    n = batch * H
    # Round inputs to bf16 first so the reference sees exactly what the device sees.
    value = torch.randn(n, value_h, value_w, D).to(torch.bfloat16).float()
    grid = (torch.rand(n, num_query * points, 1, 2) * 2.0 - 1.0).to(torch.bfloat16).float()
    attn = torch.softmax(torch.randn(n, num_query, points), dim=-1).to(torch.bfloat16).float()

    def to_dev(t):
        return U.to_tt(t.to(torch.bfloat16), mesh, layout=ttnn.ROW_MAJOR_LAYOUT)

    out = ttnn.experimental.multi_scale_deformable_attn(to_dev(value), to_dev(grid), to_dev(attn), align_corners=False)
    U.assert_pcc(_reference_msda(value, grid, attn), out, pcc=0.99, mesh_device=mesh)


# =============================================================================
# nonzero  (non_zero_indices_sc_reader)
# =============================================================================


@U.with_default_mesh()
@pytest.mark.parametrize("num_hits", [3680, 3072, 1376, 992], ids=lambda h: f"hits{h}")
def test_nonzero(ttnn_mesh_device, reset_seeds, num_hits):
    """Per-camera BEV hit indices: nonzero over the (1, 1, 1, 10000) hit-count row."""
    U.xfail_on_quasar("ttnn.nonzero has no Quasar port yet (see module docstring)")
    mesh = ttnn_mesh_device
    row = torch.zeros(1, 1, 1, Q, dtype=torch.bfloat16)
    hit_idx = torch.randperm(Q)[:num_hits]
    row[..., hit_idx] = torch.randint(1, 5, (num_hits,)).to(torch.bfloat16)
    x = U.to_tt(row, mesh, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    count_t, coords_t = ttnn.nonzero(x, memory_config=ttnn.L1_MEMORY_CONFIG)

    count = int(ttnn.to_torch(count_t).reshape(-1)[0].item())
    assert count == num_hits, f"nonzero count {count} != {num_hits}"
    # coords mirror torch.nonzero(as_tuple=False): count 4-tuples (b, n, h, w); w is the query index.
    coords = ttnn.to_torch(coords_t).reshape(-1)[: count * 4].reshape(count, 4)[:, 3].long()
    assert torch.equal(coords, torch.sort(hit_idx).values), "nonzero indices differ from torch"


# =============================================================================
# getitem  (reader/writer_getitem): query[:, idx, :] gather on dim -2
# =============================================================================


@pytest.mark.blackhole_scale  # (1, 10000, 256) source allocated in-body: too big for the emulator
@U.with_default_mesh()
@pytest.mark.parametrize("num_idx", [3680, 992], ids=lambda n: f"idx{n}")
def test_embedding_query_gather(ttnn_mesh_device, reset_seeds, num_idx):
    """vadv2's stock form of the fork's getitem: ttnn.embedding over the (10000, 256) query table."""
    mesh = ttnn_mesh_device
    table = U.torch_rand((Q, E))
    idx = torch.randint(0, Q, (1, num_idx), dtype=torch.int32)
    tt_table = U.to_tt(table, mesh, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_idx = U.to_tt(idx, mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    out = ttnn.embedding(tt_idx, tt_table, layout=ttnn.TILE_LAYOUT)
    U.assert_lossless(table[idx.long()].reshape(1, num_idx, E), out, mesh_device=mesh)


@pytest.mark.blackhole_scale  # (1, 10000, 256) source allocated in-body: too big for the emulator
@U.with_default_mesh()
@pytest.mark.parametrize("num_idx", [3680, 992], ids=lambda n: f"idx{n}")
def test_moreh_getitem_query_gather(ttnn_mesh_device, reset_seeds, num_idx):
    """The op the fork's getitem was derived from: moreh getitem, Int32 index on dim 1 of (1, 10000, 256)."""
    mesh = ttnn_mesh_device
    x = U.torch_rand((1, Q, E))
    idx = torch.randint(0, Q, (num_idx,), dtype=torch.int32)
    tt_x = U.to_tt(x, mesh, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_idx = U.to_tt(idx, mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    out = ttnn.operations.moreh.getitem(tt_x, [tt_idx], [1])
    U.assert_lossless(x[:, idx.long()], out, mesh_device=mesh)


# =============================================================================
# setitem  (setitem_*)
# =============================================================================


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize("max_len", [3680, 992], ids=lambda n: f"maxlen{n}")
def test_scatter_add_slots(ttnn_mesh_device, reset_seeds, max_len):
    """vadv2's stock form of the SCA scatter: scatter_add rows of the per-camera attention
    output into slots (10000 + 1 sentinel row, 256), padded indices routed to the sentinel."""
    mesh = ttnn_mesh_device
    slots = U.torch_rand((Q + 1, E))
    idx_rows = []
    for _ in range(U.NUM_CAMS):
        hits = torch.randperm(Q)[: max_len - 32]
        idx_rows.append(torch.cat([hits, torch.full((32,), Q)]))  # sentinel padding
    idx = torch.cat(idx_rows).unsqueeze(1).expand(-1, E).contiguous()
    src = U.torch_rand((U.NUM_CAMS * max_len, E))

    tt_slots = U.to_tt(slots, mesh)
    tt_idx = U.to_tt(idx.to(torch.int32), mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_src = U.to_tt(src, mesh)

    out = ttnn.scatter_add(tt_slots, 0, tt_idx, tt_src)
    ref = slots.float().scatter_add(0, idx.long(), src.float())
    U.assert_pcc(ref, out, pcc=0.999, mesh_device=mesh)


@U.with_default_mesh()
@pytest.mark.parametrize(
    "out_shape, value_width",
    # setitem fp32 variant ct [2, 128, 2, 64]: DRAM fp32 rows of 32 (128 B) receive 16-wide
    # (64 B) value rows. Row count is a runtime arg (lost); 300 decoder queries assumed.
    [pytest.param((1, 1, 300, 32), 16, id="fp32-rows32-write16")],
)
def test_slice_write_fp32(ttnn_mesh_device, reset_seeds, out_shape, value_width):
    mesh = ttnn_mesh_device
    dst = torch.zeros(out_shape, dtype=torch.float32)
    value = torch.randn(*out_shape[:-1], value_width, dtype=torch.float32)
    tt_dst = U.to_tt(dst, mesh, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_value = U.to_tt(
        value, mesh, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    begins = [0, 0, 0, 0]
    ends = [*out_shape[:-1], value_width]
    ttnn.experimental.slice_write(tt_value, tt_dst, begins, ends, [1, 1, 1, 1])

    ref = dst.clone()
    ref[..., :value_width] = value
    U.assert_lossless(ref, tt_dst, mesh_device=mesh)


# =============================================================================
# reshard  (reshard_same_width_reader): 16-wide bf16 rows, DRAM -> L1 shards
# =============================================================================


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape, shard_h",
    # Row count not recoverable [L]; BEV-sized 10000 x 16 over 20 cores (512-row shards).
    [pytest.param((1, 1, Q, 16), 512, id="rm-w16-hs20")],
)
def test_reshard_same_width(ttnn_mesh_device, reset_seeds, shape, shard_h):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh, layout=ttnn.ROW_MAJOR_LAYOUT)
    out = ttnn.to_memory_config(x, U.hs20(mesh, (shard_h, shape[-1])))
    assert out.is_sharded()
    U.assert_lossless(x_torch, out, mesh_device=mesh)
