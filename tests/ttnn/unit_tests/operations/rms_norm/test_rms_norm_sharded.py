# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinements 2 + 4 — WIDTH_SHARDED / BLOCK_SHARDED / HEIGHT_SHARDED rms_norm.

Covers every combine topology and shard-placement path:
  * k = 1 (one shard column), k <= 8 symmetric all-gather, k > 8 per-row owner gather + inv_rms return;
  * WIDTH groups spanning several grid rows (auto config over the whole grid);
  * ragged shards (Split.PAD last column / row), pinned perf geometries;
  * W tail (masked in the collapse reduce — the caller's shard must not change), H padding, poisoned padding;
  * ROW_MAJOR stick shards (sub-tile shard widths, local tilize), fp32 / bf16 / bf8b;
  * HEIGHT (k = 1, local reduction): row shards over the whole grid (multi-range grid, ragged last shard).
"""

import pytest
import torch
import ttnn

from eval.sharding import auto_shard_config, shard_config
from ttnn.operations.rms_norm import rms_norm

WS = ttnn.TensorMemoryLayout.WIDTH_SHARDED
BS = ttnn.TensorMemoryLayout.BLOCK_SHARDED
HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def _ref(x, gamma, eps):
    xf = x.float()
    y = xf / torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)
    return y * gamma.float().reshape(-1) if gamma is not None else y


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _run(
    device,
    shape,
    ml,
    *,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat16,
    gamma_layout="same",
    fp32_dest=True,
    fidelity=ttnn.MathFidelity.HiFi4,
    pinned=None,
    poison=None,
    pcc=0.999,
):
    torch.manual_seed(0)
    tdt = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    x = torch.randn(shape, dtype=torch.float32).to(tdt)
    W = shape[-1]
    gamma_t = None
    g = None
    if gamma_layout is not None:
        gl = layout if gamma_layout == "same" else gamma_layout
        gamma_t = torch.randn(W, dtype=torch.float32).to(torch.bfloat16)
        g = ttnn.from_torch(
            gamma_t.reshape(1, 1, 1, W),
            dtype=ttnn.bfloat16,
            layout=gl,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    if pinned is not None:
        mc = shard_config(pinned[0], pinned[1], ml, layout=layout, dtype=dtype, device=device)
    else:
        mc = auto_shard_config(list(shape), ml, layout=layout, dtype=dtype, device=device)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=mc)
    if poison is not None:
        xt = ttnn.fill_implicit_tile_padding(xt, poison)
    before = ttnn.to_torch(xt)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = fidelity
    cfg.fp32_dest_acc_en = fp32_dest
    cfg.math_approx_mode = False
    out = rms_norm(xt, gamma=g, epsilon=1e-6, compute_kernel_config=cfg, memory_config=xt.memory_config())
    assert out.memory_config() == mc
    got = ttnn.to_torch(out).float()
    if dtype == ttnn.bfloat8_b:
        x = ttnn.to_torch(xt)  # reference on the device-quantized input
    exp = _ref(x, gamma_t, 1e-6)
    p = _pcc(got, exp)
    assert p >= pcc, f"PCC {p} < {pcc}; max abs {(got - exp).abs().max()}"
    # The caller's input shard is never mutated (W-tail masking happens in compute, not in place).
    assert torch.equal(ttnn.to_torch(xt), before)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 2048),  # WIDTH k=8 (symmetric) on BH 13x10 auto: 64 tiles / 130 -> 64 one-tile shards
        (1, 1, 32, 64),  # 2 one-tile shards
        (1, 1, 32, 32),  # k = 1
        (1, 1, 128, 4096),  # k = 128 owner combine, 4 tile-rows -> 4 owners
        (1, 1, 32, 8192),  # k = 128, 2 tiles per shard, 1 tile-row
        (4, 8, 32, 256),  # many tile-rows per shard
    ],
)
def test_width_tile(device, shape):
    _run(device, shape, WS)


@pytest.mark.parametrize("shape", [(1, 1, 256, 512), (1, 1, 2048, 256), (2, 4, 128, 512), (1, 1, 1024, 1024)])
def test_block_tile(device, shape):
    _run(device, shape, BS)


@pytest.mark.parametrize("ml", [WS, BS, HS])
@pytest.mark.parametrize(
    "shape", [(1, 1, 32, 50), (2, 1, 100, 47), (1, 1, 17, 64), (1, 1, 416, 1025), (1, 1, 32, 4095)]
)
def test_non_aligned_tile(device, ml, shape):
    _run(device, shape, ml)


@pytest.mark.parametrize("ml", [WS, BS, HS])
@pytest.mark.parametrize("shape", [(1, 1, 32, 40), (1, 1, 32, 200), (1, 1, 224, 72), (1, 1, 40, 40)])
def test_poisoned_padding(device, ml, shape):
    _run(device, shape, ml, fp32_dest=False, poison=1000.0)


@pytest.mark.parametrize("ml", [WS, BS, HS])
@pytest.mark.parametrize(
    "shape", [(1, 1, 32, 64), (1, 1, 256, 512), (2, 1, 100, 47), (1, 1, 32, 4096), (1, 1, 333, 1000), (1, 32, 50)]
)
def test_row_major(device, ml, shape):
    _run(device, shape, ml, layout=ttnn.ROW_MAJOR_LAYOUT)


@pytest.mark.parametrize("ml", [WS, BS, HS])
def test_fp32_and_no_gamma(device, ml):
    _run(device, (1, 1, 64, 1024), ml, dtype=ttnn.float32, gamma_layout=None)
    _run(device, (1, 1, 64, 1000), ml, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)


@pytest.mark.parametrize("ml", [WS, BS, HS])
def test_bf8b_rm_gamma(device, ml):
    _run(device, (1, 1, 64, 2048), ml, dtype=ttnn.bfloat8_b, gamma_layout=ttnn.ROW_MAJOR_LAYOUT, pcc=0.99)


@pytest.mark.parametrize(
    "rows,W,ml,shard",
    [
        (32, 1024, WS, ([32, 128], (8, 1))),
        (32, 2304, WS, ([32, 256], (9, 1))),
        (32, 5120, WS, ([32, 160], (8, 4))),  # k = 32 owner combine over 4 grid rows
        (32, 7168, WS, ([32, 256], (7, 4))),
        (640, 6144, BS, ([64, 704], (9, 10))),  # ragged last shard column
        (640, 7168, BS, ([64, 1024], (7, 10))),
        (7168, 1024, BS, ([736, 128], (8, 10))),  # ragged last shard row
        (8192, 1024, BS, ([1024, 128], (8, 8))),
    ],
)
def test_pinned_perf_geometry(device, rows, W, ml, shard):
    _run(
        device,
        (1, 1, rows, W),
        ml,
        gamma_layout=ttnn.TILE_LAYOUT,
        fp32_dest=False,
        fidelity=ttnn.MathFidelity.HiFi2,
        pinned=shard,
        pcc=0.9995,
    )


@pytest.mark.parametrize(
    "shape,layout",
    [
        ((7136, 736), ttnn.TILE_LAYOUT),  # k = 23 owner combine, 223 tile-rows -> several row blocks
        ((1, 1, 4064, 160), ttnn.ROW_MAJOR_LAYOUT),  # k = 20 RM stick shards, several row blocks
    ],
)
def test_owner_combine_multi_block(device, shape, layout):
    """Members own different row counts per block: cb_inv_owned's read pointer must wrap correctly."""
    _run(device, shape, WS, layout=layout, fp32_dest=False)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 256, 512),  # golden loose case: 8 one-tile-row shards
        (1, 1, 32, 32),  # one shard, one tile
        (2, 4, 128, 512),  # leading dims folded into the row shards
        (1, 1, 8192, 1024),  # 256 tile-rows over the whole grid: multi-range grid + ragged last shard
        (7136, 736),  # Rt = 223 (prime): ragged last shard
        (99991, 64),  # Rt = 3125, H tail in the last shard
        (100, 5120),  # wide rows, 4 tile-rows
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_height(device, shape, layout):
    _run(device, shape, HS, layout=layout, fp32_dest=False, fidelity=ttnn.MathFidelity.HiFi2)


def test_height_rm_tile_gamma(device):
    """HEIGHT shards hold whole rows, so a ROW_MAJOR input pairs with a TILE gamma (W-split RM shards cannot)."""
    _run(device, (1, 1, 333, 1000), HS, layout=ttnn.ROW_MAJOR_LAYOUT, gamma_layout=ttnn.TILE_LAYOUT)


@pytest.mark.parametrize(
    "shape,dtype,gamma_dtype,gamma_layout",
    [
        ((1, 1, 32, 8192), ttnn.bfloat16, ttnn.bfloat16, ttnn.TILE_LAYOUT),  # 512 KB shards: resident gamma OOM
        ((1, 1, 32, 8192), ttnn.bfloat8_b, ttnn.float32, ttnn.TILE_LAYOUT),
        ((1, 1, 32, 4096), ttnn.float32, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
        ((1, 1, 160, 11008), ttnn.bfloat16, ttnn.bfloat16, ttnn.TILE_LAYOUT),  # 704 KB shards, ~52 KB left for CBs
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_height_streamed_gamma(device, shape, dtype, gamma_dtype, gamma_layout, layout):
    """Wide-W HEIGHT cells that do not fit with the resident gamma / RM chunk: gamma streams, RM chunk shrinks."""
    if layout == ttnn.ROW_MAJOR_LAYOUT and (dtype == ttnn.bfloat8_b or gamma_layout == ttnn.TILE_LAYOUT):
        pytest.skip("bfloat8_b has no ROW_MAJOR form; RM input pairs with RM gamma here")
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32)
    W = shape[-1]
    gamma_t = torch.randn(W, dtype=torch.float32)
    g = ttnn.from_torch(gamma_t.reshape(1, 1, 1, W), dtype=gamma_dtype, layout=gamma_layout, device=device)
    mc = auto_shard_config(list(shape), HS, layout=layout, dtype=dtype, device=device)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=mc)
    out = rms_norm(xt, gamma=g, epsilon=1e-6, memory_config=mc)
    exp = _ref(ttnn.to_torch(xt).float(), ttnn.to_torch(g).float().reshape(-1), 1e-6)
    p = _pcc(ttnn.to_torch(out).float(), exp)
    assert p >= (0.99 if dtype == ttnn.bfloat8_b else 0.999), p


@pytest.mark.parametrize(
    "shape,ml,chunk",
    [
        ((1, 1, 2048, 1024), HS, 5),  # several tile-rows per shard, ragged last chunk (32 = 6 * 5 + 2)
        ((1, 1, 416, 1025), HS, 4),  # W tail (masked) + ragged chunk
        ((1, 1, 1024, 1024), BS, 3),  # k > 1 owner combine + streamed gamma
        ((1, 1, 64, 4095), WS, 2),  # ragged last shard column: gap tiles in the last chunk
        # Perf 1 regression: 223 tile-rows over 130 Tensix cores -> 2 tile-rows per shard x 2 gamma windows. Each
        # window must land row-major in the shard-backed output (was packed window-major: PCC 0.50).
        ((7136, 736), HS, 16),
    ],
)
@pytest.mark.parametrize("gamma_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_forced_streamed_gamma(device, monkeypatch, shape, ml, chunk, gamma_layout):
    """The streamed-gamma regime on shapes where resident gamma fits (forced): multi-block, tails, gaps, combine."""
    from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

    monkeypatch.setattr(pd, "GAMMA_STREAM_FORCE_CHUNK", chunk)
    _run(device, shape, ml, gamma_layout=gamma_layout)


@pytest.mark.parametrize(
    "shape,ml,chunk_cap,pinned",
    [
        # 8 tile-rows per shard x 4 chunks: compute blocks on cb_out_rm while > depth gamma chunks are still due
        ((1, 1, 2048, 1024), HS, 8, ([256, 1024], (8, 1))),
        ((1, 1, 333, 1000), HS, 4, None),  # ragged chunk pitch (32 = 4 * 8) + H tail
        ((1, 1, 256, 2000), BS, 4, None),  # k > 1, W tail inside the last member's chunk
    ],
)
def test_forced_streamed_gamma_rm(device, monkeypatch, shape, ml, chunk_cap, pinned):
    """RM streamed gamma (G = C) with a lowered chunk cap: > depth chunks per block, several tile-rows per block."""
    from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

    monkeypatch.setattr(pd, "GAMMA_STREAM_FORCE_CHUNK", 1)
    monkeypatch.setattr(pd, "RM_CHUNK_CAP", chunk_cap)
    _run(device, shape, ml, layout=ttnn.ROW_MAJOR_LAYOUT, gamma_layout=ttnn.ROW_MAJOR_LAYOUT, pinned=pinned)
