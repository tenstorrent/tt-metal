# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark: achieved matmul-FLOP/s of the SDPA kernel vs the k-loop length, to test WHY the fused
neighborhood SDPA sits far below peak. Three probes on ONE chip:
  (1) a deep GEMM  -> empirical peak TFLOP/s on this device.
  (2) DENSE SDPA sweeping S (short->long k-loop, clean path: no gather/no per-element mask).
  (3) NEIGHBORHOOD block SDPA (short box k-loop + gather + per-element mask + box slack).
FLOP/s = analytic matmul FLOPs / measured device time. If (2) climbs with S and (3) << dense-long, the
k-loop length (amortization) drives it; if dense-short is already high, the neighborhood-unique overheads do.
NOT the production grid (dense is O(S^2) -> OOM there); this isolates the mechanism on tractable shapes."""
import time

import torch

import ttnn


def _sync_time(device, fn, iters=10):
    fn()  # warmup / compile
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / iters


def _gemm_peak(device):
    """Deep square GEMM -> practical peak TFLOP/s (contraction = 4096, far from attention's 64)."""
    M = K = N = 4096
    a = ttnn.from_torch(torch.randn(M, K, dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
    b = ttnn.from_torch(torch.randn(K, N, dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
    t = _sync_time(device, lambda: ttnn.matmul(a, b))
    flops = 2.0 * M * K * N
    return flops / t / 1e12


def _dense_sdpa_tflops(device, S, nh=16, d=64, q_chunk=128, k_chunk=128):
    mk = lambda: ttnn.from_torch(
        torch.randn(1, nh, S, d, dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    q, k, v = mk(), mk(), mk()
    g = device.compute_with_storage_grid_size()
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(g.x, g.y), exp_approx_mode=False, q_chunk_size=q_chunk, k_chunk_size=k_chunk
    )
    fn = lambda: ttnn.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=d**-0.5, program_config=pc
    )
    t = _sync_time(device, fn)
    # dense non-causal: every query attends all S keys. QK + PV = 2 * (S*S*d) each -> 2*2*nh*S^2*d.
    flops = 2.0 * 2.0 * nh * (S * S) * d
    return flops / t / 1e12, t


def _neighborhood_tflops(device, grid, kernel, block, nh=16, d=64, k_chunk=32):
    from models.tt_dit.layers.block_permute import to_block_order_tt

    T, H, W = grid
    kt, kh, kw = kernel
    S = T * H * W
    mk = lambda: ttnn.from_torch(
        torch.randn(1, nh, S, d, dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    q, k, v = mk(), mk(), mk()
    off = ttnn.from_torch(
        torch.zeros(1, dtype=torch.int32), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    k_rm = ttnn.reshape(ttnn.to_layout(k, ttnn.ROW_MAJOR_LAYOUT), (1, nh, T * H, W * d))
    v_rm = ttnn.reshape(ttnn.to_layout(v, ttnn.ROW_MAJOR_LAYOUT), (1, nh, T * H, W * d))
    qb = to_block_order_tt(q, grid, block)
    vol = block[0] * block[1] * block[2]
    g = device.compute_with_storage_grid_size()
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(g.x, g.y), exp_approx_mode=False, q_chunk_size=vol, k_chunk_size=k_chunk
    )
    op_block = (block[2], block[1], block[0])  # (bw,bh,bt) op order
    fn = lambda: ttnn.transformer.scaled_dot_product_attention(
        qb,
        k_rm,
        v_rm,
        is_causal=False,
        neighborhood_3d=(T, H, W, kt, kh, kw),
        neighborhood_gather=True,
        neighborhood_block=op_block,
        scale=d**-0.5,
        windowed_q_token_offset=0,
        windowed_q_token_offset_tensor=off,
        program_config=pc,
    )
    t = _sync_time(device, fn)
    # box FLOPs actually computed (dense over the box): 2*2*nh * num_q * box * d ; box = (bt+kt-1)(bh+kh-1)(bw+kw-1)
    box = (block[0] + kt - 1) * (block[1] + kh - 1) * (block[2] + kw - 1)
    box_flops = 2.0 * 2.0 * nh * S * box * d
    # useful (true window) FLOPs: 2*2*nh * num_q * (kt*kh*kw) * d
    win = kt * kh * kw
    win_flops = 2.0 * 2.0 * nh * S * win * d
    return box_flops / t / 1e12, win_flops / t / 1e12, t, box, win


def _gna_style_tflops(device, vol, box, nh=16, d=64, B=256, k_chunk=128):
    """GNA-style block-sparse = each query-block does a DENSE attention over its box tiles, NO per-element
    mask. That's a batched dense SDPA: batch = query-blocks (× heads), Sq = vol, Sk = box, is_causal=False,
    no mask. Uses the well-optimized standard path (no gather/no windowed-narrowing/no per-elem mask).
    Timing depends only on shapes, so random tensors are fine for a speed probe. Same box-FLOPs as the
    neighborhood kernel -> a faster time here is a pure UTILIZATION win (the GNA ceiling on our HW)."""
    q = ttnn.from_torch(torch.randn(B, 1, vol, d, dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
    k = ttnn.from_torch(torch.randn(B, 1, box, d, dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
    v = ttnn.from_torch(torch.randn(B, 1, box, d, dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
    g = device.compute_with_storage_grid_size()
    qc = vol if vol % 32 == 0 else ((vol + 31) // 32) * 32
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(g.x, g.y), exp_approx_mode=False, q_chunk_size=qc, k_chunk_size=k_chunk
    )
    fn = lambda: ttnn.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=d**-0.5, program_config=pc
    )
    t = _sync_time(device, fn)
    box_flops = 2.0 * 2.0 * B * vol * box * d  # same box-FLOPs the neighborhood kernel computes
    return box_flops / t / 1e12, t


def test_sdpa_flop_util(*, device):
    peak = _gemm_peak(device)
    print(f"\n=== achieved matmul-FLOP/s (bf16, one chip) ===")
    print(f"  GEMM 4096^3 (deep K)              {peak:6.1f} TFLOP/s   <- practical peak")
    print(f"  --- DENSE SDPA (clean path), head_dim=64, sweep S ---")
    for S in (256, 512, 1024, 2048, 4096, 8192):
        tf, t = _dense_sdpa_tflops(device, S)
        print(f"  dense  S={S:5d}                    {tf:6.1f} TFLOP/s   ({tf/peak*100:4.1f}% of peak, {t*1e3:.2f} ms)")
    print(f"  --- NEIGHBORHOOD block SDPA (gather + per-elem mask), k_chunk sweep — isolates chunk size ---")
    nbr_ref = {}
    for k_chunk in (32, 64, 128):
        try:
            boxtf, wintf, t, box, win = _neighborhood_tflops(
                device, (15, 32, 16), (5, 5, 5), (5, 8, 4), k_chunk=k_chunk
            )
            nbr_ref[k_chunk] = boxtf
            print(
                f"  nbr k_chunk={k_chunk:3d}  box={box:5d}  box-FLOP {boxtf:6.1f} ({boxtf/peak*100:4.1f}%)  {t*1e3:.2f} ms"
            )
        except Exception as e:
            print(f"  nbr k_chunk={k_chunk:3d}  FAILED: {str(e)[:80]}")
    nbrtf = nbr_ref.get(32, 3.4)
    print(f"  --- GNA-style block-sparse (batched DENSE over box, NO mask), k_chunk sweep ---")
    vol = 5 * 8 * 4
    for kernel in ((5, 5, 5), (11, 11, 11)):
        box = (5 + kernel[0] - 1) * (8 + kernel[1] - 1) * (4 + kernel[2] - 1)
        for k_chunk in (64, 128, 256):
            try:
                gtf, gt = _gna_style_tflops(device, vol, box, k_chunk=k_chunk)
                print(
                    f"  gna box={box:5d}(k={kernel[0]:2d}) kc={k_chunk:3d}  box-FLOP {gtf:6.1f} ({gtf/peak*100:4.1f}%)"
                    f"  -> {gtf/nbrtf:4.1f}x nbr(kc32)  {gt*1e3:.2f} ms"
                )
            except Exception as e:
                print(f"  gna box={box:5d}(k={kernel[0]:2d}) kc={k_chunk:3d}  FAILED: {str(e)[:80]}")
