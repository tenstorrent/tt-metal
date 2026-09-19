# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Decoder-wave op sweeps on one chip: SDPA chunk sizes under the mask-free view, and minimal_matmul
block configs for the four projection shapes. Min-of-N wall per op; outputs compared to today's config.

    pytest models/tt_dit/tests/models/minimax_h3/tools/decoder_block_probe.py -q -s

Not a gate; a measurement that says which configs are worth carrying into the decoder.
"""
import time

import pytest
import torch

import ttnn

SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]
HEADS, SEQ, VALID, HEAD_DIM, DIM = 32, 1824, 1797, 64, 2048
REPEATS = 15


def _timed(mesh_device, fn, n=REPEATS):
    fn()
    ttnn.synchronize_device(mesh_device)
    best = float("inf")
    for _ in range(n):
        ttnn.synchronize_device(mesh_device)
        mark = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(mesh_device)
        best = min(best, time.perf_counter() - mark)
    return out, best * 1e3


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_sdpa_chunk_sweep(mesh_device):
    """q/k chunk sizes for the decoder's SDPA, on the logical-1797 view with no mask."""
    torch.manual_seed(0)
    dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    q, k, v = (dev(torch.randn(1, HEADS, SEQ, HEAD_DIM)) for _ in range(3))
    view = lambda t: ttnn.reshape(t, ttnn.Shape([1, HEADS, VALID, HEAD_DIM]), ttnn.Shape([1, HEADS, SEQ, HEAD_DIM]))
    qv, kv, vv = view(q), view(k), view(v)
    kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
    )
    grid = mesh_device.compute_with_storage_grid_size()

    def sdpa(qc, kc):
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=False
        )
        return ttnn.transformer.scaled_dot_product_attention(
            qv, kv, vv, attn_mask=None, is_causal=False, program_config=cfg, compute_kernel_config=kernel_config
        )

    # Reference = today's config (q192/k192). A q-chunk change only moves query rows between cores, so it must be
    # bit-identical; a k-chunk change reorders the bf16 accumulation and is not.
    ref = ttnn.to_torch(_timed(mesh_device, lambda: sdpa(192, 192))[0])[..., :VALID, :]
    rows = []
    for qc in (64, 96, 128, 160, 192, 224, 256):
        for kc in (96, 128, 192, 256):
            try:
                out, ms = _timed(mesh_device, lambda: sdpa(qc, kc))
            except Exception as exc:  # noqa: BLE001
                print(f"sdpa q{qc} k{kc}: FAILED {type(exc).__name__}: {str(exc)[:120]}")
                continue
            got = ttnn.to_torch(out)[..., :VALID, :]
            maxdiff = (got.float() - ref.float()).abs().max().item()
            equal = bool(torch.equal(got, ref))
            rows.append((qc, kc, ms, maxdiff, equal))
            print(f"sdpa q{qc} k{kc}: {ms:.3f} ms  max|diff| vs q192k192 {maxdiff:.2e}  bit-identical {equal}")
    rows.sort(key=lambda r: r[2])
    print("\nsdpa best 5:")
    for qc, kc, ms, md, eq in rows[:5]:
        print(f"  q{qc} k{kc}: {ms:.3f} ms  bit-identical {eq}")
    today = [r for r in rows if r[0] == 192 and r[1] == 192]
    if today:
        print(f"  today (q192 k192): {today[0][2]:.3f} ms")
    q128 = [r for r in rows if r[0] == 128 and r[1] == 192]
    if q128 and today:
        print(f"SDPA_Q128: {q128[0][2]:.3f} ms vs today {today[0][2]:.3f} ms, bit-identical {q128[0][4]}")


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_matmul_block_sweep(mesh_device):
    """minimal_matmul block configs for to_qkv, to_out, ff1 (swiglu), ff2 at the decoder's shapes."""
    from models.tt_dit.utils.matmul import get_matmul_config, get_matmul_core_grid
    from models.tt_dit.models.vae.minimax_h3.blockings_minimax_h3_vae import register_h3_vae_decoder_blockings

    register_h3_vae_decoder_blockings()  # "today" = the decoder's registered configs, not the generic default
    torch.manual_seed(0)
    # The pipeline runs on a Galaxy, where get_matmul_core_grid clamps to 11x10 (power); a one-chip mesh returns 12x10 and
    # would miss the registered blockings, so mirror the clamp here.
    core_grid = ttnn.CoreCoord(11, 10)
    print(f"\ncore grid {core_grid.x}x{core_grid.y}")
    kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True
    )
    shapes = [("to_qkv", SEQ, DIM, 3 * DIM, False), ("to_out", SEQ, DIM, DIM, False), ("ff1 swiglu", SEQ, DIM, 8 * DIM, True), ("ff2", SEQ, 4 * DIM, DIM, False)]
    candidates = [(8, 8, 8), (8, 4, 8), (8, 2, 8), (4, 4, 8), (8, 4, 16), (4, 2, 8), (8, 2, 16), (6, 4, 12), (14, 4, 6), (14, 2, 10), (6, 2, 6)]
    for name, M, K, N, swiglu in shapes:
        x = ttnn.from_torch(torch.randn(1, M, K) * 0.1, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)
        w = ttnn.from_torch(torch.randn(K, N) * 0.02, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)
        b = ttnn.from_torch(torch.randn(1, N) * 0.02, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)
        today_cfg = get_matmul_config(M, K, N, core_grid)
        print(f"\n{name}: M{M} K{K} N{N} swiglu={swiglu}; today's config {today_cfg}")
        rows = []
        for blocks in [None] + candidates:
            try:
                cfg = today_cfg if blocks is None else get_matmul_config(M, K, N, core_grid, default_block_size=blocks)
                out, ms = _timed(
                    mesh_device,
                    lambda: ttnn.experimental.minimal_matmul(
                        input_tensor=x,
                        weight_tensor=w,
                        bias_tensor=b,
                        config=cfg,
                        compute_kernel_config=kernel_config,
                        fuse_swiglu=swiglu,
                    ),
                    n=8,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  {blocks or 'today'}: FAILED {type(exc).__name__}: {str(exc)[:100]}")
                continue
            rows.append((blocks or "today", ms))
            print(f"  {blocks or 'today'}: {ms:.3f} ms")
            ttnn.deallocate(out)
        rows.sort(key=lambda r: r[1])
        print(f"  best: {rows[0][0]} {rows[0][1]:.3f} ms; today {[r for r in rows if r[0] == 'today'][0][1]:.3f} ms")
