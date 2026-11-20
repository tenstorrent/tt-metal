# Pipeline-parallel prefill feasibility: one FULL Llama-3.3-70B layer (no tensor parallelism) on ONE Blackhole chip.
# In a 32-stage pipeline (2.5 layers per chip) with the pipeline full, a 4K prompt costs (layer time at 4K) x 80 / 32
# of Galaxy time; TP32 prefill today takes ~232 ms per 4K prompt. Times the matmuls at full width (QKV 8192->10240,
# WO 8192->8192, FF1/FF3 8192->28672, FF2 28672->8192) with the model's weight formats (bf8; FF1/FF3 bf4 LoFi) and
# causal SDPA (64 q heads, 8 kv heads). Norms, RoPE and the residual adds are left out (small next to these).
#   pytest models/demos/llama3_70b_galaxy/tests/test_pp_stage_bench.py -s     (single device)
import time

import pytest
import torch
from loguru import logger

import ttnn

H, NQ, NKV, D, FF = 8192, 64, 8, 128, 28672
ITERS = 5


def _timed(device, fn):
    out = fn()
    ttnn.synchronize_device(device)
    for o in out if isinstance(out, (list, tuple)) else [out]:
        ttnn.deallocate(o)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        out = fn()
        for o in out if isinstance(out, (list, tuple)) else [out]:
            ttnn.deallocate(o)
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / ITERS * 1e3


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("M", [512, 1024, 2048, 4096])
def test_pp_stage_bench(device, M):
    grid = device.compute_with_storage_grid_size()
    dram = ttnn.DRAM_MEMORY_CONFIG
    hifi2 = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    lofi = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    def w(k, n, dt):
        return ttnn.from_torch(
            torch.randn(1, 1, k, n) * 0.02, device=device, layout=ttnn.TILE_LAYOUT, dtype=dt, memory_config=dram
        )

    def act(m, k):
        return ttnn.from_torch(
            torch.randn(1, 1, m, k), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=dram
        )

    x, x_ff = act(M, H), act(M, FF)
    wqkv, wo = w(H, (NQ + 2 * NKV) * D, ttnn.bfloat8_b), w(H, H, ttnn.bfloat8_b)
    w1, w3, w2 = w(H, FF, ttnn.bfloat4_b), w(H, FF, ttnn.bfloat4_b), w(FF, H, ttnn.bfloat8_b)
    q = ttnn.from_torch(torch.randn(1, NQ, M, D), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    k = ttnn.from_torch(torch.randn(1, NKV, M, D), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    v = ttnn.from_torch(torch.randn(1, NKV, M, D), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

    def lin(a, b, ckc):
        return lambda: ttnn.linear(
            a,
            b,
            compute_kernel_config=ckc,
            dtype=ttnn.bfloat8_b,
            memory_config=dram,
            core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
        )

    def mm(a, b, ckc, mb=8, kb=8, nb=8, sh=1, sw=8):
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=mb,
            K_block_size=kb,
            N_block_size=nb,
            subblock_h=sh,
            subblock_w=sw,
            compute_with_storage_grid_size=grid,
        )
        return lambda: ttnn.experimental.minimal_matmul(
            input_tensor=a, weight_tensor=b, config=cfg, compute_kernel_config=ckc, memory_config=dram
        )

    ops = {
        "qkv": (x, wqkv, hifi2, 2 * M * H * (NQ + 2 * NKV) * D),
        "wo": (x, wo, hifi2, 2 * M * H * H),
        "ff1": (x, w1, lofi, 2 * M * H * FF),
        "ff3": (x, w3, lofi, 2 * M * H * FF),
        "ff2": (x_ff, w2, hifi2, 2 * M * FF * H),
    }
    total = 0.0
    for name, (a, b, ckc, flops) in ops.items():
        best = None
        for label, fn in (("linear", lin(a, b, ckc)), ("minimal", mm(a, b, ckc))):
            try:
                ms = _timed(device, fn)
            except Exception as e:
                logger.warning(f"PPSTAGE M={M} {name} {label} ERROR {str(e)[:200]}")
                continue
            logger.info(f"PPSTAGE M={M} {name} {label}: {ms:.3f} ms ({flops / ms / 1e9:.0f} TFLOPS)")
            best = ms if best is None else min(best, ms)
        total += best or 0.0
    sdpa_best = None
    for qc, kc in ((128, 256), (256, 256), (128, 512), (256, 512)):
        if M % qc or M % kc:
            continue
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=False
        )
        try:
            ms = _timed(
                device,
                lambda: ttnn.transformer.scaled_dot_product_attention(
                    q, k, v, is_causal=True, program_config=cfg, compute_kernel_config=hifi2, memory_config=dram
                ),
            )
        except Exception as e:
            logger.warning(f"PPSTAGE M={M} sdpa q{qc} k{kc} ERROR {str(e)[:200]}")
            continue
        logger.info(f"PPSTAGE M={M} sdpa q{qc} k{kc}: {ms:.3f} ms")
        sdpa_best = ms if sdpa_best is None else min(sdpa_best, ms)
    total += sdpa_best or 0.0
    per_prompt = total * 80 / 32 * 4096 / M  # ms of pipeline time per 4K prompt with 32 stages, full pipeline
    logger.info(
        f"PPSTAGE M={M} layer {total:.2f} ms -> 32-stage pipeline: {per_prompt:.1f} ms per 4K prompt "
        f"({4096 / per_prompt * 1e3:.0f} tok/s; TP32 today ~232 ms, ~17.6K tok/s)"
    )
