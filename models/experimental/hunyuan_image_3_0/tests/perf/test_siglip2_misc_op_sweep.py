# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Micro-sweep for SigLIP2's non-matmul, non-SDPA ops: LayerNorm, BinaryNg (residual
add), NlpCreateHeads, NLPConcatHeads. Isolated microbenchmarks at the tower's actual
shapes (S=1024, hidden=1152, 16 heads, head_dim=96 padded) — same rationale as
test_siglip2_matmul_sweep.py: the Tracy aggregate's per-op-code totals are unreliable
across program-cache-merged repeats, so measure each op cleanly instead.

None of these ops take a MatmulProgramConfig — the only real knob is memory_config
(DRAM vs L1, interleaved vs sharded), so the sweep is over that axis.

Run:
    pytest models/experimental/hunyuan_image_3_0/tests/perf/test_siglip2_misc_op_sweep.py -s
"""

from __future__ import annotations

import time

import torch
import ttnn

_TILE = 32
S = 1024
H = 1152
NUM_HEADS = 16
PADDED_HEAD_DIM = 96
QKV_DIM = NUM_HEADS * PADDED_HEAD_DIM  # 1536


def _bench(dev, fn, it=40):
    for _ in range(6):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    t = time.perf_counter()
    for _ in range(it):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t) / it * 1e6


def _pcc(a, b):
    a = a.float().flatten() - a.float().mean()
    b = b.float().flatten() - b.float().mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


def _report(name, results):
    base = next((us for _, us, _ in results if us is not None), None)
    print(f"\n=== {name} ===")
    for lab, us, pcc in results:
        if us is None:
            print(f"  {lab:32}    FAIL  {pcc}")
        else:
            pcc_s = f"PCC={pcc:.6f}" if isinstance(pcc, float) else "PCC=n/a"
            print(f"  {lab:32} {us:8.2f} us  ({base/us:4.2f}x)  {pcc_s}")
    ranked = sorted(((us, lab) for lab, us, _ in results if us is not None), key=lambda t: t[0])
    if ranked:
        best_us, best_lab = ranked[0]
        print(f"  >> BEST: {best_lab} @ {best_us:.2f} us ({base/best_us:.2f}x vs prod)")
        if best_lab.startswith("prod"):
            print("  >> no candidate beat production config")
        elif best_us < base * 0.98:
            print(f"  >> candidate beat prod ({base/best_us:.2f}x) — worth trying in-model")
        else:
            print(f"  >> best non-prod within noise of prod ({base/best_us:.2f}x)")


def test_layer_norm_sweep(device):
    device.enable_program_cache()
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    torch.manual_seed(0)
    x = torch.randn(1, S, H, dtype=torch.bfloat16) * 0.1
    gamma = torch.ones(1, H, dtype=torch.bfloat16)
    beta = torch.zeros(1, H, dtype=torch.bfloat16)
    ref = torch.nn.functional.layer_norm(x.float(), (H,), gamma.float().reshape(-1), beta.float().reshape(-1), 1e-6)

    x_dram = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    x_l1 = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    w = ttnn.from_torch(gamma, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(beta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    results = []
    for label, xin, mem_out in [
        ("prod DRAM-in/DRAM-out", x_dram, ttnn.DRAM_MEMORY_CONFIG),
        ("L1-in/L1-out", x_l1, ttnn.L1_MEMORY_CONFIG),
        ("L1-in/DRAM-out", x_l1, ttnn.DRAM_MEMORY_CONFIG),
        ("DRAM-in/L1-out", x_dram, ttnn.L1_MEMORY_CONFIG),
    ]:
        try:

            def fn(xin=xin, mem_out=mem_out):
                return ttnn.layer_norm(
                    xin, weight=w, bias=b, epsilon=1e-6, compute_kernel_config=ckc, memory_config=mem_out
                )

            out = ttnn.to_torch(fn()).reshape(-1, H)
            us = _bench(device, fn)
            results.append((label, us, _pcc(out, ref)))
        except Exception as e:
            results.append((label, None, str(e)[:90]))
    _report("LayerNorm [1024,1152]", results)


def test_binary_add_sweep(device):
    device.enable_program_cache()
    torch.manual_seed(0)
    a = torch.randn(1, S, H, dtype=torch.bfloat16) * 0.1
    b = torch.randn(1, S, H, dtype=torch.bfloat16) * 0.1
    ref = a.float() + b.float()

    a_dram = ttnn.from_torch(
        a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    b_dram = ttnn.from_torch(
        b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    a_l1 = ttnn.from_torch(
        a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    b_l1 = ttnn.from_torch(
        b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    results = []
    for label, x1, x2, mem_out in [
        ("prod DRAM+DRAM->DRAM", a_dram, b_dram, ttnn.DRAM_MEMORY_CONFIG),
        ("L1+L1->L1", a_l1, b_l1, ttnn.L1_MEMORY_CONFIG),
        ("L1+L1->DRAM", a_l1, b_l1, ttnn.DRAM_MEMORY_CONFIG),
        ("DRAM+DRAM->L1", a_dram, b_dram, ttnn.L1_MEMORY_CONFIG),
    ]:
        try:

            def fn(x1=x1, x2=x2, mem_out=mem_out):
                return ttnn.add(x1, x2, memory_config=mem_out)

            out = ttnn.to_torch(fn()).reshape(-1, H)
            us = _bench(device, fn)
            results.append((label, us, _pcc(out, ref)))
        except Exception as e:
            results.append((label, None, str(e)[:90]))
    _report("BinaryNg add (residual) [1024,1152]", results)


def test_nlp_create_qkv_heads_sweep(device):
    device.enable_program_cache()
    torch.manual_seed(0)
    qkv = torch.randn(1, 1, S, 3 * QKV_DIM, dtype=torch.bfloat16) * 0.1

    qkv_dram = ttnn.from_torch(
        qkv, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    qkv_l1 = ttnn.from_torch(
        qkv, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    results = []
    for label, xin, mem_out in [
        ("prod DRAM-in/DRAM-out", qkv_dram, ttnn.DRAM_MEMORY_CONFIG),
        ("L1-in/L1-out", qkv_l1, ttnn.L1_MEMORY_CONFIG),
        ("L1-in/DRAM-out", qkv_l1, ttnn.DRAM_MEMORY_CONFIG),
        ("DRAM-in/L1-out", qkv_dram, ttnn.L1_MEMORY_CONFIG),
    ]:
        try:

            def fn(xin=xin, mem_out=mem_out):
                q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                    xin, num_heads=NUM_HEADS, num_kv_heads=NUM_HEADS, transpose_k_heads=False, memory_config=mem_out
                )
                ttnn.deallocate(k)
                ttnn.deallocate(v)
                return q

            fn()  # correctness not checked (fixed-function reshuffle; PCC N/A here)
            us = _bench(device, fn)
            results.append((label, us, None))
        except Exception as e:
            results.append((label, None, str(e)[:90]))
    _report("NlpCreateHeads qkv[1,1,1024,4608]", results)


def test_nlp_concat_heads_sweep(device):
    device.enable_program_cache()
    torch.manual_seed(0)
    attn = torch.randn(1, NUM_HEADS, S, PADDED_HEAD_DIM, dtype=torch.bfloat16) * 0.1

    attn_dram = ttnn.from_torch(
        attn, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    attn_l1 = ttnn.from_torch(
        attn, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    results = []
    for label, xin, mem_out in [
        ("prod DRAM-in/DRAM-out", attn_dram, ttnn.DRAM_MEMORY_CONFIG),
        ("L1-in/L1-out", attn_l1, ttnn.L1_MEMORY_CONFIG),
        ("L1-in/DRAM-out", attn_l1, ttnn.DRAM_MEMORY_CONFIG),
        ("DRAM-in/L1-out", attn_dram, ttnn.L1_MEMORY_CONFIG),
    ]:
        try:

            def fn(xin=xin, mem_out=mem_out):
                return ttnn.experimental.nlp_concat_heads(xin, memory_config=mem_out)

            fn()  # fixed-function reshuffle; PCC N/A here
            us = _bench(device, fn)
            results.append((label, us, None))
        except Exception as e:
            results.append((label, None, str(e)[:90]))
    _report("NLPConcatHeads attn[1,16,1024,96]", results)
