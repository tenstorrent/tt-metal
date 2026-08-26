# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-perf variants for the Qwen3.6 gated-attention prefill chain (issue #50475).

One pytest node per variant so each can be captured under its own Tracy run and
the ops CSV attributed unambiguously. Correctness for every one of these
variants is already pinned in test_gated_attention_prefill.py; this file exists
only to be timed.

Driven by gated_attention_perf_sweep.sh.
"""

import math
import os

import pytest
import torch

import ttnn

QWEN36_NH = 6
QWEN36_NKV = 1
QWEN36_HD = 256

# Production prefill chunk. tp.py caps serving prefill at <=2048 per chunk.
#
# GDN_PERF_DRYRUN=1 shrinks the shapes and lifts the hardware-only guard so every
# variant can be exercised on ttsim first. The hardware window is the scarce
# resource here -- a variant that dies on a bad argument should die on the
# simulator, for free, not burn a slot on the card.
DRYRUN = os.environ.get("GDN_PERF_DRYRUN") == "1"
SEQ = 128 if DRYRUN else 2048
ITERS = 1 if DRYRUN else 5


def fa_rand(*shape):
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


# name -> (grid, exp_approx, dtype, fidelity, fused_sigmoid[, kv_dtype, chunk])
#
# Cumulative ladder: each row adds one change to the row above it, so the delta
# between adjacent rows is that change's cost in isolation.
VARIANTS = {
    # What tp.py ran before this work: 8x8 grid, exact exp, separate sigmoid.
    "a_baseline": ((8, 8), False, ttnn.bfloat16, ttnn.MathFidelity.HiFi2, False),
    "b_fullgrid": (None, False, ttnn.bfloat16, ttnn.MathFidelity.HiFi2, False),
    "c_approxexp": (None, True, ttnn.bfloat16, ttnn.MathFidelity.HiFi2, False),
    # == what tp.py runs now
    "d_fusedgate": (None, True, ttnn.bfloat16, ttnn.MathFidelity.HiFi2, True),
    # Untaken candidates, for sizing the remaining headroom.
    "e_bfp8": (None, True, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True),
    "f_bfp8_lofi": (None, True, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi, True),
    # bf16 Q with bf8 K/V only. K/V are the tensors streamed O(S^2) times, so most
    # of the bandwidth win should survive while Q keeps full precision.
    "g_bf8kv_only": (None, True, ttnn.bfloat16, ttnn.MathFidelity.HiFi2, True, ttnn.bfloat8_b, 128),
    # bf8 at q/k_chunk=256. tp.py records that 256 won in isolation but its CBs
    # clashed with the resident L1 activation; in bf16 it overflows L1 outright
    # (1,676,160 B of CBs vs a 1,572,864 B limit). bf8 halves CB footprint and
    # fits, so this asks whether bf8 also unlocks the faster chunk size.
    "h_bfp8_c256": (None, True, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True, None, 256),
    # --- q_chunk occupancy sweep -------------------------------------------------
    # sdpa_program_factory.cpp:391 distributes total_q_chunks = B*NQH*q_num_chunks
    # across cores, pair-wise when causal. At S=2048/q_chunk=128 that is
    # 1*6*16 = 96 chunks -> 48 pairs -> only 48 of ~110 cores get any work.
    # Halving q_chunk doubles the chunk count and the occupancy. tp.py's note that
    # 128 beat 64 predates the full-grid change: it was measured on an 8x8=64-core
    # grid, where 64 would have needed two passes. Different tradeoff now.
    "i_bf8_q64": (None, True, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True, None, (64, 128)),
    "j_bf8_q32": (None, True, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True, None, (32, 128)),
    "k_bf8_q64k64": (None, True, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True, None, (64, 64)),
    "l_bf16_q64": (None, True, ttnn.bfloat16, ttnn.MathFidelity.HiFi2, True, None, (64, 128)),
    # q_chunk and k_chunk DECOUPLED. Every previous chunk experiment moved them together, so this
    # combination was never tried. q=128 holds the measured occupancy optimum (48 pairs) while
    # k=256 makes Sk_chunk_t=8, which is what switches on `can_reduce_trigger` in
    # compute_streaming.hpp:1972 -- at Sk_chunk_t=4 that predicate is false and the whole
    # PACK->UNPACK early-reduce overlap path is dead code. It also halves the k-chunk iteration
    # count, which matters on a DRAM-bound op.
    "q_bf8_q128k256": (None, True, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True, None, (128, 256)),
    "r_bf16_q128k256": (None, True, ttnn.bfloat16, ttnn.MathFidelity.HiFi2, True, None, (128, 256)),
}

# Variants that drop nlp_concat_heads entirely by having SDPA emit concat-heads
# layout (fuse_concat_heads=True). Bit-identical to the unfused chain -- see
# test_gated_attention_prefill.py::test_sdpa_fused_concat_heads_is_bit_identical.
FUSED_CONCAT = {"m_bf8_fusedconcat": ttnn.bfloat8_b, "n_bf16_fusedconcat": ttnn.bfloat16}

# Same as FUSED_CONCAT but SDPA emits into L1 instead of DRAM, so the gate multiply
# reads its big operand from L1. Removes a DRAM write + DRAM read of [S, NH*HD].
# Bit-identical -- memory placement does not change values.
FUSED_CONCAT_L1 = {"o_bf8_l1attn": ttnn.bfloat8_b, "p_bf16_l1attn": ttnn.bfloat16}

# k_chunk=256 makes SDPA itself faster (reduce_trigger switches on) but its larger CBs contend
# with the L1-resident attention output, and the gate multiply pays for it. These put the output
# back in DRAM to see whether the SDPA win survives once that contention is removed.
FUSED_CONCAT_DRAM_K256 = {"s_bf8_k256_dram": ttnn.bfloat8_b, "t_bf16_k256_dram": ttnn.bfloat16}

# Best bf8 config (k=256, out DRAM) but with the GATE tensor in bfloat8_b instead of bf16.
# The gate is one of three full [S, NH*HD] passes in the multiply; halving it should take ~25%
# off that op's traffic. Separate knob from the Q/K/V dtype -- the gate is not part of the
# quadratic work.
GATE_DTYPE = {"u_bf8_gate_bf8": ttnn.bfloat8_b}


@pytest.mark.parametrize("variant", list(GATE_DTYPE.keys()))
def test_gated_attention_prefill_perf_gate_dtype(device, variant):
    """Best bf8 config with a bfloat8_b gate tensor."""
    if os.environ.get("TT_METAL_SIMULATOR") and not DRYRUN:
        pytest.skip("perf variants are hardware-only")
    gate_dt = GATE_DTYPE[variant]
    dtype = ttnn.bfloat8_b
    torch.manual_seed(1234)
    b, nh, nkv, s, d = 1, QWEN36_NH, QWEN36_NKV, SEQ, QWEN36_HD
    scale = 1.0 / math.sqrt(d)
    tt_q = ttnn.from_torch(fa_rand(b, nh, s, d), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_k = ttnn.from_torch(fa_rand(b, nkv, s, d), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_v = ttnn.from_torch(fa_rand(b, nkv, s, d), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_gate = ttnn.from_torch(fa_rand(b, 1, s, nh * d), dtype=gate_dt, layout=ttnn.TILE_LAYOUT, device=device)
    g = device.compute_with_storage_grid_size()
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(g.x, g.y), q_chunk_size=128, k_chunk_size=256, exp_approx_mode=True
    )
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    for _ in range(ITERS):
        concat = ttnn.transformer.scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            is_causal=True,
            scale=scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=pc,
            compute_kernel_config=ckc,
            fuse_concat_heads=True,
        )
        gated = ttnn.multiply(
            concat,
            tt_gate,
            input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(concat)
        ttnn.deallocate(gated)
    ttnn.synchronize_device(device)


@pytest.mark.parametrize("variant", list(FUSED_CONCAT_DRAM_K256.keys()))
def test_gated_attention_prefill_perf_k256_dram(device, variant):
    """q_chunk=128, k_chunk=256, SDPA output in DRAM (not L1)."""
    if os.environ.get("TT_METAL_SIMULATOR") and not DRYRUN:
        pytest.skip("perf variants are hardware-only")
    dtype = FUSED_CONCAT_DRAM_K256[variant]
    torch.manual_seed(1234)
    b, nh, nkv, s, d = 1, QWEN36_NH, QWEN36_NKV, SEQ, QWEN36_HD
    scale = 1.0 / math.sqrt(d)
    tt_q = ttnn.from_torch(fa_rand(b, nh, s, d), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_k = ttnn.from_torch(fa_rand(b, nkv, s, d), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_v = ttnn.from_torch(fa_rand(b, nkv, s, d), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_gate = ttnn.from_torch(fa_rand(b, 1, s, nh * d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    g = device.compute_with_storage_grid_size()
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(g.x, g.y), q_chunk_size=128, k_chunk_size=256, exp_approx_mode=True
    )
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    for _ in range(ITERS):
        concat = ttnn.transformer.scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            is_causal=True,
            scale=scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=pc,
            compute_kernel_config=ckc,
            fuse_concat_heads=True,
        )
        gated = ttnn.multiply(
            concat,
            tt_gate,
            input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(concat)
        ttnn.deallocate(gated)
    ttnn.synchronize_device(device)


@pytest.mark.parametrize("variant", list(FUSED_CONCAT_L1.keys()))
def test_gated_attention_prefill_perf_l1_attn(device, variant):
    """SDPA -> L1, gate multiply reads L1. Saves a DRAM round trip of the output."""
    if os.environ.get("TT_METAL_SIMULATOR") and not DRYRUN:
        pytest.skip("perf variants are hardware-only")
    dtype = FUSED_CONCAT_L1[variant]
    torch.manual_seed(1234)
    b, nh, nkv, s, d = 1, QWEN36_NH, QWEN36_NKV, SEQ, QWEN36_HD
    scale = 1.0 / math.sqrt(d)
    tt_q = ttnn.from_torch(fa_rand(b, nh, s, d), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_k = ttnn.from_torch(fa_rand(b, nkv, s, d), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_v = ttnn.from_torch(fa_rand(b, nkv, s, d), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_gate = ttnn.from_torch(fa_rand(b, 1, s, nh * d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    g = device.compute_with_storage_grid_size()
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(g.x, g.y), q_chunk_size=128, k_chunk_size=128, exp_approx_mode=True
    )
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    for _ in range(ITERS):
        concat = ttnn.transformer.scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            is_causal=True,
            scale=scale,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=pc,
            compute_kernel_config=ckc,
            fuse_concat_heads=True,
        )
        gated = ttnn.multiply(
            concat,
            tt_gate,
            input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(concat)
        ttnn.deallocate(gated)
    ttnn.synchronize_device(device)


@pytest.mark.parametrize("variant", list(FUSED_CONCAT.keys()))
def test_gated_attention_prefill_perf_fused_concat(device, variant):
    """Same chain minus nlp_concat_heads: SDPA writes concat-heads layout directly."""
    if os.environ.get("TT_METAL_SIMULATOR") and not DRYRUN:
        pytest.skip("perf variants are hardware-only")
    dtype = FUSED_CONCAT[variant]
    torch.manual_seed(1234)
    b, nh, nkv, s, d = 1, QWEN36_NH, QWEN36_NKV, SEQ, QWEN36_HD
    scale = 1.0 / math.sqrt(d)
    tt_q = ttnn.from_torch(fa_rand(b, nh, s, d), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_k = ttnn.from_torch(fa_rand(b, nkv, s, d), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_v = ttnn.from_torch(fa_rand(b, nkv, s, d), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_gate = ttnn.from_torch(fa_rand(b, 1, s, nh * d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    g = device.compute_with_storage_grid_size()
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(g.x, g.y), q_chunk_size=128, k_chunk_size=128, exp_approx_mode=True
    )
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    for _ in range(ITERS):
        concat = ttnn.transformer.scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            is_causal=True,
            scale=scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=pc,
            compute_kernel_config=ckc,
            fuse_concat_heads=True,
        )
        gated = ttnn.multiply(
            concat,
            tt_gate,
            input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(concat)
        ttnn.deallocate(gated)
    ttnn.synchronize_device(device)


@pytest.mark.parametrize("variant", list(VARIANTS.keys()))
def test_gated_attention_prefill_perf(device, variant):
    if os.environ.get("TT_METAL_SIMULATOR") and not DRYRUN:
        pytest.skip("perf variants are hardware-only (set GDN_PERF_DRYRUN=1 to shape-check on ttsim)")

    spec = VARIANTS[variant]
    grid, exp_approx, dtype, fidelity, fused_sigmoid = spec[:5]
    kv_dtype = spec[5] if len(spec) > 5 and spec[5] is not None else dtype
    chunk = spec[6] if len(spec) > 6 else 128
    q_chunk, k_chunk = chunk if isinstance(chunk, tuple) else (chunk, chunk)
    torch.manual_seed(1234)

    b, nh, nkv, s, d = 1, QWEN36_NH, QWEN36_NKV, SEQ, QWEN36_HD
    scale = 1.0 / math.sqrt(d)

    Q = fa_rand(b, nh, s, d)
    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)
    gate = fa_rand(b, 1, s, nh * d)

    tt_q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_k = ttnn.from_torch(K, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_v = ttnn.from_torch(V, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_gate = ttnn.from_torch(gate, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    dev_grid = device.compute_with_storage_grid_size()
    grid_arg = grid if grid is not None else (dev_grid.x, dev_grid.y)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_arg,
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=exp_approx,
    )
    # Matches what the op resolves to when no compute_kernel_config is passed --
    # init_device_compute_kernel_config(arch, cfg, HiFi2, true, false, false) at sdpa.cpp:51,
    # i.e. (fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc). Only `fidelity`
    # varies across the ladder, so a_baseline is a faithful stand-in for the old call site.
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    for _ in range(ITERS):
        attn = ttnn.transformer.scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            is_causal=True,
            scale=scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
        concat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn)

        if fused_sigmoid:
            gated = ttnn.multiply(
                concat,
                tt_gate,
                input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            sig = ttnn.sigmoid(tt_gate, memory_config=ttnn.L1_MEMORY_CONFIG)
            gated = ttnn.multiply(concat, sig, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(sig)
        ttnn.deallocate(concat)
        ttnn.deallocate(gated)

    ttnn.synchronize_device(device)
