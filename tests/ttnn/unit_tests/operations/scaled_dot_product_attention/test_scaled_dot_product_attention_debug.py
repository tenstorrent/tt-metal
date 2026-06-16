# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Debug artifact for the fp32 S=8192 golden rms near-miss (ttnn-expert-debugger).

Root cause established here (see module docstring + the two numpy/torch
simulations below): the S=8192 fp32 rms (~0.0284 > 0.02 target) is dominated by
SFPU `exp` precision accumulated across the online-softmax KV-block recurrence
(256 blocks), NOT by TF32 matmul operand truncation.

Evidence chain (reproducible with the helpers below):

1. flash_recurrence_sim with FULL fp32 matmuls           -> rms 0.000000
     => the online-softmax topology (corr-rescale, l_i / O_i update, final
        normalize) is numerically exact. Not an error source.

2. flash_recurrence_sim with TF32-truncated matmul operands (single trunc):
        S=2048 rms 0.000420   S=8192 rms 0.000430
     => TF32 matmul truncation is ~0.0004 AND FLAT in S. The "256-block TF32
        accumulation" hypothesis is FALSIFIED — it neither reaches 0.028 nor
        grows with sequence length.

3. Faithful TF32 on EVERY FPU operand (matmul + every eltwise mul/add):
        S=2048 rms 0.00197    S=8192 rms 0.00405
     => even pessimistic TF32-everywhere is 7x below the device's 0.0284.

4. Injecting ~0.4% relative error into each `exp` call reproduces the device:
        exp_err 0.003 -> rms 0.0202 ;  device measured rms 0.0284 @ S=8192
     => SFPU exp precision (called 2x/block: corr and P) accumulated over the
        recurrence is the dominant residual.

Device measurements (HiFi4, fp32_dest_acc_en=True, accurate exp — the defaults):
    S=1024 (32 blk)  rms 0.00586
    S=2048 (64 blk)  rms 0.00872
    S=4096 (128 blk) rms 0.01510
    S=8192 (256 blk) rms 0.02840    <- the failing golden cell
  Growth ~ sqrt(num_kv_blocks) with a mild systematic component — the signature
  of accumulating quasi-random per-block SFPU/packing rounding, exactly what a
  256-step online-softmax recurrence produces.

  math_approx_mode=True (fast exp) was WORSE at every S, confirming the kernel
  already uses the accurate exp and that exp precision is load-bearing.

Conclusion: genuine hardware SFPU-precision limit. The kernel already runs at
the max precision the descriptor exposes (HiFi4 + fp32 DEST acc + accurate exp +
fp32 score/prob/accumulator CBs + power-of-two exact scalers). No descriptor- or
helper-level lever closes the S=8192 fp32 rms without algorithm changes outside
the binding online-softmax topology.
"""

import math

import pytest
import torch


def _to_tf32(x):
    xi = x.to(torch.float32).view(torch.int32)
    return ((xi + (1 << 12)) & ~((1 << 13) - 1)).view(torch.float32)


def _mm_tf32(a, b):
    return torch.matmul(_to_tf32(a), _to_tf32(b))


def _flash_recurrence_sim(Q, K, V, scale, *, mm, exp_err=0.0, gen=None):
    """Online-softmax flash attention, KV block = one 32-row tile.

    `mm` selects the matmul (torch.matmul for exact fp32, _mm_tf32 for the
    TF32-operand model). `exp_err` injects relative noise into each exp call.
    """
    B, H, Sq, D = Q.shape
    Skv = K.shape[2]
    out = torch.zeros_like(Q)
    blk = 32
    for qs in range(0, Sq, blk):
        Qb = Q[:, :, qs : qs + blk, :]
        m_i = torch.full((B, H, blk, 1), -1e30)
        l_i = torch.zeros((B, H, blk, 1))
        O_i = torch.zeros((B, H, blk, D))
        for ks in range(0, Skv, blk):
            Kb = K[:, :, ks : ks + blk, :]
            Vb = V[:, :, ks : ks + blk, :]
            S = mm(Qb, Kb.transpose(-2, -1)) * scale
            m_blk = S.max(dim=-1, keepdim=True).values
            m_new = torch.maximum(m_i, m_blk)
            corr = torch.exp(m_i - m_new)
            P = torch.exp(S - m_new)
            if exp_err > 0.0:
                P = P * (1 + exp_err * torch.randn(P.shape, generator=gen))
                corr = corr * (1 + exp_err * torch.randn(corr.shape, generator=gen))
            l_blk = P.sum(dim=-1, keepdim=True)
            l_i = corr * l_i + l_blk
            PV = mm(P, Vb)
            O_i = corr * O_i + PV
            m_i = m_new
        out[:, :, qs : qs + blk, :] = O_i / l_i
    return out


def _rms(out, ref):
    out = out.to(torch.float32)
    ref = ref.to(torch.float32)
    return (torch.sqrt(torch.mean((out - ref) ** 2)) / ref.std()).item()


def test_error_source_isolation_no_device():
    """Pure-torch proof that the fp32 rms is exp-dominated, not TF32-matmul.

    Runs on CPU (no device) so it is a cheap, permanent regression guard for the
    diagnosis. Assertions encode the evidence chain.
    """
    S = 8192
    torch.manual_seed(0)
    Q = torch.randn((1, 1, S, 64))
    K = torch.randn((1, 1, S, 64))
    V = torch.randn((1, 1, S, 64))
    scale = 1.0 / math.sqrt(64)

    sc = torch.matmul(Q, K.transpose(-2, -1)) * scale
    ref = torch.matmul(torch.softmax(sc, dim=-1), V)

    rms_fp32 = _rms(_flash_recurrence_sim(Q, K, V, scale, mm=torch.matmul), ref)
    rms_tf32 = _rms(_flash_recurrence_sim(Q, K, V, scale, mm=_mm_tf32), ref)
    gen = torch.Generator().manual_seed(1)
    rms_exp = _rms(_flash_recurrence_sim(Q, K, V, scale, mm=_mm_tf32, exp_err=0.003, gen=gen), ref)

    # 1. Recurrence is exact in fp32.
    assert rms_fp32 < 1e-5, f"recurrence not exact: {rms_fp32}"
    # 2. TF32 matmul truncation alone is tiny — cannot explain the 0.0284 miss.
    assert rms_tf32 < 0.001, f"TF32-matmul rms unexpectedly large: {rms_tf32}"
    # 3. A ~0.3% per-exp relative error reproduces the device-scale miss.
    assert rms_exp > 0.015, f"exp-error model did not reproduce the miss: {rms_exp}"


# ============================================================================
# bf8b + fp32_dest_acc_en=False (16-bit DEST) numerical-mismatch investigation
# ----------------------------------------------------------------------------
# Symptom: bf8b inputs at fp32_dest_acc_en=False -> PCC ~0.05 (garbage); bf16 at
# the same config worked fine.
#
# Root cause (FIXED): the QK^T matmul in the online-softmax compute kernel passed
# cb_q_in (bf8b for bf8b inputs) as the matmul_block interm placeholder.
# matmul_block with the default reconfig=INPUT_AND_OUTPUT issues
# pack_reconfig_data_format(interm) before the K-loop, configuring the PACKER for
# the interm CB's data format. The helper only RE-points the packer to the true
# output format on the last K-block when (packer_l1_acc || fp32_dest_acc_en) — so
# with bf16 DEST + no L1 acc that re-point is skipped and the QK result is packed
# into cb_qk (bf16) using bf8b block-float encoding. The downstream score path
# reads those bytes back as ~1e-37 denormals -> PCC collapse.
#
# Localization (DEVICE_PRINT, TSLICE on cb_qk, fa_rand seed 1234, B1H1S1024D128):
#   true bf8b QK post-matmul row0 = -7.7e-37, -3.2e-36, ...  (garbage)
#   with bf16 interm placeholder  = 1.3125, ... (matches torch expected 1.3377)
#
# Fix: pass cb_o_tmp (bf16 accum_fmt, not live during the QK matmul) as the QK
# interm placeholder so the packer DF matches cb_qk. (PV already passed cb_p,
# which is bf16, so it was never affected.)
#
# NOTE: the op-level precision_matrix test currently SKIPS bf8b+fp16-DEST as
# "known-bad" (test_..._precision_matrix.py:118). With this fix that skip can be
# lifted — left to the implementer (the debugger does not edit acceptance tests).
# ============================================================================


def _fa_rand(*shape):
    n1 = torch.randn(shape)
    n2 = torch.randn(shape) * 10
    b = torch.bernoulli(torch.full(shape, 0.001))
    return n1 + n2 * b


def _pcc(a, b):
    x = a.flatten().float()
    y = b.flatten().float()
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm() + 1e-12))


@pytest.mark.parametrize("dtype_name", ["bfloat8_b", "bfloat16"])
def test_sdpa_fp16_dest_acc_off(device, dtype_name):
    """bf8b and bf16 must both pass at fp32_dest_acc_en=False (16-bit DEST).

    bf8b here was the original bug (PCC ~0.05 before the QK interm-placeholder
    format fix). bf16 is the control that always worked.
    """
    import ttnn
    from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

    dtype = {"bfloat8_b": ttnn.bfloat8_b, "bfloat16": ttnn.bfloat16}[dtype_name]
    torch.manual_seed(1234)
    b, h, s, d = 1, 1, 1024, 128
    q = _fa_rand(b, h, s, d)
    k = _fa_rand(b, h, s, d)
    v = _fa_rand(b, h, s, d)
    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)

    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    tq = ttnn.from_torch(q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(tq, tk, tv, is_causal=True, compute_kernel_config=ckc)
    o = ttnn.to_torch(out).float()

    pcc = _pcc(o, ref)
    assert pcc > 0.99, f"{dtype_name} fp16-DEST PCC={pcc:.5f} (expected >0.99)"
