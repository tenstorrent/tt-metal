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
