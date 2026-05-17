"""Multi-head SDPA at the exact SigLIP shape: 16 heads × head_dim=72.

head_dim=72 doesn't tile-align (72 = 2.25 tiles). We zero-pad head_dim
from 72 → 96 (3 tiles) for the device compute, then unpad on output:

  Q_h, K_h, V_h           torch (M, 72)      bf16
  Q_h_pad, K_h_pad, V_h_pad  torch (M, 96)   bf16, last 24 cols = 0
  scale = 1/sqrt(72)      (NOT 1/sqrt(96) — true head_dim is 72)

  Device per head (3 dispatches):
    qk_T = Q_h_pad @ K_h_pad^T * scale     (M, 256)   ← padded zeros contribute 0
    attn = softmax(qk_T, dim=-1)            (M, 256)
    out_pad = attn @ V_h_pad                (M, 96)   ← last 24 cols of out_pad are 0
  Unpad: out_h = out_pad[:, :72]            (M, 72)

  After 16 heads: stack (M, 16, 72) → reshape (M, 1152).

Compares against torch.nn.functional.scaled_dot_product_attention on the
original unpadded (M, 16, 72) Q/K/V tensors.
"""
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_fc1 import pcc  # noqa: E402

M = 256
NUM_HEADS = 16
HEAD_DIM_TRUE = 72  # SigLIP-So400m's actual head_dim
HEAD_DIM_PADDED = 96  # padded to 3 tiles for device compute
M_KV = 256
D = NUM_HEADS * HEAD_DIM_TRUE  # 1152


def make_qkv(seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(M, NUM_HEADS, HEAD_DIM_TRUE, generator=g, dtype=torch.bfloat16) * 0.5
    k = torch.randn(M_KV, NUM_HEADS, HEAD_DIM_TRUE, generator=g, dtype=torch.bfloat16) * 0.5
    v = torch.randn(M_KV, NUM_HEADS, HEAD_DIM_TRUE, generator=g, dtype=torch.bfloat16) * 0.5
    return q, k, v


def torch_multihead_sdpa(q, k, v):
    qt = q.permute(1, 0, 2).unsqueeze(0).float()
    kt = k.permute(1, 0, 2).unsqueeze(0).float()
    vt = v.permute(1, 0, 2).unsqueeze(0).float()
    out = F.scaled_dot_product_attention(qt, kt, vt)
    out = out.squeeze(0).permute(1, 0, 2).contiguous()
    return out.reshape(M, D).to(torch.bfloat16)


def pad_head_dim(t, pad_to):
    """(M, head_dim_true) → (M, pad_to) with zero-padded tail."""
    if t.shape[-1] == pad_to:
        return t
    pad_n = pad_to - t.shape[-1]
    return torch.cat([t, torch.zeros(t.shape[0], pad_n, dtype=t.dtype)], dim=-1).contiguous()


def single_head_sdpa_device_padded(device, q_h_pad, k_h_pad, v_h_pad, scale):
    """Run one head's SDPA on device at padded head_dim.

    q_h_pad, k_h_pad, v_h_pad: (M, HEAD_DIM_PADDED) bf16, real values in
    [:, :HEAD_DIM_TRUE], zeros in [:, HEAD_DIM_TRUE:].
    scale: 1/sqrt(HEAD_DIM_TRUE) — the TRUE-head-dim scaling.
    Returns: (M, HEAD_DIM_PADDED) bf16; caller unpads to first HEAD_DIM_TRUE cols.
    """
    import ttnn
    from generic_matmul_op import build_matmul_tensors_m_parallel, run_encoder_matmul
    from softmax_op import SigLIPSoftmaxOp, build_tensors_for_softmax_test

    k_scaled = (k_h_pad.float() * scale).to(torch.bfloat16)
    kT = k_scaled.T.contiguous()  # (HEAD_DIM_PADDED, M_KV)

    # qk_T = Q_h_pad @ K_h_pad^T (scaled)
    act1, w1, out1 = build_matmul_tensors_m_parallel(device, q_h_pad, kT, M=M, K=HEAD_DIM_PADDED, N=M_KV, num_cores=8)
    run_encoder_matmul(act1, w1, out1, M=M, K=HEAD_DIM_PADDED, N=M_KV, parallel="M", num_cores=8)
    qk_host = ttnn.to_torch(out1).contiguous()
    ttnn.deallocate(act1)
    ttnn.deallocate(w1)
    ttnn.deallocate(out1)

    # softmax(qk_T)
    (act_sm, scaler_sm, max_t, exp_t, sum_t, isum_t, attn_sm) = build_tensors_for_softmax_test(
        device, qk_host, num_cores=8
    )
    SigLIPSoftmaxOp.op(act_sm, scaler_sm, attn_sm, max_t, exp_t, sum_t, isum_t, M=M, K=M_KV, num_cores=8)
    attn_host = ttnn.to_torch(attn_sm).contiguous()
    for _t in (act_sm, scaler_sm, max_t, exp_t, sum_t, isum_t, attn_sm):
        ttnn.deallocate(_t)

    # out_pad = attn @ V_h_pad
    act3, w3, out3 = build_matmul_tensors_m_parallel(
        device, attn_host, v_h_pad, M=M, K=M_KV, N=HEAD_DIM_PADDED, num_cores=8
    )
    run_encoder_matmul(act3, w3, out3, M=M, K=M_KV, N=HEAD_DIM_PADDED, parallel="M", num_cores=8)
    out_host = ttnn.to_torch(out3)
    ttnn.deallocate(act3)
    ttnn.deallocate(w3)
    ttnn.deallocate(out3)
    return out_host


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_siglip_multihead_sdpa(device):
    """SigLIP-exact multi-head SDPA: 16 heads × head_dim=72 (padded to 96).

    Validates the head_dim-padding strategy against torch SDPA at true head_dim=72.
    """
    q, k, v = make_qkv(seed=42)
    y_golden = torch_multihead_sdpa(q, k, v)  # (M, D=1152)

    scale = 1.0 / math.sqrt(HEAD_DIM_TRUE)

    out_heads = []
    head_pccs = []
    for h in range(NUM_HEADS):
        q_h = q[:, h, :].contiguous()
        k_h = k[:, h, :].contiguous()
        v_h = v[:, h, :].contiguous()

        q_h_pad = pad_head_dim(q_h, HEAD_DIM_PADDED)
        k_h_pad = pad_head_dim(k_h, HEAD_DIM_PADDED)
        v_h_pad = pad_head_dim(v_h, HEAD_DIM_PADDED)

        out_pad = single_head_sdpa_device_padded(device, q_h_pad, k_h_pad, v_h_pad, scale)
        out_h = out_pad[:, :HEAD_DIM_TRUE].contiguous()  # unpad to (M, 72)

        # Per-head torch reference for diagnostics.
        out_h_torch = (
            F.scaled_dot_product_attention(
                q_h.float().unsqueeze(0).unsqueeze(0),
                k_h.float().unsqueeze(0).unsqueeze(0),
                v_h.float().unsqueeze(0).unsqueeze(0),
            )
            .squeeze(0)
            .squeeze(0)
            .to(torch.bfloat16)
        )
        p_h = pcc(out_h_torch, out_h)
        head_pccs.append(p_h)
        out_heads.append(out_h)

    y_device = torch.stack(out_heads, dim=1).reshape(M, D).to(torch.bfloat16)

    p = pcc(y_golden, y_device)
    print(f"\nPCC (SigLIP multi-head SDPA, 16×72 padded to 96) = {p:.6f}")
    print(
        f"  Per-head PCC range: min={min(head_pccs):.6f}, max={max(head_pccs):.6f}, "
        f"mean={sum(head_pccs)/len(head_pccs):.6f}"
    )

    assert p >= 0.99, f"SigLIP SDPA PCC {p} below 0.99 gate"
