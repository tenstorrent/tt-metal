"""Multi-head SDPA: per-head loop over the proven single-head pipeline.

Sandbox shape: M=256, num_heads=18, head_dim=64 (clean tile alignment, total
D=1152). Compares against torch.nn.functional.scaled_dot_product_attention.

Each head independently runs:
    qk_T = Q_h @ K_h^T (scaled)
    attn = softmax(qk_T, dim=-1)
    out  = attn @ V_h
Outputs are stacked across heads then reshaped to (M, num_heads*head_dim).

This is a correctness sandbox — 18×3 = 54 kernel dispatches with host
round-trips between stages. Production attention block will batch heads
and chain L1-to-L1.
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
NUM_HEADS = 18
HEAD_DIM = 64
M_KV = 256
D = NUM_HEADS * HEAD_DIM  # 1152


def make_qkv(seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    # (M, num_heads, head_dim) layout.
    q = torch.randn(M, NUM_HEADS, HEAD_DIM, generator=g, dtype=torch.bfloat16) * 0.5
    k = torch.randn(M_KV, NUM_HEADS, HEAD_DIM, generator=g, dtype=torch.bfloat16) * 0.5
    v = torch.randn(M_KV, NUM_HEADS, HEAD_DIM, generator=g, dtype=torch.bfloat16) * 0.5
    return q, k, v


def torch_multihead_sdpa(q, k, v):
    # q,k,v: (M, num_heads, head_dim). torch SDPA wants (B, num_heads, M, head_dim).
    qt = q.permute(1, 0, 2).unsqueeze(0).float()  # (1, NH, M, HD)
    kt = k.permute(1, 0, 2).unsqueeze(0).float()
    vt = v.permute(1, 0, 2).unsqueeze(0).float()
    out = F.scaled_dot_product_attention(qt, kt, vt)  # (1, NH, M, HD)
    out = out.squeeze(0).permute(1, 0, 2).contiguous()  # (M, NH, HD)
    return out.reshape(M, D).to(torch.bfloat16)


def single_head_sdpa_device(device, q_h, k_h, v_h, head_dim):
    """Run one head's SDPA on device. Returns torch (M, head_dim) bf16."""
    import ttnn
    from generic_matmul_op import build_matmul_tensors_m_parallel, run_encoder_matmul
    from softmax_op import SigLIPSoftmaxOp, build_tensors_for_softmax_test

    scale = 1.0 / math.sqrt(head_dim)
    k_scaled = (k_h.float() * scale).to(torch.bfloat16)
    kT = k_scaled.T.contiguous()  # (head_dim, M_KV)

    # Stage 1: qk_T = Q_h @ K_h^T (scaled)
    act1, w1, out1 = build_matmul_tensors_m_parallel(device, q_h, kT, M=M, K=head_dim, N=M_KV, num_cores=8)
    run_encoder_matmul(act1, w1, out1, M=M, K=head_dim, N=M_KV, parallel="M", num_cores=8)
    qk_host = ttnn.to_torch(out1).contiguous()

    # Stage 2: attn = softmax(qk_T)
    (act_sm, scaler_sm, max_t, exp_t, sum_t, isum_t, attn_sm) = build_tensors_for_softmax_test(
        device, qk_host, num_cores=8
    )
    SigLIPSoftmaxOp.op(act_sm, scaler_sm, attn_sm, max_t, exp_t, sum_t, isum_t, M=M, K=M_KV, num_cores=8)
    attn_host = ttnn.to_torch(attn_sm).contiguous()

    # Stage 3: out = attn @ V_h
    act3, w3, out3 = build_matmul_tensors_m_parallel(device, attn_host, v_h, M=M, K=M_KV, N=head_dim, num_cores=8)
    run_encoder_matmul(act3, w3, out3, M=M, K=M_KV, N=head_dim, parallel="M", num_cores=8)
    return ttnn.to_torch(out3)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_multihead_sdpa(device):
    """Per-head loop SDPA, vs torch multi-head SDPA reference."""
    q, k, v = make_qkv(seed=42)
    y_golden = torch_multihead_sdpa(q, k, v)  # (M, D)

    out_heads = []
    head_pccs = []
    for h in range(NUM_HEADS):
        q_h = q[:, h, :].contiguous()
        k_h = k[:, h, :].contiguous()
        v_h = v[:, h, :].contiguous()

        out_h = single_head_sdpa_device(device, q_h, k_h, v_h, head_dim=HEAD_DIM)

        # Reference for this head alone (sanity check the head-loop is working).
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
    print(f"\nPCC (multi-head SDPA, full) = {p:.6f}")
    print(
        f"  Per-head PCC range: min={min(head_pccs):.6f}, max={max(head_pccs):.6f}, mean={sum(head_pccs)/len(head_pccs):.6f}"
    )
    print(f"  num_heads={NUM_HEADS}, head_dim={HEAD_DIM}, M={M}, M_kv={M_KV}, D={D}")

    assert p >= 0.99, f"Multi-head SDPA PCC {p} below 0.99 gate"
