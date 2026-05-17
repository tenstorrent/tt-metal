"""Single-head SDPA sandbox: Q @ K^T → softmax → attn @ V.

Validates composition of the encoder-matmul + softmax kernels against torch
scaled-dot-product attention. M=256, head_dim=64 (clean tile), M_kv=256.

3 sequential device kernels, all M-parallel sharded so outputs flow into
the next op without resharding:
  qk_T = Q @ K^T        (M-parallel matmul, M=256 K=64 N=256)
  attn = softmax(qk_T)  (row-wise softmax, K=256)
  out  = attn @ V       (M-parallel matmul, M=256 K=256 N=64)
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
HEAD_DIM = 64
M_KV = 256


def make_qkv(seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(M, HEAD_DIM, generator=g, dtype=torch.bfloat16) * 0.5
    k = torch.randn(M_KV, HEAD_DIM, generator=g, dtype=torch.bfloat16) * 0.5
    v = torch.randn(M_KV, HEAD_DIM, generator=g, dtype=torch.bfloat16) * 0.5
    return q, k, v


def torch_sdpa(q, k, v):
    qf, kf, vf = q.float(), k.float(), v.float()
    scale = 1.0 / math.sqrt(HEAD_DIM)
    attn_scores = qf @ kf.T * scale  # (M, M_kv)
    attn = F.softmax(attn_scores, dim=-1)
    out = attn @ vf  # (M, HEAD_DIM)
    return out.to(torch.bfloat16)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_sdpa_qk_only(device):
    """Sanity: M-parallel matmul at qk_T shape (M=256, K=64, N=256)."""
    import ttnn
    from generic_matmul_op import build_matmul_tensors_m_parallel, run_encoder_matmul

    q, k, _ = make_qkv(seed=42)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    k_scaled = (k.float() * scale).to(torch.bfloat16)
    kT = k_scaled.T.contiguous()  # (HEAD_DIM, M_KV)

    act, w, out = build_matmul_tensors_m_parallel(device, q, kT, M=M, K=HEAD_DIM, N=M_KV, num_cores=8)
    run_encoder_matmul(act, w, out, M=M, K=HEAD_DIM, N=M_KV, parallel="M", num_cores=8)
    qk_dev = ttnn.to_torch(out)
    torch_qk = (q.float() @ k_scaled.float().T).to(torch.bfloat16)
    p = pcc(torch_qk, qk_dev)
    print(f"\nPCC (qk_T only) = {p:.6f}")
    assert p >= 0.99


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_sdpa_end_to_end(device):
    """Full SDPA: qk_T → softmax → attn @ V, vs torch fp32 reference."""
    import ttnn
    from generic_matmul_op import build_matmul_tensors_m_parallel, run_encoder_matmul
    from softmax_op import SigLIPSoftmaxOp, build_tensors_for_softmax_test

    q, k, v = make_qkv(seed=42)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    k_scaled = (k.float() * scale).to(torch.bfloat16)
    kT = k_scaled.T.contiguous()  # (HEAD_DIM, M_KV)

    y_golden = torch_sdpa(q, k, v)  # (M, HEAD_DIM)

    # ---- Stage 1: qk_T = Q @ K^T ----
    act1, w1, out1 = build_matmul_tensors_m_parallel(device, q, kT, M=M, K=HEAD_DIM, N=M_KV, num_cores=8)
    run_encoder_matmul(act1, w1, out1, M=M, K=HEAD_DIM, N=M_KV, parallel="M", num_cores=8)
    qk_dev = ttnn.to_torch(out1)
    torch_qk = (q.float() @ k_scaled.float().T).to(torch.bfloat16)
    print(f"\n  PCC (qk_T) = {pcc(torch_qk, qk_dev):.6f}")

    # ---- Stage 2: attn = softmax(qk_T) along last dim ----
    # Bring qk back to host and re-shard for softmax (cheap; production would
    # avoid this by keeping the M-parallel sharded output in-place).
    qk_host = ttnn.to_torch(out1).contiguous()
    (act_sm, scaler_sm, max_t, exp_t, sum_t, isum_t, attn_sm) = build_tensors_for_softmax_test(
        device, qk_host, num_cores=8
    )
    SigLIPSoftmaxOp.op(act_sm, scaler_sm, attn_sm, max_t, exp_t, sum_t, isum_t, M=M, K=M_KV, num_cores=8)
    attn_dev = ttnn.to_torch(attn_sm)
    torch_attn = F.softmax(torch_qk.float(), dim=-1).to(torch.bfloat16)
    print(f"  PCC (attn=softmax) = {pcc(torch_attn, attn_dev):.6f}")

    # ---- Stage 3: out = attn @ V ----
    act3, w3, out3 = build_matmul_tensors_m_parallel(device, attn_dev, v, M=M, K=M_KV, N=HEAD_DIM, num_cores=8)
    run_encoder_matmul(act3, w3, out3, M=M, K=M_KV, N=HEAD_DIM, parallel="M", num_cores=8)
    out_dev = ttnn.to_torch(out3)
    p = pcc(y_golden, out_dev)
    print(f"  PCC (SDPA end-to-end) = {p:.6f}")
    assert p >= 0.99, f"SDPA end-to-end PCC {p} below 0.99 gate"
