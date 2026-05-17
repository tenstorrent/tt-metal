"""SigLIP attention sub-block composition: LN1 → QKV → MHA-SDPA → O-proj → residual.

End-to-end orchestrator gluing today's K1.1 primitives into the SigLIP
encoder attention sub-block (without bias on QKV/O-proj for first cut).

Shape:
  M = 256, D = 1152, num_heads = 16, head_dim = 72 (padded to 96 for SDPA tiles).
  QKV out = (M, 3*D) = (256, 3456); split → Q, K, V each (M, D).

Validation: compare against a torch fp32 reference of the same sub-block.

Each device stage round-trips through host between stages (to_torch +
from_torch with new sharding). Correctness-only — production will chain L1-to-L1.
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
D = 1152
NUM_HEADS = 16
HEAD_DIM_TRUE = 72
HEAD_DIM_PADDED = 96
M_KV = 256


def make_inputs(seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, D, generator=g, dtype=torch.bfloat16) * 0.5
    ln_w = torch.ones(D, dtype=torch.bfloat16) + torch.randn(D, generator=g, dtype=torch.bfloat16) * 0.1
    ln_b = torch.randn(D, generator=g, dtype=torch.bfloat16) * 0.05
    # QKV fused weight (K=D, N=3D), no bias for first cut.
    qkv_w = torch.randn(D, 3 * D, generator=g, dtype=torch.bfloat16) * (1.0 / math.sqrt(D))
    # O-proj weight (K=D, N=D), no bias for first cut.
    o_w = torch.randn(D, D, generator=g, dtype=torch.bfloat16) * (1.0 / math.sqrt(D))
    return x, ln_w, ln_b, qkv_w, o_w


def torch_attention_subblock(x, ln_w, ln_b, qkv_w, o_w):
    """Reference attention sub-block in fp32, output bf16."""
    xf = x.float()
    ln_out = F.layer_norm(xf, (D,), ln_w.float(), ln_b.float(), eps=1e-6)
    qkv_out = ln_out @ qkv_w.float()  # (M, 3D)
    q, k, v = qkv_out.chunk(3, dim=-1)  # each (M, D)
    # Reshape to (num_heads, M, head_dim).
    q = q.reshape(M, NUM_HEADS, HEAD_DIM_TRUE).permute(1, 0, 2).unsqueeze(0)  # (1, NH, M, HD)
    k = k.reshape(M_KV, NUM_HEADS, HEAD_DIM_TRUE).permute(1, 0, 2).unsqueeze(0)
    v = v.reshape(M_KV, NUM_HEADS, HEAD_DIM_TRUE).permute(1, 0, 2).unsqueeze(0)
    mha = F.scaled_dot_product_attention(q, k, v)  # (1, NH, M, HD)
    mha = mha.squeeze(0).permute(1, 0, 2).contiguous().reshape(M, D)  # (M, D)
    oproj = mha @ o_w.float()  # (M, D)
    out = xf + oproj  # residual
    return out.to(torch.bfloat16)


def device_attention_subblock(device, x, ln_w, ln_b, qkv_w, o_w):
    """Device path through K1.1 primitives. Returns (M, D) bf16."""
    import ttnn
    from layernorm_op import SigLIPLayerNormOp, build_tensors_for_ln_test
    from qkv_op import SigLIPQKVMatmulOp, build_tensors_for_test as build_qkv
    from oproj_op import SigLIPOprojMatmulOp, build_tensors_for_oproj_test
    from residual_op import SigLIPResidualAddOp, build_tensors_for_residual_test
    from test_sdpa_siglip import single_head_sdpa_device_padded, pad_head_dim

    def _free(*tensors):
        for t in tensors:
            ttnn.deallocate(t)

    # ============ Stage 1: LN1 ============
    (
        act_tt,
        gamma_tt,
        beta_tt,
        scaler_tt,
        ones_tt,
        accum_tt,
        xmm_tt,
        xmm2_tt,
        mean_tt,
        var_tt,
        ivar_tt,
        ln_out_tt,
    ) = build_tensors_for_ln_test(device, ln_w, ln_b, x, num_cores=8)
    SigLIPLayerNormOp.op(
        act_tt,
        gamma_tt,
        beta_tt,
        scaler_tt,
        ones_tt,
        accum_tt,
        xmm_tt,
        xmm2_tt,
        mean_tt,
        var_tt,
        ivar_tt,
        ln_out_tt,
        num_cores=8,
        eps=1e-6,
    )
    ln_out_host = ttnn.to_torch(ln_out_tt).contiguous()
    _free(act_tt, gamma_tt, beta_tt, scaler_tt, ones_tt, accum_tt, xmm_tt, xmm2_tt, mean_tt, var_tt, ivar_tt, ln_out_tt)

    # ============ Stage 2: QKV matmul (M=256, K=1152, N=3456) ============
    qkv_dummy_bias = torch.zeros(3 * D, dtype=torch.bfloat16)
    act_qkv, w_qkv, qkv_out_tt = build_qkv(device, qkv_w, qkv_dummy_bias, ln_out_host, num_cores=36)
    SigLIPQKVMatmulOp.op(act_qkv, w_qkv, qkv_out_tt, num_cores=36)
    qkv_out_host = ttnn.to_torch(qkv_out_tt).contiguous()  # (M, 3D)
    _free(act_qkv, w_qkv, qkv_out_tt)

    # Split Q, K, V on host (cheap).
    q_full, k_full, v_full = qkv_out_host.chunk(3, dim=-1)  # each (M, D)

    # ============ Stage 3: 16-head SDPA loop ============
    # Reshape to (M, num_heads, head_dim).
    q_heads = q_full.reshape(M, NUM_HEADS, HEAD_DIM_TRUE).contiguous()
    k_heads = k_full.reshape(M_KV, NUM_HEADS, HEAD_DIM_TRUE).contiguous()
    v_heads = v_full.reshape(M_KV, NUM_HEADS, HEAD_DIM_TRUE).contiguous()

    scale = 1.0 / math.sqrt(HEAD_DIM_TRUE)
    out_heads = []
    for h in range(NUM_HEADS):
        q_h = q_heads[:, h, :].contiguous()
        k_h = k_heads[:, h, :].contiguous()
        v_h = v_heads[:, h, :].contiguous()
        q_h_pad = pad_head_dim(q_h, HEAD_DIM_PADDED)
        k_h_pad = pad_head_dim(k_h, HEAD_DIM_PADDED)
        v_h_pad = pad_head_dim(v_h, HEAD_DIM_PADDED)
        out_pad = single_head_sdpa_device_padded(device, q_h_pad, k_h_pad, v_h_pad, scale)
        out_heads.append(out_pad[:, :HEAD_DIM_TRUE].contiguous())
    mha_out = torch.stack(out_heads, dim=1).reshape(M, D).to(torch.bfloat16).contiguous()

    # ============ Stage 4: O-proj matmul (M=256, K=1152, N=1152) ============
    act_op, w_op, oproj_out_tt = build_tensors_for_oproj_test(device, o_w, mha_out, num_cores=36)
    SigLIPOprojMatmulOp.op(act_op, w_op, oproj_out_tt, num_cores=36)
    oproj_out_host = ttnn.to_torch(oproj_out_tt).contiguous()  # (M, D)
    _free(act_op, w_op, oproj_out_tt)

    # ============ Stage 5: Residual add (M, D) + (M, D) ============
    a_tt, b_tt, res_out_tt = build_tensors_for_residual_test(device, oproj_out_host, x, num_cores=8)
    SigLIPResidualAddOp.op(a_tt, b_tt, res_out_tt)
    result = ttnn.to_torch(res_out_tt).contiguous()
    _free(a_tt, b_tt, res_out_tt)
    return result


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_attention_subblock_synthetic(device):
    """End-to-end attention sub-block on synthetic weights.

    Pipeline: LN1 → QKV → 16-head SDPA → O-proj → residual.
    Validates composition data-flow correctness. No bias on QKV/O-proj.
    """
    x, ln_w, ln_b, qkv_w, o_w = make_inputs(seed=42)
    y_golden = torch_attention_subblock(x, ln_w, ln_b, qkv_w, o_w)

    y_device = device_attention_subblock(device, x, ln_w, ln_b, qkv_w, o_w)

    p = pcc(y_golden, y_device)
    print(f"\nPCC (attention sub-block end-to-end) = {p:.6f}")
    print(f"  M={M}, D={D}, num_heads={NUM_HEADS}, head_dim={HEAD_DIM_TRUE}→{HEAD_DIM_PADDED}")
    assert p >= 0.99, f"Attention sub-block PCC {p} below 0.99 gate"
