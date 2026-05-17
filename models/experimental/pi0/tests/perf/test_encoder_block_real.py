"""SigLIP encoder block at REAL π0.5 layer-0 weights — K1.1 acceptance test.

Loads the real layer-0 weights from
    /storage/sdawle/pi05_weights/pi05_base/model.safetensors
and runs the full encoder block end-to-end. Real activations come from
patch_embed + pos_embed (make_real_activation in golden_fc1.py).

Bias on QKV/O-proj/FC1/FC2 is added on host after each matmul (broadcast-add
on the (N,) bias to (M, N) output). Cheap and keeps the device dispatches
small. K1.2 (persistent kernel) will fuse bias into matmul pack.

Pipeline (matches HuggingFace SiglipVisionTransformer.encoder.layers[0]):
  attention:  LN1 → QKV+b → 16-head SDPA → O-proj+b → residual + x
  MLP:        LN2 → FC1+b → GELU-tanh → FC2+b → residual + x1
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_fc1 import pcc, make_real_activation  # noqa: E402

# Composition helpers
from test_attention_block import device_attention_subblock  # noqa: E402
from test_mlp_block import device_mlp_subblock  # noqa: E402

PI05_WEIGHTS = "/storage/sdawle/pi05_weights/pi05_base/model.safetensors"
VP = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.0."

M = 256
D = 1152
NUM_HEADS = 16
HEAD_DIM = 72
INTERMEDIATE = 4304
INTERMEDIATE_PADDED = 4320


def load_layer0_weights() -> dict:
    """Load all 12 weight tensors for SigLIP layer-0 attention + MLP sub-blocks."""
    sd = load_file(PI05_WEIGHTS)

    def _get(name: str) -> torch.Tensor:
        return sd[f"{VP}{name}"].to(torch.bfloat16)

    # LayerNorms (gamma + beta, shape (D,) each).
    ln1_w = _get("layer_norm1.weight")
    ln1_b = _get("layer_norm1.bias")
    ln2_w = _get("layer_norm2.weight")
    ln2_b = _get("layer_norm2.bias")

    # Attention Q/K/V projections (HF stores as (out, in), so .T to get (in, out)=(K, N)).
    wq = _get("self_attn.q_proj.weight").T.contiguous()  # (D, D)
    wk = _get("self_attn.k_proj.weight").T.contiguous()
    wv = _get("self_attn.v_proj.weight").T.contiguous()
    bq = _get("self_attn.q_proj.bias")  # (D,)
    bk = _get("self_attn.k_proj.bias")
    bv = _get("self_attn.v_proj.bias")
    qkv_w = torch.cat([wq, wk, wv], dim=1).contiguous()  # (D, 3D=3456)
    qkv_b = torch.cat([bq, bk, bv], dim=0).contiguous()  # (3D,)

    # O-proj
    o_w = _get("self_attn.out_proj.weight").T.contiguous()  # (D, D)
    o_b = _get("self_attn.out_proj.bias")

    # MLP
    fc1_w_logical = _get("mlp.fc1.weight").T.contiguous()  # (D, 4304)
    fc1_b_logical = _get("mlp.fc1.bias")  # (4304,)
    fc2_w_logical = _get("mlp.fc2.weight").T.contiguous()  # (4304, D)
    fc2_b = _get("mlp.fc2.bias")  # (D,)

    return dict(
        ln1_w=ln1_w,
        ln1_b=ln1_b,
        qkv_w=qkv_w,
        qkv_b=qkv_b,
        o_w=o_w,
        o_b=o_b,
        ln2_w=ln2_w,
        ln2_b=ln2_b,
        fc1_w_logical=fc1_w_logical,
        fc1_b_logical=fc1_b_logical,
        fc2_w_logical=fc2_w_logical,
        fc2_b=fc2_b,
    )


def torch_encoder_block_real(x, w):
    """Torch reference at real weights. Computes in fp32 for accuracy."""
    xf = x.float()
    # Attention sub-block
    ln1 = F.layer_norm(xf, (D,), w["ln1_w"].float(), w["ln1_b"].float(), eps=1e-6)
    qkv = ln1 @ w["qkv_w"].float() + w["qkv_b"].float()
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.reshape(M, NUM_HEADS, HEAD_DIM).permute(1, 0, 2).unsqueeze(0)
    k = k.reshape(M, NUM_HEADS, HEAD_DIM).permute(1, 0, 2).unsqueeze(0)
    v = v.reshape(M, NUM_HEADS, HEAD_DIM).permute(1, 0, 2).unsqueeze(0)
    mha = F.scaled_dot_product_attention(q, k, v).squeeze(0).permute(1, 0, 2).contiguous().reshape(M, D)
    oproj = mha @ w["o_w"].float() + w["o_b"].float()
    x1 = xf + oproj

    # MLP sub-block
    ln2 = F.layer_norm(x1, (D,), w["ln2_w"].float(), w["ln2_b"].float(), eps=1e-6)
    fc1 = ln2 @ w["fc1_w_logical"].float() + w["fc1_b_logical"].float()
    gelu = F.gelu(fc1, approximate="tanh")
    fc2 = gelu @ w["fc2_w_logical"].float() + w["fc2_b"].float()
    x2 = x1 + fc2

    return x2.to(torch.bfloat16)


def device_encoder_block_real(device, x, w):
    """Device path. Biases applied on host after each matmul (broadcast-add
    on (N,) → (M, N) before passing to the next stage). Cleaner than
    fusing for first cut; production K1.2 will fuse bias into matmul pack.

    To inject bias, we adapt the device_attention_subblock / device_mlp_subblock
    by inlining their stages here, with bias adds between stages.
    """
    import ttnn
    from layernorm_op import SigLIPLayerNormOp, build_tensors_for_ln_test
    from qkv_op import SigLIPQKVMatmulOp, build_tensors_for_test as build_qkv
    from oproj_op import SigLIPOprojMatmulOp, build_tensors_for_oproj_test
    from residual_op import SigLIPResidualAddOp, build_tensors_for_residual_test
    from fc1_op import SigLIPFC1MatmulOp, build_tensors_for_fc1_test
    from gelu_op import SigLIPGeluOp, build_tensors_for_gelu_test
    from fc2_op import SigLIPFC2MatmulOp, build_tensors_for_fc2_test
    from test_sdpa_siglip import single_head_sdpa_device_padded, pad_head_dim
    import math

    def _free(*ts):
        for t in ts:
            ttnn.deallocate(t)

    def _add_bias_host(matmul_out: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """matmul_out: (M, N), bias: (N,). Returns (M, N) = matmul_out + bias."""
        return (matmul_out.float() + bias.float()).to(torch.bfloat16).contiguous()

    HEAD_DIM_PADDED = 96

    # ============ Stage 1: LN1 ============
    (
        a_tt,
        gamma_tt,
        beta_tt,
        sc_tt,
        on_tt,
        ac_tt,
        xmm_tt,
        xmm2_tt,
        m_tt,
        v_tt,
        iv_tt,
        ln_out_tt,
    ) = build_tensors_for_ln_test(device, w["ln1_w"], w["ln1_b"], x, num_cores=8)
    SigLIPLayerNormOp.op(
        a_tt,
        gamma_tt,
        beta_tt,
        sc_tt,
        on_tt,
        ac_tt,
        xmm_tt,
        xmm2_tt,
        m_tt,
        v_tt,
        iv_tt,
        ln_out_tt,
        num_cores=8,
        eps=1e-6,
    )
    ln1_out = ttnn.to_torch(ln_out_tt).contiguous()
    _free(a_tt, gamma_tt, beta_tt, sc_tt, on_tt, ac_tt, xmm_tt, xmm2_tt, m_tt, v_tt, iv_tt, ln_out_tt)

    # ============ Stage 2: QKV matmul + bias ============
    dummy_b = torch.zeros(3 * D, dtype=torch.bfloat16)
    act_q, w_q, qkv_out_tt = build_qkv(device, w["qkv_w"], dummy_b, ln1_out, num_cores=36)
    SigLIPQKVMatmulOp.op(act_q, w_q, qkv_out_tt, num_cores=36)
    qkv_out = ttnn.to_torch(qkv_out_tt).contiguous()
    _free(act_q, w_q, qkv_out_tt)
    qkv_out = _add_bias_host(qkv_out, w["qkv_b"])

    # Split Q, K, V
    q_full, k_full, v_full = qkv_out.chunk(3, dim=-1)
    q_heads = q_full.reshape(M, NUM_HEADS, HEAD_DIM).contiguous()
    k_heads = k_full.reshape(M, NUM_HEADS, HEAD_DIM).contiguous()
    v_heads = v_full.reshape(M, NUM_HEADS, HEAD_DIM).contiguous()

    # ============ Stage 3: 16-head SDPA loop ============
    scale = 1.0 / math.sqrt(HEAD_DIM)
    out_heads = []
    for h in range(NUM_HEADS):
        q_h_pad = pad_head_dim(q_heads[:, h, :].contiguous(), HEAD_DIM_PADDED)
        k_h_pad = pad_head_dim(k_heads[:, h, :].contiguous(), HEAD_DIM_PADDED)
        v_h_pad = pad_head_dim(v_heads[:, h, :].contiguous(), HEAD_DIM_PADDED)
        out_pad = single_head_sdpa_device_padded(device, q_h_pad, k_h_pad, v_h_pad, scale)
        out_heads.append(out_pad[:, :HEAD_DIM].contiguous())
    mha = torch.stack(out_heads, dim=1).reshape(M, D).to(torch.bfloat16).contiguous()

    # ============ Stage 4: O-proj + bias ============
    act_o, w_o, oproj_out_tt = build_tensors_for_oproj_test(device, w["o_w"], mha, num_cores=36)
    SigLIPOprojMatmulOp.op(act_o, w_o, oproj_out_tt, num_cores=36)
    oproj_out = ttnn.to_torch(oproj_out_tt).contiguous()
    _free(act_o, w_o, oproj_out_tt)
    oproj_out = _add_bias_host(oproj_out, w["o_b"])

    # ============ Stage 5: Attention residual: x1 = x + oproj_out ============
    a_tt, b_tt, x1_tt = build_tensors_for_residual_test(device, oproj_out, x, num_cores=8)
    SigLIPResidualAddOp.op(a_tt, b_tt, x1_tt)
    x1 = ttnn.to_torch(x1_tt).contiguous()
    _free(a_tt, b_tt, x1_tt)

    # ============ Stage 6: LN2 ============
    (
        a_tt,
        gamma_tt,
        beta_tt,
        sc_tt,
        on_tt,
        ac_tt,
        xmm_tt,
        xmm2_tt,
        m_tt,
        v_tt,
        iv_tt,
        ln_out_tt,
    ) = build_tensors_for_ln_test(device, w["ln2_w"], w["ln2_b"], x1, num_cores=8)
    SigLIPLayerNormOp.op(
        a_tt,
        gamma_tt,
        beta_tt,
        sc_tt,
        on_tt,
        ac_tt,
        xmm_tt,
        xmm2_tt,
        m_tt,
        v_tt,
        iv_tt,
        ln_out_tt,
        num_cores=8,
        eps=1e-6,
    )
    ln2_out = ttnn.to_torch(ln_out_tt).contiguous()
    _free(a_tt, gamma_tt, beta_tt, sc_tt, on_tt, ac_tt, xmm_tt, xmm2_tt, m_tt, v_tt, iv_tt, ln_out_tt)

    # ============ Stage 7: FC1 + bias ============
    # Pad FC1 weight + bias to N=4320.
    pad_n = INTERMEDIATE_PADDED - INTERMEDIATE
    fc1_w_pad = torch.cat([w["fc1_w_logical"], torch.zeros(D, pad_n, dtype=torch.bfloat16)], dim=1).contiguous()
    fc1_b_pad = torch.cat([w["fc1_b_logical"], torch.zeros(pad_n, dtype=torch.bfloat16)], dim=0)
    act_f1, w_f1, fc1_out_tt = build_tensors_for_fc1_test(device, fc1_w_pad, ln2_out, num_cores=27)
    SigLIPFC1MatmulOp.op(act_f1, w_f1, fc1_out_tt, num_cores=27)
    fc1_out = ttnn.to_torch(fc1_out_tt).contiguous()  # (M, 4320)
    _free(act_f1, w_f1, fc1_out_tt)
    fc1_out = _add_bias_host(fc1_out, fc1_b_pad)

    # ============ Stage 8: GELU ============
    gin_tt, gout_tt = build_tensors_for_gelu_test(device, fc1_out, num_cores=8)
    SigLIPGeluOp.op(gin_tt, gout_tt)
    gelu_out = ttnn.to_torch(gout_tt).contiguous()
    _free(gin_tt, gout_tt)

    # ============ Stage 9: FC2 + bias ============
    fc2_w_pad = torch.cat([w["fc2_w_logical"], torch.zeros(pad_n, D, dtype=torch.bfloat16)], dim=0).contiguous()
    act_f2, w_f2, fc2_out_tt = build_tensors_for_fc2_test(device, fc2_w_pad, gelu_out)
    SigLIPFC2MatmulOp.op(act_f2, w_f2, fc2_out_tt, device)
    fc2_full = ttnn.to_torch(fc2_out_tt).contiguous()
    fc2_out = fc2_full[:M, :].contiguous()  # take row-0 replica from FC2's K-row-replicated output
    _free(act_f2, w_f2, fc2_out_tt)
    fc2_out = _add_bias_host(fc2_out, w["fc2_b"])

    # ============ Stage 10: MLP residual: x2 = x1 + fc2_out ============
    a_tt, b_tt, x2_tt = build_tensors_for_residual_test(device, fc2_out, x1, num_cores=8)
    SigLIPResidualAddOp.op(a_tt, b_tt, x2_tt)
    x2 = ttnn.to_torch(x2_tt).contiguous()
    _free(a_tt, b_tt, x2_tt)

    return x2


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_encoder_block_real_weights(device):
    """K1.1 acceptance test: full encoder block on real π0.5 layer-0 weights."""
    w = load_layer0_weights()
    x = make_real_activation(seed=42)  # (M=256, D=1152) bf16

    y_golden = torch_encoder_block_real(x, w)
    y_device = device_encoder_block_real(device, x, w)

    p = pcc(y_golden, y_device)
    print(f"\nPCC (encoder block, real π0.5 layer-0 weights) = {p:.6f}")
    print(f"  M={M}, D={D}, num_heads={NUM_HEADS}, head_dim={HEAD_DIM}")
    assert p >= 0.99, f"Real-weights encoder block PCC {p} below 0.99 gate"
