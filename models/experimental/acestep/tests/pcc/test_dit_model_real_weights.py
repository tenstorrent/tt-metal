# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: full ACE-Step v1.5 DiT model (24 layers) with REAL checkpoint weights.

The ultimate correctness signal: the entire AceStepDiTModel (dual timestep -> concat context
-> patchify -> condition embedder -> ALL 24 real DiT layers -> norm_out AdaLN -> de-patchify)
run with the genuine trained tensors from model.safetensors, validated against the real HF
AceStepDiTModel populated with the same weights.

This proves the full model handles the real bf16 weight distribution end-to-end, not just random
init. Requires model.safetensors (4.79 GB); skips cleanly if absent.
"""

import pytest
import torch

import ttnn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from models.experimental.acestep.reference.hf_reference import load_config, load_modeling_module
from models.experimental.acestep.reference.weight_utils import checkpoint_path, load_module_weights
from models.experimental.acestep.tt.dit_layer import AceStepDiTLayerConfig
from models.experimental.acestep.tt.dit_model import AceStepDiTModel, AceStepDiTModelConfig
from models.experimental.acestep.tt.dit_output import DiTOutputConfig
from models.experimental.acestep.tt.dit_stack import AceStepDiTStackConfig
from models.experimental.acestep.tt.patch_embed import PatchEmbedConfig
from models.experimental.acestep.tt.timestep_embedding import TimestepEmbeddingConfig
from models.experimental.acestep.tests.test_utils import (
    HEAD_DIM,
    HIDDEN_SIZE,
    NUM_ATTENTION_HEADS,
    NUM_KEY_VALUE_HEADS,
    RMS_NORM_EPS,
    SLIDING_WINDOW,
    assert_pcc,
    make_lazy_weight,
    require_single_device,
    to_torch,
    to_ttnn_tensor,
)

SEQ_LEN = 128
ENC_LEN = 96
IN_CHANNELS = 192
OUT_CHANNELS = 64
PATCH = 2
TIME_IN = 256
HIDDEN_CH = 64


def _have_checkpoint():
    try:
        checkpoint_path()
        return True
    except AssertionError:
        return False


def _wT(w, d):
    return make_lazy_weight(
        w.detach().clone().transpose(-1, -2).contiguous(), d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )


def _norm(w, d):
    return make_lazy_weight(w.detach().clone(), d, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)


def _bias(b, d):
    return make_lazy_weight(
        b.detach().clone().reshape(1, -1).contiguous(), d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )


def _te_cfg(te, d):
    return TimestepEmbeddingConfig(
        linear_1_weight=_wT(te.linear_1.weight, d),
        linear_1_bias=_bias(te.linear_1.bias, d),
        linear_2_weight=_wT(te.linear_2.weight, d),
        linear_2_bias=_bias(te.linear_2.bias, d),
        time_proj_weight=_wT(te.time_proj.weight, d),
        time_proj_bias=_bias(te.time_proj.bias, d),
        in_channels=TIME_IN,
        time_embed_dim=HIDDEN_SIZE,
    )


def _layer_cfg(rl, at, d):
    a, c = rl.self_attn, rl.cross_attn
    return AceStepDiTLayerConfig(
        scale_shift_table=make_lazy_weight(
            rl.scale_shift_table.detach().clone(), d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        ),
        self_attn_norm_weight=_norm(rl.self_attn_norm.weight, d),
        mlp_norm_weight=_norm(rl.mlp_norm.weight, d),
        cross_attn_norm_weight=_norm(rl.cross_attn_norm.weight, d),
        wq=_wT(a.q_proj.weight, d),
        wk=_wT(a.k_proj.weight, d),
        wv=_wT(a.v_proj.weight, d),
        wo=_wT(a.o_proj.weight, d),
        q_norm_weight=_norm(a.q_norm.weight, d),
        k_norm_weight=_norm(a.k_norm.weight, d),
        c_wq=_wT(c.q_proj.weight, d),
        c_wk=_wT(c.k_proj.weight, d),
        c_wv=_wT(c.v_proj.weight, d),
        c_wo=_wT(c.o_proj.weight, d),
        c_q_norm_weight=_norm(c.q_norm.weight, d),
        c_k_norm_weight=_norm(c.k_norm.weight, d),
        w1=_wT(rl.mlp.gate_proj.weight, d),
        w2=_wT(rl.mlp.down_proj.weight, d),
        w3=_wT(rl.mlp.up_proj.weight, d),
        n_heads=NUM_ATTENTION_HEADS,
        n_kv_heads=NUM_KEY_VALUE_HEADS,
        head_dim=HEAD_DIM,
        dim=HIDDEN_SIZE,
        eps=RMS_NORM_EPS,
        sliding_window=SLIDING_WINDOW if at == "sliding_attention" else None,
        use_cross_attention=True,
    )


@pytest.mark.skipif(not _have_checkpoint(), reason="model.safetensors not downloaded")
def test_dit_model_real_weights_vs_hf(device):
    require_single_device(device)
    torch.manual_seed(0)

    m = load_modeling_module()
    cfg = load_config()
    cfg._attn_implementation = "eager"
    dit = m.AceStepDiTModel(cfg).eval()  # full 24 layers
    load_module_weights(dit, "decoder.")  # genuine trained weights

    hidden = torch.randn(1, SEQ_LEN, HIDDEN_CH, dtype=torch.float32)
    context = torch.randn(1, SEQ_LEN, IN_CHANNELS - HIDDEN_CH, dtype=torch.float32)
    encoder = torch.randn(1, ENC_LEN, HIDDEN_SIZE, dtype=torch.float32)
    t = torch.rand(1, dtype=torch.float32)
    t_r = torch.rand(1, dtype=torch.float32)

    with torch.no_grad():
        (ref_out, *_) = dit(
            hidden_states=hidden,
            timestep=t,
            timestep_r=t_r,
            attention_mask=None,
            encoder_hidden_states=encoder,
            encoder_attention_mask=None,
            context_latents=context,
        )

    tprime = SEQ_LEN // PATCH
    rope = Qwen3RotaryEmbedding(cfg)
    position_ids = torch.arange(tprime).unsqueeze(0)
    cos, sin = rope(torch.zeros(1, tprime, HEAD_DIM), position_ids)
    sliding_mask = (
        m.create_4d_mask(
            seq_len=tprime,
            dtype=torch.float32,
            device=hidden.device,
            attention_mask=None,
            sliding_window=SLIDING_WINDOW,
            is_sliding_window=True,
            is_causal=False,
        )
        if tprime > SLIDING_WINDOW
        else None
    )

    layer_types = [rl.attention_type for rl in dit.layers]
    ct = dit.proj_out[1]
    inp, outp, k = ct.weight.shape
    proj_out_w = ct.weight.permute(2, 1, 0).reshape(k * outp, inp).transpose(0, 1).contiguous()
    conv = dit.proj_in[1]
    oc, ic, kk = conv.weight.shape
    proj_in_w = conv.weight.reshape(oc, ic * kk).transpose(0, 1).contiguous()

    model = AceStepDiTModel(
        AceStepDiTModelConfig(
            time_embed=_te_cfg(dit.time_embed, device),
            time_embed_r=_te_cfg(dit.time_embed_r, device),
            patch_embed=PatchEmbedConfig(
                weight=make_lazy_weight(proj_in_w, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
                bias=_bias(conv.bias, device),
                in_channels=IN_CHANNELS,
                out_channels=HIDDEN_SIZE,
                patch_size=PATCH,
            ),
            condition_embedder_weight=_wT(dit.condition_embedder.weight, device),
            condition_embedder_bias=_bias(dit.condition_embedder.bias, device),
            stack=AceStepDiTStackConfig(
                layer_configs=[_layer_cfg(rl, at, device) for rl, at in zip(dit.layers, layer_types)],
                layer_types=layer_types,
            ),
            output=DiTOutputConfig(
                scale_shift_table=make_lazy_weight(
                    dit.scale_shift_table.detach().clone(), device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                ),
                norm_out_weight=_norm(dit.norm_out.weight, device),
                proj_out_weight=make_lazy_weight(proj_out_w, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
                proj_out_bias=_bias(ct.bias, device),
                dim=HIDDEN_SIZE,
                out_channels=OUT_CHANNELS,
                patch_size=PATCH,
                eps=RMS_NORM_EPS,
            ),
            dim=HIDDEN_SIZE,
        )
    )

    hidden_tt = to_ttnn_tensor(hidden.reshape(1, 1, SEQ_LEN, HIDDEN_CH), device)
    context_tt = to_ttnn_tensor(context.reshape(1, 1, SEQ_LEN, IN_CHANNELS - HIDDEN_CH), device)
    encoder_tt = to_ttnn_tensor(encoder.reshape(1, 1, ENC_LEN, HIDDEN_SIZE), device)
    cos_tt = ttnn.from_torch(cos.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sin_tt = ttnn.from_torch(sin.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sliding_tt = (
        None
        if sliding_mask is None
        else ttnn.from_torch(sliding_mask, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    )

    out_tt = model.forward(hidden_tt, context_tt, t, t_r, cos_tt, sin_tt, encoder_tt, sliding_mask=sliding_tt)
    out = to_torch(out_tt, expected_shape=(1, 1, SEQ_LEN, OUT_CHANNELS)).reshape(1, SEQ_LEN, OUT_CHANNELS)

    # Real 24-layer model e2e. Threshold 0.95 (the user's e2e requirement).
    assert_pcc(ref_out, out, 0.95)
