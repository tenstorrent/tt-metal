# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: full ACE-Step v1.5 DiT model (integration) vs genuine HF AceStepDiTModel.

End-to-end single denoise step: dual timestep -> concat context -> proj_in -> condition
embedder -> DiT layer stack -> norm_out AdaLN -> proj_out. Uses num_hidden_layers=2 to bound
runtime while exercising the entire wiring (both attention types, cross-attention, dual
timestep, patch/de-patch). Threshold 0.95 (deepest composition in the suite).
"""

import torch

import ttnn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from models.experimental.acestep.reference.hf_reference import load_config, load_modeling_module
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

N_LAYERS = 2
SEQ_LEN = 128  # divisible by patch_size=2
ENC_LEN = 96
IN_CHANNELS = 192
OUT_CHANNELS = 64
PATCH = 2
TIME_IN = 256
HIDDEN_CH = 64  # hidden_states channels; context_latents = 192 - 64 = 128


def _wT(w, device):
    return make_lazy_weight(
        w.detach().clone().transpose(-1, -2).contiguous(), device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )


def _norm(w, device):
    return make_lazy_weight(w.detach().clone(), device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)


def _bias(b, device):
    return make_lazy_weight(
        b.detach().clone().reshape(1, -1).contiguous(), device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )


def _te_cfg(ref_te, device):
    return TimestepEmbeddingConfig(
        linear_1_weight=_wT(ref_te.linear_1.weight, device),
        linear_1_bias=_bias(ref_te.linear_1.bias, device),
        linear_2_weight=_wT(ref_te.linear_2.weight, device),
        linear_2_bias=_bias(ref_te.linear_2.bias, device),
        time_proj_weight=_wT(ref_te.time_proj.weight, device),
        time_proj_bias=_bias(ref_te.time_proj.bias, device),
        in_channels=TIME_IN,
        time_embed_dim=HIDDEN_SIZE,
    )


def _layer_cfg(rl, at, device):
    a, c = rl.self_attn, rl.cross_attn
    return AceStepDiTLayerConfig(
        scale_shift_table=make_lazy_weight(
            rl.scale_shift_table.detach().clone(), device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        ),
        self_attn_norm_weight=_norm(rl.self_attn_norm.weight, device),
        mlp_norm_weight=_norm(rl.mlp_norm.weight, device),
        cross_attn_norm_weight=_norm(rl.cross_attn_norm.weight, device),
        wq=_wT(a.q_proj.weight, device),
        wk=_wT(a.k_proj.weight, device),
        wv=_wT(a.v_proj.weight, device),
        wo=_wT(a.o_proj.weight, device),
        q_norm_weight=_norm(a.q_norm.weight, device),
        k_norm_weight=_norm(a.k_norm.weight, device),
        c_wq=_wT(c.q_proj.weight, device),
        c_wk=_wT(c.k_proj.weight, device),
        c_wv=_wT(c.v_proj.weight, device),
        c_wo=_wT(c.o_proj.weight, device),
        c_q_norm_weight=_norm(c.q_norm.weight, device),
        c_k_norm_weight=_norm(c.k_norm.weight, device),
        w1=_wT(rl.mlp.gate_proj.weight, device),
        w2=_wT(rl.mlp.down_proj.weight, device),
        w3=_wT(rl.mlp.up_proj.weight, device),
        n_heads=NUM_ATTENTION_HEADS,
        n_kv_heads=NUM_KEY_VALUE_HEADS,
        head_dim=HEAD_DIM,
        dim=HIDDEN_SIZE,
        eps=RMS_NORM_EPS,
        sliding_window=SLIDING_WINDOW if at == "sliding_attention" else None,
        use_cross_attention=True,
    )


def _init(dit):
    with torch.no_grad():
        for te in (dit.time_embed, dit.time_embed_r):
            for lin in (te.linear_1, te.linear_2, te.time_proj):
                lin.weight.copy_(0.02 * torch.randn_like(lin.weight))
                lin.bias.copy_(0.01 * torch.randn_like(lin.bias))
        conv = dit.proj_in[1]
        conv.weight.copy_(0.02 * torch.randn_like(conv.weight))
        conv.bias.copy_(0.01 * torch.randn_like(conv.bias))
        dit.condition_embedder.weight.copy_(0.02 * torch.randn_like(dit.condition_embedder.weight))
        dit.condition_embedder.bias.copy_(0.01 * torch.randn_like(dit.condition_embedder.bias))
        dit.norm_out.weight.copy_(1.0 + 0.02 * torch.randn_like(dit.norm_out.weight))
        dit.scale_shift_table.copy_(0.05 * torch.randn_like(dit.scale_shift_table))
        ct = dit.proj_out[1]
        ct.weight.copy_(0.02 * torch.randn_like(ct.weight))
        ct.bias.copy_(0.01 * torch.randn_like(ct.bias))
        for rl in dit.layers:
            for a in (rl.self_attn, rl.cross_attn):
                for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
                    lin.weight.copy_(0.02 * torch.randn_like(lin.weight))
                a.q_norm.weight.copy_(1.0 + 0.02 * torch.randn_like(a.q_norm.weight))
                a.k_norm.weight.copy_(1.0 + 0.02 * torch.randn_like(a.k_norm.weight))
            for lin in (rl.mlp.gate_proj, rl.mlp.up_proj, rl.mlp.down_proj):
                lin.weight.copy_(0.02 * torch.randn_like(lin.weight))
            for nm in (rl.self_attn_norm, rl.cross_attn_norm, rl.mlp_norm):
                nm.weight.copy_(1.0 + 0.02 * torch.randn_like(nm.weight))
            rl.scale_shift_table.copy_(0.05 * torch.randn_like(rl.scale_shift_table))


def test_dit_model_vs_hf(device):
    require_single_device(device)
    torch.manual_seed(0)

    m = load_modeling_module()
    cfg = load_config()
    cfg._attn_implementation = "eager"
    cfg.num_hidden_layers = N_LAYERS
    dit = m.AceStepDiTModel(cfg).eval()
    _init(dit)

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
        )  # [1, SEQ_LEN, out_channels]

    # RoPE over patched sequence (T/p positions).
    tprime = SEQ_LEN // PATCH
    rope = Qwen3RotaryEmbedding(cfg)
    position_ids = torch.arange(tprime).unsqueeze(0)
    cos, sin = rope(torch.zeros(1, tprime, HEAD_DIM), position_ids)

    sliding_mask = m.create_4d_mask(
        seq_len=tprime,
        dtype=torch.float32,
        device=hidden.device,
        attention_mask=None,
        sliding_window=SLIDING_WINDOW,
        is_sliding_window=True,
        is_causal=False,
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
    sliding_tt = ttnn.from_torch(sliding_mask, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    out_tt = model.forward(hidden_tt, context_tt, t, t_r, cos_tt, sin_tt, encoder_tt, sliding_mask=sliding_tt)
    out = to_torch(out_tt, expected_shape=(1, 1, SEQ_LEN, OUT_CHANNELS)).reshape(1, SEQ_LEN, OUT_CHANNELS)

    assert_pcc(ref_out, out, 0.95)
