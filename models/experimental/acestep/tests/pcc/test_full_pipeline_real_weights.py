# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: FULL real-weight pipeline e2e — ConditionEncoder -> full 24-layer DiT.

The ultimate correctness signal: genuine trained weights (model.safetensors) through BOTH
subsystems chained end-to-end, at the model's real depth (all 24 DiT layers):

    ctx = ConditionEncoder(text, lyric, timbre)     # real encoder.* weights
    out = AceStepDiTModel(..., encoder_hidden_states=ctx)   # real decoder.* weights, 24 layers

Both the reference chain and the TT chain use the SAME genuine checkpoint weights, so any
divergence is our implementation across the entire generation compute graph. Requires
model.safetensors; skips if absent. Marked slow (heaviest test in the suite).
"""

import pytest
import torch

import ttnn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from models.experimental.acestep.reference.hf_reference import load_config, load_modeling_module
from models.experimental.acestep.reference.weight_utils import checkpoint_path, load_module_weights
from models.experimental.acestep.tt.condition_encoder import (
    AceStepConditionEncoder,
    AceStepConditionEncoderConfig,
)
from models.experimental.acestep.tt.dit_model import AceStepDiTModel, AceStepDiTModelConfig
from models.experimental.acestep.tt.dit_output import DiTOutputConfig
from models.experimental.acestep.tt.dit_stack import AceStepDiTStackConfig
from models.experimental.acestep.tt.patch_embed import PatchEmbedConfig
from models.experimental.acestep.tests.pcc.test_integration_seam import (
    DIT_SEQ,
    HIDDEN_CH,
    IN_CHANNELS,
    LYRIC_LEN,
    OUT_CHANNELS,
    PATCH,
    TEXT_LEN,
    TIMBRE_HIDDEN_DIM,
    TIMBRE_LEN,
    _bias,
    _dit_layer_cfg,
    _encode_core,
    _lyric_cfg,
    _mask_tt,
    _norm,
    _te_cfg,
    _wT,
)
from models.experimental.acestep.tests.test_utils import (
    HEAD_DIM,
    HIDDEN_SIZE,
    RMS_NORM_EPS,
    SLIDING_WINDOW,
    TEXT_HIDDEN_DIM,
    assert_pcc,
    make_lazy_weight,
    require_single_device,
    to_torch,
    to_ttnn_tensor,
)


def _have_checkpoint():
    try:
        checkpoint_path()
        return True
    except AssertionError:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _have_checkpoint(), reason="model.safetensors not downloaded")
def test_full_pipeline_real_weights(device):
    require_single_device(device)
    torch.manual_seed(0)

    m = load_modeling_module()
    cfg = load_config()
    cfg._attn_implementation = "eager"
    # FULL depth: all 24 DiT layers (default num_hidden_layers=24).

    ce = m.AceStepConditionEncoder(cfg).eval()
    load_module_weights(ce, "encoder.")
    dit = m.AceStepDiTModel(cfg).eval()
    load_module_weights(dit, "decoder.")

    text = torch.randn(1, TEXT_LEN, TEXT_HIDDEN_DIM)
    lyric = torch.randn(1, LYRIC_LEN, TEXT_HIDDEN_DIM)
    timbre = torch.randn(1, TIMBRE_LEN, TIMBRE_HIDDEN_DIM)
    hidden = torch.randn(1, DIT_SEQ, HIDDEN_CH)
    context = torch.randn(1, DIT_SEQ, IN_CHANNELS - HIDDEN_CH)
    t = torch.rand(1)
    t_r = torch.rand(1)

    # Reference chain: real ConditionEncoder pieces -> real full-24L DiT.
    with torch.no_grad():
        text_ref = text @ ce.text_projector.weight.t()
        lyric_ref = _encode_core(ce.lyric_encoder, m, lyric, LYRIC_LEN)
        timbre_ref = _encode_core(ce.timbre_encoder, m, timbre, TIMBRE_LEN)
        ctx_ref = torch.cat([lyric_ref, timbre_ref, text_ref], dim=1)
        (ref_out, *_) = dit(
            hidden_states=hidden,
            timestep=t,
            timestep_r=t_r,
            attention_mask=None,
            encoder_hidden_states=ctx_ref,
            encoder_attention_mask=None,
            context_latents=context,
        )

    # TT chain: ConditionEncoder -> DiT (both real weights).
    tt_ce = AceStepConditionEncoder(
        AceStepConditionEncoderConfig(
            text_projector_weight=_wT(ce.text_projector.weight, device),
            lyric_encoder=_lyric_cfg(ce.lyric_encoder, device),
            timbre_encoder=_lyric_cfg(ce.timbre_encoder, device),
        )
    )

    ct = dit.proj_out[1]
    inp, outp, k = ct.weight.shape
    proj_out_w = ct.weight.permute(2, 1, 0).reshape(k * outp, inp).transpose(0, 1).contiguous()
    conv = dit.proj_in[1]
    oc, ic, kk = conv.weight.shape
    proj_in_w = conv.weight.reshape(oc, ic * kk).transpose(0, 1).contiguous()
    dit_layer_types = [rl.attention_type for rl in dit.layers]

    tt_dit = AceStepDiTModel(
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
                layer_configs=[_dit_layer_cfg(rl, at, device) for rl, at in zip(dit.layers, dit_layer_types)],
                layer_types=dit_layer_types,
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

    rope = Qwen3RotaryEmbedding(cfg)
    lcos, lsin = rope(torch.zeros(1, LYRIC_LEN, HEAD_DIM), torch.arange(LYRIC_LEN).unsqueeze(0))
    tcos, tsin = rope(torch.zeros(1, TIMBRE_LEN, HEAD_DIM), torch.arange(TIMBRE_LEN).unsqueeze(0))
    tprime = DIT_SEQ // PATCH
    dcos, dsin = rope(torch.zeros(1, tprime, HEAD_DIM), torch.arange(tprime).unsqueeze(0))
    dit_sliding = _mask_tt(m, tprime, hidden.device, device) if tprime > SLIDING_WINDOW else None

    def tf(x, s, c):
        return to_ttnn_tensor(x.reshape(1, 1, s, c), device)

    def rope_tt(cos, sin):
        return (
            ttnn.from_torch(cos.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            ttnn.from_torch(sin.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        )

    lcos_tt, lsin_tt = rope_tt(lcos, lsin)
    tcos_tt, tsin_tt = rope_tt(tcos, tsin)
    dcos_tt, dsin_tt = rope_tt(dcos, dsin)

    ctx_tt = tt_ce.forward(
        tf(text, TEXT_LEN, TEXT_HIDDEN_DIM),
        tf(lyric, LYRIC_LEN, TEXT_HIDDEN_DIM),
        tf(timbre, TIMBRE_LEN, TIMBRE_HIDDEN_DIM),
        lcos_tt,
        lsin_tt,
        tcos_tt,
        tsin_tt,
        lyric_sliding=_mask_tt(m, LYRIC_LEN, lyric.device, device),
        timbre_sliding=_mask_tt(m, TIMBRE_LEN, timbre.device, device),
    )

    out_tt = tt_dit.forward(
        tf(hidden, DIT_SEQ, HIDDEN_CH),
        tf(context, DIT_SEQ, IN_CHANNELS - HIDDEN_CH),
        t,
        t_r,
        dcos_tt,
        dsin_tt,
        ctx_tt,
        sliding_mask=dit_sliding,
    )
    out = to_torch(out_tt, expected_shape=(1, 1, DIT_SEQ, OUT_CHANNELS)).reshape(1, DIT_SEQ, OUT_CHANNELS)

    # Full real-weight pipeline e2e (24 layers). User's e2e requirement: >= 0.95.
    assert_pcc(ref_out, out, 0.95)
