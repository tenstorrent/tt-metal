# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Consolidated PCC sanity test for all 24 TTNN blocks against the real HF
SeamlessM4T-v2-Large checkpoint.

For each block:

1. Loads the real HF weights via ``tt.weight_loader.load_hf_state_dict()``
   (once per pytest session, cached at module scope).
2. Extracts the per-block nested state-dict via the matching
   ``weight_loader.<block>_weights(...)`` helper.
3. Computes a "golden" reference output via the bit-identical
   ``reference.functional.<block>_forward(...)`` -- the standalone reference
   is what was used to verify the HF-class equivalence in
   ``reference/test_functional_*.py``, so this is equivalent to comparing
   against HuggingFace directly.
4. Runs the TTNN block with the same real weights + same input.
5. Asserts PCC > 0.99.

Sub-models (text_encoder, text_decoder, speech_encoder, etc.) and the
top-level T2TT model use **reduced layer counts (2 layers)** to keep this
phase-1 test under ~5 minutes; the per-layer composites are independently
covered. Phase 2 will scale these up to full config.

Run with:

    pytest models/demos/facebook_seamless_m4t_v2_large/tests/test_real_hf_weights.py -v
"""

from __future__ import annotations

import math
import re

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.facebook_seamless_m4t_v2_large.reference import functional as ref
from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl

# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def hf_sd():
    """Load the full HF checkpoint once per session (~2.3 GB on disk)."""
    return wl.load_hf_state_dict()


@pytest.fixture(scope="function")
def device():
    """Open a fresh ttnn device per test.

    The TTNN block instances allocate weights into device DRAM; with a
    session-scoped device those allocations accumulate across tests and the
    full-config blocks (especially ``code_hifigan_vocoder`` and the
    sub-models) run out of memory near the end of the run. Per-test device
    open/close costs ~1 second each but provides clean isolation.
    """
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pcc_value(ref_t: torch.Tensor, tt_t: torch.Tensor) -> float:
    """Run ``comp_pcc`` and parse the returned PCC value as a float."""
    _, msg = comp_pcc(ref_t, tt_t, 0.99)
    s = str(msg).strip()
    try:
        return float(s)
    except ValueError:
        m = re.search(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", s)
        return float(m.group(0)) if m else float("nan")


def _to_tt(device, t: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    return ttnn.from_torch(
        t,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# ---------------------------------------------------------------------------
# Per-block test bodies. Each returns the PCC float.
# ---------------------------------------------------------------------------


def _t_layernorm(device, hf_sd) -> float:
    from models.demos.facebook_seamless_m4t_v2_large.tt.layernorm import LayerNorm

    # Use a representative encoder LN: ``text_encoder.layer_norm``.
    sd = wl._ln_sd(hf_sd, "text_encoder.layer_norm")
    weight, bias = sd["weight"], sd["bias"]
    dim = weight.shape[-1]
    eps = 1e-5

    torch.manual_seed(123)
    x = torch.randn(1, 128, dim, dtype=torch.float32)

    ref_out = ref.layernorm_forward(x, weight, bias, eps=eps)

    tt_block = LayerNorm(device=device, dim=dim, weight=weight, bias=bias, eps=eps, weight_dtype=ttnn.bfloat16)
    tt_in = _to_tt(device, x)
    tt_out = ttnn.to_torch(tt_block(tt_in)).to(torch.float32).reshape(ref_out.shape)
    return _pcc_value(ref_out, tt_out)


def _t_scaled_word_embedding(device, hf_sd) -> float:
    from models.demos.facebook_seamless_m4t_v2_large.tt.scaled_word_embedding import ScaledWordEmbedding

    weight = wl.shared_embedding_weight(hf_sd)
    vocab, hidden = weight.shape
    scale = math.sqrt(hidden)  # 32.0
    padding_idx = 0

    torch.manual_seed(123)
    # Keep ids small; vocab is 256102.
    input_ids = torch.randint(low=2, high=512, size=(1, 16), dtype=torch.long)

    ref_out = ref.scaled_word_embedding_forward(input_ids, weight, scale=scale, padding_idx=padding_idx)

    tt_block = ScaledWordEmbedding(
        device=device, weight=weight, scale=scale, padding_idx=padding_idx, weight_dtype=ttnn.bfloat16
    )
    tt_in = ttnn.from_torch(
        input_ids.to(torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_out = ttnn.to_torch(tt_block(tt_in)).to(torch.float32).reshape(ref_out.shape)
    return _pcc_value(ref_out, tt_out)


def _t_sinusoidal_positional_embedding(device, hf_sd) -> float:
    from models.demos.facebook_seamless_m4t_v2_large.tt.sinusoidal_positional_embedding import (
        SinusoidalPositionalEmbedding,
    )

    hidden = 1024
    padding_idx = 1
    weights = wl.build_sinusoidal_positional_embedding_weights(
        num_embeddings=4096 + 2, embedding_dim=hidden, padding_idx=padding_idx
    )

    torch.manual_seed(123)
    input_ids = torch.randint(low=2, high=1000, size=(1, 128), dtype=torch.long)

    ref_out = ref.sinusoidal_positional_embedding_forward(
        weights, input_ids=input_ids, padding_idx=padding_idx, past_key_values_length=0
    )

    tt_block = SinusoidalPositionalEmbedding(
        device=device, weights=weights, padding_idx=padding_idx, weight_dtype=ttnn.bfloat16
    )
    tt_out = ttnn.to_torch(tt_block(input_ids=input_ids, past_key_values_length=0)).to(torch.float32)
    tt_out = tt_out.reshape(ref_out.shape)
    return _pcc_value(ref_out, tt_out)


def _build_realistic_post_ln_input(hf_sd, ln_prefix, batch=1, seq_len=64, embed_dim=1024, padding_idx=0):
    """Build a post-LN realistic input by going through the embed + position +
    LayerNorm chain a real layer would see. Random N(0,1) input is out-of-
    distribution for trained weights and produces ~1000x amplified outputs that
    swamp bf16 precision; this helper produces post-LN-scale activations that
    match what the layer actually sees in production."""
    ln_sd = wl._ln_sd(hf_sd, ln_prefix)
    shared = wl.shared_embedding_weight(hf_sd)
    input_ids = torch.randint(low=2, high=512, size=(batch, seq_len), dtype=torch.long)
    pre_ln = F.embedding(input_ids, shared) * math.sqrt(embed_dim)
    sin_w = wl.build_sinusoidal_positional_embedding_weights(
        num_embeddings=4096 + 2, embedding_dim=embed_dim, padding_idx=padding_idx
    )
    pre_ln = pre_ln + ref.sinusoidal_positional_embedding_forward(sin_w, input_ids=input_ids, padding_idx=padding_idx)
    return F.layer_norm(pre_ln, (embed_dim,), weight=ln_sd["weight"], bias=ln_sd["bias"])


def _t_seamless_mha(device, hf_sd) -> float:
    from models.demos.facebook_seamless_m4t_v2_large.tt.seamless_mha import SeamlessMHA

    # Use text encoder layer 0's self_attn (BART-style q/k/v/out_proj with bias).
    sd = wl._bart_attn_sd(hf_sd, "text_encoder.layers.0.self_attn")
    embed_dim = sd["q_proj"]["weight"].shape[-1]  # 1024
    num_heads, head_dim = 16, 64

    # Use realistic post-LN-scale input (random N(0,1) input is OOD for
    # trained weights and saturates bf16 precision; see helper docstring).
    torch.manual_seed(123)
    x = _build_realistic_post_ln_input(
        hf_sd, ln_prefix="text_encoder.layers.0.self_attn_layer_norm", batch=1, seq_len=64, embed_dim=embed_dim
    )

    ref_out = ref.seamless_mha_forward(
        x, sd, num_heads=num_heads, head_dim=head_dim, encoder_hidden_states=None, attention_mask=None
    )

    tt_block = SeamlessMHA(
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        state_dict=sd,
        weight_dtype=ttnn.bfloat16,
    )
    tt_in = _to_tt(device, x)
    tt_out = ttnn.to_torch(tt_block(tt_in, encoder_hidden_states=None, attention_mask=None)).to(torch.float32)
    return _pcc_value(ref_out, tt_out.reshape(ref_out.shape))


def _t_seamless_ffn(device, hf_sd) -> float:
    from models.demos.facebook_seamless_m4t_v2_large.tt.seamless_ffn import SeamlessFfn

    sd = wl._nllb_ffn_sd(hf_sd, "text_encoder.layers.0.ffn")
    fc1_w, fc1_b = sd["fc1"]["weight"], sd["fc1"]["bias"]
    fc2_w, fc2_b = sd["fc2"]["weight"], sd["fc2"]["bias"]
    hidden = fc1_w.shape[-1]

    torch.manual_seed(123)
    x = torch.randn(1, 64, hidden, dtype=torch.float32)

    ref_out = ref.seamless_ffn_forward(x, fc1_w, fc1_b, fc2_w, fc2_b)

    tt_block = SeamlessFfn(
        device=device,
        fc1_weight=fc1_w,
        fc1_bias=fc1_b,
        fc2_weight=fc2_w,
        fc2_bias=fc2_b,
        weight_dtype=ttnn.bfloat16,
    )
    tt_in = _to_tt(device, x)
    tt_out = ttnn.to_torch(tt_block(tt_in)).to(torch.float32).reshape(ref_out.shape)
    return _pcc_value(ref_out, tt_out)


def _t_conformer_ffn(device, hf_sd) -> float:
    from models.demos.facebook_seamless_m4t_v2_large.tt.conformer_ffn import ConformerFfn

    sd = wl._conformer_ffn_sd(hf_sd, "speech_encoder.encoder.layers.0.ffn1")
    iw, ib = sd["intermediate_dense"]["weight"], sd["intermediate_dense"]["bias"]
    ow, ob = sd["output_dense"]["weight"], sd["output_dense"]["bias"]
    hidden = iw.shape[-1]

    torch.manual_seed(123)
    x = torch.randn(1, 128, hidden, dtype=torch.float32)

    ref_out = ref.conformer_ffn_forward(x, sd, act_fn="swish")

    tt_block = ConformerFfn(
        device=device,
        intermediate_weight=iw,
        intermediate_bias=ib,
        output_weight=ow,
        output_bias=ob,
        weight_dtype=ttnn.bfloat16,
    )
    tt_in = _to_tt(device, x)
    tt_out = ttnn.to_torch(tt_block(tt_in)).to(torch.float32).reshape(ref_out.shape)
    return _pcc_value(ref_out, tt_out)


def _t_conformer_self_attention(device, hf_sd) -> float:
    from models.demos.facebook_seamless_m4t_v2_large.tt.conformer_self_attention import ConformerSelfAttention

    sd = wl._conformer_attn_sd(hf_sd, "speech_encoder.encoder.layers.0.self_attn")
    distance_embedding_weight = hf_sd["speech_encoder.encoder.layers.0.self_attn.distance_embedding.weight"]
    embed_dim = sd["linear_q"]["weight"].shape[-1]
    num_heads, head_dim = 16, 64
    left_max, right_max = 64, 8

    torch.manual_seed(123)
    seq_len = 128
    x = torch.randn(1, seq_len, embed_dim, dtype=torch.float32)

    ref_out = ref.conformer_self_attention_forward(
        x,
        sd,
        num_heads=num_heads,
        head_dim=head_dim,
        distance_embedding_weight=distance_embedding_weight,
        left_max_position_embeddings=left_max,
        right_max_position_embeddings=right_max,
        position_embeddings_type="relative_key",
        attention_mask=None,
    )

    tt_block = ConformerSelfAttention(
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        state_dict=sd,
        distance_embedding_weight=distance_embedding_weight,
        left_max_position_embeddings=left_max,
        right_max_position_embeddings=right_max,
        position_embeddings_type="relative_key",
        batch_size=1,
        weight_dtype=ttnn.bfloat16,
    )
    tt_in = _to_tt(device, x)
    tt_out = ttnn.to_torch(tt_block(tt_in, attention_mask=None)).to(torch.float32).reshape(ref_out.shape)
    return _pcc_value(ref_out, tt_out)


def _t_conformer_convolution_module(device, hf_sd) -> float:
    from models.demos.facebook_seamless_m4t_v2_large.tt.conformer_convolution_module import ConformerConvolutionModule

    # Lifted from conformer_encoder_layer_weights' conv_module sub-dict.
    p = "speech_encoder.encoder.layers.0.conv_module"
    sd = {
        "layer_norm": wl._ln_sd(hf_sd, f"{p}.layer_norm"),
        "pointwise_conv1": wl._conv_sd(hf_sd, f"{p}.pointwise_conv1", with_bias=False),
        "depthwise_conv": wl._conv_sd(hf_sd, f"{p}.depthwise_conv", with_bias=False),
        "depthwise_layer_norm": wl._ln_sd(hf_sd, f"{p}.depthwise_layer_norm"),
        "pointwise_conv2": wl._conv_sd(hf_sd, f"{p}.pointwise_conv2", with_bias=False),
    }
    hidden = sd["layer_norm"]["weight"].shape[-1]
    kernel_size = 31
    eps = 1e-5

    torch.manual_seed(123)
    x = torch.randn(1, 128, hidden, dtype=torch.float32)

    ref_out = ref.conformer_convolution_module_forward(x, sd, kernel_size=kernel_size, eps=eps, attention_mask=None)

    tt_block = ConformerConvolutionModule(
        device=device,
        layer_norm_weight=sd["layer_norm"]["weight"],
        layer_norm_bias=sd["layer_norm"]["bias"],
        pointwise_conv1_weight=sd["pointwise_conv1"]["weight"],
        depthwise_conv_weight=sd["depthwise_conv"]["weight"],
        depthwise_layer_norm_weight=sd["depthwise_layer_norm"]["weight"],
        depthwise_layer_norm_bias=sd["depthwise_layer_norm"]["bias"],
        pointwise_conv2_weight=sd["pointwise_conv2"]["weight"],
        kernel_size=kernel_size,
        eps=eps,
        weight_dtype=ttnn.bfloat16,
    )
    tt_in = _to_tt(device, x)
    tt_out = ttnn.to_torch(tt_block(tt_in, attention_mask=None)).to(torch.float32).reshape(ref_out.shape)
    return _pcc_value(ref_out, tt_out)


def _t_variance_predictor(device, hf_sd) -> float:
    from models.demos.facebook_seamless_m4t_v2_large.tt.variance_predictor import VariancePredictor

    # T2U duration predictor (matches `_variance_predictor_sd`).
    sd = wl._variance_predictor_sd(hf_sd, "t2u_model.model.decoder.duration_predictor")
    embed_dim = sd["conv1"]["weight"].shape[1]  # 1024
    hidden_dim = sd["conv1"]["weight"].shape[0]  # 256
    kernel_size = 3
    eps = 1e-5

    torch.manual_seed(123)
    x = torch.randn(1, 64, embed_dim, dtype=torch.float32)

    ref_out = ref.variance_predictor_forward(x, sd, kernel_size=kernel_size, eps=eps, padding_mask=None)

    tt_block = VariancePredictor(
        device=device,
        conv1_weight=sd["conv1"]["weight"],
        conv1_bias=sd["conv1"]["bias"],
        ln1_weight=sd["ln1"]["weight"],
        ln1_bias=sd["ln1"]["bias"],
        conv2_weight=sd["conv2"]["weight"],
        conv2_bias=sd["conv2"]["bias"],
        ln2_weight=sd["ln2"]["weight"],
        ln2_bias=sd["ln2"]["bias"],
        proj_weight=sd["proj"]["weight"],
        proj_bias=sd["proj"]["bias"],
        kernel_size=kernel_size,
        eps=eps,
        weight_dtype=ttnn.bfloat16,
    )
    tt_in = _to_tt(device, x)
    tt_out = ttnn.to_torch(tt_block(tt_in, padding_mask=None)).to(torch.float32).reshape(ref_out.shape)
    return _pcc_value(ref_out, tt_out)


def _t_hifigan_residual_block(device, hf_sd) -> float:
    from models.demos.facebook_seamless_m4t_v2_large.tt.hifigan_residual_block import HifiGanResidualBlock

    sd = wl.hifigan_residual_block_weights(hf_sd, block_idx=0, num_inner_convs=3)
    # convs1[0] gives the channel count.
    channels = sd["convs1"][0]["weight"].shape[0]
    kernel_size = 3
    dilation = (1, 3, 5)
    leaky_relu_slope = 0.1

    torch.manual_seed(123)
    x = torch.randn(1, channels, 80, dtype=torch.float32)

    ref_out = ref.hifigan_residual_block_forward(
        x, sd, kernel_size=kernel_size, dilation=dilation, leaky_relu_slope=leaky_relu_slope
    )

    convs1_weights = [c["weight"] for c in sd["convs1"]]
    convs1_biases = [c["bias"] for c in sd["convs1"]]
    convs2_weights = [c["weight"] for c in sd["convs2"]]
    convs2_biases = [c["bias"] for c in sd["convs2"]]

    tt_block = HifiGanResidualBlock(
        device=device,
        convs1_weights=convs1_weights,
        convs1_biases=convs1_biases,
        convs2_weights=convs2_weights,
        convs2_biases=convs2_biases,
        kernel_size=kernel_size,
        dilation=dilation,
        leaky_relu_slope=leaky_relu_slope,
        weight_dtype=ttnn.bfloat16,
    )
    tt_in = _to_tt(device, x)
    tt_out = ttnn.to_torch(tt_block(tt_in)).to(torch.float32).reshape(ref_out.shape)
    return _pcc_value(ref_out, tt_out)


def _t_conformer_feature_projection(device, hf_sd) -> float:
    from models.demos.facebook_seamless_m4t_v2_large.tt.conformer_feature_projection import ConformerFeatureProjection

    sd = {
        "layer_norm": wl._ln_sd(hf_sd, "speech_encoder.feature_projection.layer_norm"),
        "projection": wl._linear_sd(hf_sd, "speech_encoder.feature_projection.projection"),
    }
    feature_size = sd["layer_norm"]["weight"].shape[-1]  # 160
    eps = 1e-5

    torch.manual_seed(123)
    x = torch.randn(1, 64, feature_size, dtype=torch.float32)

    ref_out = ref.conformer_feature_projection_forward(x, sd, eps=eps)

    tt_block = ConformerFeatureProjection(
        device=device,
        layer_norm_weight=sd["layer_norm"]["weight"],
        layer_norm_bias=sd["layer_norm"]["bias"],
        projection_weight=sd["projection"]["weight"],
        projection_bias=sd["projection"]["bias"],
        eps=eps,
        weight_dtype=ttnn.bfloat16,
    )
    tt_in = _to_tt(device, x)
    tt_out = ttnn.to_torch(tt_block(tt_in)).to(torch.float32).reshape(ref_out.shape)
    return _pcc_value(ref_out, tt_out)


def _t_conformer_encoder_layer(device, hf_sd) -> float:
    from models.demos.facebook_seamless_m4t_v2_large.tt.conformer_encoder_layer import ConformerEncoderLayer

    sd_with_dist = wl.conformer_encoder_layer_weights(hf_sd, layer_idx=0)
    distance_embedding_weight = sd_with_dist["distance_embedding_weight"]
    # Pop the sibling so the inner state_dict matches the reference functional shape.
    sd = {k: v for k, v in sd_with_dist.items() if k != "distance_embedding_weight"}

    embed_dim = sd["ffn1_layer_norm"]["weight"].shape[-1]
    num_heads, head_dim = 16, 64
    left_max, right_max = 64, 8
    conv_kernel_size = 31
    eps = 1e-5
    seq_len = 64

    torch.manual_seed(123)
    x = torch.randn(1, seq_len, embed_dim, dtype=torch.float32)

    ref_out = ref.conformer_encoder_layer_forward(
        x,
        sd,
        num_heads=num_heads,
        head_dim=head_dim,
        distance_embedding_weight=distance_embedding_weight,
        left_max_position_embeddings=left_max,
        right_max_position_embeddings=right_max,
        position_embeddings_type="relative_key",
        conv_kernel_size=conv_kernel_size,
        act_fn="swish",
        eps=eps,
        attention_mask=None,
        conv_attention_mask=None,
    )

    tt_block = ConformerEncoderLayer(
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        state_dict=sd,
        distance_embedding_weight=distance_embedding_weight,
        left_max_position_embeddings=left_max,
        right_max_position_embeddings=right_max,
        position_embeddings_type="relative_key",
        conv_kernel_size=conv_kernel_size,
        eps=eps,
        batch_size=1,
        weight_dtype=ttnn.bfloat16,
    )
    tt_in = _to_tt(device, x)
    tt_out = ttnn.to_torch(tt_block(tt_in, attention_mask=None, conv_attention_mask=None)).to(torch.float32)
    return _pcc_value(ref_out, tt_out.reshape(ref_out.shape))


def _t_text_encoder_layer(device, hf_sd) -> float:
    from models.demos.facebook_seamless_m4t_v2_large.tt.text_encoder_layer import TextEncoderLayer

    sd = wl.text_encoder_layer_weights(hf_sd, layer_idx=0)
    embed_dim = sd["self_attn_layer_norm"]["weight"].shape[-1]
    num_heads, head_dim = 16, 64
    eps = 1e-5

    # Use realistic embed-derived input (matches what the layer sees in production).
    torch.manual_seed(123)
    shared = wl.shared_embedding_weight(hf_sd)
    input_ids = torch.randint(low=2, high=512, size=(1, 64), dtype=torch.long)
    x = F.embedding(input_ids, shared) * math.sqrt(embed_dim)
    sin_w = wl.build_sinusoidal_positional_embedding_weights(
        num_embeddings=4096 + 2, embedding_dim=embed_dim, padding_idx=0
    )
    x = x + ref.sinusoidal_positional_embedding_forward(sin_w, input_ids=input_ids, padding_idx=0)

    ref_out = ref.text_encoder_layer_forward(
        x, sd, num_heads=num_heads, head_dim=head_dim, attention_mask=None, eps=eps
    )

    tt_block = TextEncoderLayer(
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        state_dict=sd,
        eps=eps,
        weight_dtype=ttnn.bfloat16,
    )
    tt_in = _to_tt(device, x)
    tt_out = ttnn.to_torch(tt_block(tt_in, attention_mask=None)).to(torch.float32)
    return _pcc_value(ref_out, tt_out.reshape(ref_out.shape))


def _t_text_decoder_layer(device, hf_sd) -> float:
    from models.demos.facebook_seamless_m4t_v2_large.tt.text_decoder_layer import TextDecoderLayer

    sd = wl.text_decoder_layer_weights(hf_sd, layer_idx=0)
    embed_dim = sd["self_attn_layer_norm"]["weight"].shape[-1]
    num_heads, head_dim = 16, 64
    eps = 1e-5

    # Use realistic embed-derived inputs (random N(0,1) is OOD for trained weights).
    # For the encoder_hidden_states, run a small (2-layer) real text encoder
    # so the cross-attention sees in-distribution K/V values.
    torch.manual_seed(123)
    shared = wl.shared_embedding_weight(hf_sd)
    dec_ids = torch.randint(low=2, high=512, size=(1, 32), dtype=torch.long)
    enc_ids = torch.randint(low=2, high=512, size=(1, 64), dtype=torch.long)
    sin_w = wl.build_sinusoidal_positional_embedding_weights(
        num_embeddings=4096 + 2, embedding_dim=embed_dim, padding_idx=0
    )
    x = F.embedding(dec_ids, shared) * math.sqrt(embed_dim)
    x = x + ref.sinusoidal_positional_embedding_forward(sin_w, input_ids=dec_ids, padding_idx=0)
    # Encoder hidden states from the real (2-layer reduced) text encoder.
    enc_sd = wl.text_encoder_weights(hf_sd, num_layers=2)
    enc = ref.text_encoder_forward(
        enc_ids,
        enc_sd,
        num_heads=num_heads,
        head_dim=head_dim,
        attention_mask=None,
        eps=eps,
        padding_idx=0,
        embed_scale=math.sqrt(embed_dim),
    )

    ref_out = ref.text_decoder_layer_forward(
        x,
        sd,
        num_heads=num_heads,
        head_dim=head_dim,
        encoder_hidden_states=enc,
        self_attention_mask=None,
        encoder_attention_mask=None,
        eps=eps,
    )

    tt_block = TextDecoderLayer(
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        state_dict=sd,
        eps=eps,
        weight_dtype=ttnn.bfloat16,
    )
    tt_x = _to_tt(device, x)
    tt_enc = _to_tt(device, enc)
    tt_out = ttnn.to_torch(
        tt_block(tt_x, encoder_hidden_states=tt_enc, self_attention_mask=None, encoder_attention_mask=None)
    ).to(torch.float32)
    return _pcc_value(ref_out, tt_out.reshape(ref_out.shape))


def _t_t2u_decoder_layer(device, hf_sd) -> float:
    from models.demos.facebook_seamless_m4t_v2_large.tt.t2u_decoder_layer import T2UDecoderLayer

    sd = wl.t2u_decoder_layer_weights(hf_sd, layer_idx=0)
    embed_dim = sd["self_attn_layer_norm"]["weight"].shape[-1]
    num_heads, head_dim = 16, 64
    conv_kernel_size = 7
    eps = 1e-5

    torch.manual_seed(123)
    x = torch.randn(1, 64, embed_dim, dtype=torch.float32)

    ref_out = ref.t2u_decoder_layer_forward(
        x,
        sd,
        num_heads=num_heads,
        head_dim=head_dim,
        attention_mask=None,
        padding_mask=None,
        eps=eps,
        conv_kernel_size=conv_kernel_size,
    )

    tt_block = T2UDecoderLayer(
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        state_dict=sd,
        conv_kernel_size=conv_kernel_size,
        eps=eps,
        weight_dtype=ttnn.bfloat16,
    )
    tt_x = _to_tt(device, x)
    tt_out = ttnn.to_torch(tt_block(tt_x, attention_mask=None, padding_mask=None)).to(torch.float32)
    return _pcc_value(ref_out, tt_out.reshape(ref_out.shape))


def _t_conformer_adapter_layer(device, hf_sd) -> float:
    from models.demos.facebook_seamless_m4t_v2_large.tt.conformer_adapter_layer import ConformerAdapterLayer

    sd = wl.conformer_adapter_layer_weights(hf_sd, layer_idx=0)
    embed_dim = sd["residual_layer_norm"]["weight"].shape[-1]
    num_heads, head_dim = 16, 64
    kernel_size, stride = 8, 8
    eps = 1e-5
    seq_len = 128

    torch.manual_seed(123)
    x = torch.randn(1, seq_len, embed_dim, dtype=torch.float32)

    ref_out = ref.conformer_adapter_layer_forward(
        x,
        sd,
        num_heads=num_heads,
        head_dim=head_dim,
        kernel_size=kernel_size,
        stride=stride,
        eps=eps,
        attention_mask=None,
    )

    sub_seq_len = ref_out.shape[1]
    tt_block = ConformerAdapterLayer(
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        sub_seq_len=sub_seq_len,
        state_dict=sd,
        kernel_size=kernel_size,
        stride=stride,
        eps=eps,
        batch_size=1,
        weight_dtype=ttnn.bfloat16,
    )
    tt_in = _to_tt(device, x)
    tt_out = ttnn.to_torch(tt_block(tt_in, attention_mask=None)).to(torch.float32)
    return _pcc_value(ref_out, tt_out.reshape(ref_out.shape))


# ---------------------------------------------------------------------------
# Sub-models (reduced layer counts).
# ---------------------------------------------------------------------------

# To keep this test fast we override layer counts via the weight_loader's
# ``num_layers`` arguments (so we only materialize state dicts for layer 0..N-1).
SUBMODEL_LAYERS = 2


def _t_speech_encoder(device, hf_sd) -> float:
    from models.demos.facebook_seamless_m4t_v2_large.tt.speech_encoder import SpeechEncoder

    sd = wl.speech_encoder_weights(hf_sd, num_encoder_layers=SUBMODEL_LAYERS, num_adapter_layers=1)

    feature_size = 160
    hidden = 1024
    num_heads, head_dim = 16, 64
    eps = 1e-5

    torch.manual_seed(123)
    x = torch.randn(1, 64, feature_size, dtype=torch.float32)

    ref_out = ref.speech_encoder_forward(
        x,
        sd,
        num_heads=num_heads,
        head_dim=head_dim,
        attention_mask=None,
        eps=eps,
        speech_encoder_hidden_act="swish",
        left_max_position_embeddings=64,
        right_max_position_embeddings=8,
        position_embeddings_type="relative_key",
        conv_depthwise_kernel_size=31,
        adaptor_kernel_size=8,
        adaptor_stride=8,
        speech_encoder_chunk_size=20000,
        speech_encoder_left_chunk_num=128,
        add_adapter=True,
    )

    tt_block = SpeechEncoder(
        device=device,
        state_dict=sd,
        feature_size=feature_size,
        hidden=hidden,
        num_heads=num_heads,
        head_dim=head_dim,
        seq_len=64,
        batch_size=1,
        eps=eps,
        speech_encoder_hidden_act="swish",
        left_max_position_embeddings=64,
        right_max_position_embeddings=8,
        position_embeddings_type="relative_key",
        conv_depthwise_kernel_size=31,
        adaptor_kernel_size=8,
        adaptor_stride=8,
        speech_encoder_chunk_size=20000,
        speech_encoder_left_chunk_num=128,
        add_adapter=True,
        weight_dtype=ttnn.bfloat16,
    )
    tt_in = _to_tt(device, x)
    tt_out = ttnn.to_torch(tt_block(tt_in, attention_mask_2d=None)).to(torch.float32)
    return _pcc_value(ref_out, tt_out.reshape(ref_out.shape))


def _t_text_encoder(device, hf_sd) -> float:
    from models.demos.facebook_seamless_m4t_v2_large.tt.text_encoder import TextEncoder

    sd = wl.text_encoder_weights(hf_sd, num_layers=SUBMODEL_LAYERS)
    embed_dim = 1024
    num_heads, head_dim = 16, 64
    eps = 1e-5
    padding_idx = 0
    embed_scale = math.sqrt(embed_dim)

    torch.manual_seed(123)
    input_ids = torch.randint(low=2, high=512, size=(1, 16), dtype=torch.long)

    ref_out = ref.text_encoder_forward(
        input_ids,
        sd,
        num_heads=num_heads,
        head_dim=head_dim,
        attention_mask=None,
        eps=eps,
        padding_idx=padding_idx,
        embed_scale=embed_scale,
    )

    tt_block = TextEncoder(
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        embed_tokens_weight=sd["embed_tokens"]["weight"],
        embed_positions_weights=sd["embed_positions_weights"],
        layers_state_dict=sd["layers"],
        final_layer_norm_state_dict=sd["final_layer_norm"],
        eps=eps,
        padding_idx=padding_idx,
        embed_scale=embed_scale,
        weight_dtype=ttnn.bfloat16,
    )
    tt_out = ttnn.to_torch(tt_block(input_ids, attention_mask_torch=None)).to(torch.float32)
    return _pcc_value(ref_out, tt_out.reshape(ref_out.shape))


def _t_text_decoder(device, hf_sd) -> float:
    from models.demos.facebook_seamless_m4t_v2_large.tt.text_decoder import TextDecoder

    sd = wl.text_decoder_weights(hf_sd, num_layers=SUBMODEL_LAYERS)
    embed_dim = 1024
    num_heads, head_dim = 16, 64
    eps = 1e-5
    padding_idx = 0

    torch.manual_seed(123)
    input_ids = torch.randint(low=2, high=512, size=(1, 8), dtype=torch.long)
    enc = torch.randn(1, 16, embed_dim, dtype=torch.float32)

    ref_out = ref.text_decoder_forward(
        input_ids,
        sd,
        num_heads=num_heads,
        head_dim=head_dim,
        encoder_hidden_states=enc,
        attention_mask=None,
        encoder_attention_mask=None,
        eps=eps,
        padding_idx=padding_idx,
    )

    tt_block = TextDecoder(
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        embed_tokens_weight=sd["embed_tokens"]["weight"],
        embed_positions_weights=sd["embed_positions_weights"],
        layers_state_dict=sd["layers"],
        final_layer_norm_state_dict=sd["layer_norm"],
        eps=eps,
        padding_idx=padding_idx,
        weight_dtype=ttnn.bfloat16,
    )
    tt_out = ttnn.to_torch(
        tt_block(input_ids, encoder_hidden_states_torch=enc, attention_mask=None, encoder_attention_mask=None)
    ).to(torch.float32)
    return _pcc_value(ref_out, tt_out.reshape(ref_out.shape))


def _t_t2u_encoder(device, hf_sd) -> float:
    from models.demos.facebook_seamless_m4t_v2_large.tt.t2u_encoder import T2uEncoder

    sd = wl.t2u_encoder_weights(hf_sd, num_layers=SUBMODEL_LAYERS)
    embed_dim = 1024
    num_heads, head_dim = 16, 64
    eps = 1e-5

    torch.manual_seed(123)
    inputs_embeds = torch.randn(1, 16, embed_dim, dtype=torch.float32)

    ref_out = ref.t2u_encoder_forward(
        inputs_embeds, sd, num_heads=num_heads, head_dim=head_dim, attention_mask=None, eps=eps
    )

    tt_block = T2uEncoder(
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        layers_state_dict=sd["layers"],
        final_layer_norm_state_dict=sd["final_layer_norm"],
        eps=eps,
        weight_dtype=ttnn.bfloat16,
    )
    tt_out = ttnn.to_torch(tt_block(inputs_embeds, attention_mask_torch=None)).to(torch.float32)
    return _pcc_value(ref_out, tt_out.reshape(ref_out.shape))


def _t_t2u_decoder(device, hf_sd) -> float:
    from models.demos.facebook_seamless_m4t_v2_large.tt.t2u_decoder import T2uDecoder

    bundle = wl.t2u_decoder_weights(hf_sd, num_layers=SUBMODEL_LAYERS)
    sd = bundle["state_dict"]
    char_pos = bundle["char_positional_weights"]
    pos = bundle["positional_weights"]

    embed_dim = 1024
    num_heads, head_dim = 16, 64
    eps = 1e-5
    embed_scale = math.sqrt(embed_dim)
    padding_idx = 1
    conv_kernel_size = 7
    variance_predictor_kernel_size = 3

    torch.manual_seed(123)
    encoder_seq_len = 4
    char_seq_len = 8
    char_input_ids = torch.randint(low=2, high=512, size=(1, char_seq_len), dtype=torch.long)
    # All-ones char counts (each text token -> 1 char). Total chars = encoder_seq_len
    # so sum == char_seq_len when count_per_id is shape (1, encoder_seq_len).
    char_count_per_id = torch.ones((1, encoder_seq_len), dtype=torch.long) * (char_seq_len // encoder_seq_len)
    encoder_hidden_states = torch.randn(1, encoder_seq_len, embed_dim, dtype=torch.float32)

    ref_out_dict = ref.t2u_decoder_forward(
        char_input_ids=char_input_ids,
        char_count_per_id=char_count_per_id,
        encoder_hidden_states=encoder_hidden_states,
        state_dict=sd,
        num_heads=num_heads,
        head_dim=head_dim,
        embed_scale=embed_scale,
        char_positional_weights=char_pos,
        positional_weights=pos,
        padding_idx=padding_idx,
        eps=eps,
        variance_predictor_kernel_size=variance_predictor_kernel_size,
        conv_kernel_size=conv_kernel_size,
    )
    ref_last = ref_out_dict["last_hidden_state"]

    tt_block = T2uDecoder(
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        state_dict=sd,
        char_positional_weights=char_pos,
        positional_weights=pos,
        embed_scale=embed_scale,
        padding_idx=padding_idx,
        eps=eps,
        variance_predictor_kernel_size=variance_predictor_kernel_size,
        conv_kernel_size=conv_kernel_size,
        weight_dtype=ttnn.bfloat16,
    )
    tt_out = tt_block(
        char_input_ids=char_input_ids,
        char_count_per_id=char_count_per_id,
        encoder_hidden_states=encoder_hidden_states,
    )
    tt_last = ttnn.to_torch(tt_out["last_hidden_state"]).to(torch.float32).reshape(ref_last.shape)
    return _pcc_value(ref_last, tt_last)


def _t_hifigan_vocoder(device, hf_sd) -> float:
    from models.demos.facebook_seamless_m4t_v2_large.tt.hifigan_vocoder import HifiGanVocoder

    sd = wl.hifigan_vocoder_weights(hf_sd)
    # conv_pre input channels = 1792 (unit + spkr + lang embed dims).
    model_in_dim = sd["conv_pre"]["weight"].shape[1]

    upsample_rates = (5, 4, 4, 2, 2)
    upsample_kernel_sizes = (11, 8, 8, 4, 4)
    resblock_kernel_sizes = (3, 7, 11)
    resblock_dilation_sizes = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    leaky_relu_slope = 0.1

    torch.manual_seed(123)
    x = torch.randn(1, model_in_dim, 16, dtype=torch.float32)

    ref_out = ref.hifigan_vocoder_forward(
        x,
        sd,
        upsample_rates=upsample_rates,
        upsample_kernel_sizes=upsample_kernel_sizes,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        leaky_relu_slope=leaky_relu_slope,
    )

    tt_block = HifiGanVocoder(
        device=device,
        state_dict=sd,
        upsample_rates=upsample_rates,
        upsample_kernel_sizes=upsample_kernel_sizes,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        leaky_relu_slope=leaky_relu_slope,
        weight_dtype=ttnn.bfloat16,
    )
    tt_in = _to_tt(device, x)
    tt_out = ttnn.to_torch(tt_block(tt_in)).to(torch.float32).reshape(ref_out.shape)
    return _pcc_value(ref_out, tt_out)


def _t_code_hifigan_vocoder(device, hf_sd) -> float:
    from models.demos.facebook_seamless_m4t_v2_large.tt.code_hifigan_vocoder import CodeHifiGanVocoder

    sd = wl.code_hifigan_vocoder_weights(hf_sd)
    pad_token_id = 1
    kernel_size = 3
    upsample_rates = (5, 4, 4, 2, 2)
    upsample_kernel_sizes = (11, 8, 8, 4, 4)
    resblock_kernel_sizes = (3, 7, 11)
    resblock_dilation_sizes = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    leaky_relu_slope = 0.1

    torch.manual_seed(123)
    # Small input: B=1, T=2 unit ids (vocab 10000); single speaker / lang.
    input_ids = torch.randint(low=2, high=1000, size=(1, 2), dtype=torch.long)
    speaker_id = torch.tensor([[0]], dtype=torch.long)
    lang_id = torch.tensor([[0]], dtype=torch.long)

    ref_out = ref.code_hifigan_vocoder_forward(
        input_ids=input_ids,
        speaker_id=speaker_id,
        lang_id=lang_id,
        state_dict=sd,
        pad_token_id=pad_token_id,
        variance_predictor_kernel_size=kernel_size,
        upsample_rates=upsample_rates,
        upsample_kernel_sizes=upsample_kernel_sizes,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        leaky_relu_slope=leaky_relu_slope,
    )
    # code_hifigan_vocoder_forward returns a dict with "waveform", "lengths", "dur_out".
    if isinstance(ref_out, dict):
        ref_waveform = ref_out["waveform"]
    elif isinstance(ref_out, tuple):
        ref_waveform = ref_out[0]
    else:
        ref_waveform = ref_out

    tt_block = CodeHifiGanVocoder(
        device=device,
        state_dict=sd,
        pad_token_id=pad_token_id,
        variance_predictor_kernel_size=kernel_size,
        upsample_rates=upsample_rates,
        upsample_kernel_sizes=upsample_kernel_sizes,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        leaky_relu_slope=leaky_relu_slope,
        weight_dtype=ttnn.bfloat16,
    )
    tt_waveform = tt_block(input_ids=input_ids, speaker_id=speaker_id, lang_id=lang_id)
    tt_waveform_torch = ttnn.to_torch(tt_waveform).to(torch.float32).reshape(ref_waveform.shape)
    return _pcc_value(ref_waveform, tt_waveform_torch)


def _t_seamless_m4t_v2(device, hf_sd) -> float:
    from models.demos.facebook_seamless_m4t_v2_large.tt.seamless_m4t_v2 import SeamlessM4Tv2

    bundle = wl.seamless_m4t_v2_t2tt_weights(
        hf_sd, num_encoder_layers=SUBMODEL_LAYERS, num_decoder_layers=SUBMODEL_LAYERS
    )
    text_enc_sd = bundle["text_encoder_state_dict"]
    text_dec_sd = bundle["text_decoder_state_dict"]
    lm_head_sd = bundle["lm_head_state_dict"]

    embed_dim = 1024
    num_heads, head_dim = 16, 64
    eps = 1e-5
    encoder_padding_idx = 0
    decoder_padding_idx = 0

    torch.manual_seed(123)
    input_ids = torch.randint(low=2, high=512, size=(1, 8), dtype=torch.long)
    decoder_input_ids = torch.randint(low=2, high=512, size=(1, 4), dtype=torch.long)

    # Reference: combine text_encoder + text_decoder + lm_head.
    # The top-level reference functional expects an outer state_dict.
    full_sd = {
        "text_encoder": text_enc_sd,
        "text_decoder": text_dec_sd,
        "lm_head": lm_head_sd,
    }
    ref_out_dict = ref.seamless_m4t_v2_forward(
        input_ids,
        decoder_input_ids,
        full_sd,
        num_heads=num_heads,
        head_dim=head_dim,
        attention_mask=None,
        decoder_attention_mask=None,
        eps=eps,
        encoder_padding_idx=encoder_padding_idx,
        decoder_padding_idx=decoder_padding_idx,
    )
    # The reference returns either a dict or a tensor depending on version; handle both.
    if isinstance(ref_out_dict, dict):
        ref_logits = ref_out_dict.get("logits", ref_out_dict.get("lm_logits"))
    else:
        ref_logits = ref_out_dict

    tt_block = SeamlessM4Tv2(
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        text_encoder_state_dict=text_enc_sd,
        text_decoder_state_dict=text_dec_sd,
        lm_head_state_dict=lm_head_sd,
        eps=eps,
        encoder_padding_idx=encoder_padding_idx,
        decoder_padding_idx=decoder_padding_idx,
        weight_dtype=ttnn.bfloat16,
    )
    tt_logits = tt_block(input_ids, decoder_input_ids, attention_mask=None, decoder_attention_mask=None)
    tt_logits_torch = ttnn.to_torch(tt_logits).to(torch.float32).reshape(ref_logits.shape)
    return _pcc_value(ref_logits, tt_logits_torch)


# ---------------------------------------------------------------------------
# Parametrized dispatch table
# ---------------------------------------------------------------------------

# Order: 10 leaves -> 6 layer composites -> 7 sub-models -> 1 top-level.
BLOCKS = [
    # --- 10 leaves ---
    ("layernorm", _t_layernorm),
    ("scaled_word_embedding", _t_scaled_word_embedding),
    ("sinusoidal_positional_embedding", _t_sinusoidal_positional_embedding),
    ("seamless_mha", _t_seamless_mha),
    ("seamless_ffn", _t_seamless_ffn),
    ("conformer_ffn", _t_conformer_ffn),
    ("conformer_self_attention", _t_conformer_self_attention),
    ("conformer_convolution_module", _t_conformer_convolution_module),
    ("variance_predictor", _t_variance_predictor),
    ("hifigan_residual_block", _t_hifigan_residual_block),
    # --- 6 layer composites ---
    ("conformer_feature_projection", _t_conformer_feature_projection),
    ("conformer_encoder_layer", _t_conformer_encoder_layer),
    ("text_encoder_layer", _t_text_encoder_layer),
    ("text_decoder_layer", _t_text_decoder_layer),
    ("t2u_decoder_layer", _t_t2u_decoder_layer),
    ("conformer_adapter_layer", _t_conformer_adapter_layer),
    # --- 7 sub-models (reduced layer counts) ---
    ("speech_encoder", _t_speech_encoder),
    ("text_encoder", _t_text_encoder),
    ("text_decoder", _t_text_decoder),
    ("t2u_encoder", _t_t2u_encoder),
    ("t2u_decoder", _t_t2u_decoder),
    ("hifigan_vocoder", _t_hifigan_vocoder),
    ("code_hifigan_vocoder", _t_code_hifigan_vocoder),
    # --- 1 top-level ---
    ("seamless_m4t_v2", _t_seamless_m4t_v2),
]


@pytest.mark.parametrize("block_name,block_fn", BLOCKS, ids=[b[0] for b in BLOCKS])
def test_real_hf_weights(block_name, block_fn, device, hf_sd):
    """PCC > 0.99 with real HF weights for ``block_name``."""
    pcc = block_fn(device, hf_sd)
    print(f"[{block_name}] real-HF PCC = {pcc:.6f}")
    assert pcc > 0.99, f"{block_name}: PCC {pcc} <= 0.99"
