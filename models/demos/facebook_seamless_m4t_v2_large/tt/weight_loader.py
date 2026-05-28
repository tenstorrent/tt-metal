# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Map the HF SeamlessM4T-v2 checkpoint into per-TTNN-module state_dict dicts.

Pure PyTorch / safetensors. No TTNN. No device touches.

The HF checkpoint exposes 2232 flat keys under these top-level groups (see
``safetensors`` repo on the hub)::

    shared.weight                                                    (tied embedding)
    text_encoder.<...>                                               NLLB text encoder
    text_decoder.<...>                                               NLLB text decoder
    speech_encoder.<...>                                             W2v-BERT-2.0 + adapter
    t2u_model.model.encoder.<...>                                    T2U encoder
    t2u_model.model.decoder.<...>                                    NAR T2U decoder
    vocoder.<...>                                                    code HiFi-GAN

This module produces nested ``state_dict`` dicts in the EXACT shape each
TTNN module under ``models/demos/facebook_seamless_m4t_v2_large/tt/*.py``
expects (matching the ``_extract_*`` helpers used by the reference tests).
The TTNN ``__init__`` signatures are the contract; see the per-loader
docstrings below for the leaf shapes.

Notes on derived values:

* ``shared.weight`` is the text-encoder + text-decoder token embedding AND
  the LM head (HF ties them all). The loader returns it once via
  :func:`shared_embedding_weight` and embeds it into the per-sub-model dicts
  via :func:`text_encoder_weights`, :func:`text_decoder_weights`, and
  :func:`lm_head_weights`.
* The HF sinusoidal positional embedding for the NLLB encoder/decoder/T2U
  encoder/T2U decoder is **NOT** stored in the checkpoint -- it is a
  deterministic function of ``num_positions`` + ``embedding_dim`` +
  ``padding_idx`` (see ``SeamlessM4Tv2SinusoidalPositionalEmbedding.get_embedding``).
  :func:`build_sinusoidal_positional_embedding_weights` rebuilds it on
  demand at the standard ``max_position_embeddings + 2`` (HF offset=2).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import load_file

HF_PATH = "/local/ttuser/.cache/huggingface/hub/models--facebook--seamless-m4t-v2-large/snapshots/5f8cc790b19fc3f67a61c105133b20b34e3dcb76"

# ---------------------------------------------------------------------------
# Top-level checkpoint loading
# ---------------------------------------------------------------------------


def load_hf_state_dict(hf_path: str = HF_PATH) -> Dict[str, torch.Tensor]:
    """Concatenate the two safetensors shards into a single flat ``{key: tensor}`` dict."""
    sd: Dict[str, torch.Tensor] = {}
    for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        path = Path(hf_path) / shard
        sd.update(load_file(str(path)))
    return sd


# ---------------------------------------------------------------------------
# Sinusoidal positional embedding (HF stores nothing for this -- rebuild it)
# ---------------------------------------------------------------------------


def build_sinusoidal_positional_embedding_weights(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int | None = None,
) -> torch.Tensor:
    """Verbatim port of ``SeamlessM4Tv2SinusoidalPositionalEmbedding.get_embedding``."""
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
    emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return emb.to(torch.get_default_dtype())


# ---------------------------------------------------------------------------
# Tiny helpers to assemble {"weight", "bias"} groups
# ---------------------------------------------------------------------------


def _ln_sd(hf_sd: Dict[str, torch.Tensor], prefix: str) -> dict:
    """Return {"weight": gamma, "bias": beta} for a LayerNorm at ``prefix``."""
    return {"weight": hf_sd[f"{prefix}.weight"], "bias": hf_sd[f"{prefix}.bias"]}


def _linear_sd(hf_sd: Dict[str, torch.Tensor], prefix: str, with_bias: bool = True) -> dict:
    sd = {"weight": hf_sd[f"{prefix}.weight"]}
    if with_bias and f"{prefix}.bias" in hf_sd:
        sd["bias"] = hf_sd[f"{prefix}.bias"]
    return sd


def _conv_sd(hf_sd: Dict[str, torch.Tensor], prefix: str, with_bias: bool = True) -> dict:
    sd = {"weight": hf_sd[f"{prefix}.weight"]}
    if with_bias and f"{prefix}.bias" in hf_sd:
        sd["bias"] = hf_sd[f"{prefix}.bias"]
    return sd


def _bart_attn_sd(hf_sd: Dict[str, torch.Tensor], prefix: str) -> dict:
    """BART-style 4-projection MHA (q/k/v/out_proj) with bias=True.

    Consumed by ``SeamlessMHA`` in
    ``models/demos/facebook_seamless_m4t_v2_large/tt/seamless_mha.py``.
    """
    return {
        "q_proj": _linear_sd(hf_sd, f"{prefix}.q_proj"),
        "k_proj": _linear_sd(hf_sd, f"{prefix}.k_proj"),
        "v_proj": _linear_sd(hf_sd, f"{prefix}.v_proj"),
        "out_proj": _linear_sd(hf_sd, f"{prefix}.out_proj"),
    }


def _conformer_attn_sd(hf_sd: Dict[str, torch.Tensor], prefix: str) -> dict:
    """Conformer self-attention with ``linear_q/k/v/out`` (bias=True).

    Consumed by ``ConformerSelfAttention`` in
    ``models/demos/facebook_seamless_m4t_v2_large/tt/conformer_self_attention.py``.
    The relative-key ``distance_embedding`` lives inside the same HF
    sub-module but is passed as a SIBLING arg by the encoder-layer TTNN
    module, so we expose it here via :func:`conformer_encoder_layer_weights`.
    """
    return {
        "linear_q": _linear_sd(hf_sd, f"{prefix}.linear_q"),
        "linear_k": _linear_sd(hf_sd, f"{prefix}.linear_k"),
        "linear_v": _linear_sd(hf_sd, f"{prefix}.linear_v"),
        "linear_out": _linear_sd(hf_sd, f"{prefix}.linear_out"),
    }


def _conformer_ffn_sd(hf_sd: Dict[str, torch.Tensor], prefix: str) -> dict:
    return {
        "intermediate_dense": _linear_sd(hf_sd, f"{prefix}.intermediate_dense"),
        "output_dense": _linear_sd(hf_sd, f"{prefix}.output_dense"),
    }


def _nllb_ffn_sd(hf_sd: Dict[str, torch.Tensor], prefix: str) -> dict:
    return {
        "fc1": _linear_sd(hf_sd, f"{prefix}.fc1"),
        "fc2": _linear_sd(hf_sd, f"{prefix}.fc2"),
    }


# ---------------------------------------------------------------------------
# Shared / embedding helpers
# ---------------------------------------------------------------------------


def shared_embedding_weight(hf_sd: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Return ``shared.weight`` -- the (vocab, hidden) token embedding shared by
    the text encoder, text decoder, and LM head."""
    return hf_sd["shared.weight"]


# ---------------------------------------------------------------------------
# Per-block loaders -- NLLB text encoder/decoder side
# ---------------------------------------------------------------------------


def text_encoder_layer_weights(hf_sd: Dict[str, torch.Tensor], layer_idx: int) -> dict:
    """state_dict for one :class:`TextEncoderLayer`.

    Keys: ``self_attn_layer_norm``, ``self_attn``, ``ffn_layer_norm``, ``ffn``.
    """
    p = f"text_encoder.layers.{layer_idx}"
    return {
        "self_attn_layer_norm": _ln_sd(hf_sd, f"{p}.self_attn_layer_norm"),
        "self_attn": _bart_attn_sd(hf_sd, f"{p}.self_attn"),
        "ffn_layer_norm": _ln_sd(hf_sd, f"{p}.ffn_layer_norm"),
        "ffn": _nllb_ffn_sd(hf_sd, f"{p}.ffn"),
    }


def text_decoder_layer_weights(hf_sd: Dict[str, torch.Tensor], layer_idx: int) -> dict:
    """state_dict for one :class:`TextDecoderLayer`.

    Keys: ``self_attn_layer_norm``, ``self_attn``, ``cross_attention_layer_norm``,
    ``cross_attention``, ``ffn_layer_norm``, ``ffn``.
    """
    p = f"text_decoder.layers.{layer_idx}"
    return {
        "self_attn_layer_norm": _ln_sd(hf_sd, f"{p}.self_attn_layer_norm"),
        "self_attn": _bart_attn_sd(hf_sd, f"{p}.self_attn"),
        "cross_attention_layer_norm": _ln_sd(hf_sd, f"{p}.cross_attention_layer_norm"),
        "cross_attention": _bart_attn_sd(hf_sd, f"{p}.cross_attention"),
        "ffn_layer_norm": _ln_sd(hf_sd, f"{p}.ffn_layer_norm"),
        "ffn": _nllb_ffn_sd(hf_sd, f"{p}.ffn"),
    }


def text_encoder_weights(
    hf_sd: Dict[str, torch.Tensor],
    num_layers: int = 24,
    hidden_size: int = 1024,
    max_position_embeddings: int = 4096,
    padding_idx: int = 1,
) -> dict:
    """Full state_dict for :class:`TextEncoder`.

    Shape consumed by ``SeamlessM4Tv2`` (top-level) ``text_encoder_state_dict`` arg.
    """
    sin_offset = 2  # HF SeamlessM4Tv2SinusoidalPositionalEmbedding.offset
    embed_positions_weights = build_sinusoidal_positional_embedding_weights(
        num_embeddings=max_position_embeddings + sin_offset,
        embedding_dim=hidden_size,
        padding_idx=padding_idx,
    )
    return {
        "embed_tokens": {"weight": shared_embedding_weight(hf_sd)},
        "embed_positions_weights": embed_positions_weights,
        "layers": [text_encoder_layer_weights(hf_sd, i) for i in range(num_layers)],
        "final_layer_norm": _ln_sd(hf_sd, "text_encoder.layer_norm"),
    }


def text_decoder_weights(
    hf_sd: Dict[str, torch.Tensor],
    num_layers: int = 24,
    hidden_size: int = 1024,
    max_position_embeddings: int = 4096,
    padding_idx: int = 0,
) -> dict:
    """Full state_dict for :class:`TextDecoder`.

    HF puts the final LayerNorm under ``text_decoder.layer_norm`` (note the
    *singular* "layer_norm", matching the HF attribute name).
    """
    sin_offset = 2
    embed_positions_weights = build_sinusoidal_positional_embedding_weights(
        num_embeddings=max_position_embeddings + sin_offset,
        embedding_dim=hidden_size,
        padding_idx=padding_idx,
    )
    return {
        "embed_tokens": {"weight": shared_embedding_weight(hf_sd)},
        "embed_positions_weights": embed_positions_weights,
        "layers": [text_decoder_layer_weights(hf_sd, i) for i in range(num_layers)],
        "layer_norm": _ln_sd(hf_sd, "text_decoder.layer_norm"),
    }


def lm_head_weights(hf_sd: Dict[str, torch.Tensor]) -> dict:
    """state_dict for the LM head (tied to the shared embedding, no bias).

    Consumed by :class:`SeamlessM4Tv2` top-level ``lm_head_state_dict`` arg.
    """
    return {"weight": shared_embedding_weight(hf_sd)}


# ---------------------------------------------------------------------------
# Per-block loaders -- speech encoder (W2v-BERT-2.0)
# ---------------------------------------------------------------------------


def conformer_encoder_layer_weights(hf_sd: Dict[str, torch.Tensor], layer_idx: int) -> dict:
    """state_dict for one :class:`ConformerEncoderLayer`.

    Mirrors ``_extract_encoder_layer_state_dict`` from
    ``reference/test_functional_speech_encoder.py`` (NOT the
    ``test_functional_conformer_encoder_layer.py`` variant, which omits the
    sibling ``distance_embedding_weight``).
    """
    p = f"speech_encoder.encoder.layers.{layer_idx}"
    return {
        "ffn1_layer_norm": _ln_sd(hf_sd, f"{p}.ffn1_layer_norm"),
        "ffn1": _conformer_ffn_sd(hf_sd, f"{p}.ffn1"),
        "self_attn_layer_norm": _ln_sd(hf_sd, f"{p}.self_attn_layer_norm"),
        "self_attn": _conformer_attn_sd(hf_sd, f"{p}.self_attn"),
        "conv_module": {
            "layer_norm": _ln_sd(hf_sd, f"{p}.conv_module.layer_norm"),
            # conv_module's pointwise / depthwise convs have bias=False.
            "pointwise_conv1": _conv_sd(hf_sd, f"{p}.conv_module.pointwise_conv1", with_bias=False),
            "depthwise_conv": _conv_sd(hf_sd, f"{p}.conv_module.depthwise_conv", with_bias=False),
            "depthwise_layer_norm": _ln_sd(hf_sd, f"{p}.conv_module.depthwise_layer_norm"),
            "pointwise_conv2": _conv_sd(hf_sd, f"{p}.conv_module.pointwise_conv2", with_bias=False),
        },
        "ffn2_layer_norm": _ln_sd(hf_sd, f"{p}.ffn2_layer_norm"),
        "ffn2": _conformer_ffn_sd(hf_sd, f"{p}.ffn2"),
        "final_layer_norm": _ln_sd(hf_sd, f"{p}.final_layer_norm"),
        # Sibling: relative-key distance embedding (also lives under self_attn
        # in HF but is passed separately to ConformerEncoderLayer).
        "distance_embedding_weight": hf_sd[f"{p}.self_attn.distance_embedding.weight"],
    }


def conformer_adapter_layer_weights(hf_sd: Dict[str, torch.Tensor], layer_idx: int) -> dict:
    """state_dict for one :class:`ConformerAdapterLayer`.

    Mirrors ``_extract_adapter_layer_state_dict`` from the speech-encoder reference test.
    The adapter conv1d layers (residual_conv / self_attn_conv) have bias=True.
    """
    p = f"speech_encoder.adapter.layers.{layer_idx}"
    return {
        "residual_layer_norm": _ln_sd(hf_sd, f"{p}.residual_layer_norm"),
        "residual_conv": _conv_sd(hf_sd, f"{p}.residual_conv"),
        "self_attn_layer_norm": _ln_sd(hf_sd, f"{p}.self_attn_layer_norm"),
        "self_attn_conv": _conv_sd(hf_sd, f"{p}.self_attn_conv"),
        "self_attn": _conformer_attn_sd(hf_sd, f"{p}.self_attn"),
        "ffn_layer_norm": _ln_sd(hf_sd, f"{p}.ffn_layer_norm"),
        "ffn": _conformer_ffn_sd(hf_sd, f"{p}.ffn"),
    }


def speech_encoder_weights(
    hf_sd: Dict[str, torch.Tensor],
    num_encoder_layers: int = 24,
    num_adapter_layers: int = 1,
) -> dict:
    """Full state_dict for :class:`SpeechEncoder` (W2v-BERT-2.0 + adapter)."""
    sd: dict = {
        "feature_projection": {
            "layer_norm": _ln_sd(hf_sd, "speech_encoder.feature_projection.layer_norm"),
            "projection": _linear_sd(hf_sd, "speech_encoder.feature_projection.projection"),
        },
        "encoder": {
            "layers": [conformer_encoder_layer_weights(hf_sd, i) for i in range(num_encoder_layers)],
            "final_layer_norm": _ln_sd(hf_sd, "speech_encoder.encoder.layer_norm"),
        },
        "intermediate_ffn": _conformer_ffn_sd(hf_sd, "speech_encoder.intermediate_ffn"),
        "inner_layer_norm": _ln_sd(hf_sd, "speech_encoder.inner_layer_norm"),
    }
    if num_adapter_layers > 0:
        sd["adapter"] = {
            "layers": [conformer_adapter_layer_weights(hf_sd, i) for i in range(num_adapter_layers)],
        }
    return sd


# ---------------------------------------------------------------------------
# Per-block loaders -- T2U encoder/decoder
# ---------------------------------------------------------------------------


def t2u_encoder_layer_weights(hf_sd: Dict[str, torch.Tensor], layer_idx: int) -> dict:
    """state_dict for one T2U encoder layer.

    Structurally identical to a text encoder layer (consumed by
    :class:`TextEncoderLayer` via :class:`T2uEncoder`).
    """
    p = f"t2u_model.model.encoder.layers.{layer_idx}"
    return {
        "self_attn_layer_norm": _ln_sd(hf_sd, f"{p}.self_attn_layer_norm"),
        "self_attn": _bart_attn_sd(hf_sd, f"{p}.self_attn"),
        "ffn_layer_norm": _ln_sd(hf_sd, f"{p}.ffn_layer_norm"),
        "ffn": _nllb_ffn_sd(hf_sd, f"{p}.ffn"),
    }


def t2u_decoder_layer_weights(hf_sd: Dict[str, torch.Tensor], layer_idx: int) -> dict:
    """state_dict for one :class:`T2UDecoderLayer` (NAR, post-norm, conv branch)."""
    p = f"t2u_model.model.decoder.layers.{layer_idx}"
    return {
        "self_attn": _bart_attn_sd(hf_sd, f"{p}.self_attn"),
        "self_attn_layer_norm": _ln_sd(hf_sd, f"{p}.self_attn_layer_norm"),
        "conv1": _conv_sd(hf_sd, f"{p}.conv1"),
        "conv2": _conv_sd(hf_sd, f"{p}.conv2"),
        "conv_layer_norm": _ln_sd(hf_sd, f"{p}.conv_layer_norm"),
    }


def t2u_encoder_weights(hf_sd: Dict[str, torch.Tensor], num_layers: int = 6) -> dict:
    """Full state_dict for :class:`T2uEncoder` (no token/position embeddings)."""
    return {
        "layers": [t2u_encoder_layer_weights(hf_sd, i) for i in range(num_layers)],
        "final_layer_norm": _ln_sd(hf_sd, "t2u_model.model.encoder.layer_norm"),
    }


def _variance_predictor_sd(hf_sd: Dict[str, torch.Tensor], prefix: str) -> dict:
    """state_dict for one :class:`VariancePredictor` (duration / variance)."""
    return {
        "conv1": _conv_sd(hf_sd, f"{prefix}.conv1"),
        "ln1": _ln_sd(hf_sd, f"{prefix}.ln1"),
        "conv2": _conv_sd(hf_sd, f"{prefix}.conv2"),
        "ln2": _ln_sd(hf_sd, f"{prefix}.ln2"),
        "proj": _linear_sd(hf_sd, f"{prefix}.proj"),
    }


def t2u_decoder_weights(
    hf_sd: Dict[str, torch.Tensor],
    num_layers: int = 6,
    hidden_size: int = 1024,
    max_position_embeddings: int = 4096,
    t2u_padding_idx: int = 1,
) -> dict:
    """Full state_dict for :class:`T2uDecoder`.

    Returns the ``state_dict`` arg shape AND the two positional-embedding
    tables consumed as separate args (``char_positional_weights`` and
    ``positional_weights``). The caller passes those into ``T2uDecoder``'s
    ``__init__`` as siblings to the ``state_dict``.

    The returned dict has keys::

        "state_dict":              consumed as T2uDecoder(state_dict=...)
        "char_positional_weights": consumed as T2uDecoder(char_positional_weights=...)
        "positional_weights":      consumed as T2uDecoder(positional_weights=...)
    """
    sin_offset = 2
    char_positional_weights = build_sinusoidal_positional_embedding_weights(
        num_embeddings=max_position_embeddings + sin_offset,
        embedding_dim=hidden_size,
        padding_idx=t2u_padding_idx,
    )
    positional_weights = build_sinusoidal_positional_embedding_weights(
        num_embeddings=max_position_embeddings + sin_offset,
        embedding_dim=hidden_size,
        padding_idx=t2u_padding_idx,
    )
    state_dict = {
        "embed_char": {"weight": hf_sd["t2u_model.model.decoder.embed_char.weight"]},
        "pos_emb_alpha_char": hf_sd["t2u_model.model.decoder.pos_emb_alpha_char"],
        "pos_emb_alpha": hf_sd["t2u_model.model.decoder.pos_emb_alpha"],
        "duration_predictor": _variance_predictor_sd(hf_sd, "t2u_model.model.decoder.duration_predictor"),
        "layers": [t2u_decoder_layer_weights(hf_sd, i) for i in range(num_layers)],
        "layer_norm": _ln_sd(hf_sd, "t2u_model.model.decoder.layer_norm"),
    }
    return {
        "state_dict": state_dict,
        "char_positional_weights": char_positional_weights,
        "positional_weights": positional_weights,
    }


# ---------------------------------------------------------------------------
# Per-block loaders -- code HiFi-GAN vocoder
# ---------------------------------------------------------------------------


def hifigan_residual_block_weights(
    hf_sd: Dict[str, torch.Tensor],
    block_idx: int,
    num_inner_convs: int = 3,
) -> dict:
    """state_dict for one :class:`HifiGanResidualBlock` at ``resblocks[block_idx]``."""
    p = f"vocoder.hifi_gan.resblocks.{block_idx}"
    convs1 = [_conv_sd(hf_sd, f"{p}.convs1.{j}") for j in range(num_inner_convs)]
    convs2 = [_conv_sd(hf_sd, f"{p}.convs2.{j}") for j in range(num_inner_convs)]
    return {"convs1": convs1, "convs2": convs2}


def hifigan_vocoder_weights(
    hf_sd: Dict[str, torch.Tensor],
    num_upsamples: int = 5,
    num_kernels: int = 3,
) -> dict:
    """Full state_dict for :class:`HifiGanVocoder` (vocoder.hifi_gan)."""
    conv_pre = _conv_sd(hf_sd, "vocoder.hifi_gan.conv_pre")
    conv_post = _conv_sd(hf_sd, "vocoder.hifi_gan.conv_post")
    upsampler = [_conv_sd(hf_sd, f"vocoder.hifi_gan.upsampler.{i}") for i in range(num_upsamples)]
    resblocks = [
        hifigan_residual_block_weights(hf_sd, i, num_inner_convs=num_kernels)
        for i in range(num_upsamples * num_kernels)
    ]
    return {
        "conv_pre": conv_pre,
        "upsampler": upsampler,
        "resblocks": resblocks,
        "conv_post": conv_post,
    }


def code_hifigan_vocoder_weights(
    hf_sd: Dict[str, torch.Tensor],
    num_upsamples: int = 5,
    num_kernels: int = 3,
) -> dict:
    """Full state_dict for :class:`CodeHifiGanVocoder`."""
    return {
        "unit_embedding": {"weight": hf_sd["vocoder.unit_embedding.weight"]},
        "speaker_embedding": {"weight": hf_sd["vocoder.speaker_embedding.weight"]},
        "language_embedding": {"weight": hf_sd["vocoder.language_embedding.weight"]},
        "dur_predictor": _variance_predictor_sd(hf_sd, "vocoder.dur_predictor"),
        "hifi_gan": hifigan_vocoder_weights(hf_sd, num_upsamples=num_upsamples, num_kernels=num_kernels),
    }


# ---------------------------------------------------------------------------
# Top-level convenience: T2TT state_dict for SeamlessM4Tv2
# ---------------------------------------------------------------------------


def seamless_m4t_v2_t2tt_weights(
    hf_sd: Dict[str, torch.Tensor],
    num_encoder_layers: int = 24,
    num_decoder_layers: int = 24,
) -> dict:
    """Top-level T2TT bundle for :class:`SeamlessM4Tv2`.

    Returns a dict with three keys (``text_encoder_state_dict``,
    ``text_decoder_state_dict``, ``lm_head_state_dict``) -- each shaped
    exactly as the corresponding TTNN top-level ``__init__`` arg.
    """
    return {
        "text_encoder_state_dict": text_encoder_weights(hf_sd, num_layers=num_encoder_layers),
        "text_decoder_state_dict": text_decoder_weights(hf_sd, num_layers=num_decoder_layers),
        "lm_head_state_dict": lm_head_weights(hf_sd),
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _walk(d, prefix: str = ""):
    """Yield (path, tensor) for every torch.Tensor leaf in a nested dict/list."""
    if isinstance(d, torch.Tensor):
        yield prefix, d
        return
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _walk(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            yield from _walk(v, f"{prefix}[{i}]")


def _count_leaves(d) -> int:
    return sum(1 for _ in _walk(d))


if __name__ == "__main__":
    hf_sd = load_hf_state_dict()
    total_bytes = sum(t.numel() * t.element_size() for t in hf_sd.values())
    print(f"Loaded {len(hf_sd)} keys, {total_bytes / 1e9:.2f} GB")

    # Per-block loaders -- pick small layer counts to keep the test cheap.
    print("\n--- Per-layer loaders (layer 0) ---")
    per_layer_loaders = [
        ("text_encoder_layer_weights", lambda: text_encoder_layer_weights(hf_sd, 0)),
        ("text_decoder_layer_weights", lambda: text_decoder_layer_weights(hf_sd, 0)),
        ("conformer_encoder_layer_weights", lambda: conformer_encoder_layer_weights(hf_sd, 0)),
        ("conformer_adapter_layer_weights", lambda: conformer_adapter_layer_weights(hf_sd, 0)),
        ("t2u_encoder_layer_weights", lambda: t2u_encoder_layer_weights(hf_sd, 0)),
        ("t2u_decoder_layer_weights", lambda: t2u_decoder_layer_weights(hf_sd, 0)),
        ("hifigan_residual_block_weights", lambda: hifigan_residual_block_weights(hf_sd, 0)),
    ]
    for name, fn in per_layer_loaders:
        result = fn()
        n = _count_leaves(result)
        print(f"  {name}: {n} leaves, ok")

    print("\n--- Sub-model loaders ---")
    sub_model_loaders = [
        ("text_encoder_weights", lambda: text_encoder_weights(hf_sd, num_layers=24)),
        ("text_decoder_weights", lambda: text_decoder_weights(hf_sd, num_layers=24)),
        ("speech_encoder_weights", lambda: speech_encoder_weights(hf_sd, num_encoder_layers=24, num_adapter_layers=1)),
        ("t2u_encoder_weights", lambda: t2u_encoder_weights(hf_sd, num_layers=6)),
        ("t2u_decoder_weights", lambda: t2u_decoder_weights(hf_sd, num_layers=6)),
        ("hifigan_vocoder_weights", lambda: hifigan_vocoder_weights(hf_sd)),
        ("code_hifigan_vocoder_weights", lambda: code_hifigan_vocoder_weights(hf_sd)),
        ("lm_head_weights", lambda: lm_head_weights(hf_sd)),
        (
            "seamless_m4t_v2_t2tt_weights",
            lambda: seamless_m4t_v2_t2tt_weights(hf_sd, num_encoder_layers=24, num_decoder_layers=24),
        ),
    ]
    for name, fn in sub_model_loaders:
        result = fn()
        n = _count_leaves(result)
        print(f"  {name}: {n} leaves, ok")

    # Spot check: shared embedding shape.
    shared = shared_embedding_weight(hf_sd)
    print(f"\nshared.weight shape: {tuple(shared.shape)} (expect (256102, 1024))")
    assert tuple(shared.shape) == (256102, 1024), shared.shape

    # Spot check: text_encoder_layer 0 self_attn.q_proj shape.
    tel0 = text_encoder_layer_weights(hf_sd, 0)
    q_w = tel0["self_attn"]["q_proj"]["weight"]
    q_b = tel0["self_attn"]["q_proj"]["bias"]
    assert tuple(q_w.shape) == (1024, 1024), q_w.shape
    assert tuple(q_b.shape) == (1024,), q_b.shape

    # Spot check: conformer encoder layer 0 distance_embedding shape.
    cel0 = conformer_encoder_layer_weights(hf_sd, 0)
    de_w = cel0["distance_embedding_weight"]
    assert tuple(de_w.shape) == (73, 64), de_w.shape

    # Spot check: hifigan resblock 0 has 3 convs1 and 3 convs2.
    rb0 = hifigan_residual_block_weights(hf_sd, 0)
    assert len(rb0["convs1"]) == 3 and len(rb0["convs2"]) == 3, (len(rb0["convs1"]), len(rb0["convs2"]))

    # Spot check: sinusoidal builder shape.
    sin = build_sinusoidal_positional_embedding_weights(4098, 1024, padding_idx=0)
    assert tuple(sin.shape) == (4098, 1024), sin.shape

    print("\nAll loaders OK.")
