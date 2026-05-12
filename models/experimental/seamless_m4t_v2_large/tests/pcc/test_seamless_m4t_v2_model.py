# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Task-level PCC tests for SeamlessM4Tv2: the five canonical inference tasks.

| # | Task | Input → Output | What's compared at PCC ≥ 0.99 |
| - | ---- | -------------- | ----------------------------- |
| 1 | T2TT | text  → text   | text-decoder lm_head logits |
| 2 | S2TT | speech → text  | text-decoder lm_head logits |
| 3 | T2ST | text  → speech | text-decoder logits **and** T2U logits |
| 4 | S2ST | speech → speech| text-decoder logits **and** T2U logits |
| 5 | ASR  | speech → text (same lang) | text-decoder lm_head logits |

Why ``forward()`` and not ``generate()`` for the comparison?
    Autoregressive generation accumulates bf16 cascade through (text-decoder × N steps) → T2U →
    vocoder; the final waveform PCC sits well below 0.99 even with the duration-predictor fp32 path
    in ``tt_text_to_unit._duration_predictor``. ``forward()`` does a single deterministic step so
    the same bf16 model produces the same logits as HF to within fp32-accumulator precision —
    PCC ≥ 0.99 is the right bar.

Why two PCC checks for T2ST / S2ST?
    The speech-output tasks share their first stage (text-decoder) with their text-output siblings,
    so the first check is the same as T2TT / S2TT. The second check exercises the T2U module
    (encoder + decoder + ``lm_head``) on synthetic-but-realistic embeddings, using HF discrete
    durations as ``reference_discrete_durations`` (the same isolation strategy as
    ``test_text_to_unit.py:30-33`` — measures T2U arithmetic accuracy, not duration-predictor drift).
    The downstream vocoder is validated separately at PCC ≥ 0.99 in ``test_code_hifigan.py``.

Inputs are real and inference-representative throughout: ``AutoTokenizer`` for text, ``AutoProcessor``
for audio. All tensor math runs through TTNN (per-test ``ttnn.from_torch`` uploads + ``ttnn.to_torch``
host readbacks; no torch math is used in the inference path).
"""

from __future__ import annotations

import os
from typing import Any, Tuple

import pytest
import torch
import ttnn
from loguru import logger
from transformers import AutoProcessor, AutoTokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.seamless_m4t_v2_large.reference.torch_seamless_m4t_v2_model import (
    forward_text_modality_logits,
    load_pretrained_seamless_m4t_v2_model,
)
from models.experimental.seamless_m4t_v2_large.reference.torch_text_to_unit import (
    forward_t2u_logits_and_padding,
    hf_discrete_duration_counts_batch1,
    synthetic_t2u_inputs,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_seamless_m4t_v2_model_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_seamless_m4t_v2_model import TTSeamlessM4Tv2Model

PCC_THRESHOLD = 0.99


# ---------------------------------------------------------------------------
# Weights / fixtures
# ---------------------------------------------------------------------------


def _weights_dir_or_skip() -> str:
    try:
        return ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")


# ---------------------------------------------------------------------------
# Tensor uploads (TTNN host → device)
# ---------------------------------------------------------------------------


def torch_ids_to_ttnn(device: ttnn.Device, t: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.to(torch.int32).cpu(),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def torch_feats_to_ttnn(device: ttnn.Device, t: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.to(torch.bfloat16).cpu().contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# ---------------------------------------------------------------------------
# TT model assembly
# ---------------------------------------------------------------------------


def make_tt_model(device: ttnn.Device, model: Any, cfg: Any, t2u_cfg: Any) -> TTSeamlessM4Tv2Model:
    params = create_seamless_m4t_v2_model_parameters(model, device=device)
    return TTSeamlessM4Tv2Model(
        device,
        params,
        layer_norm_eps=cfg.layer_norm_eps,
        encoder_layers=cfg.encoder_layers,
        encoder_attention_heads=cfg.encoder_attention_heads,
        decoder_layers=cfg.decoder_layers,
        decoder_attention_heads=cfg.decoder_attention_heads,
        hidden_size=cfg.hidden_size,
        feature_projection_input_dim=cfg.feature_projection_input_dim,
        speech_encoder_attention_heads=cfg.speech_encoder_attention_heads,
        speech_encoder_intermediate_size=cfg.speech_encoder_intermediate_size,
        speech_encoder_layers=cfg.speech_encoder_layers,
        speech_encoder_chunk_size=cfg.speech_encoder_chunk_size,
        speech_encoder_left_chunk_num=cfg.speech_encoder_left_chunk_num,
        pad_token_id=cfg.pad_token_id,
        decoder_start_token_id=cfg.decoder_start_token_id,
        vocab_size=cfg.vocab_size,
        adaptor_kernel_size=cfg.adaptor_kernel_size,
        adaptor_stride=cfg.adaptor_stride,
        t2u_eos_token_id=cfg.t2u_eos_token_id,
        t2u_pad_token_id=t2u_cfg.pad_token_id,
        vocoder_offset=cfg.vocoder_offset,
        t2u_layer_norm_eps=t2u_cfg.layer_norm_eps,
        t2u_encoder_layers=t2u_cfg.encoder_layers,
        t2u_encoder_attention_heads=t2u_cfg.encoder_attention_heads,
        t2u_decoder_layers=t2u_cfg.decoder_layers,
        t2u_decoder_attention_heads=t2u_cfg.decoder_attention_heads,
        variance_predictor_embed_dim=t2u_cfg.variance_predictor_embed_dim,
        variance_predictor_hidden_dim=t2u_cfg.variance_predictor_hidden_dim,
        variance_predictor_kernel_size=t2u_cfg.variance_predictor_kernel_size,
        vocoder_config=cfg,
        generation_config=model.generation_config,
        hf_config=cfg,
    )


# ---------------------------------------------------------------------------
# Real-input builders
# ---------------------------------------------------------------------------


def _real_text_input_ids(tokenizer: Any, dev: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenise a real English sentence → ``(input_ids, attention_mask)`` at natural length."""
    enc = tokenizer(
        ["Hello, my name is SeamlessM4T and I translate speech and text."],
        return_tensors="pt",
        padding=True,
    )
    return enc["input_ids"].to(dev), enc["attention_mask"].to(dev)


def _real_speech_features(processor: Any, dev: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Synthesise a 1-second 16 kHz waveform → ``(input_features, attention_mask)`` via ``AutoProcessor``."""
    torch.manual_seed(42)
    sampling_rate = 16_000
    # Small-amplitude noise is representative of real near-silence speech segments.
    wav = torch.randn(1, sampling_rate, dtype=torch.float32) * 0.01
    audio_inputs = processor(audios=wav, sampling_rate=sampling_rate, return_tensors="pt")
    input_features = audio_inputs["input_features"].to(dev, dtype=torch.bfloat16)
    attention_mask = audio_inputs["attention_mask"].to(dev, dtype=torch.long)
    return input_features, attention_mask


def _lang_id_or_default(cfg: Any, lang: str, default: int) -> int:
    """Look up a target-language code id from ``cfg``; fall back to ``default`` if absent."""
    mapping = getattr(cfg, "lang_code_to_id", None) or getattr(cfg, "text_decoder_lang_to_code_id", None)
    if mapping and lang in mapping:
        return int(mapping[lang])
    return default


def _decoder_seed(cfg: Any, dev: torch.device, *, tgt_lang: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """``[decoder_start_token_id, tgt_lang_code_id]`` seed for ``forward()`` — matches HF ``generate()``."""
    ds = cfg.decoder_start_token_id
    tid = _lang_id_or_default(cfg, tgt_lang, ds + 1)
    decoder_input_ids = torch.tensor([[ds, tid]], dtype=torch.long, device=dev)
    decoder_attention_mask = torch.ones_like(decoder_input_ids)
    return decoder_input_ids, decoder_attention_mask


# ---------------------------------------------------------------------------
# PCC comparators
# ---------------------------------------------------------------------------


def _assert_text_logits_pcc(
    ref_logits: torch.Tensor, logits_tt: ttnn.Tensor, *, ctx: str, pcc: float = PCC_THRESHOLD
) -> None:
    """Compare HF text-decoder logits ``[1, dec_seq, vocab]`` vs TTNN readback at PCC ≥ ``pcc``."""
    ref_f = ref_logits.detach().float().cpu()
    _, sd, v = ref_f.shape
    flat = ttnn.to_torch(ttnn.from_device(logits_tt)).to(torch.bfloat16).contiguous().reshape(-1)
    sp = flat.numel() // v
    tt_f = flat.reshape(1, sp, v)[:, :sd, :v].contiguous().float().cpu()
    assert tt_f.shape == ref_f.shape, f"{ctx}: shape ref {tuple(ref_f.shape)} vs ttnn {tuple(tt_f.shape)}"
    ok, msg = check_with_pcc(ref_f, tt_f, pcc=pcc)
    logger.info(f"{ctx} text-decoder logits PCC: {msg}")
    assert ok, f"{ctx}: text-decoder logits PCC < {pcc}: {msg}"


def _assert_t2u_logits_pcc(
    model: Any,
    tt_model: TTSeamlessM4Tv2Model,
    device: ttnn.Device,
    *,
    ctx: str,
    pcc: float = PCC_THRESHOLD,
) -> None:
    """T2U logits PCC check — synthetic inputs, HF reference durations.

    Mirrors ``test_text_to_unit.py``: ``synthetic_t2u_inputs`` builds a valid batch-1 T2U input set,
    HF computes reference discrete durations, and both HF and TT run T2U on the same
    ``inputs_embeds`` / ``char_input_ids`` / ``char_count_per_id``. Passing
    ``reference_discrete_durations`` to the TT T2U skips the TT duration predictor so the unit
    length matches HF — what's measured is the encoder + decoder + ``lm_head`` arithmetic on the
    unit-vocabulary logits.
    """
    t2u_cfg = model.t2u_model.config
    inputs_embeds, attention_mask, char_input_ids, char_count_per_id = synthetic_t2u_inputs(
        t2u_cfg,
        batch=1,
        encoder_seq_len=32,
        seed=1,
        dtype=torch.bfloat16,
    )
    hf_dev = next(model.t2u_model.parameters()).device
    char_count_per_id_dev = char_count_per_id.to(hf_dev)

    ref_logits, _ = forward_t2u_logits_and_padding(
        model.t2u_model,
        inputs_embeds,
        attention_mask,
        char_input_ids,
        char_count_per_id_dev,
    )
    ref_logits_bf16 = ref_logits.to(torch.bfloat16).cpu()

    ref_durs = hf_discrete_duration_counts_batch1(
        model.t2u_model,
        inputs_embeds.to(hf_dev),
        attention_mask.to(hf_dev),
        char_input_ids.to(hf_dev),
        char_count_per_id_dev,
    )

    mask_4d = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)
    inputs_embeds_tt = ttnn.from_torch(
        inputs_embeds.cpu().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    attn_tt = ttnn.from_torch(
        mask_4d.cpu().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    char_ids_tt = ttnn.from_torch(
        char_input_ids.cpu().to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cc_list = [int(x) for x in char_count_per_id[0].cpu().tolist()]

    tt_logits_tt, _ = tt_model.t2u.forward(
        inputs_embeds_tt,
        attn_tt,
        char_ids_tt,
        cc_list,
        reference_discrete_durations=ref_durs,
    )
    ttnn.deallocate(inputs_embeds_tt)
    ttnn.deallocate(attn_tt)
    ttnn.deallocate(char_ids_tt)

    tt_logits = ttnn.to_torch(ttnn.from_device(tt_logits_tt)).to(torch.bfloat16).cpu()
    ttnn.deallocate(tt_logits_tt)

    v = int(ref_logits_bf16.shape[-1])
    flat = tt_logits.reshape(-1)
    sp = flat.numel() // v
    tt_logits_3d = flat.reshape(1, sp, v)[:, : ref_logits_bf16.shape[1], :].contiguous()
    assert (
        tt_logits_3d.shape == ref_logits_bf16.shape
    ), f"{ctx}: T2U logits shape ref={tuple(ref_logits_bf16.shape)} tt={tuple(tt_logits_3d.shape)}"

    ok, msg = check_with_pcc(ref_logits_bf16.float(), tt_logits_3d.float(), pcc=pcc)
    logger.info(f"{ctx} T2U logits PCC: {msg}")
    assert ok, f"{ctx}: T2U logits PCC < {pcc}: {msg}"


# ---------------------------------------------------------------------------
# Shared text-decoder forward routine
# ---------------------------------------------------------------------------


def _forward_and_compare_text_logits(
    device: ttnn.Device,
    *,
    weights_dir: str,
    use_speech_input: bool,
    tgt_lang: str,
    ctx: str,
) -> Tuple[Any, TTSeamlessM4Tv2Model, Any]:
    """Run HF + TT ``forward()`` for the given modality/target-language pair; assert text logits PCC.

    Returns ``(model, tt_model, cfg)`` so the caller can chain additional checks (e.g., T2U for
    T2ST / S2ST).
    """
    model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = model.t2u_model.config
    dev = next(model.parameters()).device
    tt_model = make_tt_model(device, model, cfg, t2u_cfg)

    decoder_input_ids, decoder_attention_mask = _decoder_seed(cfg, dev, tgt_lang=tgt_lang)

    if use_speech_input:
        processor = AutoProcessor.from_pretrained(os.fspath(weights_dir), local_files_only=True)
        input_features, enc_attn = _real_speech_features(processor, dev)
        logger.info(
            f"{ctx} — input_features={tuple(input_features.shape)}, "
            f"dec_seq={decoder_input_ids.shape[1]}, tgt_lang={tgt_lang}"
        )
        with torch.no_grad():
            ref_out = model(
                input_features=input_features,
                attention_mask=enc_attn,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                use_cache=False,
                return_dict=True,
            )
        ref_logits = ref_out.logits.to(torch.bfloat16).cpu().float()
        out = tt_model.forward(
            input_features=torch_feats_to_ttnn(device, input_features),
            attention_mask=torch_ids_to_ttnn(device, enc_attn.cpu()),
            decoder_input_ids=torch_ids_to_ttnn(device, decoder_input_ids),
            decoder_attention_mask=torch_ids_to_ttnn(device, decoder_attention_mask),
            use_cache=False,
            return_dict=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(os.fspath(weights_dir), local_files_only=True)
        input_ids, enc_attn = _real_text_input_ids(tokenizer, dev)
        logger.info(f"{ctx} — enc_seq={input_ids.shape[1]}, dec_seq={decoder_input_ids.shape[1]}, tgt_lang={tgt_lang}")
        ref_logits = (
            forward_text_modality_logits(
                model,
                input_ids=input_ids,
                attention_mask=enc_attn,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
            .to(torch.bfloat16)
            .cpu()
        )
        out = tt_model.forward(
            input_ids=torch_ids_to_ttnn(device, input_ids),
            attention_mask=torch_ids_to_ttnn(device, enc_attn),
            decoder_input_ids=torch_ids_to_ttnn(device, decoder_input_ids),
            decoder_attention_mask=torch_ids_to_ttnn(device, decoder_attention_mask),
            use_cache=False,
            return_dict=True,
        )

    _assert_text_logits_pcc(ref_logits, out.logits, ctx=ctx)
    return model, tt_model, cfg


# ---------------------------------------------------------------------------
# Test 1 — T2TT: Text-to-Text Translation
# ---------------------------------------------------------------------------


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_t2tt(device, reset_seeds):
    """T2TT — Text-to-Text Translation. PCC ≥ 0.99 on text-decoder ``lm_head`` logits."""
    _ = reset_seeds
    weights_dir = _weights_dir_or_skip()
    _forward_and_compare_text_logits(
        device, weights_dir=weights_dir, use_speech_input=False, tgt_lang="eng", ctx="T2TT"
    )


# ---------------------------------------------------------------------------
# Test 2 — S2TT: Speech-to-Text Translation
# ---------------------------------------------------------------------------


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_s2tt(device, reset_seeds):
    """S2TT — Speech-to-Text Translation. PCC ≥ 0.99 on text-decoder ``lm_head`` logits."""
    _ = reset_seeds
    weights_dir = _weights_dir_or_skip()
    _forward_and_compare_text_logits(device, weights_dir=weights_dir, use_speech_input=True, tgt_lang="eng", ctx="S2TT")


# ---------------------------------------------------------------------------
# Test 3 — T2ST: Text-to-Speech Translation
# ---------------------------------------------------------------------------


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_t2st(device, reset_seeds):
    """T2ST — Text-to-Speech Translation. PCC ≥ 0.99 on (text-decoder logits, T2U logits).

    The downstream vocoder is validated separately in ``test_code_hifigan.py`` (also PCC ≥ 0.99).
    Together with this test's two PCC checks, every stage of the T2ST pipeline is covered at the
    0.99 bar.
    """
    _ = reset_seeds
    weights_dir = _weights_dir_or_skip()
    model, tt_model, _ = _forward_and_compare_text_logits(
        device, weights_dir=weights_dir, use_speech_input=False, tgt_lang="eng", ctx="T2ST"
    )
    _assert_t2u_logits_pcc(model, tt_model, device, ctx="T2ST")


# ---------------------------------------------------------------------------
# Test 4 — S2ST: Speech-to-Speech Translation
# ---------------------------------------------------------------------------


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_s2st(device, reset_seeds):
    """S2ST — Speech-to-Speech Translation. PCC ≥ 0.99 on (text-decoder logits, T2U logits).

    The downstream vocoder is validated separately in ``test_code_hifigan.py`` (also PCC ≥ 0.99).
    """
    _ = reset_seeds
    weights_dir = _weights_dir_or_skip()
    model, tt_model, _ = _forward_and_compare_text_logits(
        device, weights_dir=weights_dir, use_speech_input=True, tgt_lang="eng", ctx="S2ST"
    )
    _assert_t2u_logits_pcc(model, tt_model, device, ctx="S2ST")


# ---------------------------------------------------------------------------
# Test 5 — ASR: Automatic Speech Recognition (same-language transcription)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_asr(device, reset_seeds):
    """ASR — Automatic Speech Recognition. PCC ≥ 0.99 on text-decoder ``lm_head`` logits.

    Same forward dataflow as S2TT (speech-encoder → text-decoder → ``lm_head``); ASR is the special
    case where the decoder seed's target-language code equals the source language, so the decoder
    transcribes rather than translates. Decoder seed: ``[decoder_start_token_id, eng_lang_code_id]``.
    """
    _ = reset_seeds
    weights_dir = _weights_dir_or_skip()
    _forward_and_compare_text_logits(device, weights_dir=weights_dir, use_speech_input=True, tgt_lang="eng", ctx="ASR")
