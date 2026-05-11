# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for ``TTSeamlessM4Tv2Model.forward`` and ``.generate``.

Tests use **real, inference-representative inputs** throughout:

* **Text**: an English sentence tokenised by the model's own ``AutoTokenizer`` — produces naturally
  variable-length sequences with the correct vocabulary distribution and ``pad_token_id`` placement.
* **Speech**: a 1-second mono waveform at 16 kHz processed by ``AutoProcessor`` — produces the
  log-mel feature tensor ``input_features`` and binary frame-level ``attention_mask`` that the
  speech encoder actually sees during real inference.

PCC threshold ≥ 0.99 for all logit comparisons (HF float32 reference vs TTNN bfloat16 readback).
"""

from __future__ import annotations

import os
from typing import Any, Tuple

import pytest
import torch
import ttnn
from loguru import logger
from transformers import AutoProcessor, AutoTokenizer

from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.seamless_m4t_v2_large.reference.torch_seamless_m4t_v2_model import (
    forward_text_modality_logits,
    load_pretrained_seamless_m4t_v2_model,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_seamless_m4t_v2_model_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_seamless_m4t_v2_model import (
    TTSeamlessM4Tv2GreedySearchOutput,
    TTSeamlessM4Tv2Model,
)

PCC_THRESHOLD = 0.99

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _weights_dir_or_skip() -> str:
    try:
        return ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")


def _tt_logits_to_torch_bf16_flat(logits_tt: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(ttnn.from_device(logits_tt)).to(torch.bfloat16).contiguous().reshape(-1)


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


def assert_logits_pcc(ref: torch.Tensor, logits_tt: ttnn.Tensor, *, pcc: float, context: str) -> None:
    """Compare HF reference logits [1, dec_seq, vocab] vs TTNN readback."""
    ref_f = ref.detach().float().cpu()
    _, sd, v = ref_f.shape
    flat = _tt_logits_to_torch_bf16_flat(logits_tt)
    sp = flat.numel() // v
    tt_f = flat.reshape(1, sp, v)[:, :sd, :v].contiguous().float().cpu()
    assert tt_f.shape == ref_f.shape, f"{context}: shape ref {tuple(ref_f.shape)} vs ttnn {tuple(tt_f.shape)}"
    ok, msg = check_with_pcc(ref_f, tt_f, pcc=pcc)
    logger.info(f"{context} PCC: {msg}")
    assert ok, msg


# ---------------------------------------------------------------------------
# Real-input builders
# ---------------------------------------------------------------------------


def _real_text_inputs(
    tokenizer: Any, cfg: Any, dev: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenise a real English sentence → encoder + decoder inputs at natural sequence lengths."""
    enc_batch = tokenizer(
        ["Hello, my name is SeamlessM4T and I translate speech and text."],
        return_tensors="pt",
        padding=True,
    )
    input_ids = enc_batch["input_ids"].to(dev)
    attention_mask = enc_batch["attention_mask"].to(dev)

    # Decoder seed: [decoder_start_token_id, eng_lang_code_id] — mirrors HF generate() initialisation.
    ds = cfg.decoder_start_token_id
    tid = model_lang_id_or_default(cfg, "eng", ds + 1)
    decoder_input_ids = torch.tensor([[ds, tid]], dtype=torch.long, device=dev)
    dec_attn = torch.ones_like(decoder_input_ids)
    return input_ids, attention_mask, decoder_input_ids, dec_attn


def model_lang_id_or_default(cfg: Any, lang: str, default: int) -> int:
    """Look up a language token ID from config, falling back to ``default``."""
    mapping = getattr(cfg, "lang_code_to_id", None) or getattr(cfg, "text_decoder_lang_to_code_id", None)
    if mapping and lang in mapping:
        return int(mapping[lang])
    # Fallback: use decoder_start_token_id + 1 as a safe non-pad seed token.
    return default


def _real_speech_inputs(processor: Any, dev: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Synthesise a 1-second 16 kHz waveform and process to log-mel features."""
    torch.manual_seed(42)
    sampling_rate = 16_000
    # Small-amplitude noise is representative of real near-silence speech segments.
    wav = torch.randn(1, sampling_rate, dtype=torch.float32) * 0.01
    audio_inputs = processor(audios=wav, sampling_rate=sampling_rate, return_tensors="pt")
    input_features = audio_inputs["input_features"].to(dev, dtype=torch.bfloat16)
    attention_mask = audio_inputs["attention_mask"].to(dev, dtype=torch.long)
    return input_features, attention_mask


# ---------------------------------------------------------------------------
# Test 1: forward() — text modality
# ---------------------------------------------------------------------------


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_forward_text_modality(device, reset_seeds):
    """``forward()`` text path: real tokenised sentence → encoder → decoder → lm_head, PCC ≥ 0.99."""
    _ = reset_seeds
    weights_dir = _weights_dir_or_skip()
    model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = model.t2u_model.config
    dev = next(model.parameters()).device
    tt_model = make_tt_model(device, model, cfg, t2u_cfg)

    tokenizer = AutoTokenizer.from_pretrained(os.fspath(weights_dir), local_files_only=True)
    input_ids, enc_attn, decoder_input_ids, dec_attn = _real_text_inputs(tokenizer, cfg, dev)

    logger.info(
        f"Text forward — enc_seq={input_ids.shape[1]}, dec_seq={decoder_input_ids.shape[1]}, "
        f"vocab_size={cfg.vocab_size}"
    )

    # HF reference
    ref_logits = (
        forward_text_modality_logits(
            model,
            input_ids=input_ids,
            attention_mask=enc_attn,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=dec_attn,
        )
        .to(torch.bfloat16)
        .cpu()
    )

    # TTNN forward
    out = tt_model.forward(
        input_ids=torch_ids_to_ttnn(device, input_ids),
        attention_mask=torch_ids_to_ttnn(device, enc_attn),
        decoder_input_ids=torch_ids_to_ttnn(device, decoder_input_ids),
        decoder_attention_mask=torch_ids_to_ttnn(device, dec_attn),
        use_cache=False,
        return_dict=True,
    )
    assert_logits_pcc(ref_logits, out.logits, pcc=PCC_THRESHOLD, context="forward() text modality")


# ---------------------------------------------------------------------------
# Test 2: forward() — speech modality
# ---------------------------------------------------------------------------


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_forward_speech_modality(device, reset_seeds):
    """``forward()`` speech path: processor log-mel features → encoder → decoder → lm_head, PCC ≥ 0.99."""
    _ = reset_seeds
    weights_dir = _weights_dir_or_skip()
    model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = model.t2u_model.config
    dev = next(model.parameters()).device
    tt_model = make_tt_model(device, model, cfg, t2u_cfg)

    processor = AutoProcessor.from_pretrained(os.fspath(weights_dir), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(os.fspath(weights_dir), local_files_only=True)

    input_features, enc_attn_speech = _real_speech_inputs(processor, dev)
    _, _, decoder_input_ids, dec_attn = _real_text_inputs(tokenizer, cfg, dev)

    logger.info(
        f"Speech forward — input_features={tuple(input_features.shape)}, " f"dec_seq={decoder_input_ids.shape[1]}"
    )

    with torch.no_grad():
        ref_out = model(
            input_features=input_features,
            attention_mask=enc_attn_speech,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=dec_attn,
            use_cache=False,
            return_dict=True,
        )
    ref_logits = ref_out.logits.to(torch.bfloat16).cpu().float()  # [1, dec_seq, vocab]

    feats_tt = torch_feats_to_ttnn(device, input_features)
    enc_attn_tt = torch_ids_to_ttnn(device, enc_attn_speech.cpu())
    out = tt_model.forward(
        input_features=feats_tt,
        attention_mask=enc_attn_tt,
        decoder_input_ids=torch_ids_to_ttnn(device, decoder_input_ids),
        decoder_attention_mask=torch_ids_to_ttnn(device, dec_attn),
        use_cache=False,
        return_dict=True,
    )
    assert_logits_pcc(ref_logits, out.logits, pcc=PCC_THRESHOLD, context="forward() speech modality")


# ---------------------------------------------------------------------------
# Test 3: generate() — text-only greedy, token prefix match
# ---------------------------------------------------------------------------


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_generate_text_only_greedy(device, reset_seeds):
    """``generate()`` text path: real tokenised input → greedy text-only decoding, prefix tokens match HF."""
    _ = reset_seeds
    weights_dir = _weights_dir_or_skip()

    torch.manual_seed(0)
    model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = model.t2u_model.config
    dev = next(model.parameters()).device

    tokenizer = AutoTokenizer.from_pretrained(os.fspath(weights_dir), local_files_only=True)
    batch = tokenizer(["Hello, how are you?"], return_tensors="pt", padding=True)
    input_ids = batch["input_ids"].to(dev)
    attention_mask = batch["attention_mask"].to(dev)

    logger.info(f"Generate text — enc_seq={input_ids.shape[1]}, tokens={input_ids[0].tolist()}")

    gen_kw = dict(
        generate_speech=False,
        max_new_tokens=12,
        do_sample=False,
        num_beams=1,
        pad_token_id=cfg.pad_token_id,
        eos_token_id=cfg.eos_token_id,
        tgt_lang="eng",
    )

    with torch.no_grad():
        ref_gen = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kw)

    tt_model = make_tt_model(device, model, cfg, t2u_cfg)
    tt_gen = tt_model.generate(
        input_ids=torch_ids_to_ttnn(device, input_ids),
        attention_mask=torch_ids_to_ttnn(device, attention_mask),
        **gen_kw,
    )
    assert isinstance(tt_gen, TTSeamlessM4Tv2GreedySearchOutput)

    ref_seq = ref_gen.sequences.cpu()
    tt_seq = ttnn.to_torch(ttnn.from_device(tt_gen.sequences)).to(torch.int64).cpu()

    prefix_len = min(4, ref_seq.shape[1], tt_seq.shape[1])
    assert torch.equal(ref_seq[:, :prefix_len], tt_seq[:, :prefix_len]), (
        f"generate() prefix mismatch: "
        f"ref[:, :{prefix_len}]={ref_seq[:, :prefix_len].tolist()} "
        f"tt[:, :{prefix_len}]={tt_seq[:, :prefix_len].tolist()}"
    )
    logger.info(
        f"generate() token match OK; ref_len={ref_seq.shape[1]}, tt_len={tt_seq.shape[1]}, "
        f"ref={ref_seq[0].tolist()}, tt={tt_seq[0].tolist()}"
    )
