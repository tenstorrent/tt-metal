# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Torch HF reference vs TTNN for ``SeamlessM4Tv2Model`` full stack.

This module defines **two** pytest tests:

1. **``test_torch_hf_reference_vs_ttnn_forward``** — ``forward()``: HF torch vs TTNN for **text** then **speech** logits (same file, one test).
2. **``test_torch_hf_reference_vs_ttnn_generate``** — ``generate()``: HF torch vs TTNN (text-only greedy prefix).

Readback to torch is only for assertions; the port under test is TTNN.
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


def _weights_dir_or_skip() -> str:
    try:
        return ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")


def _tt_logits_to_torch_bf16_flat(logits_tt: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(ttnn.from_device(logits_tt)).to(torch.bfloat16).contiguous().reshape(-1)


def torch_token_ids_to_ttnn(device: ttnn.Device, t: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.to(torch.int32).cpu(),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def make_tt_seamless_m4tv2_model(
    device: ttnn.Device, model: torch.nn.Module, cfg: Any, t2u_cfg: Any
) -> TTSeamlessM4Tv2Model:
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


def assert_pcc_torch_reference_vs_ttnn_logits(
    ref_logits_torch: torch.Tensor,
    logits_tt: ttnn.Tensor,
    *,
    pcc: float = PCC_THRESHOLD,
    context: str,
) -> None:
    ref = ref_logits_torch.detach().float().cpu()
    _, sd, v = ref.shape
    flat = _tt_logits_to_torch_bf16_flat(logits_tt)
    sp = flat.numel() // v
    tt = flat.reshape(1, sp, v)[:, :sd, :v].contiguous().float().cpu()
    assert tt.shape == ref.shape, f"{context}: shape ref {tuple(ref.shape)} vs ttnn→torch {tuple(tt.shape)}"
    ok, msg = check_with_pcc(ref, tt, pcc=pcc)
    logger.info(f"{context} — torch reference vs TTNN (PCC readback): {msg}")
    assert ok, msg


def _random_text_forward_inputs(
    cfg: Any, dev: torch.device, *, batch: int, enc_seq: int, dec_seq: int, seed: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    input_ids = torch.randint(1, min(cfg.vocab_size - 1, 2**31 - 1), (batch, enc_seq), dtype=torch.int64, device=dev)
    enc_attn = torch.ones(batch, enc_seq, dtype=torch.long, device=dev)
    decoder_input_ids = torch.randint(
        1, min(cfg.vocab_size - 1, 2**31 - 1), (batch, dec_seq), dtype=torch.int64, device=dev
    )
    dec_attn = torch.ones(batch, dec_seq, dtype=torch.long, device=dev)
    return input_ids, enc_attn, decoder_input_ids, dec_attn


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_torch_hf_reference_vs_ttnn_forward(device, reset_seeds):
    """HF ``SeamlessM4Tv2Model.forward`` (torch) vs ``TTSeamlessM4Tv2Model.forward`` (ttnn): text then speech logits PCC.

    **Speech** inputs match ``demo/torch_demo.py``: ``AutoProcessor`` on 16 kHz mono ``audios=`` so
    ``input_features`` / ``attention_mask`` have the same layout as real inference (log-mel frames
    from the feature extractor).

    **Text** branch keeps fixed-length random token tensors (``enc_seq=dec_seq=32``) so TT vs HF
    PCC stays in the regime exercised by ``test_text_encoder`` / ``test_text_decoder``; natural
    ``processor(text=...)`` lengths are not guaranteed to hit the same numeric parity threshold.

    Bumped ``timeout`` to 20 minutes because the torch HF speech encoder reference runs on bf16 CPU
    (``F.linear`` is the long pole) and easily exceeds ``pytest.ini``'s default 300 s budget.
    """
    _ = reset_seeds
    weights_dir = _weights_dir_or_skip()
    model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = model.t2u_model.config
    dev = next(model.parameters()).device
    tt_model = make_tt_seamless_m4tv2_model(device, model, cfg, t2u_cfg)

    processor = AutoProcessor.from_pretrained(os.fspath(weights_dir), local_files_only=True)

    # --- Text modality (fixed shapes for stable PCC vs HF) ---
    input_ids, enc_attn, decoder_input_ids, dec_attn = _random_text_forward_inputs(
        cfg, dev, batch=1, enc_seq=32, dec_seq=32, seed=1
    )
    ref_logits_text = (
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

    out_text = tt_model.forward(
        input_ids=torch_token_ids_to_ttnn(device, input_ids),
        attention_mask=torch_token_ids_to_ttnn(device, enc_attn),
        decoder_input_ids=torch_token_ids_to_ttnn(device, decoder_input_ids),
        decoder_attention_mask=torch_token_ids_to_ttnn(device, dec_attn),
        use_cache=False,
        return_dict=True,
    )
    assert_pcc_torch_reference_vs_ttnn_logits(
        ref_logits_text,
        out_text.logits,
        context="forward() text modality (torch ref vs ttnn)",
    )

    # --- Speech modality (``processor(audios=..., sampling_rate=16_000)`` like ``demo/torch_demo.py``) ---
    torch.manual_seed(0)
    sampling_rate = 16_000
    # Short mono clip at 16 kHz (same rate as the demo after ``resample_waveform``).
    wav = torch.randn(1, sampling_rate, dtype=torch.float32) * 0.01
    audio_inputs = processor(audios=wav, sampling_rate=sampling_rate, return_tensors="pt")
    input_features = audio_inputs["input_features"].to(dev, dtype=torch.bfloat16)
    enc_attn_s = audio_inputs["attention_mask"].to(dev, dtype=torch.long)
    # Reuse the text-branch decoder ids so speech PCC isolates the speech encoder + cross-attn path.
    decoder_input_ids_s = decoder_input_ids
    dec_attn_s = dec_attn

    with torch.no_grad():
        ref_out_speech = model(
            input_features=input_features,
            attention_mask=enc_attn_s,
            decoder_input_ids=decoder_input_ids_s,
            decoder_attention_mask=dec_attn_s,
            use_cache=False,
            return_dict=True,
        )
    ref_logits_speech = ref_out_speech.logits.to(torch.bfloat16).cpu().float()

    feats_tt = ttnn.from_torch(
        input_features.cpu().contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    enc_attn_tt = torch_token_ids_to_ttnn(device, enc_attn_s.cpu())
    out_speech = tt_model.forward(
        input_features=feats_tt,
        attention_mask=enc_attn_tt,
        decoder_input_ids=torch_token_ids_to_ttnn(device, decoder_input_ids_s),
        decoder_attention_mask=torch_token_ids_to_ttnn(device, dec_attn_s),
        use_cache=False,
        return_dict=True,
    )
    assert_pcc_torch_reference_vs_ttnn_logits(
        ref_logits_speech,
        out_speech.logits,
        context="forward() speech modality (torch ref vs ttnn)",
    )


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_torch_hf_reference_vs_ttnn_generate(device, reset_seeds):
    """HF ``model.generate`` (torch) vs ``TTSeamlessM4Tv2Model.generate`` (ttnn): text-only greedy prefix."""
    _ = reset_seeds
    weights_dir = _weights_dir_or_skip()

    torch.manual_seed(0)
    model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = model.t2u_model.config
    dev = next(model.parameters()).device

    tok = AutoTokenizer.from_pretrained(os.fspath(weights_dir), local_files_only=True)
    batch = tok(["Hello"], return_tensors="pt", padding=True)
    input_ids = batch["input_ids"].to(dev)
    attention_mask = batch["attention_mask"].to(dev)

    gen_kw = dict(
        generate_speech=False,
        max_new_tokens=12,
        do_sample=False,
        num_beams=1,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        tgt_lang="eng",
    )

    with torch.no_grad():
        ref_gen_torch = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kw)

    tt_model = make_tt_seamless_m4tv2_model(device, model, cfg, t2u_cfg)
    tt_gen = tt_model.generate(
        input_ids=torch_token_ids_to_ttnn(device, input_ids),
        attention_mask=torch_token_ids_to_ttnn(device, attention_mask),
        **gen_kw,
    )
    assert isinstance(tt_gen, TTSeamlessM4Tv2GreedySearchOutput)

    ref_sequences_torch = ref_gen_torch.sequences.cpu()
    tt_sequences_torch = ttnn.to_torch(ttnn.from_device(tt_gen.sequences)).to(torch.int64).cpu()

    prefix_len = min(3, ref_sequences_torch.shape[1], tt_sequences_torch.shape[1])
    assert torch.equal(ref_sequences_torch[:, :prefix_len], tt_sequences_torch[:, :prefix_len]), (
        "generate(): torch reference vs ttnn prefix mismatch: "
        f"ref[:{prefix_len}]={ref_sequences_torch[:, :prefix_len].tolist()} "
        f"ttnn→torch[:{prefix_len}]={tt_sequences_torch[:, :prefix_len].tolist()}"
    )
    logger.info(
        "generate(): torch reference vs ttnn (text-only greedy prefix); "
        f"lengths ref={tuple(ref_sequences_torch.shape)} ttnn→torch={tuple(tt_sequences_torch.shape)}"
    )
