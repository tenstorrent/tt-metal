# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Realistic text-decoder PCC inputs derived from the HF / TT ``generate()`` path.

In production the decoder never sees random ``encoder_hidden`` or random ``input_ids``:

  * **T2TT / T2ST** — ``encoder_hidden_states`` = ``text_encoder`` output on a tokenized
    source prompt (``processor(text=..., src_lang=...)`` in the demo). Decoder prefill is the
    two-token seed ``[decoder_start_token_id, tgt_lang_code_id]`` (see ``TTSeamlessM4Tv2Model.generate``).
  * **S2TT / S2ST / ASR** — ``encoder_hidden_states`` = ``speech_encoder`` output (includes the
    length adaptor) on processor ``input_features``. ``encoder_attention_mask`` is rebuilt from
    subsampled mel lengths (HF ``_compute_new_attention_mask``), not the raw processor mask.

These helpers build that distribution on the HF reference model. The PCC test runner then
tile-pads encoder/decoder timelines the same way ``TTSeamlessM4Tv2Model._encode_text`` /
``_prefill_text_decoder_kv_cache`` do before calling the TT decoder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import torch
from transformers import AutoProcessor, AutoTokenizer, SeamlessM4Tv2Model
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import _compute_new_attention_mask

from models.experimental.seamless_m4t_v2_large.tt.common import TILE

# Same prompt / language pairing as ``test_seamless_m4t_v2_model.py`` and the demo.
DEFAULT_PROMPT = "Hello, my name is SeamlessM4T."
DEFAULT_SRC_LANG = "eng"
DEFAULT_TGT_LANG = "hin"
# Unit phrase repeated when synthesizing a longer tokenized source for max-encoder-seq sweeps.
_LONG_TEXT_UNIT = "Hello, my name is SeamlessM4T. "


@dataclass(frozen=True)
class TextDecoderPccInputs:
    """Decoder forward tensors matching HF ``SeamlessM4Tv2Decoder`` argument names (logical lengths)."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    encoder_hidden_states: torch.Tensor
    encoder_attention_mask: torch.Tensor


@dataclass(frozen=True)
class TextDecoderPccAligned:
    """Tile-padded tensors matching ``TTSeamlessM4Tv2Model`` prefill before ``text_decoder.forward``."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    encoder_hidden_states: torch.Tensor
    encoder_attention_mask: torch.Tensor
    logical_dec_seq: int
    logical_enc_seq: int
    padded_dec_seq: int
    padded_enc_seq: int


def tile_align(seq: int) -> int:
    return ((seq + TILE - 1) // TILE) * TILE


def load_hf_model_and_processor(
    weights_dir: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[SeamlessM4Tv2Model, AutoProcessor, AutoTokenizer]:
    path = os.fspath(weights_dir)
    model = SeamlessM4Tv2Model.from_pretrained(
        path,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if dtype is not None:
        model.to(dtype)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return model, processor, tokenizer


def decoder_seed_ids(model: SeamlessM4Tv2Model, tgt_lang: str) -> torch.Tensor:
    """``[decoder_start_token_id, text_decoder_lang_to_code_id[tgt_lang]]`` — production prefill seed."""
    ds = int(model.config.decoder_start_token_id)
    tid = int(model.generation_config.text_decoder_lang_to_code_id[tgt_lang])
    return torch.tensor([[ds, tid]], dtype=torch.long)


def _hf_text_encoder_hidden(
    model: SeamlessM4Tv2Model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    p0 = next(model.parameters())
    ii = input_ids.to(device=p0.device)
    am = attention_mask.to(device=p0.device)
    with torch.no_grad():
        enc = model.text_encoder(input_ids=ii, attention_mask=am, return_dict=True)
    return enc.last_hidden_state.to(dtype=p0.dtype)


def _hf_speech_encoder_hidden_and_mask(
    model: SeamlessM4Tv2Model,
    input_features: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Speech encoder output + subsampled encoder mask (HF ``SeamlessM4Tv2Model.forward`` speech path)."""
    p0 = next(model.parameters())
    feats = input_features.to(device=p0.device, dtype=p0.dtype)
    am = attention_mask.to(device=p0.device)
    with torch.no_grad():
        enc = model.speech_encoder(input_features=feats, attention_mask=am, return_dict=True)
    hidden = enc.last_hidden_state.to(dtype=p0.dtype)
    sub_lens = model._compute_sub_sample_lengths_from_attention_mask(am).to(hidden.device)
    enc_attn = _compute_new_attention_mask(hidden_states=hidden, seq_lens=sub_lens)
    return hidden, enc_attn.to(torch.long)


def tokenize_source_text(
    processor: AutoProcessor,
    text: str,
    src_lang: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Demo-style source tokenization: ``processor(text=..., src_lang=...)``."""
    enc = processor(text=text, src_lang=src_lang, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def source_text_for_enc_len(
    processor: AutoProcessor,
    target_enc_len: int,
    *,
    src_lang: str = DEFAULT_SRC_LANG,
    unit: str = _LONG_TEXT_UNIT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a tokenized source with exactly ``target_enc_len`` non-pad tokens (repeat + trim)."""
    if target_enc_len < 1:
        raise ValueError(f"target_enc_len must be >= 1, got {target_enc_len}")
    text = unit
    input_ids, attention_mask = tokenize_source_text(processor, text, src_lang)
    while int(attention_mask.sum().item()) < target_enc_len:
        text += unit
        input_ids, attention_mask = tokenize_source_text(processor, text, src_lang)
    valid = int(attention_mask.sum().item())
    if valid > target_enc_len:
        input_ids = input_ids[:, :target_enc_len]
        attention_mask = attention_mask[:, :target_enc_len]
    return input_ids, attention_mask


def make_t2tt_decoder_pcc_inputs(
    model: SeamlessM4Tv2Model,
    processor: AutoProcessor,
    *,
    text: str = DEFAULT_PROMPT,
    src_lang: str = DEFAULT_SRC_LANG,
    tgt_lang: str = DEFAULT_TGT_LANG,
    enc_seq_len: int | None = None,
) -> TextDecoderPccInputs:
    """T2TT-style decoder PCC inputs: text-encoder hidden states + two-token decoder seed."""
    if enc_seq_len is None:
        src_ids, src_mask = tokenize_source_text(processor, text, src_lang)
    else:
        src_ids, src_mask = source_text_for_enc_len(processor, enc_seq_len, src_lang=src_lang)
    encoder_hidden = _hf_text_encoder_hidden(model, src_ids, src_mask)
    dec_ids = decoder_seed_ids(model, tgt_lang)
    dec_mask = torch.ones_like(dec_ids)
    return TextDecoderPccInputs(
        input_ids=dec_ids,
        attention_mask=dec_mask,
        encoder_hidden_states=encoder_hidden,
        encoder_attention_mask=src_mask.to(encoder_hidden.device),
    )


def _s2tt_encoder_timeline_from_wav(
    model: SeamlessM4Tv2Model,
    processor: AutoProcessor,
    *,
    wav_seconds: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Run HF speech encoder on processor features; return hidden, mask, logical enc seq."""
    torch.manual_seed(seed)
    n_samples = max(1, int(16_000 * wav_seconds))
    wav = (torch.randn(n_samples, dtype=torch.float32) * 0.01).numpy().reshape(-1)
    audio = processor(audios=wav, sampling_rate=16_000, return_tensors="pt")
    input_features = audio["input_features"].to(dtype=next(model.parameters()).dtype)
    mel_mask = audio["attention_mask"]
    encoder_hidden, enc_attn = _hf_speech_encoder_hidden_and_mask(model, input_features, mel_mask)
    return encoder_hidden, enc_attn, int(encoder_hidden.shape[1])


def _truncate_encoder_timeline(
    encoder_hidden: torch.Tensor,
    enc_attn: torch.Tensor,
    target_enc_seq: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if int(encoder_hidden.shape[1]) < target_enc_seq:
        raise RuntimeError(f"encoder timeline {encoder_hidden.shape[1]} < target {target_enc_seq}; use longer audio")
    if int(encoder_hidden.shape[1]) > target_enc_seq:
        encoder_hidden = encoder_hidden[:, :target_enc_seq, :].contiguous()
        enc_attn = enc_attn[:, :target_enc_seq].contiguous()
    return encoder_hidden, enc_attn


def make_s2tt_decoder_pcc_inputs(
    model: SeamlessM4Tv2Model,
    processor: AutoProcessor,
    *,
    tgt_lang: str = DEFAULT_TGT_LANG,
    wav_seconds: float = 1.0,
    enc_seq_len: int | None = None,
    seed: int = 42,
) -> TextDecoderPccInputs:
    """S2TT-style inputs: 16 kHz audio → processor features → speech-encoder hidden + decoder seed.

    When ``enc_seq_len`` is set, lengthen audio until the subsampled speech-encoder timeline is at
    least that long, then truncate to exactly ``enc_seq_len`` (decoder PCC at a fixed cross-attn K).
    """
    if enc_seq_len is not None:
        if enc_seq_len < 1:
            raise ValueError(f"enc_seq_len must be >= 1, got {enc_seq_len}")
        seconds = max(wav_seconds, 1.0)
        encoder_hidden, enc_attn, got = _s2tt_encoder_timeline_from_wav(
            model, processor, wav_seconds=seconds, seed=seed
        )
        while got < enc_seq_len:
            seconds *= 1.5
            encoder_hidden, enc_attn, got = _s2tt_encoder_timeline_from_wav(
                model, processor, wav_seconds=seconds, seed=seed
            )
        encoder_hidden, enc_attn = _truncate_encoder_timeline(encoder_hidden, enc_attn, enc_seq_len)
    else:
        encoder_hidden, enc_attn, _ = _s2tt_encoder_timeline_from_wav(
            model, processor, wav_seconds=wav_seconds, seed=seed
        )

    dec_ids = decoder_seed_ids(model, tgt_lang)
    dec_mask = torch.ones_like(dec_ids)
    return TextDecoderPccInputs(
        input_ids=dec_ids,
        attention_mask=dec_mask,
        encoder_hidden_states=encoder_hidden,
        encoder_attention_mask=enc_attn,
    )


def align_case_for_tt_prefill(case: TextDecoderPccInputs, pad_token_id: int) -> TextDecoderPccAligned:
    """Right-pad encoder/decoder to tile boundaries like ``TTSeamlessM4Tv2Model`` prefill."""
    input_ids = case.input_ids
    dec_mask = case.attention_mask
    encoder_hidden = case.encoder_hidden_states
    enc_mask = case.encoder_attention_mask

    batch = int(input_ids.shape[0])
    logical_dec = int(input_ids.shape[1])
    logical_enc = int(encoder_hidden.shape[1])
    padded_dec = tile_align(logical_dec)
    padded_enc = tile_align(logical_enc)

    if padded_dec > logical_dec:
        tail = torch.full(
            (batch, padded_dec - logical_dec),
            pad_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        input_ids = torch.cat([input_ids, tail], dim=1)
        # Production prefill uses an all-ones decoder mask over the tile-padded length.
        dec_mask = torch.ones(batch, padded_dec, dtype=dec_mask.dtype, device=dec_mask.device)

    if padded_enc > logical_enc:
        h_dim = int(encoder_hidden.shape[2])
        enc_pad = torch.zeros(
            batch,
            padded_enc - logical_enc,
            h_dim,
            dtype=encoder_hidden.dtype,
            device=encoder_hidden.device,
        )
        encoder_hidden = torch.cat([encoder_hidden, enc_pad], dim=1)
        mask_pad = torch.zeros(
            batch,
            padded_enc - logical_enc,
            dtype=enc_mask.dtype,
            device=enc_mask.device,
        )
        enc_mask = torch.cat([enc_mask, mask_pad], dim=1)

    return TextDecoderPccAligned(
        input_ids=input_ids,
        attention_mask=dec_mask,
        encoder_hidden_states=encoder_hidden,
        encoder_attention_mask=enc_mask,
        logical_dec_seq=logical_dec,
        logical_enc_seq=logical_enc,
        padded_dec_seq=padded_dec,
        padded_enc_seq=padded_enc,
    )
