# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Seamless M4T v2 — TTNN demo of all five inference tasks.

Tasks demonstrated (every one runs through ``TTSeamlessM4Tv2Model.generate``):

  1. **T2TT** Text-to-Text Translation        (English text → Hindi text)
  2. **T2ST** Text-to-Speech Translation      (English text → Hindi speech)
  3. **S2TT** Speech-to-Text Translation      (Hindi speech → English text)
  4. **S2ST** Speech-to-Speech Translation    (Hindi speech → Spanish speech)
  5. **ASR**  Automatic Speech Recognition    (Hindi speech → Hindi text)

The Hindi speech used as the input to tasks 3–5 is produced by task 2, so the demo is fully
self-contained — no external audio files needed. Output audio is written next to this file:

  * ``outputs/t2st_hindi_speech.wav``  — task 2 output (re-used as input for tasks 3-5)
  * ``outputs/s2st_spanish_speech.wav`` — task 4 output

Run from repo root:

  python models/experimental/seamless_m4t_v2_large/demo/ttnn_demo.py

Optional: ``SEAMLESS_M4T_V2_WEIGHTS=/path/to/seamless-m4t-v2-large`` if not using the default tree.
"""

from __future__ import annotations

import os
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
import ttnn
from transformers import AutoProcessor, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.experimental.seamless_m4t_v2_large.reference.torch_seamless_m4t_v2_model import (
    load_pretrained_seamless_m4t_v2_model,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_seamless_m4t_v2_model_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_seamless_m4t_v2_model import (
    TTSeamlessM4Tv2GenerationOutput,
    TTSeamlessM4Tv2GreedySearchOutput,
    TTSeamlessM4Tv2Model,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
T2ST_WAV = OUTPUT_DIR / "t2st_hindi_speech.wav"
S2ST_WAV = OUTPUT_DIR / "s2st_spanish_speech.wav"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _weights_dir() -> Path:
    env = os.environ.get("SEAMLESS_M4T_V2_WEIGHTS")
    if env:
        return Path(env).expanduser().resolve()
    return ensure_seamless_m4t_v2_large_weights()


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


def make_tt_model(device: ttnn.Device, model: torch.nn.Module, cfg, t2u_cfg) -> TTSeamlessM4Tv2Model:
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


def _waveform_to_mono_fp32(waveform_tt: ttnn.Tensor, lengths_tt: ttnn.Tensor) -> np.ndarray:
    """Read a TT vocoder waveform back to host as a 1-D fp32 numpy array, trimmed to valid length.

    TT vocoder output shape: ``[B, T_max, 1]`` (right-padded with zeros to the batch max). The valid
    sample count per row is in ``lengths_tt`` — we trim to that to drop trailing silence padding.
    """
    arr = ttnn.to_torch(ttnn.from_device(waveform_tt)).float().squeeze().cpu().numpy()
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    valid_len = int(ttnn.to_torch(ttnn.from_device(lengths_tt)).long().reshape(-1)[0].item())
    if 0 < valid_len <= arr.size:
        arr = arr[:valid_len]
    return arr


def _save_wav(path: Path, waveform_np: np.ndarray, sample_rate: int) -> None:
    """Save a mono fp32 waveform to ``path`` as a 16-bit PCM WAV (stdlib ``wave`` only)."""
    arr = np.clip(waveform_np, -1.0, 1.0)
    pcm16 = (arr * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())


def _decode(tokenizer: Any, sequences_tt: ttnn.Tensor) -> str:
    """Read a TT decoder sequence back to host and decode to a single string (special tokens skipped)."""
    ids = ttnn.to_torch(ttnn.from_device(sequences_tt)).to(torch.int64).cpu()
    return tokenizer.batch_decode(ids, skip_special_tokens=True)[0]


def _print_header(idx: int, name: str, abbrev: str, src: str, tgt: str) -> None:
    print()
    print("=" * 78)
    print(f"  {idx}. {abbrev}  —  {name}  ({src} → {tgt})")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    weights_dir = _weights_dir()
    path = os.fspath(weights_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    torch.manual_seed(0)
    model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = model.t2u_model.config
    sample_rate = int(getattr(cfg, "sampling_rate", 16000))

    # ---- Single English prompt drives the entire demo chain ----
    src_text = "Hello, my dog is cute"
    src_lang = "eng"
    tgt_translate = "hin"  # task 1, 2: translate eng → hin
    tgt_back_text = "eng"  # task 3: speech in hin → text in eng (back-translation)
    tgt_speech_other = "spa"  # task 4: speech in hin → speech in spa
    tgt_asr = "hin"  # task 5: transcribe the hin audio in hin

    text_inputs = processor(text=src_text, src_lang=src_lang, return_tensors="pt")
    input_ids = text_inputs["input_ids"]
    input_text_attn = text_inputs["attention_mask"]

    gen_common = dict(
        max_new_tokens=48,
        do_sample=False,
        num_beams=1,
        pad_token_id=cfg.pad_token_id,
        eos_token_id=cfg.eos_token_id,
    )

    try:
        original_default = ttnn.GetDefaultDevice()
    except Exception:
        original_default = None
    # 65536 B L1_SMALL is the value used by the speech-generation PCC tests
    # (``test_code_hifigan.py`` and ``test_seamless_m4t_v2_model.py``). 32768 B works for the
    # text-only path but is not enough for S2ST: speech-encoder + T2U + vocoder chained back-to-back
    # exhausts L1_SMALL before the vocoder's ``_resblock`` conv1d can allocate.
    device = ttnn.open_device(device_id=0, l1_small_size=65536)
    ttnn.SetDefaultDevice(device)

    try:
        tt_model = make_tt_model(device, model, cfg, t2u_cfg)

        # =========================================================================
        # 1. T2TT — Text-to-Text Translation (English → Hindi)
        # =========================================================================
        _print_header(1, "Text-to-Text Translation", "T2TT", "eng", tgt_translate)
        print(f"  Input text  ({src_lang}): {src_text}")
        t2tt_out = tt_model.generate(
            input_ids=torch_ids_to_ttnn(device, input_ids),
            attention_mask=torch_ids_to_ttnn(device, input_text_attn),
            generate_speech=False,
            tgt_lang=tgt_translate,
            **gen_common,
        )
        if not isinstance(t2tt_out, TTSeamlessM4Tv2GreedySearchOutput):
            raise TypeError(f"T2TT expected TTSeamlessM4Tv2GreedySearchOutput, got {type(t2tt_out)}")
        print(f"  Output text ({tgt_translate}): {_decode(tokenizer, t2tt_out.sequences)}")

        # =========================================================================
        # 2. T2ST — Text-to-Speech Translation (English text → Hindi speech)
        # =========================================================================
        _print_header(2, "Text-to-Speech Translation", "T2ST", "eng", tgt_translate)
        print(f"  Input text  ({src_lang}): {src_text}")
        t2st_out = tt_model.generate(
            input_ids=torch_ids_to_ttnn(device, input_ids),
            attention_mask=torch_ids_to_ttnn(device, input_text_attn),
            generate_speech=True,
            return_intermediate_token_ids=True,
            tgt_lang=tgt_translate,
            speaker_id=0,
            **gen_common,
        )
        if not isinstance(t2st_out, TTSeamlessM4Tv2GenerationOutput):
            raise TypeError(f"T2ST expected TTSeamlessM4Tv2GenerationOutput, got {type(t2st_out)}")
        print(f"  Intermediate text ({tgt_translate}): {_decode(tokenizer, t2st_out.sequences)}")
        hindi_wav_np = _waveform_to_mono_fp32(t2st_out.waveform, t2st_out.waveform_lengths)
        _save_wav(T2ST_WAV, hindi_wav_np, sample_rate=sample_rate)
        print(f"  Output audio ({tgt_translate}, {sample_rate} Hz, {hindi_wav_np.size} samples)")
        print(f"  Saved to: {T2ST_WAV}")

        # The Hindi speech from T2ST becomes the input for tasks 3-5.
        audio_inputs = processor(audios=hindi_wav_np, sampling_rate=sample_rate, return_tensors="pt")
        input_features = audio_inputs["input_features"]
        input_speech_attn = audio_inputs["attention_mask"]

        # =========================================================================
        # 3. S2TT — Speech-to-Text Translation (Hindi speech → English text)
        # =========================================================================
        _print_header(3, "Speech-to-Text Translation", "S2TT", tgt_translate, tgt_back_text)
        print(f"  Input audio ({tgt_translate}): {T2ST_WAV} ({sample_rate} Hz)")
        s2tt_out = tt_model.generate(
            input_features=torch_feats_to_ttnn(device, input_features),
            attention_mask=torch_ids_to_ttnn(device, input_speech_attn),
            generate_speech=False,
            tgt_lang=tgt_back_text,
            **gen_common,
        )
        if not isinstance(s2tt_out, TTSeamlessM4Tv2GreedySearchOutput):
            raise TypeError(f"S2TT expected TTSeamlessM4Tv2GreedySearchOutput, got {type(s2tt_out)}")
        print(f"  Output text ({tgt_back_text}): {_decode(tokenizer, s2tt_out.sequences)}")

        # =========================================================================
        # 4. S2ST — Speech-to-Speech Translation (Hindi speech → Spanish speech)
        # =========================================================================
        _print_header(4, "Speech-to-Speech Translation", "S2ST", tgt_translate, tgt_speech_other)
        print(f"  Input audio ({tgt_translate}): {T2ST_WAV} ({sample_rate} Hz)")
        s2st_out = tt_model.generate(
            input_features=torch_feats_to_ttnn(device, input_features),
            attention_mask=torch_ids_to_ttnn(device, input_speech_attn),
            generate_speech=True,
            return_intermediate_token_ids=True,
            tgt_lang=tgt_speech_other,
            speaker_id=0,
            **gen_common,
        )
        if not isinstance(s2st_out, TTSeamlessM4Tv2GenerationOutput):
            raise TypeError(f"S2ST expected TTSeamlessM4Tv2GenerationOutput, got {type(s2st_out)}")
        print(f"  Intermediate text ({tgt_speech_other}): {_decode(tokenizer, s2st_out.sequences)}")
        spanish_wav_np = _waveform_to_mono_fp32(s2st_out.waveform, s2st_out.waveform_lengths)
        _save_wav(S2ST_WAV, spanish_wav_np, sample_rate=sample_rate)
        print(f"  Output audio ({tgt_speech_other}, {sample_rate} Hz, {spanish_wav_np.size} samples)")
        print(f"  Saved to: {S2ST_WAV}")

        # =========================================================================
        # 5. ASR — Automatic Speech Recognition (Hindi speech → Hindi text)
        # =========================================================================
        _print_header(5, "Automatic Speech Recognition", "ASR", tgt_translate, tgt_asr)
        print(f"  Input audio ({tgt_translate}): {T2ST_WAV} ({sample_rate} Hz)")
        asr_out = tt_model.generate(
            input_features=torch_feats_to_ttnn(device, input_features),
            attention_mask=torch_ids_to_ttnn(device, input_speech_attn),
            generate_speech=False,
            tgt_lang=tgt_asr,
            **gen_common,
        )
        if not isinstance(asr_out, TTSeamlessM4Tv2GreedySearchOutput):
            raise TypeError(f"ASR expected TTSeamlessM4Tv2GreedySearchOutput, got {type(asr_out)}")
        print(f"  Output text ({tgt_asr}): {_decode(tokenizer, asr_out.sequences)}")

        print()
        print("=" * 78)
        print("  ok — all five tasks completed")
        print("=" * 78)
        print(f"  Audio outputs saved under: {OUTPUT_DIR}")

    finally:
        if original_default is not None:
            ttnn.SetDefaultDevice(original_default)
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
