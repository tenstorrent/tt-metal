#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Profiler harness for the SeamlessM4T-v2 S2ST (speech-to-speech) pipeline.

Goal: characterize the TTNN S2ST pipeline (speech encoder + AR text
decoder + T2U NAR generator + code HiFi-GAN vocoder) under tracy so we
can identify the hottest ops and decide where to apply targeted
optimization.

This pipeline is essentially the S2TT path (speech encoder + Conformer +
adapter + NLLB decoder + AR loop) composed with the T2ST tail (T2U +
vocoder NAR stages). The TTNN model file is
``tt/speech_to_speech_model.py``. Per-call structure:

    1. processor.feature_extractor   (host: 80-mel + stride-2 stacking)
    2. speech_encoder                (TTNN, one-shot, ~57 ms on p150a)
    3. AR text decoder + LM head     (TTNN, ~17 ms/step traced)
    4. text_decoder rerun            (host HF fp32, ~138 ms)
    5. char input prep               (host HF tokenizer helpers)
    6. T2U encoder+decoder+LM-head   (TTNN, one-shot)
    7. code HiFi-GAN vocoder         (TTNN, one-shot)

Just like T2ST, T2U + vocoder are NAR one-shot stages, so the right
primary metric is **end-to-end wall-clock per synthesize**, not
per-decode-step. Cross-call AR trace reuse is BLOCKED by the post-AR
NAR stage allocations (documented in PERF_NOTES.md::T2ST), so this
harness defaults to the untraced production path.

Usage (no tracy, just wall-clock)::

    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && \\
        source python_env/bin/activate && export ARCH_NAME=blackhole
    python models/demos/facebook_seamless_m4t_v2_large/tt/profile_s2st.py

Usage (under tracy)::

    python -m tracy -p -v -r --op-support-count 3000 \\
        -n seamless_m4t_s2st \\
        models/demos/facebook_seamless_m4t_v2_large/tt/profile_s2st.py

The script runs 1 warmup ``synthesize`` + N timed ``synthesize`` calls
and prints per-stage timings:

    feature_extractor_ms      host: load wav + 80-mel + stride-2 stack
    speech_encoder_ms         TTNN speech encoder forward (Conformer + adapter)
    ar_text_ms                AR text decoder full generate (+ LM head argmax)
    hf_rerun_ms               host HF text_decoder rerun (cross-attn over sub-sampled speech mask)
    char_prep_ms              host HF char-input prep
    t2u_ms                    T2U encoder+decoder+LM-head+argmax (TTNN)
    vocoder_ms                code HiFi-GAN vocoder forward (TTNN)
    total_ms                  end-to-end synthesize wall-clock
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import ttnn

INPUTS_DIR = Path(__file__).resolve().parents[1] / "demo" / "inputs"

# Default profiling budget for the AR text-decoder. Includes prefix tokens.
DEFAULT_MAX_NEW_TOKENS = 32

# Default WAV sample to profile. Short hello-world style.
DEFAULT_SAMPLE = "sample_hello.wav"
DEFAULT_SRC_LANG = "eng"
DEFAULT_TGT_LANG = "fra"
DEFAULT_MAX_AUDIO_SECONDS = 5.0

SAMPLING_RATE = 16000

_TILE = 32


def _pad_to_tile(n: int, tile: int = _TILE) -> int:
    return (tile - n % tile) % tile


# ---------------------------------------------------------------------------
# Per-stage instrumented synthesize, mirroring SpeechToSpeechModel.synthesize
# but inserting ttnn.synchronize_device boundaries between every stage so we
# can attribute time on-device-perceived rather than dispatch enqueue.
# ---------------------------------------------------------------------------


def _run_one_synthesize(
    model,
    audio_path: str,
    src_lang: str,
    tgt_lang: str,
    max_new_tokens: int,
    max_audio_seconds: float,
    timings: Dict[str, float],
    waveform_ref: List[Optional[np.ndarray]],
) -> Tuple[np.ndarray, int]:
    """Run one TTNN synthesize and time it per-stage.

    Reimplements ``SpeechToSpeechModel.synthesize`` so we can drop
    ``ttnn.synchronize_device`` between stages without changing the
    model. Numerical results are bit-identical to ``synthesize()``.
    """
    from models.demos.facebook_seamless_m4t_v2_large.tt.speech_to_speech_model import EMBED_DIM
    from models.demos.facebook_seamless_m4t_v2_large.tt.speech_to_text_model import (
        _extract_features,
        _load_wav_to_16k_mono,
    )

    # ----------------------------------------- 1. feature extractor (host)
    t0 = time.perf_counter()
    audio = _load_wav_to_16k_mono(audio_path)
    input_features, attention_mask_2d, _logical_len = _extract_features(
        audio,
        processor=model.processor,
        target_seq_len=model.audio_seq_len,
        max_audio_seconds=max_audio_seconds,
    )
    timings["feature_extractor_ms"] = (time.perf_counter() - t0) * 1000.0

    # ----------------------------------------- 2. speech encoder (TTNN)
    t1 = time.perf_counter()
    feat_tt = model._to_tt_features(input_features)
    enc_hidden_tt = model.speech_encoder(feat_tt, attention_mask_2d=attention_mask_2d)
    enc_hidden_torch = ttnn.to_torch(enc_hidden_tt).to(torch.float32)
    ttnn.deallocate(enc_hidden_tt)
    while enc_hidden_torch.dim() > 3:
        enc_hidden_torch = enc_hidden_torch.squeeze(0)
    enc_hidden_torch = enc_hidden_torch.reshape(1, model.sub_seq_len, EMBED_DIM)
    ttnn.synchronize_device(model.device)
    timings["speech_encoder_ms"] = (time.perf_counter() - t1) * 1000.0

    # Build post-adapter cross-attention mask + tile-pad enc hidden.
    sub_mask_2d, logical_sub = model._post_adapter_attention_mask(attention_mask_2d)
    pad_needed = _pad_to_tile(model.sub_seq_len)
    if pad_needed > 0:
        zeros = torch.zeros((1, pad_needed, EMBED_DIM), dtype=enc_hidden_torch.dtype)
        enc_hidden_padded = torch.cat([enc_hidden_torch, zeros], dim=1)
        mask_pad = torch.zeros((1, pad_needed), dtype=sub_mask_2d.dtype)
        sub_mask_2d_padded = torch.cat([sub_mask_2d, mask_pad], dim=1)
    else:
        enc_hidden_padded = enc_hidden_torch
        sub_mask_2d_padded = sub_mask_2d

    # ----------------------------------------- 3. AR text decoder + LM head
    text_tgt_lang_id = model._resolve_text_tgt_lang_id(tgt_lang)
    max_total = min(int(max_new_tokens), model.max_decode_seq_len)

    t2 = time.perf_counter()
    text_tokens = model.text_generator.generate(
        encoder_hidden_states=enc_hidden_padded,
        encoder_attention_mask=sub_mask_2d_padded,
        decoder_start_token_id=model.decoder_start_token_id,
        tgt_lang_id=text_tgt_lang_id,
        eos_token_id=model.eos_token_id,
        max_new_tokens=max_total,
        do_sample=False,
    )
    ttnn.synchronize_device(model.device)
    timings["ar_text_ms"] = (time.perf_counter() - t2) * 1000.0
    tokens_generated = int(len(text_tokens)) if hasattr(text_tokens, "__len__") else 0

    if isinstance(text_tokens, torch.Tensor):
        seq = text_tokens.to(torch.int64).view(1, -1)
    else:
        seq = torch.tensor(text_tokens, dtype=torch.int64).view(1, -1)
    if int(seq[0, -1].item()) != model.eos_token_id:
        seq = torch.cat([seq, torch.tensor([[model.eos_token_id]], dtype=torch.int64)], dim=1)

    # ----------------------------------------- 4. HF text_decoder rerun (host)
    hf_model = model._load_hf_helper_model()
    enc_hidden_logical = enc_hidden_torch[:, :logical_sub, :].to(torch.float32)
    encoder_attention_mask_logical = sub_mask_2d[:, :logical_sub].to(torch.long)

    t3 = time.perf_counter()
    text_dec_out = hf_model.text_decoder(
        input_ids=seq[:, :-1],
        encoder_hidden_states=enc_hidden_logical,
        encoder_attention_mask=encoder_attention_mask_logical,
    )
    t2u_input_embeds = text_dec_out.last_hidden_state  # [1, T_text, H]
    T_text = int(t2u_input_embeds.shape[1])
    timings["hf_rerun_ms"] = (time.perf_counter() - t3) * 1000.0

    # ----------------------------------------- 5. Char input prep (host)
    t4 = time.perf_counter()
    seq_lens = (seq[:, :-1] != model.pad_token_id).int().sum(1)
    t2u_attention_mask = model._compute_new_attention_mask(seq_lens, batch=1, mask_seq_len=T_text)
    t2u_input_ids = seq[:, 2:-1].clone()
    t2u_input_ids = torch.masked_fill(t2u_input_ids, t2u_input_ids == model.eos_token_id, model.pad_token_id)
    t2u_subwords = hf_model._indices_to_subwords(t2u_input_ids)
    t2u_char_count_per_id = hf_model._count_character_length_in_subword(
        t2u_input_ids, t2u_subwords, pad_token_id=model.pad_token_id
    )
    pad_zero = t2u_char_count_per_id.new_zeros((t2u_char_count_per_id.shape[0], 1))
    t2u_char_count_per_id = torch.cat([pad_zero, t2u_char_count_per_id, pad_zero], dim=1)
    t2u_char_input_ids = hf_model._get_char_input_ids(
        t2u_input_ids,
        t2u_subwords,
        t2u_char_count_per_id,
        pad_token_id=model.pad_token_id,
    )
    timings["char_prep_ms"] = (time.perf_counter() - t4) * 1000.0

    # ----------------------------------------- 6. T2U generator (TTNN)
    t5 = time.perf_counter()
    t2u_out = model.t2u_generator.synthesize_units(
        text_decoder_hidden=t2u_input_embeds,
        char_input_ids=t2u_char_input_ids,
        char_count_per_id=t2u_char_count_per_id,
        t2u_attention_mask=t2u_attention_mask,
    )
    ttnn.synchronize_device(model.device)
    timings["t2u_ms"] = (time.perf_counter() - t5) * 1000.0
    unit_token_ids: torch.Tensor = t2u_out["unit_token_ids"]

    # ----------------------------------------- 7. Code HiFi-GAN vocoder (TTNN)
    vocoder_lang_id = model._resolve_vocoder_lang_id(tgt_lang)
    speaker_t = torch.tensor([[0]] * unit_token_ids.shape[0], dtype=torch.int64)
    lang_t = torch.tensor([[vocoder_lang_id]] * unit_token_ids.shape[0], dtype=torch.int64)
    t6 = time.perf_counter()
    waveform_tt = model.vocoder(
        input_ids=unit_token_ids,
        speaker_id=speaker_t,
        lang_id=lang_t,
    )
    waveform_torch = ttnn.to_torch(waveform_tt).to(torch.float32)
    ttnn.deallocate(waveform_tt)
    ttnn.synchronize_device(model.device)
    timings["vocoder_ms"] = (time.perf_counter() - t6) * 1000.0

    # Trim to valid sample count.
    while waveform_torch.dim() > 1 and waveform_torch.shape[0] == 1:
        waveform_torch = waveform_torch.squeeze(0)
    last_lengths = model.vocoder.last_lengths
    if last_lengths is not None:
        try:
            valid = int(last_lengths.item() if last_lengths.dim() == 0 else last_lengths.view(-1)[0].item())
            valid = max(0, min(valid, int(waveform_torch.shape[-1])))
            waveform_torch = waveform_torch[..., :valid].contiguous()
        except Exception:
            pass
    audio_np = waveform_torch.detach().cpu().numpy().astype(np.float32)
    waveform_ref[0] = audio_np
    return audio_np, tokens_generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help=(f"Path to a 16 kHz mono WAV. Default: " f"{INPUTS_DIR}/{DEFAULT_SAMPLE}."),
    )
    parser.add_argument("--src-lang", type=str, default=DEFAULT_SRC_LANG)
    parser.add_argument("--tgt-lang", type=str, default=DEFAULT_TGT_LANG)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="AR text-decoder generation budget (includes 2-token prefix).",
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=DEFAULT_MAX_AUDIO_SECONDS,
        help="Cap on the input audio length (seconds). Defaults to 5.",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="ttnn device id to open (default 0).",
    )
    parser.add_argument(
        "--num-timed",
        type=int,
        default=1,
        help=(
            "Number of timed synthesize() calls AFTER the 1-call warmup. "
            "Reported numbers are medians across these calls."
        ),
    )
    args = parser.parse_args()

    from transformers import AutoProcessor

    from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl
    from models.demos.facebook_seamless_m4t_v2_large.tt.speech_to_speech_model import SpeechToSpeechModel

    audio_path = args.audio if args.audio else str(INPUTS_DIR / DEFAULT_SAMPLE)
    if not Path(audio_path).is_file():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    print(f"[profile_s2st] loading HF state dict ...")
    hf_sd = wl.load_hf_state_dict()
    processor = AutoProcessor.from_pretrained(wl.HF_PATH)

    print(f"[profile_s2st] opening device {args.device_id} ...")
    device = ttnn.open_device(
        device_id=args.device_id,
        l1_small_size=32768,
    )

    try:
        print(f"[profile_s2st] building SpeechToSpeechModel ...")
        t_build0 = time.perf_counter()
        model = SpeechToSpeechModel(device=device, hf_state_dict=hf_sd, processor=processor)
        t_build = time.perf_counter() - t_build0
        print(f"[profile_s2st] model build = {t_build*1000:.0f} ms")

        # ------- 1. WARMUP -------
        print(
            f"[profile_s2st] WARMUP synthesize: audio={audio_path!r} "
            f"src_lang={args.src_lang} tgt_lang={args.tgt_lang}"
        )
        warmup_timings: Dict[str, float] = {}
        warmup_audio: List[Optional[np.ndarray]] = [None]
        t_w0 = time.perf_counter()
        _, warmup_tokens = _run_one_synthesize(
            model,
            audio_path=audio_path,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            max_new_tokens=args.max_new_tokens,
            max_audio_seconds=args.max_audio_seconds,
            timings=warmup_timings,
            waveform_ref=warmup_audio,
        )
        warmup_total_ms = (time.perf_counter() - t_w0) * 1000.0
        print(
            f"[profile_s2st] WARMUP tokens={warmup_tokens}, "
            f"audio samples={warmup_audio[0].shape[-1] if warmup_audio[0] is not None else 0}"
        )
        print(f"[profile_s2st] WARMUP total_ms={warmup_total_ms:.1f}")
        print(f"[profile_s2st] WARMUP stage breakdown: " + ", ".join(f"{k}={v:.1f}" for k, v in warmup_timings.items()))

        # ------- 2. TIMED -------
        all_timings: List[Dict[str, float]] = []
        all_totals: List[float] = []
        all_audio_samples: List[int] = []
        all_tokens: List[int] = []
        for i in range(args.num_timed):
            t: Dict[str, float] = {}
            audio_ref: List[Optional[np.ndarray]] = [None]
            print(f"[profile_s2st] TIMED[{i+1}/{args.num_timed}] synthesize ...")
            t_t0 = time.perf_counter()
            _, tokens = _run_one_synthesize(
                model,
                audio_path=audio_path,
                src_lang=args.src_lang,
                tgt_lang=args.tgt_lang,
                max_new_tokens=args.max_new_tokens,
                max_audio_seconds=args.max_audio_seconds,
                timings=t,
                waveform_ref=audio_ref,
            )
            total_ms = (time.perf_counter() - t_t0) * 1000.0
            all_timings.append(t)
            all_totals.append(total_ms)
            all_tokens.append(tokens)
            all_audio_samples.append(int(audio_ref[0].shape[-1]) if audio_ref[0] is not None else 0)

        # ------- 3. Report -------
        def _stat(name: str, xs: List[float]) -> str:
            if not xs:
                return f"{name}=<empty>"
            return (
                f"{name}: n={len(xs)}, min={min(xs):.2f}, "
                f"p50={statistics.median(xs):.2f}, mean={statistics.mean(xs):.2f}, "
                f"max={max(xs):.2f}"
            )

        keys = [
            "feature_extractor_ms",
            "speech_encoder_ms",
            "ar_text_ms",
            "hf_rerun_ms",
            "char_prep_ms",
            "t2u_ms",
            "vocoder_ms",
        ]
        per_key: Dict[str, List[float]] = {k: [d.get(k, float("nan")) for d in all_timings] for k in keys}

        print("")
        print("=" * 64)
        print("[profile_s2st] RESULTS")
        print("=" * 64)
        print(f"audio:           {audio_path!r} ({args.src_lang} -> {args.tgt_lang})")
        print(f"max_new_tokens:  {args.max_new_tokens}")
        print(f"warmup_total_ms: {warmup_total_ms:.1f}")
        print(f"warmup_tokens:   {warmup_tokens}")
        if warmup_audio[0] is not None:
            print(
                f"warmup_samples:  {warmup_audio[0].shape[-1]} " f"(~{warmup_audio[0].shape[-1] / SAMPLING_RATE:.3f} s)"
            )
        print("")
        for k in keys:
            print(_stat(k, per_key[k]))
        print(_stat("total_ms                ", all_totals))
        print(f"tokens_generated_per_call: {all_tokens}")
        print(f"audio_samples_per_call:    {all_audio_samples}")
        print("=" * 64)

        # Final single-line summary.
        def _med(xs: List[float]) -> float:
            return statistics.median(xs) if xs else float("nan")

        total_p50 = _med(all_totals)
        print(
            f"[profile_s2st] SUMMARY "
            + " ".join(f"{k}={_med(per_key[k]):.2f}" for k in keys)
            + f" total_ms={total_p50:.2f}"
            + f" tokens_generated={all_tokens[0] if all_tokens else 0}"
        )

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
