#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Profiler harness for the SeamlessM4T-v2 T2ST (text-to-speech) pipeline.

Goal: characterize the TTNN T2ST pipeline (text encoder + AR text decoder
+ T2U encoder/decoder + code HiFi-GAN vocoder) under tracy so we can
identify the hottest ops and decide where to apply targeted optimization.

Unlike T2TT/S2TT (where the AR text decoder dominates and the
``--traced`` win lives there), T2ST adds two substantial NAR stages
*after* the AR text decoder:

    1. T2U generator (encoder + NAR decoder with duration upsample) --
       a one-shot forward pass over the AR text output.
    2. Code HiFi-GAN vocoder -- ConvTranspose1d stack that turns unit
       embeddings into a waveform.

Both run once per ``synthesize(...)`` call (no AR loop inside), so the
right primary metric here is **end-to-end wall-clock per synthesize**,
not per-decode-step.

Usage (no tracy, just wall-clock)::

    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && \\
        source python_env/bin/activate && export ARCH_NAME=blackhole
    python models/demos/facebook_seamless_m4t_v2_large/tt/profile_t2st.py

Usage (under tracy)::

    python -m tracy -p -v -r --op-support-count 3000 \\
        -n seamless_m4t_t2st \\
        models/demos/facebook_seamless_m4t_v2_large/tt/profile_t2st.py

The script runs 1 warmup ``synthesize`` + N timed ``synthesize`` calls
and prints per-stage timings:

    encoder_ms                text encoder forward
    ar_text_ms                AR text decoder full generate (+ LM head argmax)
    hf_rerun_ms               host HF text_decoder rerun for last_hidden_state
    char_prep_ms              host HF char-input prep
    t2u_ms                    T2U encoder+decoder+LM-head+argmax (TTNN)
    vocoder_ms                code HiFi-GAN vocoder forward (TTNN)
    total_ms                  end-to-end synthesize wall-clock

The decode path under T2ST is identical to T2TT, so the ``--traced``
flag wires the same Phase 9c metal-trace replay through the AR text
decoder. T2U + vocoder are not traced (they are one-shot per call).
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import ttnn

DEFAULT_SRC = "The quick brown fox jumps over the lazy dog and then runs through the forest."
DEFAULT_SRC_LANG = "eng"
DEFAULT_TGT_LANG = "fra"

# Default profiling budget for the AR text-decoder. Includes prefix tokens.
# Use a deliberately generous budget so the single-call trace pattern has
# enough steady-state steps (>= ~28) to amortise its recapture cost.
DEFAULT_MAX_NEW_TOKENS = 48

SAMPLING_RATE = 16000


# ---------------------------------------------------------------------------
# Per-stage instrumented synthesize, mirroring TextToSpeechModel.synthesize
# but inserting ttnn.synchronize_device boundaries between every stage so
# we can attribute time on-device-perceived rather than dispatch enqueue.
# ---------------------------------------------------------------------------


def _run_one_synthesize(
    model,
    src: str,
    src_lang: str,
    tgt_lang: str,
    max_new_tokens: int,
    use_trace: bool,
    timings: Dict[str, float],
    waveform_ref: List[Optional[np.ndarray]],
) -> Tuple[np.ndarray, int]:
    """Run one TTNN synthesize and time it per-stage.

    Reimplements the logic of ``TextToSpeechModel.synthesize`` so we can
    drop ``ttnn.synchronize_device`` between stages without changing the
    model. Numerical results are bit-identical to ``synthesize()``.
    """
    from models.demos.facebook_seamless_m4t_v2_large.tt.text_to_speech_model import _tile_pad_encoder_hidden

    # --------------------------------------------------------------- 1. tokenize
    toks = model.processor(text=src, src_lang=src_lang, return_tensors="pt")
    input_ids: torch.Tensor = toks["input_ids"]
    attn_mask: torch.Tensor = toks["attention_mask"]
    src_logical = int(attn_mask.sum().item())
    input_ids = input_ids[:, :src_logical]
    attn_mask = attn_mask[:, :src_logical]

    # ---------------------------------------------------------- 2. text encoder
    t0 = time.perf_counter()
    enc_hidden_logical = model._run_text_encoder(input_ids, attn_mask)
    ttnn.synchronize_device(model.device)
    timings["encoder_ms"] = (time.perf_counter() - t0) * 1000.0

    enc_hidden_padded = _tile_pad_encoder_hidden(enc_hidden_logical[:, :src_logical, :])
    encoder_seq_len_padded = int(enc_hidden_padded.shape[1])

    # ----------------------------------------- 3. AR text decoder + LM head
    gen = model._get_or_build_text_generator(encoder_seq_len_padded)
    text_tgt_lang_id = model._resolve_text_tgt_lang_id(tgt_lang)
    max_total = min(int(max_new_tokens), model.max_decode_seq_len)

    t1 = time.perf_counter()
    text_tokens = gen.generate(
        encoder_hidden_states=enc_hidden_padded,
        encoder_attention_mask=attn_mask,
        decoder_start_token_id=model.decoder_start_token_id,
        tgt_lang_id=text_tgt_lang_id,
        eos_token_id=model.eos_token_id,
        max_new_tokens=max_total,
        do_sample=False,
        use_trace=use_trace,
    )
    ttnn.synchronize_device(model.device)
    timings["ar_text_ms"] = (time.perf_counter() - t1) * 1000.0
    tokens_generated = int(len(text_tokens)) if hasattr(text_tokens, "__len__") else 0

    # CRITICAL: release the AR trace BEFORE T2U+vocoder allocate fresh
    # device buffers. This is the single-call trace pattern -- the next
    # synthesize() call re-captures. Without this, the post-AR
    # allocations either warn or corrupt under trace replay.
    if use_trace:
        t_rel = time.perf_counter()
        gen.release_trace()
        ttnn.synchronize_device(model.device)
        timings["release_ms"] = (time.perf_counter() - t_rel) * 1000.0
    else:
        timings["release_ms"] = 0.0

    if isinstance(text_tokens, torch.Tensor):
        seq = text_tokens.to(torch.int64).view(1, -1)
    else:
        seq = torch.tensor(text_tokens, dtype=torch.int64).view(1, -1)
    if int(seq[0, -1].item()) != model.eos_token_id:
        seq = torch.cat([seq, torch.tensor([[model.eos_token_id]], dtype=torch.int64)], dim=1)

    # ---------------------------------------- 4. HF text_decoder rerun (host)
    hf_model = model._load_hf_helper_model()
    t2 = time.perf_counter()
    encoder_attention_mask = attn_mask
    text_dec_out = hf_model.text_decoder(
        input_ids=seq[:, :-1],
        encoder_hidden_states=enc_hidden_logical.to(torch.float32)[:, :src_logical, :],
        encoder_attention_mask=encoder_attention_mask,
    )
    t2u_input_embeds = text_dec_out.last_hidden_state  # [1, T_text, H]
    T_text = int(t2u_input_embeds.shape[1])
    timings["hf_rerun_ms"] = (time.perf_counter() - t2) * 1000.0

    # ---------------------------------------- 5. Char input prep (host)
    t3 = time.perf_counter()
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
    timings["char_prep_ms"] = (time.perf_counter() - t3) * 1000.0

    # ---------------------------------------- 6. T2U generator (TTNN)
    t4 = time.perf_counter()
    t2u_out = model.t2u_generator.synthesize_units(
        text_decoder_hidden=t2u_input_embeds,
        char_input_ids=t2u_char_input_ids,
        char_count_per_id=t2u_char_count_per_id,
        t2u_attention_mask=t2u_attention_mask,
    )
    ttnn.synchronize_device(model.device)
    timings["t2u_ms"] = (time.perf_counter() - t4) * 1000.0
    unit_token_ids: torch.Tensor = t2u_out["unit_token_ids"]

    # ---------------------------------------- 7. Code HiFi-GAN vocoder (TTNN)
    vocoder_lang_id = model._resolve_vocoder_lang_id(tgt_lang)
    speaker_t = torch.tensor([[0]] * unit_token_ids.shape[0], dtype=torch.int64)
    lang_t = torch.tensor([[vocoder_lang_id]] * unit_token_ids.shape[0], dtype=torch.int64)
    t5 = time.perf_counter()
    waveform_tt = model.vocoder(
        input_ids=unit_token_ids,
        speaker_id=speaker_t,
        lang_id=lang_t,
    )
    waveform_torch = ttnn.to_torch(waveform_tt).to(torch.float32)
    ttnn.deallocate(waveform_tt)
    ttnn.synchronize_device(model.device)
    timings["vocoder_ms"] = (time.perf_counter() - t5) * 1000.0

    # Trim to valid sample count
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
        "--src",
        type=str,
        default=DEFAULT_SRC,
        help="Source text to synthesize.",
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
        "--traced",
        action="store_true",
        help=(
            "Enable Phase 9c metal-trace replay on the AR text decoder. "
            "T2U and vocoder are one-shot per call and are NOT traced."
        ),
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
        default=3,
        help=(
            "Number of timed synthesize() calls AFTER the 1-call warmup. "
            "Reported numbers are medians across these calls."
        ),
    )
    args = parser.parse_args()

    if args.traced:
        print("[profile_t2st] --traced ENABLED: AR text-decoder steps run " "under metal trace replay.")

    from transformers import AutoProcessor

    from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl
    from models.demos.facebook_seamless_m4t_v2_large.tt.text_to_speech_model import TextToSpeechModel

    print(f"[profile_t2st] loading HF state dict ...")
    hf_sd = wl.load_hf_state_dict()
    processor = AutoProcessor.from_pretrained(wl.HF_PATH)

    print(f"[profile_t2st] opening device {args.device_id} ...")
    device = ttnn.open_device(
        device_id=args.device_id,
        # When --traced, allocations from the AR-loop's captured trace can
        # leave less L1_SMALL headroom for the post-AR vocoder Conv1d
        # halos. Give a bigger L1_SMALL pool in traced mode so the
        # vocoder doesn't OOM on fragmentation.
        l1_small_size=65536 if args.traced else 32768,
        trace_region_size=256_000_000 if args.traced else 0,
    )

    try:
        print(f"[profile_t2st] building TextToSpeechModel ...")
        t_build0 = time.perf_counter()
        model = TextToSpeechModel(device=device, hf_state_dict=hf_sd, processor=processor)
        t_build = time.perf_counter() - t_build0
        print(f"[profile_t2st] model build = {t_build*1000:.0f} ms")

        # ------- 1. WARMUP -------
        print(
            f"[profile_t2st] WARMUP synthesize: src={args.src!r} " f"src_lang={args.src_lang} tgt_lang={args.tgt_lang}"
        )
        warmup_timings: Dict[str, float] = {}
        warmup_audio: List[Optional[np.ndarray]] = [None]
        t_w0 = time.perf_counter()
        _, warmup_tokens = _run_one_synthesize(
            model,
            src=args.src,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            max_new_tokens=args.max_new_tokens,
            use_trace=args.traced,
            timings=warmup_timings,
            waveform_ref=warmup_audio,
        )
        warmup_total_ms = (time.perf_counter() - t_w0) * 1000.0
        print(f"[profile_t2st] WARMUP tokens={warmup_tokens}, audio samples={warmup_audio[0].shape[-1]}")
        print(f"[profile_t2st] WARMUP total_ms={warmup_total_ms:.1f}")
        print(f"[profile_t2st] WARMUP stage breakdown: " + ", ".join(f"{k}={v:.1f}" for k, v in warmup_timings.items()))

        # ------- 2. TIMED -------
        all_timings: List[Dict[str, float]] = []
        all_totals: List[float] = []
        all_audio_samples: List[int] = []
        all_tokens: List[int] = []
        for i in range(args.num_timed):
            t: Dict[str, float] = {}
            audio_ref: List[Optional[np.ndarray]] = [None]
            print(f"[profile_t2st] TIMED[{i+1}/{args.num_timed}] synthesize ...")
            t_t0 = time.perf_counter()
            _, tokens = _run_one_synthesize(
                model,
                src=args.src,
                src_lang=args.src_lang,
                tgt_lang=args.tgt_lang,
                max_new_tokens=args.max_new_tokens,
                use_trace=args.traced,
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
            "encoder_ms",
            "ar_text_ms",
            "release_ms",
            "hf_rerun_ms",
            "char_prep_ms",
            "t2u_ms",
            "vocoder_ms",
        ]
        per_key: Dict[str, List[float]] = {k: [d.get(k, float("nan")) for d in all_timings] for k in keys}

        print("")
        print("=" * 64)
        print("[profile_t2st] RESULTS")
        print("=" * 64)
        print(f"src:             {args.src!r} ({args.src_lang} -> {args.tgt_lang})")
        print(f"max_new_tokens:  {args.max_new_tokens}")
        print(f"warmup_total_ms: {warmup_total_ms:.1f}")
        print(f"warmup_tokens:   {warmup_tokens}")
        print(f"warmup_samples:  {warmup_audio[0].shape[-1]} (~{warmup_audio[0].shape[-1] / SAMPLING_RATE:.3f} s)")
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
        # Per-step ms: AR generate() runs (tokens_generated - 1) forward
        # passes (positions 0..N-2). Position 0's logits are discarded;
        # positions 1..N-2 each produce one sampled token.
        ar_p50 = _med(per_key["ar_text_ms"])
        step_count = max(1, (all_tokens[0] if all_tokens else 1) - 1)
        per_step_ms = ar_p50 / float(step_count) if step_count > 0 else float("nan")
        print(
            f"[profile_t2st] SUMMARY "
            + " ".join(f"{k}={_med(per_key[k]):.2f}" for k in keys)
            + f" total_ms={total_p50:.2f}"
            + f" ar_step_ms={per_step_ms:.2f}"
            + f" tokens_generated={all_tokens[0] if all_tokens else 0}"
            + f" (steps={step_count})"
        )

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
