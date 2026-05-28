#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Profiler harness for the SeamlessM4T-v2 S2TT (speech-to-text) pipeline.

Goal: characterize the TTNN S2TT pipeline (speech encoder + Conformer +
adapter + NLLB decoder + AR loop) under tracy so we can identify the
hottest ops and decide where to apply targeted optimization.

The decode path reuses the SAME ``TextGenerator`` used by T2TT, so the
metal-trace + ``paged_update_cache`` machinery from Phase 9c is also
exercised here when ``--traced`` is passed. The structural difference
versus T2TT is the prefill: a 256-step audio-feature SpeechEncoder
(24 Conformer layers + 1 adapter) instead of a small text-encoder
forward.

Usage (no tracy, just wall-clock)::

    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && \\
        source python_env/bin/activate && export ARCH_NAME=blackhole
    python models/demos/facebook_seamless_m4t_v2_large/tt/profile_s2tt.py

Usage (under tracy)::

    python -m tracy -p -v -r --no-device-data-capture \\
        -n seamless_m4t_s2tt \\
        models/demos/facebook_seamless_m4t_v2_large/tt/profile_s2tt.py

The script runs 1 warmup ``translate`` + N timed ``translate`` calls and
prints:

    prefill_ms              speech_encoder + cross-attn cache populate
    warmup_total_ms         total wall-clock of the warmup call
    steady_decode_step_ms   median per-step time of the timed call (AR loop)
    timed_total_ms          total wall-clock of the timed call
    tokens_generated        # of decode_step invocations (including 2 prefix)

The ``--traced`` flag enables single-trace replay of the AR decode step
(Phase 9c: ``paged_update_cache(update_idxs_tensor=cur_pos_tt)`` + all
per-step inputs through persistent buffers). Cross-call trace reuse is
on by default when ``--traced`` is set.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

import ttnn

INPUTS_DIR = Path(__file__).resolve().parents[1] / "demo" / "inputs"

# Default profiling budget. Includes the 2 prefix tokens.
DEFAULT_MAX_NEW_TOKENS = 32

# Default WAV sample to profile. Hello-world style, short enough that the
# AR loop dominates the wall-clock budget rather than the encoder.
DEFAULT_SAMPLE = "sample_hello.wav"
DEFAULT_SRC_LANG = "eng"
DEFAULT_TGT_LANG = "fra"
DEFAULT_MAX_AUDIO_SECONDS = 5.0


def _pad_to_tile(n: int, tile: int = 32) -> int:
    return (tile - n % tile) % tile


def _run_one_translate(
    model,
    audio_path: str,
    src_lang: str,
    tgt_lang: str,
    max_new_tokens: int,
    max_audio_seconds: float,
    per_step_log: Optional[List] = None,
    prefill_log: Optional[List[float]] = None,
    use_trace: bool = False,
) -> Tuple[str, int]:
    """Run one TTNN translate, timing prefill + each AR step.

    Per-step timing is performed by either monkey-patching the
    generator's ``decode_step`` method (untraced path) or by attaching a
    ``step_callback`` hook on the generator (traced path). Both paths
    insert ``ttnn.synchronize_device`` boundaries so we measure
    host-perceived latency rather than dispatch enqueue.
    """
    from models.demos.facebook_seamless_m4t_v2_large.tt.speech_to_text_model import (
        EMBED_DIM,
        _extract_features,
        _load_wav_to_16k_mono,
    )

    # ---------- 1. Host-side audio I/O + feature extraction ----------
    audio = _load_wav_to_16k_mono(audio_path)
    input_features, attention_mask_2d, logical_len = _extract_features(
        audio,
        processor=model.processor,
        target_seq_len=model.audio_seq_len,
        max_audio_seconds=max_audio_seconds,
    )

    # ---------- 2. Speech encoder forward (TTNN) ----------
    t0 = time.perf_counter()
    feat_tt = model._to_tt_features(input_features)
    enc_hidden_tt = model.speech_encoder(feat_tt, attention_mask_2d=attention_mask_2d)
    enc_hidden_torch = ttnn.to_torch(enc_hidden_tt).to(torch.float32)
    ttnn.deallocate(enc_hidden_tt)
    while enc_hidden_torch.dim() > 3:
        enc_hidden_torch = enc_hidden_torch.squeeze(0)
    enc_hidden_torch = enc_hidden_torch.reshape(1, model.sub_seq_len, EMBED_DIM)
    ttnn.synchronize_device(model.device)
    t_enc = time.perf_counter() - t0

    # ---------- 3. Post-adapter cross-attention mask + tile pad ----------
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

    # ---------- 4. Cross-attn cache populate (prefill) ----------
    gen = model.generator
    t1 = time.perf_counter()
    gen.past_key_values.self_attn.reset()
    src_logical_pad = gen.populate_encoder_cache(enc_hidden_padded, encoder_attention_mask=sub_mask_2d_padded)
    ttnn.synchronize_device(model.device)
    t_xpop = time.perf_counter() - t1

    prefill_ms = (t_enc + t_xpop) * 1000.0
    if prefill_log is not None:
        prefill_log.append(prefill_ms)

    # ---------- 5. AR loop ----------
    tgt_lang_id = model._resolve_tgt_lang_id(tgt_lang)
    max_total = min(int(max_new_tokens), model.max_decode_seq_len)

    # Wrap decode_step to time each call (sync inside).
    original_decode_step = gen.decode_step
    step_count = [0]

    def timed_decode_step(*args, **kwargs):
        t_a = time.perf_counter()
        out = original_decode_step(*args, **kwargs)
        ttnn.synchronize_device(model.device)
        t_b = time.perf_counter()
        if per_step_log is not None:
            per_step_log.append((t_b - t_a) * 1000.0)
        step_count[0] += 1
        return out

    gen.decode_step = timed_decode_step

    # When use_trace=True, the traced AR loop bypasses decode_step. Use
    # the step_callback hook on the generator to record per-step latency
    # (kind in {"warmup", "capture", "replay"}). Only "replay" entries
    # represent steady-state cost.
    original_callback = gen.step_callback

    def trace_callback(position: int, ms: float, kind: str) -> None:
        if per_step_log is not None:
            per_step_log.append((ms, kind, position))
        step_count[0] += 1

    if use_trace:
        gen.step_callback = trace_callback
    try:
        tokens = gen.generate(
            encoder_hidden_states=enc_hidden_padded,
            encoder_attention_mask=sub_mask_2d_padded,
            decoder_start_token_id=model.decoder_start_token_id,
            tgt_lang_id=tgt_lang_id,
            eos_token_id=model.eos_token_id,
            max_new_tokens=max_total,
            do_sample=False,
            use_trace=use_trace,
        )
    finally:
        gen.decode_step = original_decode_step
        if use_trace:
            gen.step_callback = original_callback

    text = model.processor.decode(tokens, skip_special_tokens=True)
    return text, step_count[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help=(f"Path to a 16 kHz mono WAV. Default: " f"{INPUTS_DIR}/{DEFAULT_SAMPLE}."),
    )
    parser.add_argument("--src-lang", type=str, default=DEFAULT_SRC_LANG)
    parser.add_argument(
        "--tgt-lang",
        type=str,
        default=DEFAULT_TGT_LANG,
        help="Target language for translation. Pass src-lang for ASR.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="AR generation budget (includes the 2-token prefix).",
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=DEFAULT_MAX_AUDIO_SECONDS,
        help="Cap on the input audio length (seconds). Defaults to 5.",
    )
    parser.add_argument(
        "--traced",
        action="store_true",
        help=(
            "Capture a SINGLE metal trace for the AR decode step and "
            "replay it for every position across every timed call. "
            "Enabled by Phase 9c (paged_update_cache + tensor-valued "
            "cur_pos + persistent self/encoder masks). Cross-call "
            "trace reuse is automatic — no separate flag needed."
        ),
    )
    parser.add_argument(
        "--reuse-trace",
        action="store_true",
        help="[Compat] Alias for --traced.",
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
            "Number of timed translate() calls AFTER the 1-call warmup. "
            "Reported steady-state numbers are medians across these calls."
        ),
    )
    args = parser.parse_args()
    if args.reuse_trace:
        args.traced = True

    if args.traced:
        print(
            "[profile_s2tt] --traced ENABLED: AR decode steps run under a "
            "single re-usable metal trace (cross-call reuse is the default)."
        )

    # Lazy import so help/parse runs even if heavy deps are missing.
    from transformers import AutoProcessor

    from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl
    from models.demos.facebook_seamless_m4t_v2_large.tt.speech_to_text_model import SpeechToTextModel

    audio_path = args.audio if args.audio else str(INPUTS_DIR / DEFAULT_SAMPLE)
    if not Path(audio_path).is_file():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    print(f"[profile_s2tt] loading HF state dict ...")
    hf_sd = wl.load_hf_state_dict()
    processor = AutoProcessor.from_pretrained(wl.HF_PATH)

    print(f"[profile_s2tt] opening device {args.device_id} ...")
    device = ttnn.open_device(
        device_id=args.device_id,
        l1_small_size=32768,
        trace_region_size=256_000_000 if args.traced else 0,
    )

    try:
        print(f"[profile_s2tt] building SpeechToTextModel ...")
        t_build0 = time.perf_counter()
        model = SpeechToTextModel(device=device, hf_state_dict=hf_sd, processor=processor)
        t_build = time.perf_counter() - t_build0
        print(f"[profile_s2tt] model build = {t_build*1000:.0f} ms")

        # ------- 1. WARMUP -------
        print(
            f"[profile_s2tt] WARMUP translate: audio={audio_path!r} "
            f"src_lang={args.src_lang} tgt_lang={args.tgt_lang}"
        )
        warmup_step_log: List = []
        warmup_prefill: List[float] = []
        t_w0 = time.perf_counter()
        warmup_text, warmup_steps = _run_one_translate(
            model,
            audio_path=audio_path,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            max_new_tokens=args.max_new_tokens,
            max_audio_seconds=args.max_audio_seconds,
            per_step_log=warmup_step_log,
            prefill_log=warmup_prefill,
            use_trace=args.traced,
        )
        warmup_total_ms = (time.perf_counter() - t_w0) * 1000.0
        print(f"[profile_s2tt] WARMUP output: {warmup_text!r}")
        print(f"[profile_s2tt] WARMUP steps={warmup_steps}, total_ms={warmup_total_ms:.1f}")

        # ------- 2. TIMED -------
        all_step_logs: List[List] = []
        all_prefill_logs: List[float] = []
        all_totals: List[float] = []
        all_tokens: List[int] = []
        all_texts: List[str] = []
        for i in range(args.num_timed):
            step_log: List = []
            prefill_log: List[float] = []
            print(f"[profile_s2tt] TIMED[{i+1}/{args.num_timed}] translate ...")
            t_t0 = time.perf_counter()
            text, steps = _run_one_translate(
                model,
                audio_path=audio_path,
                src_lang=args.src_lang,
                tgt_lang=args.tgt_lang,
                max_new_tokens=args.max_new_tokens,
                max_audio_seconds=args.max_audio_seconds,
                per_step_log=step_log,
                prefill_log=prefill_log,
                use_trace=args.traced,
            )
            total_ms = (time.perf_counter() - t_t0) * 1000.0
            all_step_logs.append(step_log)
            all_prefill_logs.extend(prefill_log)
            all_totals.append(total_ms)
            all_tokens.append(steps)
            all_texts.append(text)

        # Steady-state per-step: when use_trace, only "replay" entries count.
        # Untraced: skip the first two prefix steps (cache warmup).
        steady_steps: List[float] = []
        capture_costs: List[float] = []
        warmup_costs: List[float] = []
        for log in all_step_logs:
            if log and isinstance(log[0], tuple):
                for ms, kind, _pos in log:
                    if kind == "replay":
                        steady_steps.append(ms)
                    elif kind == "capture":
                        capture_costs.append(ms)
                    else:
                        warmup_costs.append(ms)
            else:
                if len(log) > 2:
                    steady_steps.extend(log[2:])
                else:
                    steady_steps.extend(log)

        # ------- 3. Report -------
        def _stat(name: str, xs: List[float]) -> str:
            if not xs:
                return f"{name}=<empty>"
            return (
                f"{name}: n={len(xs)}, min={min(xs):.2f}, "
                f"p50={statistics.median(xs):.2f}, mean={statistics.mean(xs):.2f}, "
                f"max={max(xs):.2f}"
            )

        print("")
        print("=" * 64)
        print("[profile_s2tt] RESULTS")
        print("=" * 64)
        print(f"audio:           {audio_path!r} ({args.src_lang} -> {args.tgt_lang})")
        print(f"max_new_tokens:  {args.max_new_tokens}")
        print(f"warmup_total_ms: {warmup_total_ms:.1f}")
        print(f"warmup_steps:    {warmup_steps}")
        if warmup_step_log:
            _first = warmup_step_log[0]
            _last = warmup_step_log[-1]
            _first_ms = _first[0] if isinstance(_first, tuple) else _first
            _last_ms = _last[0] if isinstance(_last, tuple) else _last
            print(f"warmup_first_step_ms: {_first_ms:.2f}")
            print(f"warmup_last_step_ms:  {_last_ms:.2f}")
        print("")
        print(_stat("prefill_ms       (speech_encoder + cross-attn populate)", all_prefill_logs))
        print(_stat("steady_decode_step_ms (AR loop, position >= 2)", steady_steps))
        if args.traced:
            print(_stat("capture_step_ms (first-hit-of-pos trace capture)", capture_costs))
            print(_stat("warmup_step_ms  (first-hit-of-pos untraced warmup)", warmup_costs))
        print(_stat("timed_total_ms                              ", all_totals))
        print(f"tokens_generated_per_call: {all_tokens}")
        for i, text in enumerate(all_texts):
            print(f"text[{i}]: {text!r}")
        print("=" * 64)

        prefill_p50 = statistics.median(all_prefill_logs) if all_prefill_logs else float("nan")
        steady_p50 = statistics.median(steady_steps) if steady_steps else float("nan")
        total_p50 = statistics.median(all_totals) if all_totals else float("nan")
        print(
            f"[profile_s2tt] SUMMARY prefill_ms={prefill_p50:.2f} "
            f"steady_decode_step_ms={steady_p50:.2f} "
            f"total_ms={total_p50:.2f} "
            f"tokens_generated={all_tokens[0] if all_tokens else 0}"
        )

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
