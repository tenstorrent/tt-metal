#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Profiler harness for the SeamlessM4T-v2 T2TT (text-to-text) pipeline.

Goal: characterize the TTNN T2TT pipeline (encoder + decoder + AR loop)
under tracy so we can identify the hottest ops and decide where to
apply targeted optimization (typically metal trace+replay on the AR
decode step).

Usage (no tracy, just wall-clock):

    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && \
        source python_env/bin/activate && export ARCH_NAME=blackhole
    python models/demos/facebook_seamless_m4t_v2_large/tt/profile_t2tt.py

Usage (under tracy):

    python -m tracy -p -v -r --no-device-data-capture \
        -n seamless_m4t_t2tt \
        models/demos/facebook_seamless_m4t_v2_large/tt/profile_t2tt.py

The script runs 1 warmup ``translate`` + 1 timed ``translate`` and
prints:

    prefill_ms              encoder + cross-attn cache populate (1-time)
    warmup_total_ms         total wall-clock of the warmup call
    steady_decode_step_ms   median per-step time of the timed call (AR loop)
    timed_total_ms          total wall-clock of the timed call
    tokens_generated        # of decode_step invocations (including 2 prefix)

The ``--traced`` flag is reserved as the path-forward for metal trace
replay. The current TextDecoder.decode_step builds masks and runs
ttnn.from_torch host-side per step, which blocks a single captured
trace from being re-used across positions. See PERF_NOTES.md for
details and a proposed refactor.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import List, Optional, Tuple

import ttnn

SAMPLES_PATH = Path(__file__).resolve().parents[1] / "demo" / "inputs" / "t2tt_samples.json"

# Default profiling budget. Includes the 2 prefix tokens.
DEFAULT_MAX_NEW_TOKENS = 32


def _load_samples() -> List[dict]:
    if not SAMPLES_PATH.is_file():
        raise FileNotFoundError(f"samples file not found: {SAMPLES_PATH}")
    with open(SAMPLES_PATH) as f:
        return json.load(f)


def _run_one_translate(
    model,
    src: str,
    src_lang: str,
    tgt_lang: str,
    max_new_tokens: int,
    per_step_log: Optional[List[float]] = None,
    prefill_log: Optional[List[float]] = None,
    use_trace: bool = False,
) -> Tuple[str, int]:
    """Run one TTNN translate and time it with optional per-step instrumentation.

    Per-step timing is performed by monkey-patching the generator's
    ``decode_step`` method for the duration of this call. The timer
    wraps the call in a ttnn.synchronize_device(...) so we measure the
    on-device latency rather than dispatch enqueue.
    """
    from models.demos.facebook_seamless_m4t_v2_large.tt.text_to_text_model import _tile_pad_encoder_hidden

    # Tokenize first so the encoder-call boundary is clearly delineated.
    toks = model.processor(text=src, src_lang=src_lang, return_tensors="pt")
    input_ids = toks["input_ids"]
    attn_mask = toks["attention_mask"]
    src_logical = int(attn_mask.sum().item())
    input_ids = input_ids[:, :src_logical]
    attn_mask = attn_mask[:, :src_logical]

    # --- 1. Encoder phase (prefill) ---
    t0 = time.perf_counter()
    enc_hidden_logical = model._run_encoder(input_ids, attn_mask)
    ttnn.synchronize_device(model.device)
    t_enc = time.perf_counter() - t0

    enc_hidden_padded = _tile_pad_encoder_hidden(enc_hidden_logical[:, :src_logical, :])
    encoder_seq_len_padded = int(enc_hidden_padded.shape[1])
    gen = model._get_or_build_generator(encoder_seq_len_padded)
    tgt_lang_id = model._resolve_tgt_lang_id(tgt_lang)

    # --- 2. Cross-attn cache populate (prefill) ---
    t1 = time.perf_counter()
    gen.past_key_values.self_attn.reset()
    src_logical_pad = gen.populate_encoder_cache(enc_hidden_padded, encoder_attention_mask=attn_mask)
    ttnn.synchronize_device(model.device)
    t_xpop = time.perf_counter() - t1

    prefill_ms = (t_enc + t_xpop) * 1000.0
    if prefill_log is not None:
        prefill_log.append(prefill_ms)

    # --- 3. AR loop ---
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
    # represent steady-state cost; we tag the log so the reporting code
    # can filter.
    original_callback = gen.step_callback

    def trace_callback(position: int, ms: float, kind: str) -> None:
        if per_step_log is not None:
            # Annotate the entry with the kind via a tuple; the reporting
            # code below splits replay-only stats from warmup/capture.
            per_step_log.append((ms, kind, position))
        step_count[0] += 1

    if use_trace:
        gen.step_callback = trace_callback
    try:
        tokens = gen.generate(
            encoder_hidden_states=enc_hidden_padded,
            encoder_attention_mask=attn_mask,
            decoder_start_token_id=model.decoder_start_token_id,
            tgt_lang_id=tgt_lang_id,
            eos_token_id=model.eos_token_id,
            max_new_tokens=max_new_tokens,
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
        "--src",
        type=str,
        default=None,
        help=("Source sentence to translate. Default: first prompt in " "demo/inputs/t2tt_samples.json."),
    )
    parser.add_argument("--src-lang", type=str, default="eng")
    parser.add_argument("--tgt-lang", type=str, default="fra")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="AR generation budget (includes the 2-token prefix).",
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
        help=(
            "[Compat] Alias for --traced. Trace reuse across generate() "
            "calls is now the default behaviour of --traced; this flag "
            "is provided so older invocations keep working."
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
            "[profile_t2tt] --traced ENABLED: AR decode steps run under a "
            "single re-usable metal trace (cross-call reuse is the default)."
        )

    # Lazy import so help/parse runs even if heavy deps are missing.
    from transformers import AutoProcessor

    from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl
    from models.demos.facebook_seamless_m4t_v2_large.tt.text_to_text_model import TextToTextModel

    samples = _load_samples()
    if args.src is None:
        sample = samples[0]
        src = sample["src"]
        src_lang = sample["src_lang"]
        tgt_lang = sample["tgt_lang"]
    else:
        src, src_lang, tgt_lang = args.src, args.src_lang, args.tgt_lang

    print(f"[profile_t2tt] loading HF state dict ...")
    hf_sd = wl.load_hf_state_dict()
    processor = AutoProcessor.from_pretrained(wl.HF_PATH)

    print(f"[profile_t2tt] opening device {args.device_id} ...")
    # Allocate enough trace region so we can capture multiple per-position
    # decode traces (one per AR slot we hit). 256 MB is comfortable
    # headroom for max_decode_seq_len=128 with ~365 ops/step.
    device = ttnn.open_device(
        device_id=args.device_id,
        l1_small_size=32768,
        trace_region_size=256_000_000 if args.traced else 0,
    )

    try:
        print(f"[profile_t2tt] building TextToTextModel ...")
        t_build0 = time.perf_counter()
        model = TextToTextModel(device=device, hf_state_dict=hf_sd, processor=processor)
        t_build = time.perf_counter() - t_build0
        print(f"[profile_t2tt] model build = {t_build*1000:.0f} ms")

        # ------- 1. WARMUP -------
        print(f"[profile_t2tt] WARMUP translate: src={src!r} src_lang={src_lang} tgt_lang={tgt_lang}")
        warmup_step_log: List[float] = []
        warmup_prefill: List[float] = []
        t_w0 = time.perf_counter()
        warmup_text, warmup_steps = _run_one_translate(
            model,
            src=src,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_new_tokens=args.max_new_tokens,
            per_step_log=warmup_step_log,
            prefill_log=warmup_prefill,
            use_trace=args.traced,
        )
        warmup_total_ms = (time.perf_counter() - t_w0) * 1000.0
        print(f"[profile_t2tt] WARMUP output: {warmup_text!r}")
        print(f"[profile_t2tt] WARMUP steps={warmup_steps}, total_ms={warmup_total_ms:.1f}")

        # ------- 2. TIMED -------
        all_step_logs: List[List[float]] = []
        all_prefill_logs: List[float] = []
        all_totals: List[float] = []
        all_tokens: List[int] = []
        all_texts: List[str] = []
        for i in range(args.num_timed):
            step_log: List[float] = []
            prefill_log: List[float] = []
            print(f"[profile_t2tt] TIMED[{i+1}/{args.num_timed}] translate ...")
            t_t0 = time.perf_counter()
            text, steps = _run_one_translate(
                model,
                src=src,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                max_new_tokens=args.max_new_tokens,
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

        # Steady-state per-step: when use_trace, only "replay" entries count
        # (warmup + capture are first-hit-of-position costs). When not
        # tracing, exclude the very first decode_step of each call
        # (positions 0 and 1 are warmup-of-the-cache style; the LM head
        # path is the same but cache state diverges).
        steady_steps: List[float] = []
        capture_costs: List[float] = []
        warmup_costs: List[float] = []
        for log in all_step_logs:
            if log and isinstance(log[0], tuple):
                # Trace mode: filter by kind tag.
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
        print("[profile_t2tt] RESULTS")
        print("=" * 64)
        print(f"prompt:          {src!r} ({src_lang} -> {tgt_lang})")
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
        print(_stat("prefill_ms       (encoder + cross-attn populate)", all_prefill_logs))
        print(_stat("steady_decode_step_ms (AR loop, position >= 2)", steady_steps))
        if args.traced:
            print(_stat("capture_step_ms (first-hit-of-pos trace capture)", capture_costs))
            print(_stat("warmup_step_ms  (first-hit-of-pos untraced warmup)", warmup_costs))
        print(_stat("timed_total_ms                              ", all_totals))
        print(f"tokens_generated_per_call: {all_tokens}")
        for i, text in enumerate(all_texts):
            print(f"text[{i}]: {text!r}")
        print("=" * 64)

        # Final single-line summary mirroring the reporting contract.
        prefill_p50 = statistics.median(all_prefill_logs) if all_prefill_logs else float("nan")
        steady_p50 = statistics.median(steady_steps) if steady_steps else float("nan")
        total_p50 = statistics.median(all_totals) if all_totals else float("nan")
        print(
            f"[profile_t2tt] SUMMARY prefill_ms={prefill_p50:.2f} "
            f"steady_decode_step_ms={steady_p50:.2f} "
            f"total_ms={total_p50:.2f} "
            f"tokens_generated={all_tokens[0] if all_tokens else 0}"
        )

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
