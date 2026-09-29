# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Synthesize the fixed corpus on a Tenstorrent device: wavs, results.json and a timing table.

    python models/experimental/cosyvoice2/demo/demo.py --inputs <dir> --out <dir> [--cases a,b] [--seed 1986] \\
        [--config reported|eager] [--hift-source-dtype float32|bfloat16] [--warmup buckets|sentence|none] [--stream]

`--inputs` (or `COSYVOICE2_INPUTS`) is the directory `scripts/prepare_inputs.py` wrote: one `.npz` per corpus
case, made once in the reference venv, because the frontend (ONNX speech tokenizer, CAM++, mel filterbank) is not
something this port runs. Each case synthesizes its target sentence in its prompt speaker's voice through
`tt.pipeline.CosyVoice2TTNN.synthesize`, zero-shot, non-streaming, all three stages on the device. What returns
to the host between stages: the sampled token ids (RAS runs on the host), the mel, and the waveform.

The output directory has the layout `scripts/run_reference.py` writes, so one scoring command handles the
PyTorch reference and this port alike. The table reports each utterance's device-synchronized stage times and its
RTF (wall time / audio time).

Warm-up (`--warmup`):
- `buckets` (the default, and the Stage 1 protocol): `warmup_buckets()` runs every flow and HiFT bucket and every
  LLM prefill length once, in a fixed order, before the first request. Its time is reported as the start-up cost.
  Every corpus utterance that follows is a distinct sentence that lands in an already-warmed bucket.
- `sentence`: one throwaway sentence first, reported as the process's cold call.
- `none`: the first corpus utterance is the cold first request.

`--stream`: streaming synthesis (`CosyVoice2TTNN.synthesize_stream`, tt/streaming.py). Chunks of audio are produced
while the LLM generates, on upstream's chunk schedule. It needs `--warmup buckets`, which then warms the streaming
set too (`warmup_streaming`): a chunk's flow and HiFT run while the LLM's decode trace is alive, where nothing may
compile or allocate (docs/VALIDATION.md, "Streaming, measured"). The table then reports each utterance's time to
first audio, the first chunk's breakdown and the RTF; `results.json` keeps every chunk's times.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import soundfile

import ttnn
from models.experimental.cosyvoice2.tt.pipeline import (
    SAMPLE_RATE,
    CosyVoice2Config,
    CosyVoice2TTNN,
    Synthesis,
    device_memory,
)
from models.experimental.cosyvoice2.tt.prompt import PromptContext, RandomSources

L1_SMALL_SIZE = 65536
TRACE_REGION_SIZE = 50_000_000  # the LLM decode trace; the CFM trace is off in every preset


def prompt_wav_abs(case: dict) -> str | None:
    base = os.environ.get("COSYVOICE2_REPO" if case["set"] == "cosyvoice1_parity" else "LIBRISPEECH_ROOT")
    return os.path.join(base, case["prompt_wav"]) if base else None


def row(name: str, syn: Synthesis) -> str:
    t = syn.stage_totals()
    n_tok = len(syn.tokens)
    tok_s = n_tok / t["llm_decode"] if t.get("llm_decode") else float("nan")
    return (
        f"| {name} | {syn.audio_s:.2f} | {n_tok} | {t['llm_prefill']:.3f} | {t['llm_decode']:.3f} | {tok_s:.1f} | "
        f"{t['flow_encoder']:.3f} | {t['flow_cfm']:.3f} | {t['hift']:.3f} | {syn.wall_s:.3f} | {syn.rtf:.3f} |"
    )


def stream_row(name: str, syn: Synthesis) -> str:
    """Streaming: the first chunk's breakdown (LLM until the chunk starts, flow with its CFM, HiFT) and the totals."""
    c = syn.chunks[0]
    return (
        f"| {name} | {syn.audio_s:.2f} | {len(syn.tokens)} | {len(syn.chunks)} | {c['hop']} | {c['start_s']:.3f} | "
        f"{c['flow']:.3f} | {c['cfm']:.3f} | {c['hift']:.3f} | **{syn.first_audio_s:.3f}** | {syn.wall_s:.3f} | "
        f"{syn.rtf:.3f} |"
    )


STREAM_TABLE_HEAD = (
    "| utterance | audio s | tokens | chunks | first chunk tokens | until it starts s (text + LLM) | its flow s | its CFM s "
    "| its HiFT s | first audio s | wall s | RTF |\n|---|---|---|---|---|---|---|---|---|---|---|---|"
)

TABLE_HEAD = (
    "| utterance | audio s | tokens | LLM prefill s | LLM decode s | tok/s | flow encoder s | CFM s | HiFT s "
    "| wall s | RTF |\n|---|---|---|---|---|---|---|---|---|---|---|"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", default=os.environ.get("COSYVOICE2_INPUTS"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default="", help="comma-separated case ids (default: every librispeech case)")
    ap.add_argument("--parity", action="store_true", help="also run the CosyVoice1-parity case")
    ap.add_argument("--seed", type=int, default=1986)
    ap.add_argument("--config", choices=("reported", "eager"), default="reported")
    ap.add_argument("--hift-source-dtype", choices=("float32", "bfloat16"), default=None)
    ap.add_argument("--warmup", choices=("buckets", "sentence", "none"), default="buckets")
    ap.add_argument(
        "--stream",
        action="store_true",
        help="streaming synthesis (synthesize_stream); needs --warmup buckets, which then warms the streaming set too",
    )
    args = ap.parse_args()
    if args.stream and args.warmup != "buckets":
        ap.error("--stream needs --warmup buckets: synthesize_stream() refuses to run before warmup_streaming()")
    if not args.inputs:
        raise SystemExit("pass --inputs (or set COSYVOICE2_INPUTS) to scripts/prepare_inputs.py's --out-dir")

    paths = sorted(glob.glob(os.path.join(args.inputs, "*.npz")))
    ctxs = [PromptContext.from_npz(p) for p in paths]
    wanted = set(filter(None, args.cases.split(",")))
    ctxs = [
        c
        for c in ctxs
        if (c.meta["case"]["case_id"] in wanted)
        or (not wanted and (c.meta["case"]["set"] == "librispeech" or args.parity))
    ]
    if not ctxs:
        raise SystemExit(f"no matching cases in {args.inputs}")
    # CosyVoice2Config is frozen; presets are replaced field by field.
    cfg = getattr(CosyVoice2Config, args.config)()
    if args.hift_source_dtype:
        from dataclasses import replace

        cfg = replace(cfg, hift_source_dtype=args.hift_source_dtype)
    os.makedirs(args.out, exist_ok=True)

    trace_region = TRACE_REGION_SIZE if (cfg.llm_decode_trace or cfg.cfm_trace) else 0
    device = ttnn.open_device(device_id=0, l1_small_size=L1_SMALL_SIZE, trace_region_size=trace_region)
    arch = str(device.arch())
    lines, results, cold = [], [], None
    try:
        t0 = time.perf_counter()
        pipe = CosyVoice2TTNN(device, cfg)
        build_s = time.perf_counter() - t0
        print(f"built in {build_s:.1f} s; config {json.dumps(cfg.describe())}", flush=True)
        mem0 = device_memory(device)
        warmup_s = warmup_clock = stream_warmup_s = stream_clock = None
        if args.warmup == "buckets":
            t0 = time.perf_counter()
            warmup_clock = pipe.warmup_buckets()
            warmup_s = time.perf_counter() - t0
            parts = {k: sum(v for n, v in warmup_clock.items() if n.startswith(k)) for k in ("llm", "flow", "hift")}
            print(
                f"warmed every bucket in {warmup_s:.1f} s: " + ", ".join(f"{k} {v:.1f} s" for k, v in parts.items()),
                flush=True,
            )
            if args.stream:
                t0 = time.perf_counter()
                stream_clock = pipe.warmup_streaming()
                stream_warmup_s = time.perf_counter() - t0
                print(f"warmed the streaming set in {stream_warmup_s:.1f} s", flush=True)
        elif args.warmup == "sentence":
            cold = pipe.warmup(ctxs[0])
            lines.append(row("(warm-up, cold: first call in this process)", cold))
            print(lines[-1], flush=True)
        for ctx in ctxs:
            case = ctx.meta["case"]
            synth = pipe.synthesize_stream if args.stream else pipe.synthesize
            syn = synth(ctx, case["text"], rng=RandomSources(llm_seed=args.seed))
            assert not pipe.live_traces(), pipe.live_traces()
            name = f"{case['case_id']}.wav"
            soundfile.write(os.path.join(args.out, name), syn.audio, SAMPLE_RATE)
            mem = device_memory(device)
            results.append(
                {
                    **case,
                    "wav": name,
                    "prompt_wav_abs": prompt_wav_abs(case),
                    "seed": args.seed,
                    "audio_s": round(syn.audio_s, 3),
                    "wall_s": round(syn.wall_s, 3),
                    "rtf": round(syn.rtf, 3),
                    "segment_tokens": [s.tokens for s in syn.segments],
                    "segments": [s.text for s in syn.segments],
                    "stage_s": {k: round(v, 4) for k, v in syn.stage_totals().items()},
                    "notes": syn.notes,
                    "device_memory_after": mem,
                    **({"first_audio_s": round(syn.first_audio_s, 4), "chunks": syn.chunks} if args.stream else {}),
                }
            )
            lines.append((stream_row if args.stream else row)(case["case_id"], syn))
            print(lines[-1], f" DRAM {mem['dram'] / 2**20:.1f} MiB/bank, L1_SMALL {mem['l1_small']} B/bank", flush=True)
        evictions = pipe.conv_cache_evictions()
        pipe.release()
    finally:
        ttnn.close_device(device)

    warm = results  # every corpus utterance; after the warm-up unless --warmup none
    total_wall, total_audio = sum(r["wall_s"] for r in warm), sum(r["audio_s"] for r in warm)
    run = {
        "backend": "ttnn",
        "device": arch,
        "meta": {
            "config": cfg.describe(),
            "build_s": round(build_s, 1),
            "warmup": args.warmup,
            "warmup_buckets_s": None if warmup_s is None else round(warmup_s, 1),
            "warmup_buckets_by_geometry_s": (
                None if warmup_clock is None else {k: round(v, 2) for k, v in warmup_clock.items()}
            ),
            "stream": args.stream,
            "warmup_streaming_s": None if stream_warmup_s is None else round(stream_warmup_s, 1),
            "warmup_streaming_by_geometry_s": (
                None if stream_clock is None else {k: round(v, 2) for k, v in stream_clock.items()}
            ),
            "warmup_sentence": (
                None if cold is None else {"audio_s": round(cold.audio_s, 3), "wall_s": round(cold.wall_s, 3)}
            ),
            "device_memory_before_first_call": mem0,
            "conv_cache_evictions": evictions,
            "rtf_aggregate": round(total_wall / total_audio, 3),
        },
        "results": results,
    }
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(run, fh, indent=2, ensure_ascii=False)
    table = "\n".join([STREAM_TABLE_HEAD if args.stream else TABLE_HEAD, *lines])
    summary = (
        f"{table}\n\nDistinct utterances: {len(warm)}, audio {total_audio:.2f} s, wall {total_wall:.2f} s, "
        f"aggregate RTF {total_wall / total_audio:.3f}, worst {max(r['rtf'] for r in warm):.3f}. "
        f"Config: {args.config}, HiFT F0/source {cfg.hift_source_dtype}, bucketing {cfg.bucketing}, "
        f"warm-up {args.warmup}" + ("" if warmup_s is None else f" ({warmup_s:.1f} s)") + f", seed {args.seed}."
    )
    if args.stream:
        firsts = [r["first_audio_s"] for r in warm]
        summary += (
            f" Streaming: first audio {min(firsts):.3f}-{max(firsts):.3f} s"
            + ("" if stream_warmup_s is None else f"; streaming warm-up {stream_warmup_s:.1f} s")
            + "."
        )
    with open(os.path.join(args.out, "timings.md"), "w") as fh:
        fh.write(summary + "\n")
    print("\n" + summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
