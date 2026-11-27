# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Vocoder noise draws over fixed tokens on the device, Stage 1 and streaming: one run directory per draw and mode, in
the layout scripts/eval_wer_sim.py scores. scripts/eval_draws.py reports the mean and range over the draws, so no WER
or similarity claim rests on one draw (notes: B28, where one draw's trailing "you" moved the corpus WER).

    python models/experimental/cosyvoice2/scripts/noise_draws.py --inputs <dir> --out <dir> \\
        [--noise-seeds 1,2,3,4,5] [--seed 1986] [--cases a,b]

Every draw samples the demo's tokens (`--seed` seeds the LLM through torch's global RNG); the vocoder's noise comes
from a generator of its own (`RandomSources.noise_seed`). The warm-ups are the demo's (the buckets, then the
streaming set), so no request compiles. It writes `<out>/tt_stage1_seed<N>/` and `<out>/tt_stream_seed<N>/`, and
checks that every draw of a case sampled the same tokens.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import soundfile

import ttnn
from models.experimental.cosyvoice2.tt.pipeline import SAMPLE_RATE, CosyVoice2Config, CosyVoice2TTNN
from models.experimental.cosyvoice2.tt.prompt import PromptContext, RandomSources


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=os.environ.get("COSYVOICE2_INPUTS"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--noise-seeds", default="1,2,3,4,5")
    ap.add_argument("--seed", type=int, default=1986)
    ap.add_argument("--cases", default="", help="comma-separated case ids (default: every librispeech case)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.noise_seeds.split(",")]
    wanted = set(filter(None, args.cases.split(",")))
    ctxs = [PromptContext.from_npz(p) for p in sorted(glob.glob(os.path.join(args.inputs, "*.npz")))]
    ctxs = [
        c
        for c in ctxs
        if (c.meta["case"]["case_id"] in wanted) or (not wanted and c.meta["case"]["set"] == "librispeech")
    ]
    assert ctxs, f"no matching cases in {args.inputs}"

    device = ttnn.open_device(device_id=0, l1_small_size=65536, trace_region_size=50_000_000)  # the demo's
    tokens_by_case: dict[str, list] = {}
    try:
        pipe = CosyVoice2TTNN(device, CosyVoice2Config.reported())
        pipe.warmup_buckets()
        pipe.warmup_streaming()
        for seed in seeds:
            for mode in ("stage1", "stream"):
                out_dir = os.path.join(args.out, f"tt_{mode}_seed{seed}")
                os.makedirs(out_dir, exist_ok=True)
                synth = pipe.synthesize if mode == "stage1" else pipe.synthesize_stream
                results = []
                for ctx in ctxs:
                    case = ctx.meta["case"]
                    syn = synth(ctx, case["text"], rng=RandomSources(llm_seed=args.seed, noise_seed=seed))
                    assert not pipe.live_traces(), pipe.live_traces()
                    name = f"{case['case_id']}.wav"
                    soundfile.write(os.path.join(out_dir, name), syn.audio, SAMPLE_RATE)
                    tokens = [list(map(int, s.tokens)) for s in syn.segments]
                    tokens_by_case.setdefault(case["case_id"], tokens)
                    assert (
                        tokens == tokens_by_case[case["case_id"]]
                    ), f"{case['case_id']}: the draws sampled different tokens"
                    results.append({**case, "wav": name, "audio_s": round(syn.audio_s, 3), "rtf": round(syn.rtf, 3),
                                    "noise_seed": seed, "segment_tokens": tokens})  # fmt: skip
                    print(f"  seed {seed} {mode:6s} {case['case_id']:<34} audio {syn.audio_s:6.2f} s", flush=True)
                with open(os.path.join(out_dir, "results.json"), "w") as fh:
                    json.dump({"backend": f"ttnn-{mode}", "llm_seed": args.seed, "noise_seed": seed, "results": results},
                              fh, indent=2, ensure_ascii=False)  # fmt: skip
    finally:
        if "pipe" in locals():
            pipe.release()
        ttnn.close_device(device)
    print(
        f"every draw of each case sampled the same tokens ({len(tokens_by_case)} cases, {len(seeds)} draws x 2 modes)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
