# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What does the *first* request at an unwarmed prompt length cost?

The readiness runners build a fresh generator and issue exactly one request, so
the TTFT they report is a first-request TTFT at whatever length the reference
happens to be. Two setup costs land inside it and neither is steady state:

* the programs that depend on the exact prompt length compile on first use;
* because they compile while the decode traces are live, the generator
  re-captures those traces before it replays over the new program buffers
  (work log FM-016).

Both are one-time per shape, so quoting the first-request number as "TTFT"
without the second request next to it overstates steady-state latency. This
probe measures both on the full 47-layer model, at the readiness reference
length by default.

    python models/autoports/zai_org_glm_4_7_flash/probe/first_use_ttft_probe.py

Writes ``doc/full_model/first_use_ttft.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tt.generator import build_generator
from models.autoports.zai_org_glm_4_7_flash.tt.model import source_manifest

MODEL_DIR = Path(__file__).resolve().parents[1]
OUT = MODEL_DIR / "doc" / "full_model" / "first_use_ttft.json"
#: 154 is the AIME24 chat-template reference length, which is what
#: run_teacher_forcing reports TTFT for and which sits inside one prefill
#: chunk, so `warmup_terminal_shapes` should leave it with no first-use cost.
#: 4300 is two whole chunks plus a bucketed tail, whose chunk-offset programs
#: cannot be warmed cheaply and so must pay one recapture.
DEFAULT_LENGTHS = (154, 4300)


def _ids(gen, seq):
    text = (
        "Tenstorrent builds AI accelerators. "
        "This paragraph exists so the tokenizer produces a long, ordinary, in-distribution prompt "
        "for the first-use TTFT probe. "
    ) * 200
    ids = gen.tokenizer.encode(text, add_special_tokens=True)
    while len(ids) < seq:
        ids = ids + ids
    return ids[:seq]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default=",".join(str(n) for n in DEFAULT_LENGTHS))
    ap.add_argument("--layers", default="all", help="HF layer indices, or 'all'")
    ap.add_argument("--seq-cap", type=int, default=202752)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    layers = None if args.layers == "all" else [int(v) for v in args.layers.split(",")]
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=350_000_000)
    rows = []
    try:
        gen = build_generator(
            MODEL_DIR, dev, max_batch_size=1, max_seq_len=args.seq_cap, layer_indices=layers, progress=print
        )
        for seq in [int(v) for v in args.lengths.split(",")]:
            ids = _ids(gen, seq)
            row = {"prompt_len": seq, "physical_len": gen.model.prefill_physical_len(seq)}
            for label in ("first_request", "second_request"):
                gen.reset()
                gen.reset_counters()
                t0 = time.perf_counter()
                _, timing = gen.generate(ids, 2, enable_trace=True, stop_on_eos=False, return_timing=True)
                wall = time.perf_counter() - t0
                row[label] = {
                    # What the readiness runners measure: wall clock from the
                    # generate() call to the first token being handed back.
                    "harness_style_ttft_ms": round(
                        (timing["reset_s"] + timing["ttft_s"] + timing["recapture_s"]) * 1000, 1
                    ),
                    "reset_ms": round(timing["reset_s"] * 1000, 1),
                    "prefill_plus_first_token_ms": round(timing["ttft_s"] * 1000, 1),
                    "trace_recapture_ms": round(timing["recapture_s"] * 1000, 1),
                    "trace_recaptures": gen.counters["trace_recaptures"],
                    "wall_ms": round(wall * 1000, 1),
                }
            first, second = row["first_request"], row["second_request"]
            row["first_use_penalty_ms"] = round(first["harness_style_ttft_ms"] - second["harness_style_ttft_ms"], 1)
            row["note"] = (
                "The penalty is one program compile for this exact prompt length plus one trace recapture; "
                "both are one-time per shape. build_generator(warmup_prefill_lens=[...]) pre-pays them "
                "before the traces are captured."
            )
            rows.append(row)
            print(json.dumps(row, indent=2), flush=True)
        gen.teardown()
    finally:
        ttnn.close_mesh_device(dev)

    payload = {"source_manifest": source_manifest([__file__]), "layers": args.layers, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
