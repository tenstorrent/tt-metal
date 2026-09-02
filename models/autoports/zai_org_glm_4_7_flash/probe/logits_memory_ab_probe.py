# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""L1 or DRAM for the sampler-ready decode logits? Both arms, with repeats.

`TTSampling`'s first `ttnn.split` cannot fit four 154880-wide chunks in L1, so
producing the logits in L1 makes it log

    ttnn.split: L1 budget exceeded (need ~9945088 B, have 1229824 B for 4 chunks); DRAM downgrade
    ttnn.split: migrating L1 input (9912320 B) to DRAM before slice fallback

on the measured token-out path. `GLM47FlashModel(decode_logits_in_dram=True)`
removes the migration by producing the logits in DRAM instead, and the profiler
attributes a 40.4 us/step `CopyDeviceOperation` to the L1 arm, so the two arms
are not obviously ordered. The first version of this measurement was one sample
per arm and 34 us apart, which is not enough to choose on
(review round 7). This runs both arms with repeats, on the same process and the
same cache geometry, and records the spread alongside the means.

Both arms must produce identical tokens; that is the correctness half and it is
asserted, not just reported.

    python models/autoports/zai_org_glm_4_7_flash/probe/logits_memory_ab_probe.py

Writes ``doc/full_model/logits_memory_ab.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tt.generator import GLM47FlashGenerator
from models.autoports.zai_org_glm_4_7_flash.tt.model import GLM47FlashModel, source_manifest

MODEL_DIR = Path(__file__).resolve().parents[1]
OUT = MODEL_DIR / "doc" / "full_model" / "logits_memory_ab.json"


def _bench(fn, dev, iters):
    fn()
    ttnn.synchronize_device(dev)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        samples.append((time.perf_counter() - t0) * 1000)
    return samples


def _arm(dev, *, in_dram, layers, seq_cap, position, tokens, iters):
    model = GLM47FlashModel.from_pretrained(
        dev,
        max_batch_size=1,
        max_seq_len=seq_cap,
        layer_indices=layers,
        decode_logits_in_dram=in_dram,
        progress=lambda m: None,
    )
    gen = GLM47FlashGenerator(model)
    try:
        gen._ensure_owned_state()
        gen.capture_decode_trace()
        gen.reset()
        gen.set_decode_tokens([11])
        gen.set_decode_positions([position])
        model_only = _bench(gen.replay_decode_trace, dev, iters)
        gen.set_decode_positions([position])
        token_out = _bench(gen.decode_step_traced, dev, iters)

        gen.reset()
        gen.set_decode_tokens([11])
        gen.set_decode_positions([position])
        got = []
        for _ in range(tokens):
            gen.decode_step_traced()
            got.append(gen.read_decode_tokens(1)[0])
        return {
            "decode_logits_memory_config": "DRAM" if in_dram else "L1",
            "iterations": iters,
            "traced_model_only_ms": round(statistics.mean(model_only), 3),
            "traced_model_only_spread_ms": round(max(model_only) - min(model_only), 3),
            "traced_token_out_ms": round(statistics.mean(token_out), 3),
            "traced_token_out_spread_ms": round(max(token_out) - min(token_out), 3),
            "sampling_ms": round(statistics.mean(token_out) - statistics.mean(model_only), 3),
            "tokens": got,
        }
    finally:
        gen.teardown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,1", help="HF layer indices, or 'all'")
    ap.add_argument("--seq-cap", type=int, default=202752)
    ap.add_argument("--position", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--iters", type=int, default=64)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    layers = None if args.layers == "all" else [int(v) for v in args.layers.split(",")]
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=350_000_000)
    try:
        arms = {}
        for label, in_dram in (("l1", False), ("dram", True)):
            arms[label] = _arm(
                dev,
                in_dram=in_dram,
                layers=layers,
                seq_cap=args.seq_cap,
                position=args.position,
                tokens=args.tokens,
                iters=args.iters,
            )
            print(json.dumps(arms[label]), flush=True)
    finally:
        ttnn.close_mesh_device(dev)

    l1, dram = arms["l1"], arms["dram"]
    payload = {
        "source_manifest": source_manifest([__file__]),
        "note": (
            "Reduced 2-layer probe at the full 202752-token cache, batch 1, one decode position. The L1 arm "
            "logs ttnn.split 'L1 budget exceeded ... DRAM downgrade' and 'migrating L1 input (9912320 B) to "
            "DRAM' inside the captured sampling graph; the DRAM arm does not. Read the deltas against the "
            "per-arm spreads, not as exact figures."
        ),
        "l1": l1,
        "dram": dram,
        "token_out_delta_ms_dram_minus_l1": round(dram["traced_token_out_ms"] - l1["traced_token_out_ms"], 3),
        "sampling_delta_ms_dram_minus_l1": round(dram["sampling_ms"] - l1["sampling_ms"], 3),
        "same_tokens": l1["tokens"] == dram["tokens"],
        "decision": (
            "Whichever arm is faster, the choice is only defensible if the tokens match; they do. The "
            "shipped default is L1 (decode_logits_in_dram=False) and the other arm is one constructor "
            "argument away."
        ),
    }
    assert payload["same_tokens"], (l1["tokens"], dram["tokens"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
