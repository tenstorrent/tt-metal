# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What the per-call KV-cache signature comparison costs the public entry points.

Round 9 of the stage review made ``_invalidate_traces_if_cache_moved`` unconditional, because
gating it on ``kv_cache is not None`` missed an out-of-band ``model.set_kv_cache``.  Round 10
pointed out -- correctly -- that the cost of that was asserted rather than measured, and that
``decode_forward`` is a per-token serving API: ``_kv_cache_signature`` reads
``2 x num_layers`` = **104** buffer addresses across the pybind boundary on this build.

So it is measured here, in the three states that exist:

* **no trace captured** -- the short-circuit round 10 asked for, which is two truth tests;
* **a trace captured, cache unmoved** -- the steady state: 104 address reads and a tuple
  compare, per call;
* **the signature alone**, for attribution.

Read-only with respect to the model: it builds, captures, times, and tears down.

Usage::

    python doc/optimized_full_model/bench/invalidation_cost_probe.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))

from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (  # noqa: E402
    DEFAULT_TRACE_REGION_SIZE,
    build_generator,
    clear_generator_cache,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (  # noqa: E402
    close_multichip_mesh,
    open_multichip_mesh,
)

OUT = ROOT / "doc/optimized_full_model"


def say(*args) -> None:
    print(*args, flush=True)


def time_us(fn, reps: int) -> dict:
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e6)
    return {
        "min_us": min(samples),
        "median_us": statistics.median(samples),
        "mean_us": statistics.fmean(samples),
        "reps": reps,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--out", default="invalidation_cost_probe.json")
    args = parser.parse_args()

    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    mesh = open_multichip_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    summary: dict = {"reps": args.reps}
    generator = None
    try:
        generator = build_generator(ROOT, mesh, max_seq_len=args.max_seq_len)
        summary["layers"] = len(generator.model.layers)
        summary["address_reads_per_signature"] = 2 * len(generator.model.layers)

        # (1) No trace captured: the short-circuit.
        assert generator._trace_id is None and not generator._prefill_traces
        summary["no_trace_captured"] = time_us(generator._invalidate_traces_if_cache_moved, args.reps)
        say(f"no trace captured: {summary['no_trace_captured']}")

        # Capture the decode trace the shipped default runs on.
        prompt = [11, 22, 33, 44] * 16
        generator.prefill_forward(torch.tensor([prompt]), kv_cache=generator.model.kv_cache, prompt_lens=[len(prompt)])
        generator.decode_forward(
            tokens=torch.tensor([[prompt[-1]]], dtype=torch.long),
            start_pos=torch.tensor([len(prompt)], dtype=torch.int32),
            kv_cache=generator.model.kv_cache,
        )
        assert generator._trace_id is not None

        # (2) The steady state a serving caller pays per call.
        summary["trace_captured_cache_unmoved"] = time_us(generator._invalidate_traces_if_cache_moved, args.reps)
        say(f"trace captured, cache unmoved: {summary['trace_captured_cache_unmoved']}")

        # (3) The signature alone, for attribution.
        summary["signature_only"] = time_us(generator._kv_cache_signature, args.reps)
        say(f"signature only: {summary['signature_only']}")

        step_ms = 23.298  # the reported token-out step, doc/optimized_full_model/evidence_perf.json
        summary["share_of_token_out_step_percent"] = (
            summary["trace_captured_cache_unmoved"]["median_us"] / 1e3 / step_ms * 100
        )
        summary["step_ms_compared_against"] = step_ms
        say(f"share of a {step_ms} ms token-out step: {summary['share_of_token_out_step_percent']:.4f} %")
        summary["note"] = (
            "generate()'s own decode loop calls _decode_step_traced directly and pays this once per "
            "generate(), not per token; decode_forward is the per-token serving API and pays it per call."
        )
    finally:
        if generator is not None:
            generator.teardown()
            generator.model.deallocate()
        clear_generator_cache()
        close_multichip_mesh(mesh)

    (OUT / args.out).write_text(json.dumps(summary, indent=2) + "\n")
    say(f"wrote {OUT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
