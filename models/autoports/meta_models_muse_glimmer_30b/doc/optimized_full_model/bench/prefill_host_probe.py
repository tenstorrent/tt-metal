# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Is the prefill's missing 16 ms host dispatch, and if so, whose?

``ttft_breakdown.py`` put the 128-token prefill's 52-layer stack at 60.3 ms of
wall time against ~43.7 ms of device time (52 x the reduced-variant Tracy
window's 0.84 ms/layer).  The gap is ~0.32 ms per layer.  This probe splits it:

* ``dispatch``  -- per-layer wall time with **no** synchronisation, i.e. the time
  the host spends issuing the layer's ~30 ops (plus any queue backpressure);
* ``op_floor``  -- the host cost of one trivial ttnn op on this mesh, repeated,
  which is the per-op floor nothing in the model can go below;
* ``profile``   -- a ``cProfile`` of one whole prefill, so the host time is
  attributed to functions rather than guessed at.

Usage::

    python doc/optimized_full_model/bench/prefill_host_probe.py --length 128
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pathlib
import pstats
import sys
import time

import torch

import ttnn

ROOT = pathlib.Path(__file__).resolve().parents[3]
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))

from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (  # noqa: E402
    DEFAULT_TRACE_REGION_SIZE,
    build_generator,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (  # noqa: E402
    close_multichip_mesh,
    open_multichip_mesh,
)

OUT = ROOT / "doc/optimized_full_model"


def say(*args) -> None:
    print(*args, flush=True)


def op_floor(mesh, *, reps: int) -> dict:
    """Host cost of one trivial ttnn op on this mesh, with and without a sync."""
    a = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    ttnn.add(a, a)  # compile
    ttnn.synchronize_device(mesh)
    started = time.perf_counter()
    outs = [ttnn.add(a, a) for _ in range(reps)]
    issued = (time.perf_counter() - started) / reps * 1e6
    ttnn.synchronize_device(mesh)
    done = (time.perf_counter() - started) / reps * 1e6
    for o in outs:
        ttnn.deallocate(o)
    ttnn.deallocate(a)
    return {"issue_us": issued, "issue_plus_sync_us": done, "reps": reps}


def dispatch_scan(generator, prompt: list[int], *, rounds: int) -> dict:
    """Per-layer wall time with no per-layer synchronisation."""
    model = generator.model
    mesh = generator.mesh_device
    table = generator._coerce_page_table(None)
    per_layer: list[list[float]] = []
    totals: list[float] = []
    for _ in range(rounds):
        generator.reset()
        tt_tokens, _ = model.prefill_tokens_to_device(prompt)
        tt_page_table = model.page_table_to_device(table)
        hidden = model.embed_prefill(tt_tokens)
        ttnn.deallocate(tt_tokens)
        ttnn.synchronize_device(mesh)
        marks = []
        started = time.perf_counter()
        for layer in model.layers:
            t0 = time.perf_counter()
            out = layer.prefill_forward(hidden, page_table=tt_page_table, user_id=0, start_pos=0)
            ttnn.deallocate(hidden)
            hidden = out
            marks.append((time.perf_counter() - t0) * 1e3)
        issued = (time.perf_counter() - started) * 1e3
        ttnn.synchronize_device(mesh)
        drained = (time.perf_counter() - started) * 1e3
        per_layer.append(marks)
        totals.append((issued, drained))
        ttnn.deallocate(hidden)
        ttnn.deallocate(tt_page_table)
    best = min(totals, key=lambda t: t[1])
    flat = [v for r in per_layer for v in r]
    return {
        "issue_ms": best[0],
        "issue_plus_drain_ms": best[1],
        "per_layer_issue_ms_mean": sum(flat) / len(flat),
        "per_layer_issue_ms_min": min(flat),
        "rounds": totals,
    }


def profile_prefill(generator, prompt: list[int], *, top: int = 35) -> str:
    model = generator.model
    table = generator._coerce_page_table(None)

    def once():
        tt_tokens, _ = model.prefill_tokens_to_device(prompt)
        tt_page_table = model.page_table_to_device(table)
        hidden = model.embed_prefill(tt_tokens)
        ttnn.deallocate(tt_tokens)
        out = model.prefill_forward(hidden, page_table=tt_page_table, user_id=0)
        ttnn.synchronize_device(generator.mesh_device)
        ttnn.deallocate(out)
        ttnn.deallocate(tt_page_table)

    generator.reset()
    once()
    generator.reset()
    profiler = cProfile.Profile()
    profiler.enable()
    once()
    profiler.disable()
    buf = io.StringIO()
    pstats.Stats(profiler, stream=buf).sort_stats("tottime").print_stats(top)
    return buf.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--op-floor-reps", type=int, default=500)
    parser.add_argument("--out", default="prefill_host_probe.json")
    args = parser.parse_args()

    layer_indices = None if args.layers == "all" else [int(i) for i in args.layers.split(",")]
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    mesh = open_multichip_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    summary: dict = {"length": args.length, "max_seq_len": args.max_seq_len}
    generator = None
    try:
        summary["op_floor"] = op_floor(mesh, reps=args.op_floor_reps)
        say(f"PH op floor issue={summary['op_floor']['issue_us']:.2f} us/op")

        started = time.perf_counter()
        generator = build_generator(ROOT, mesh, max_seq_len=args.max_seq_len, layer_indices=layer_indices)
        summary["build_seconds"] = round(time.perf_counter() - started, 1)
        summary["layer_count"] = len(generator.model.layers)

        torch.manual_seed(17)
        vocab = generator.model.config.vocab_size
        prompt = [int(t) for t in torch.randint(0, vocab, (args.length,)).tolist()]
        generator.reset()
        generator.generate(prompt_token_ids=prompt, max_new_tokens=2, enable_trace=True)

        summary["dispatch"] = dispatch_scan(generator, prompt, rounds=args.rounds)
        d = summary["dispatch"]
        say(
            f"PH dispatch issue={d['issue_ms']:.2f} ms drain={d['issue_plus_drain_ms']:.2f} ms "
            f"per-layer issue mean={d['per_layer_issue_ms_mean']:.3f} ms"
        )

        text = profile_prefill(generator, prompt)
        (OUT / "logs" / f"prefill_cprofile_{args.length}.txt").write_text(text)
        say(text[:4000])
        say("PH_OK")
        return 0
    finally:
        path = OUT / args.out
        path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        say(f"PH summary -> {path}")
        if generator is not None:
            generator.teardown()
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
