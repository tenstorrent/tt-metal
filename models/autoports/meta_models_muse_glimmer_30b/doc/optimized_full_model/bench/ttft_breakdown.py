# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where the batch-1 TTFT goes, on the real 52-layer build.

The full-model stage reported TTFT 65.48 ms at prompt 128 and never split it.
The reduced-variant Tracy profile prices *one* prefill layer at ~0.84 ms of
device time, so 52 of them plus the terminal path is ~44 ms -- which leaves ~21 ms
unattributed.  This probe attributes it, by timing each phase of the prefill with
a device synchronisation around it, and by measuring the same phases at several
prompt lengths so a fixed per-call host cost can be separated from the part that
scales with the prompt.

The decode loop is already at its floor (two traces account for the step to within
4 us), so prefill is the only place a large avoidable host gap can still be hiding.

Usage::

    python doc/optimized_full_model/bench/ttft_breakdown.py --lengths 128,256,512,1024,2048
"""

from __future__ import annotations

import argparse
import json
import pathlib
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


def sync(mesh) -> None:
    ttnn.synchronize_device(mesh)


def phase_breakdown(generator, prompt: list[int], *, rounds: int) -> dict:
    """Time each prefill phase with a synchronisation around it.

    Synchronising between phases *serialises* host dispatch against device work,
    so the sum of the phases is an upper bound on the real e2e time and each phase
    is a lower bound on its own device time.  The interesting number is the
    comparison against the unsynchronised e2e call: whatever the phases add up to
    that the pipelined call does not is dispatch that was already overlapped.
    """
    model = generator.model
    mesh = generator.mesh_device
    table = generator._coerce_page_table(None)
    out: dict[str, list[float]] = {k: [] for k in ("stage_tokens", "embed", "layers", "terminal", "sample", "total")}
    for _ in range(rounds):
        generator.reset()
        sync(mesh)
        t0 = time.perf_counter()
        tt_tokens, _padded = model.prefill_tokens_to_device(prompt)
        tt_page_table = model.page_table_to_device(table)
        sync(mesh)
        t1 = time.perf_counter()
        embedded = model.embed_prefill(tt_tokens)
        ttnn.deallocate(tt_tokens)
        sync(mesh)
        t2 = time.perf_counter()
        hidden = model.prefill_forward(embedded, page_table=tt_page_table, user_id=0)
        sync(mesh)
        t3 = time.perf_counter()
        logits = model.prefill_logits(hidden, last_token_index=len(prompt) - 1)
        ttnn.deallocate(hidden)
        ttnn.deallocate(tt_page_table)
        sync(mesh)
        t4 = time.perf_counter()
        generator._allocate_device_inputs()
        sampled = generator._sample_eager(logits, into_tokens=False)
        ttnn.deallocate(logits)
        sync(mesh)
        t5 = time.perf_counter()
        del sampled
        out["stage_tokens"].append((t1 - t0) * 1e3)
        out["embed"].append((t2 - t1) * 1e3)
        out["layers"].append((t3 - t2) * 1e3)
        out["terminal"].append((t4 - t3) * 1e3)
        out["sample"].append((t5 - t4) * 1e3)
        out["total"].append((t5 - t0) * 1e3)
    return {k: {"min": min(v), "mean": sum(v) / len(v), "rounds": v} for k, v in out.items()}


def layer_scan(generator, prompt: list[int], *, rounds: int) -> dict:
    """Per-layer prefill wall time, synchronised, so the stack cost is attributable."""
    model = generator.model
    mesh = generator.mesh_device
    table = generator._coerce_page_table(None)
    sliding: list[float] = []
    full: list[float] = []
    for _ in range(rounds):
        generator.reset()
        tt_tokens, _ = model.prefill_tokens_to_device(prompt)
        tt_page_table = model.page_table_to_device(table)
        hidden = model.embed_prefill(tt_tokens)
        ttnn.deallocate(tt_tokens)
        sync(mesh)
        for layer in model.layers:
            started = time.perf_counter()
            out = layer.prefill_forward(hidden, page_table=tt_page_table, user_id=0, start_pos=0)
            sync(mesh)
            elapsed = (time.perf_counter() - started) * 1e3
            (sliding if layer.config.is_sliding else full).append(elapsed)
            ttnn.deallocate(hidden)
            hidden = out
        ttnn.deallocate(hidden)
        ttnn.deallocate(tt_page_table)
    return {
        "sliding_ms": {"min": min(sliding), "mean": sum(sliding) / len(sliding), "n": len(sliding)},
        "full_ms": {"min": min(full), "mean": sum(full) / len(full), "n": len(full)},
        "synchronised_stack_ms": sum(sliding) / rounds + sum(full) / rounds,
    }


def e2e(generator, prompt: list[int], *, rounds: int) -> dict:
    mesh = generator.mesh_device
    vals = []
    for _ in range(rounds):
        generator.reset()
        sync(mesh)
        started = time.perf_counter()
        generator.generate(prompt_token_ids=prompt, max_new_tokens=1, enable_trace=True)
        vals.append((time.perf_counter() - started) * 1e3)
    return {"min": min(vals), "mean": sum(vals) / len(vals), "rounds": vals}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="128,256,512,1024,2048")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--layer-scan", action="store_true")
    parser.add_argument("--out", default="ttft_breakdown.json")
    args = parser.parse_args()

    lengths = [int(v) for v in args.lengths.split(",") if v.strip()]
    layer_indices = None if args.layers == "all" else [int(i) for i in args.layers.split(",")]
    (OUT / "logs").mkdir(parents=True, exist_ok=True)

    mesh = open_multichip_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    summary: dict = {"lengths": lengths, "rounds": args.rounds, "max_seq_len": args.max_seq_len, "by_length": {}}
    generator = None
    try:
        started = time.perf_counter()
        generator = build_generator(ROOT, mesh, max_seq_len=args.max_seq_len, layer_indices=layer_indices)
        summary["build_seconds"] = round(time.perf_counter() - started, 1)
        summary["layer_count"] = len(generator.model.layers)
        say(f"TB built {summary['layer_count']} layers in {summary['build_seconds']}s")

        torch.manual_seed(17)
        vocab = generator.model.config.vocab_size
        for length in lengths:
            prompt = [int(t) for t in torch.randint(0, vocab, (length,)).tolist()]
            # Warm every program for this length before timing anything.
            generator.reset()
            generator.generate(prompt_token_ids=prompt, max_new_tokens=2, enable_trace=True)
            entry = {
                "e2e_ttft_ms": e2e(generator, prompt, rounds=args.rounds),
                "phases_ms": phase_breakdown(generator, prompt, rounds=args.rounds),
            }
            if args.layer_scan:
                entry["layer_scan"] = layer_scan(generator, prompt, rounds=1)
            summary["by_length"][str(length)] = entry
            ph = entry["phases_ms"]
            say(
                f"TB len={length} e2e={entry['e2e_ttft_ms']['min']:.2f} ms | "
                f"tokens={ph['stage_tokens']['min']:.2f} embed={ph['embed']['min']:.2f} "
                f"layers={ph['layers']['min']:.2f} terminal={ph['terminal']['min']:.2f} "
                f"sample={ph['sample']['min']:.2f} sum={ph['total']['min']:.2f}"
            )
        say("TB_OK")
        return 0
    finally:
        path = OUT / args.out
        path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        say(f"TB summary -> {path}")
        if generator is not None:
            generator.teardown()
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
