# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does tracing the prefill actually recover the host-dispatch gap, and what does it cost?

`prefill_host_probe.py` / `prefill_opcount.py` / `ccl_host_probe.py` establish that the
128-token prefill is host-issue bound (54.9 ms of issue against 55.1 ms to drain, over
4122 ttnn dispatches at 9-60 us each) and that no collective implementation or
persistent-buffer variant moves the per-call cost.  Tracing is the only mechanism that
removes host issue, so the README's decision not to ship a prefill trace should rest on a
*measurement* of the upside and the cost, not on an estimate.

This probe measures it, on the real 52-layer build, at one padded prompt length:

* is the prefill even trace-safe -- no host write, read or synchronisation inside the
  captured region;
* what a warmed replay costs against the eager path;
* what capture costs, since that is the other half of the trade;
* whether replay is correct: the logits from a replay are compared against the eager
  logits over the same persistent inputs.

It is a probe, not a shipped path.  The generator does not gain a prefill trace: see
`README.md` "Where TTFT actually goes" for why a per-32-row-bucket trace with retained
intermediates does not pay back on non-repeating serving prompt lengths.

Usage::

    python doc/optimized_full_model/bench/prefill_trace_probe.py --length 128 --replays 10
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
from models.autoports.meta_models_muse_glimmer_30b.tt.model import dram_capacity_bytes  # noqa: E402
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (  # noqa: E402
    close_multichip_mesh,
    open_multichip_mesh,
)

OUT = ROOT / "doc/optimized_full_model"


def say(*args) -> None:
    print(*args, flush=True)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--layers", default="all")
    parser.add_argument(
        "--with-decode-traces",
        action="store_true",
        help="capture the decode and sampling traces first, i.e. the state a shipped prefill trace would have to fit alongside",
    )
    parser.add_argument("--out", default="prefill_trace_probe.json")
    args = parser.parse_args()

    layer_indices = None if args.layers == "all" else [int(i) for i in args.layers.split(",")]
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    mesh = open_multichip_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    summary: dict = {"length": args.length, "replays": args.replays, "max_seq_len": args.max_seq_len}
    generator = None
    trace_id = None
    try:
        generator = build_generator(ROOT, mesh, max_seq_len=args.max_seq_len, layer_indices=layer_indices)
        model = generator.model
        summary["layer_count"] = len(model.layers)
        torch.manual_seed(17)
        prompt = [int(t) for t in torch.randint(0, model.config.vocab_size, (args.length,)).tolist()]

        def dram_free() -> int:
            view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
            return int(view.total_bytes_free_per_bank) * int(view.num_banks)

        summary["dram_capacity_bytes"] = dram_capacity_bytes(mesh)
        summary["with_decode_traces"] = bool(args.with_decode_traces)
        if args.with_decode_traces:
            # The state a shipped prefill trace would have to coexist with: the decode
            # trace and the sampling trace already captured in the same trace region.
            generator.generate(prompt_token_ids=prompt, max_new_tokens=4, enable_trace=True)
            ttnn.synchronize_device(mesh)
            summary["decode_trace_id"] = str(generator._trace_id)
            summary["sampling_trace_captured"] = generator._sampling_captured
        summary["dram_free_before_capture_bytes"] = dram_free()

        # Persistent trace inputs, allocated before capture.
        tokens, padded_len = model.prefill_tokens_to_device(prompt)
        page_table = model.page_table_to_device(model.normalize_page_table(None))
        summary["padded_len"] = padded_len
        last = args.length - 1

        def forward():
            hidden = model.embed_prefill(tokens)
            out = model.prefill_forward(hidden, page_table=page_table, user_id=0)
            logits = model.prefill_logits(out, last_token_index=last)
            ttnn.deallocate(out)
            return logits

        # ---- eager reference, warmed
        generator.reset()
        ttnn.deallocate(forward())
        ttnn.synchronize_device(mesh)
        generator.reset()
        reference = model.logits_to_torch(forward())
        ttnn.synchronize_device(mesh)

        eager = []
        for _ in range(args.rounds):
            generator.reset()
            ttnn.synchronize_device(mesh)
            started = time.perf_counter()
            logits = forward()
            ttnn.synchronize_device(mesh)
            eager.append((time.perf_counter() - started) * 1e3)
            ttnn.deallocate(logits)
        summary["eager_ms"] = {"min": min(eager), "rounds": eager}
        say(f"PT eager prefill min={min(eager):.2f} ms over {args.rounds} rounds")

        # ---- capture
        generator.reset()
        ttnn.synchronize_device(mesh)
        started = time.perf_counter()
        try:
            trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
            traced_logits = forward()
            ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh)
        except Exception as exc:  # noqa: BLE001
            summary["capture_error"] = str(exc)[:2000]
            say(f"PT capture FAILED: {str(exc).splitlines()[0][:300]}")
            say("PT_OK")
            return 0
        summary["capture_ms"] = (time.perf_counter() - started) * 1e3
        summary["dram_free_after_capture_bytes"] = dram_free()
        summary["capture_retained_dram_bytes"] = (
            summary["dram_free_before_capture_bytes"] - summary["dram_free_after_capture_bytes"]
        )
        say(
            f"PT capture took {summary['capture_ms']:.2f} ms and retained "
            f"{summary['capture_retained_dram_bytes'] / 1e6:.1f} MB of DRAM per device"
        )

        # ---- replay, warmed
        generator.reset()
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        replayed = model.logits_to_torch(traced_logits)
        summary["replay_vs_eager"] = {
            "pcc": round(pcc(replayed, reference), 9),
            "bit_identical": bool(torch.equal(replayed, reference)),
            "max_abs_diff": round(float((replayed - reference).abs().max()), 6),
            "eager_argmax": int(reference[model.row_within_tile(last)].argmax()),
            "replay_argmax": int(replayed[model.row_within_tile(last)].argmax()),
        }
        say(f"PT replay vs eager {summary['replay_vs_eager']}")

        traced = []
        for _ in range(args.rounds):
            generator.reset()
            ttnn.synchronize_device(mesh)
            started = time.perf_counter()
            for _ in range(args.replays):
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            traced.append((time.perf_counter() - started) / args.replays * 1e3)
        summary["traced_ms"] = {"min": min(traced), "rounds": traced}
        summary["speedup"] = round(min(eager) / min(traced), 4)
        summary["payback_replays"] = round(summary["capture_ms"] / max(min(eager) - min(traced), 1e-9), 2)
        say(
            f"PT traced replay min={min(traced):.2f} ms -> {summary['speedup']:.2f}x, "
            f"capture pays back after {summary['payback_replays']:.1f} replays of this bucket"
        )
        say("PT_OK")
        return 0
    finally:
        path = OUT / args.out
        path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        say(f"PT summary -> {path}")
        if trace_id is not None:
            try:
                ttnn.release_trace(mesh, trace_id)
            except Exception as exc:  # noqa: BLE001
                say(f"PT failed to release the prefill trace: {exc}")
        if generator is not None:
            generator.teardown()
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
