# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Which ttnn calls the prefill spends its host time in, by name.

``prefill_host_probe.py`` established that the 128-token prefill is *host-issue
bound*: 54.9 ms of host issue against 55.1 ms to drain, i.e. the device finishes
0.2 ms after the last dispatch, and ``cProfile`` puts 89 % of the wall time inside
one frame -- ``ttnn/decorators.py::FastOperation.__call__``.  That frame is every
ttnn op, so it says nothing about *which* ops.

This probe patches that one frame to count and time calls by
``python_fully_qualified_name`` over exactly one prefill.  The output is the
op-count budget: at ~13 us of host issue per call, op count is the prefill's
performance contract, and any call that does not produce device work is pure
overhead.

Usage::

    python doc/optimized_full_model/bench/prefill_opcount.py --length 128
"""

from __future__ import annotations

import argparse
import collections
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


class Counter:
    """Patches ``FastOperation.__call__`` for the duration of a ``with`` block.

    ``drain`` synchronises the device *before* each named op, so that op's recorded
    time is its own host issue with the command queue empty -- i.e. dispatch without
    the backpressure a call that follows a large matmul inherits.  This is what
    separates "the op is expensive to issue" from "the host was waiting for the
    device".  It makes the *total* meaningless (every sync serialises the pipeline),
    which is why it is a separate pass rather than the default.
    """

    def __init__(self, mesh=None, drain: tuple[str, ...] = ()) -> None:
        self.counts: collections.Counter = collections.Counter()
        self.times: collections.Counter = collections.Counter()
        self._mesh = mesh
        self._drain = drain
        self._original = None

    def __enter__(self):
        from ttnn.decorators import FastOperation

        counts, times = self.counts, self.times
        self._original = FastOperation.__call__

        original = self._original

        mesh, drain = self._mesh, self._drain

        def counting(op_self, *a, **k):
            if drain and op_self.python_fully_qualified_name in drain:
                ttnn.synchronize_device(mesh)
            started = time.perf_counter()
            try:
                return original(op_self, *a, **k)
            finally:
                name = op_self.python_fully_qualified_name
                counts[name] += 1
                times[name] += time.perf_counter() - started

        FastOperation.__call__ = counting
        return self

    def __exit__(self, *exc):
        from ttnn.decorators import FastOperation

        FastOperation.__call__ = self._original
        return False

    def report(self, top: int = 30) -> dict:
        total_calls = sum(self.counts.values())
        total_ms = sum(self.times.values()) * 1e3
        rows = []
        for name, ms in sorted(self.times.items(), key=lambda kv: -kv[1])[:top]:
            n = self.counts[name]
            rows.append(
                {
                    "op": name,
                    "calls": n,
                    "ms": round(ms * 1e3, 3),
                    "us_per_call": round(ms * 1e6 / n, 2),
                    "pct": round(ms * 1e3 / total_ms * 100, 2),
                }
            )
        return {"total_calls": total_calls, "total_ms": round(total_ms, 3), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--out", default="prefill_opcount.json")
    args = parser.parse_args()

    layer_indices = None if args.layers == "all" else [int(i) for i in args.layers.split(",")]
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    mesh = open_multichip_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    summary: dict = {"length": args.length}
    generator = None
    try:
        generator = build_generator(ROOT, mesh, max_seq_len=args.max_seq_len, layer_indices=layer_indices)
        summary["layer_count"] = len(generator.model.layers)
        torch.manual_seed(17)
        vocab = generator.model.config.vocab_size
        prompt = [int(t) for t in torch.randint(0, vocab, (args.length,)).tolist()]
        generator.reset()
        generator.generate(prompt_token_ids=prompt, max_new_tokens=2, enable_trace=True)

        model = generator.model
        table = generator._coerce_page_table(None)

        def one_prefill():
            tt_tokens, _ = model.prefill_tokens_to_device(prompt)
            tt_page_table = model.page_table_to_device(table)
            hidden = model.embed_prefill(tt_tokens)
            ttnn.deallocate(tt_tokens)
            out = model.prefill_forward(hidden, page_table=tt_page_table, user_id=0)
            ttnn.deallocate(out)
            ttnn.deallocate(tt_page_table)

        generator.reset()
        one_prefill()
        ttnn.synchronize_device(mesh)

        generator.reset()
        with Counter() as counter:
            started = time.perf_counter()
            one_prefill()
            issued_ms = (time.perf_counter() - started) * 1e3
        ttnn.synchronize_device(mesh)
        summary["prefill"] = counter.report()
        summary["prefill"]["wall_issue_ms"] = round(issued_ms, 3)
        say(f"OC prefill calls={summary['prefill']['total_calls']} wall_issue={issued_ms:.2f} ms")
        for row in summary["prefill"]["rows"]:
            say(f"OC   {row['op']:<44} n={row['calls']:>5} {row['ms']:>8.2f} ms {row['us_per_call']:>7.2f} us/call")
        say(
            f"OC unattributed = wall_issue - sum(per-op) = "
            f"{issued_ms - summary['prefill']['total_ms']:.2f} ms "
            f"(the patched frame times the ttnn call only; the model's own Python between calls is the rest)"
        )

        # ---- the same prefill with the device drained before each collective, so the
        # collectives' recorded time is dispatch without queue backpressure.
        collectives = ("ttnn.experimental.reduce_scatter_minimal_async", "ttnn.experimental.all_gather_async")
        generator.reset()
        with Counter(mesh=mesh, drain=collectives) as drained:
            one_prefill()
        ttnn.synchronize_device(mesh)
        summary["prefill_drained_collectives"] = drained.report()
        summary["prefill_drained_collectives"]["drained_ops"] = list(collectives)
        by_op = {row["op"]: row for row in summary["prefill_drained_collectives"]["rows"]}
        for op in collectives:
            loaded = next(r for r in summary["prefill"]["rows"] if r["op"] == op)
            say(
                f"OC drained {op:<44} {by_op[op]['us_per_call']:>7.2f} us/call "
                f"against {loaded['us_per_call']:>7.2f} us/call in the pipelined pass"
            )

        # The traced decode step, for contrast: the same op budget with zero host issue.
        say("OC_OK")
        return 0
    finally:
        path = OUT / args.out
        path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        say(f"OC summary -> {path}")
        if generator is not None:
            generator.teardown()
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
