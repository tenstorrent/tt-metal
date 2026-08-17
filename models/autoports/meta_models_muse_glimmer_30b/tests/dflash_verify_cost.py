# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What does one target prefill forward actually cost, as a function of its shape?

The verify forward is the whole remaining DFlash bottleneck (63.6 ms of a 117.4 ms
iteration against a 62.9 ms break-even budget), and every attempt to reason about it
from end-to-end numbers has been confounded:

* changing ``page_block_size`` invalidates every compiled program, so the run that
  tests it is **cold** and looks 3x slower than it is;
* ttnn caches kernels on disk across processes, so the same shape measured second is
  faster than measured first;
* the from-zero and aligned paths differ in row count *and* in which SDPA op runs
  (plain over the chunk vs chunked over the paged cache), so comparing them conflates
  two variables.

So this measures the forward directly on a grid of ``(start_pos, rows)``, repeating
each point until warm and reporting the warm median.  It answers the questions the
end-to-end runs cannot:

* how much of the cost is **fixed** per forward (52 layers of eager dispatch) versus
  **per-row**, which decides whether re-forwarding fewer committed rows is worth
  anything at all;
* whether the **chunked** SDPA path at ``start_pos > 0`` is more expensive than the
  plain path at ``start_pos == 0`` for the same row count -- i.e. whether the aligned
  restart pays for itself.

Usage::

    python -m models.autoports.meta_models_muse_glimmer_30b.tests.dflash_verify_cost
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from loguru import logger

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (
    build_generator,
    close_generator_mesh,
    open_generator_mesh,
)

#: ``(start_pos, rows)`` points.  ``start_pos=0`` uses the plain SDPA path over the
#: chunk; ``start_pos>0`` uses chunked SDPA over the paged cache, which is the aligned
#: verify's path.  Row counts are tile multiples so nothing is silently re-padded.
DEFAULT_GRID = (
    (0, 32),
    (0, 64),
    (0, 128),
    (0, 256),
    (64, 32),
    (64, 64),
    (128, 32),
    (128, 64),
    (256, 32),
    (256, 64),
    (512, 32),
    (512, 64),
    (1024, 32),
    (1024, 64),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=4, help="timed repeats per point, after one warmup")
    parser.add_argument("--page-block", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    mesh = open_generator_mesh()
    try:
        build_kwargs = {"max_batch_size": 1, "max_seq_len": args.max_seq_len}
        if args.page_block:
            build_kwargs["page_block_size"] = args.page_block
        gen = build_generator(".", mesh, **build_kwargs)
        model = gen.model
        block = int(model.config.page_block_size)
        logger.info(f"page_block_size={block}")

        table = gen._coerce_page_table(None)
        slot_row = model.page_table_row(table, 0)
        tt_page_table = model.page_table_row_to_device(slot_row)

        results = []
        for start_pos, rows in DEFAULT_GRID:
            if start_pos % block:
                logger.info(f"skip start_pos={start_pos}: not a multiple of page_block {block}")
                continue
            if start_pos + rows > args.max_seq_len:
                continue
            ids = [1] * rows
            timings: list[float] = []
            for repeat in range(args.repeats + 1):
                model.release_sliding_tails()
                tt_tokens, _ = model.prefill_tokens_to_device(ids)
                embedded = model.embed_prefill(tt_tokens)
                ttnn.deallocate(tt_tokens)
                ttnn.synchronize_device(model.mesh_device)
                started = time.perf_counter()
                hidden = model.prefill_forward(embedded, page_table=tt_page_table, user_id=0, start_pos=start_pos)
                ttnn.synchronize_device(model.mesh_device)
                elapsed = time.perf_counter() - started
                ttnn.deallocate(hidden)
                if repeat:  # first pass populates the program cache
                    timings.append(elapsed)
            median = statistics.median(timings)
            path = "plain" if start_pos == 0 else "chunked"
            results.append({"start_pos": start_pos, "rows": rows, "path": path, "median_ms": 1000.0 * median})
            print(f"  start_pos={start_pos:>5d} rows={rows:>4d} [{path:>7s}]  {1000.0 * median:8.2f} ms", flush=True)
        ttnn.deallocate(tt_page_table)

        print("\n" + "=" * 72)
        # Fixed vs per-row, fitted on the chunked path at one start_pos, which is what
        # the aligned verify actually runs.
        for sp in sorted({r["start_pos"] for r in results if r["start_pos"] > 0}):
            pts = sorted([r for r in results if r["start_pos"] == sp], key=lambda r: r["rows"])
            if len(pts) >= 2:
                (r0, t0), (r1, t1) = (pts[0]["rows"], pts[0]["median_ms"]), (pts[1]["rows"], pts[1]["median_ms"])
                per_row = (t1 - t0) / (r1 - r0)
                fixed = t0 - per_row * r0
                print(f"start_pos={sp:>5d}: fixed {fixed:7.2f} ms + {per_row:6.3f} ms/row")
        zero = [r for r in results if r["start_pos"] == 0]
        if len(zero) >= 2:
            (r0, t0), (r1, t1) = (zero[0]["rows"], zero[0]["median_ms"]), (zero[1]["rows"], zero[1]["median_ms"])
            per_row = (t1 - t0) / (r1 - r0)
            print(f"start_pos=    0: fixed {t0 - per_row * r0:7.2f} ms + {per_row:6.3f} ms/row  (plain SDPA)")
        print("=" * 72)

        out = Path(args.out) if args.out else Path(__file__).with_name("dflash_verify_cost.json")
        out.write_text(json.dumps({"page_block_size": block, "results": results}, indent=2))
        print(f"wrote {out}")
    finally:
        close_generator_mesh(mesh)


if __name__ == "__main__":
    main()
