#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 prefill MATRIX producer: (new tokens) x (cached tokens) against a live pipeline runner.

Driven by run_matrix.sh / matrix_row.sh (see README.md); direct use:

Run ON rank 0's host after every rank logs "setup complete, entering request loop":
  srun --jobid <id> --overlap -N1 -n1 -w <rank0 host> env <PREFILL_* env> python3 matrix_producer.py \
      --cached 61440 --new 640,1600,3072,5120,6900,32768,51200 --iters 3 --timing-dir <dir> --out results.jsonl

Semantics of one row (cached = C, a chunk multiple):
  1. prefill [0, C) into slot 0 once (C / chunk chunks), wait until the last rank (--last-rank) finished it;
  2. for each N: ITERS times push ceil(N / chunk) chunks at actual_start = C + k*chunk, actual_end =
     min(C + (k+1)*chunk, C + N), waiting for the last rank to finish before the next push burst. Every
     iteration overwrites the same positions [C, C+N), so the cached prefix [0, C) stays intact and every
     cell sees exactly C cached tokens on an IDLE pipeline (no queueing behind earlier chunks).
  TTFT = last-rank end of the burst's last chunk - producer push time of the first chunk (the producer and
  rank 0 share a host; the last rank is on another host, so this relies on NTP; h2d_ms = rank 0 compute_start -
  push time is the skew-free host->device latency). tok/s = N / TTFT.
  3. LOADED mode (--users U --reqs M, or --target-chunks): slots 1..U-1 also get the cached prefix once, then for
     each N the producer streams U*M requests of N new tokens back-to-back, round-robin across slots at CHUNK
     granularity (every slot advances one chunk per round), never waiting -> a fully filled pipeline. Reported:
       * steady-state throughput from the LAST rank's chunk cadence with the first and last `stages` chunks
         dropped (pipeline fill / drain excluded): new tok/s = (new tokens of the kept chunks) / window;
       * aggregate over the whole stream incl. tails (R*N / wall);
       * per-request TTFT under load (push of its first chunk -> last-rank end of its last chunk): median/p90
         over the requests whose first chunk entered a full pipeline (the first `stages` chunks of the stream
         are excluded). The push blocks when the runner's H2D FIFO is full, so this is the closed-loop TTFT
         of U zero-think-time users, not an open-loop arrival TTFT.
     The runner must be launched with PREFILL_NUM_USERS >= U (each slot holds its own cache; capacity per slot).
Completion is read from the runner's PREFILL_TIMING_DIR CSVs (rank<r>.csv: rank,c,compute_start,compute_ms;
one unbuffered line per chunk, c = per-rank running chunk index). Requires PREFILL_SYNC_PER_CHUNK=1.
"""

import argparse
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(
    0, os.environ.get("TT_METAL_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5)))
)

from loguru import logger  # noqa: E402
from matrix_math import read_rank_csv, steady_state, ttft_stats, wait_for_chunk  # noqa: E402  (same directory)

import ttnn  # noqa: E402
from models.demos.common.prefill.runners import prefill_producer as pp  # noqa: E402  (reads PREFILL_* env at import)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", type=int, required=True, help="cached tokens C (chunk multiple)")
    ap.add_argument("--new", type=str, required=True, help="comma list of new-token counts N")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--timing-dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True, help="results JSONL (appended)")
    ap.add_argument(
        "--trace",
        type=str,
        default=os.environ.get(
            "PREFILL_TRACE_DIR", "/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill/golden/longbook_56320"
        ),
        help="golden trace dir whose token_ids are tiled to the cache capacity (default: PREFILL_TRACE_DIR or the weka 55k golden)",
    )
    ap.add_argument("--last-rank", type=int, default=15)
    ap.add_argument("--timeout", type=float, default=900.0, help="per-burst completion timeout (s)")
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--users", type=int, default=0, help="LOADED mode: number of slots streamed round-robin (0 = off)")
    ap.add_argument(
        "--reqs", type=int, default=0, help="LOADED mode: requests per user (0 -> derive from --target-chunks)"
    )
    ap.add_argument("--target-chunks", type=int, default=240, help="LOADED mode: chunks per stream when --reqs is 0")
    ap.add_argument(
        "--stages", type=int, default=0, help="pipeline depth for the fill/drain exclusion (0 -> last_rank+1)"
    )
    ap.add_argument("--skip-idle", action="store_true", help="LOADED mode: skip the idle-pipeline cells")
    args = ap.parse_args()
    stages = args.stages or (args.last_rank + 1)

    chunk = pp.CHUNK_SIZE
    max_seq = pp.MAX_SEQ_LEN
    C = args.cached
    news = [int(x) for x in args.new.split(",") if x]
    assert C % chunk == 0, f"cached {C} must be a multiple of chunk {chunk}"
    need = C + math.ceil(max(news) / chunk) * chunk  # the runner writes whole chunks
    assert need <= max_seq, f"C + max N rounded up to chunks = {need} exceeds PREFILL_MAX_SEQ_LEN {max_seq}"
    pool = pp._load_token_pool(args.trace, max_seq)  # trace tiled cyclically to the cache capacity
    logger.info(
        f"[matrix] cached={C} new={news} iters={args.iters} chunk={chunk} max_seq={max_seq} "
        f"timing_dir={args.timing_dir} pool={len(pool)} tokens (trace {args.trace})"
    )

    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    timeout_s = int(os.environ.get("PREFILL_H2D_CONNECT_TIMEOUT", "60"))
    service = ttnn.H2DStreamService.connect(service_id, timeout_ms=timeout_s * 1000)
    payload_bytes = service.payload_size_bytes()
    logger.info(f"[matrix] attached to {service_id}; payload={payload_bytes}B")

    # The runner's per-rank chunk counter continues across producer sessions: start from what is there. Every
    # rank processes every chunk exactly once, so rank 0 and the last rank must have logged the same chunk set,
    # otherwise the pipeline is not idle (or a CSV row was lost) and our chunk indices would silently drift.
    rows0 = read_rank_csv(args.timing_dir, 0)
    rows_last = read_rank_csv(args.timing_dir, args.last_rank)
    n_prev = max(list(rows0) + list(rows_last), default=-1) + 1
    assert set(rows0) == set(rows_last) == set(range(n_prev)), (
        f"rank0 logged {len(rows0)} chunks, rank{args.last_rank} {len(rows_last)} (expected both = 0..{n_prev - 1}): "
        "pipeline not idle or timing CSVs incomplete; use a fresh PREFILL_TIMING_DIR"
    )
    next_c = n_prev
    new_tokens = {}  # chunk index -> new tokens carried (actual_end - actual_start)
    logger.info(f"[matrix] first chunk index c={next_c}")

    def push(actual_start: int, actual_end: int, slot: int = 0) -> float:
        nonlocal next_c
        arr = pp._chunk_to_host_array(pool[actual_start : actual_start + chunk])
        assert arr.nbytes == payload_bytes
        t = time.time()
        service.forward_to_tensor_bytes(arr, metadata=pp._pack_metadata(slot, actual_start, actual_end))
        new_tokens[next_c] = actual_end - actual_start
        next_c += 1
        return t

    def burst(start: int, n_tokens: int, slot: int = 0) -> dict:
        """Push the chunks covering [start, start+n_tokens) and wait for the last one on the last rank."""
        n_chunks = math.ceil(n_tokens / chunk)
        c_first = next_c
        t_push = []
        for k in range(n_chunks):
            a = start + k * chunk
            t_push.append(push(a, min(a + chunk, start + n_tokens), slot))
        c_last = next_c - 1
        last_start, last_ms = wait_for_chunk(args.timing_dir, args.last_rank, c_last, args.timeout)
        # Rank 0 shares our host and writes its row before forwarding the chunk, so its row for c_first must exist
        # and must not predate our push: that would mean our chunk index is not the runner's (stale CSVs).
        r0_first = read_rank_csv(args.timing_dir, 0).get(c_first, (None, None))[0]
        assert (
            r0_first is not None
        ), f"rank0 has no row for c={c_first} although rank{args.last_rank} finished c={c_last}"
        assert r0_first >= t_push[0] - 0.05, (
            f"rank0 started c={c_first} at {r0_first:.3f}, {t_push[0] - r0_first:.3f} s BEFORE we pushed it: chunk "
            "index misaligned with the runner (stale timing CSVs?)"
        )
        t_end = last_start + last_ms / 1000.0
        return {
            "n_chunks": n_chunks,
            "c_first": c_first,
            "c_last": c_last,
            "t_push_first": t_push[0],
            "t_rank0_start_first": r0_first,
            "t_last_end": t_end,
            "ttft_ms": (t_end - t_push[0]) * 1000.0,
            "h2d_ms": (r0_first - t_push[0]) * 1000.0,
            "last_rank_compute_ms": last_ms,
        }

    def loaded_stream(N: int, users: int, reqs: int) -> dict:
        """Stream users*reqs requests of N new tokens, round-robin across slots at chunk granularity, no waits."""
        n_chunks = math.ceil(N / chunk)
        req = {}  # (slot, k) -> dict(c_first, c_last, t_push_first)
        c0 = next_c
        for k in range(reqs):
            for j in range(n_chunks):
                a = C + j * chunk
                for s in range(users):
                    t = push(a, min(a + chunk, C + N), s)
                    r = req.setdefault((s, k), {"c_first": next_c - 1, "t_push_first": t})
                    r["c_last"] = next_c - 1
        c_end = next_c - 1
        t_stream0 = req[(0, 0)]["t_push_first"]
        wait_for_chunk(args.timing_dir, args.last_rank, c_end, args.timeout * 4)
        rl = read_rank_csv(args.timing_dir, args.last_rank)
        ends = {c: rl[c][0] + rl[c][1] / 1000.0 for c in range(c0, c_end + 1) if c in rl}
        assert len(ends) == c_end - c0 + 1, f"last rank missing {c_end - c0 + 1 - len(ends)} chunk rows"
        total_chunks = c_end - c0 + 1
        R = users * reqs
        wall = ends[c_end] - t_stream0
        steady = steady_state(ends, c0, c_end, stages, new_tokens, chunk)
        out = {
            "mode": "loaded",
            "cached": C,
            "new": N,
            "users": users,
            "reqs_per_user": reqs,
            "requests": R,
            "chunks_per_req": n_chunks,
            "total_chunks": total_chunks,
            "stages": stages,
            "wall_s": wall,
            "aggregate_new_tps": R * N / wall,
            "aggregate_processed_tps": total_chunks * chunk / wall,
            "label": args.label,
        }
        out.update(ttft_stats(req, ends, c_full=c0 + stages))
        if steady:
            out.update(steady)
        return out

    users = args.users
    if C > 0:
        logger.info(f"[matrix] === prefilling the cached prefix [0, {C}) : {C // chunk} chunks (slot 0)")
        t0 = time.perf_counter()
        b = burst(0, C)
        logger.info(
            f"[matrix] cached prefix done in {b['ttft_ms']:.0f} ms ({C / (b['ttft_ms'] / 1000):.0f} tok/s), wall {time.perf_counter() - t0:.1f} s"
        )
        for s in range(1, users):
            b = burst(0, C, s)
            logger.info(f"[matrix] cached prefix for slot {s} done in {b['ttft_ms']:.0f} ms")
    for N in news:
        if not args.skip_idle:
            cell = []
            for it in range(args.iters):
                b = burst(C, N)
                b.update(
                    {
                        "mode": "idle",
                        "cached": C,
                        "new": N,
                        "iter": it,
                        "tps": N / (b["ttft_ms"] / 1000.0),
                        "label": args.label,
                    }
                )
                logger.info(
                    f"[matrix] cell cached={C} new={N} iter={it}: TTFT {b['ttft_ms']:.1f} ms (h2d {b['h2d_ms']:.1f}) "
                    f"-> {b['tps']:.0f} new tok/s ({b['n_chunks']} chunks, last-rank compute {b['last_rank_compute_ms']:.0f} ms)"
                )
                with open(args.out, "a") as fh:
                    fh.write(json.dumps(b) + "\n")
                cell.append(b)
                time.sleep(0.5)  # margin only: the last rank's row already implies every rank finished the burst
            steady = cell[1:] if len(cell) > 1 else cell
            med = statistics.median(x["ttft_ms"] for x in steady)
            over = f"iters 1..{len(cell) - 1}" if len(cell) > 1 else "iter 0 only"
            logger.info(
                f"[matrix] CELL cached={C} new={N}: median TTFT {med:.1f} ms over {over} "
                f"(iter0 {cell[0]['ttft_ms']:.1f}) -> {N / (med / 1000):.0f} new tok/s"
            )
        if users > 0:
            n_chunks = math.ceil(N / chunk)
            reqs = args.reqs or max(3, math.ceil(args.target_chunks / (users * n_chunks)))
            logger.info(
                f"[matrix] === LOADED cached={C} new={N}: {users} users x {reqs} reqs = {users * reqs * n_chunks} chunks"
            )
            lr = loaded_stream(N, users, reqs)
            logger.info(
                f"[matrix] LOADED cached={C} new={N}: steady {lr.get('steady_new_tps', float('nan')):.0f} new tok/s "
                f"({lr.get('steady_processed_tps', float('nan')):.0f} processed, period {lr.get('chunk_period_ms_median', float('nan')):.1f} ms, "
                f"{lr.get('mid_chunks', 0)} mid chunks) | aggregate incl. tails {lr['aggregate_new_tps']:.0f} new tok/s over {lr['wall_s']:.1f} s | "
                f"TTFT under load median {lr['ttft_under_load_ms_median']:.0f} ms p90 {lr['ttft_under_load_ms_p90']:.0f} "
                f"[{lr['ttft_under_load_ms_min']:.0f}, {lr['ttft_under_load_ms_max']:.0f}] ({lr['ttft_under_load_requests']} of {lr['requests']} requests)"
            )
            with open(args.out, "a") as fh:
                fh.write(json.dumps(lr) + "\n")
            time.sleep(0.5)
    logger.info(f"[matrix] ROW cached={C} done; tables: matrix_table.py {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
