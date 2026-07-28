#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Replay a served run's degeneracy telemetry through the current guard and diff the verdicts.

The tt-shield eval of 2026-07-27 (run 30285823000) ended 130 requests on
``DegenerateBlockError``. This replays every canvas the run measured -- from the ``DG_DEGENERACY``
telemetry in the vLLM server log -- through :mod:`degeneracy` and reports which verdicts change.

Each telemetry record carries ``distinct``/``top_id``/``top_frac``/``max_run`` but not the token
ids, so the canvas is RECONSTRUCTED: same length, same dominant id, same ``top_frac`` and same
``max_run``, with the dominant run placed at the tail -- the shape the server printed, an answer
followed by a wall of <eos>. Those are exactly the features the decision rule reads, so the replay
is a faithful test of the rule. Two honest limits: ``distinct`` is NOT reproduced (the identity of
the non-dominant ids is not recorded and the rule does not read it), and a future rule that looked
at which ids fill the prefix would need the raw canvases instead of this log.

Usage:
    python replay_degeneracy_verdicts.py <vllm_server.log> [--stop-ids 1,106,50]
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from models.experimental.diffusion_gemma.tt import degeneracy as DG  # noqa: E402

TELEMETRY = re.compile(
    r"DG_DEGENERACY start_pos=(?P<start>\d+) attempt=(?P<attempt>\d+) "
    r"distinct=(?P<distinct>\d+)/(?P<tokens>\d+) top_id=(?P<top_id>\d+) "
    r"top_frac=(?P<top_frac>[\d.]+) max_run=(?P<max_run>\d+)"
)
ENDED = re.compile(r"ending request at block (?P<block>\d+): degenerate committed canvas")


def reconstruct(record: dict) -> torch.Tensor:
    """A canvas with the recorded length, dominant id, ``top_frac`` and ``max_run``.

    The dominant id takes one trailing run of ``max_run``; whatever share ``top_frac`` still needs
    is spread over the prefix in chunks small enough not to beat that run, each chunk separated by
    a filler id so the chunks cannot merge with each other or with the tail.
    """
    total, top_id = record["tokens"], record["top_id"]
    top_count = round(record["top_frac"] * total)
    tail = min(record["max_run"], top_count)
    fillers = total - top_count
    remaining = top_count - tail
    filler_ids = iter(range(900_000, 900_000 + total + 1))  # ids that cannot collide with a real one
    prefix: list[int] = []
    if fillers == 0:
        prefix = [top_id] * remaining  # the whole canvas is the dominant id
    else:
        # `fillers` separators give `fillers` chunk slots; spread `remaining` evenly over them and
        # always end the prefix with a filler so the last chunk cannot extend the tail run.
        base, extra = divmod(remaining, fillers)
        for slot in range(fillers):
            prefix.extend([top_id] * (base + (1 if slot < extra else 0)))
            prefix.append(next(filler_ids))
    ids = (prefix + [top_id] * tail)[:total]
    if len(ids) < total:  # rounding slack: pad with fillers, never with the dominant id
        ids = [next(filler_ids) for _ in range(total - len(ids))] + ids
    return torch.tensor(ids, dtype=torch.long).reshape(1, total)


def parse(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text(errors="ignore").splitlines():
        match = TELEMETRY.search(line)
        if match:
            records.append(
                {
                    "start": int(match["start"]),
                    "distinct": int(match["distinct"]),
                    "tokens": int(match["tokens"]),
                    "top_id": int(match["top_id"]),
                    "top_frac": float(match["top_frac"]),
                    "max_run": int(match["max_run"]),
                    "ended_the_request": False,
                }
            )
        elif ENDED.search(line) and records:
            records[-1]["ended_the_request"] = True
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--stop-ids", default="1,106,50", help="checkpoint eos_token_id set")
    args = parser.parse_args()

    stop_ids = {int(x) for x in args.stop_ids.split(",") if x.strip()}
    records = parse(args.log)
    if not records:
        print(f"no DG_DEGENERACY telemetry in {args.log}", file=sys.stderr)
        return 1

    rows = []
    for record in records:
        canvas = reconstruct(record)
        before = DG.is_degenerate(DG.block_degeneracy(canvas))
        after_stats = DG.block_degeneracy(canvas, stop_token_ids=stop_ids)
        rows.append(
            {
                **record,
                "before": before,
                "after": DG.is_degenerate(after_stats, stop_token_ids=stop_ids),
                "stop_tail": after_stats.get("stop_tail", 0),
                "content_tokens": after_stats.get("content_tokens", record["tokens"]),
                "reconstruction_ok": abs(DG.block_degeneracy(canvas)["top_frac"] - record["top_frac"]) < 0.01
                and DG.block_degeneracy(canvas)["max_run"] == record["max_run"],
            }
        )

    bad = [r for r in rows if not r["reconstruction_ok"]]
    ended = [r for r in rows if r["ended_the_request"]]
    freed = [r for r in ended if not r["after"]]
    kept = [r for r in ended if r["after"]]
    new_trips = [r for r in rows if r["after"] and not r["before"]]

    print(f"canvases measured in the run:        {len(rows)}")
    print(f"  reconstruction mismatches:         {len(bad)}")
    print(f"requests the run ended (guard trip): {len(ended)}")
    print(f"  now allowed to commit:             {len(freed)}  <- normal completions restored")
    print(f"  still rejected:                    {len(kept)}  <- real content collapse")
    print(f"newly rejected (regression check):   {len(new_trips)}")
    if freed:
        content = [r["content_tokens"] for r in freed]
        print(
            f"  real tokens per restored block:   median {statistics.median(content):.0f}, "
            f"max {max(content)}, total {sum(content)}"
        )
    for row in kept:
        print(
            f"  still rejected: top_id={row['top_id']} top_frac={row['top_frac']:.3f} "
            f"max_run={row['max_run']} content={row['content_tokens']}/{row['tokens']}"
        )
    healthy = [r for r in rows if not r["ended_the_request"]]
    print(f"healthy blocks in the run:           {len(healthy)}")
    print(f"  newly rejected among them:         {sum(1 for r in healthy if r['after'] and not r['before'])}")
    return 0 if not bad and not new_trips else 2


if __name__ == "__main__":
    raise SystemExit(main())
