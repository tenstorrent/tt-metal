#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Live view of a running (or finished) DiffusionGemma GPQA run.

``lm_eval`` writes its results and samples only at the very end, so for the three-plus hours a
198-question run takes the only visible state is a tqdm line and a growing server log. Everything
worth watching is in there — per-block latency, denoise steps, halt behaviour, the degeneracy
verdicts, DRAM — but reading it means remembering half a dozen grep incantations. This prints it.

Usage::

    watch_gpqa.py /home/zni/dg_runs/flip_8192/both            # one shot
    watch_gpqa.py /home/zni/dg_runs/flip_8192/both -f         # refresh every 30 s
    watch_gpqa.py /home/zni/dg_runs/flip_8192/both -f -i 10   # every 10 s

The run directory is an ``OUTPUT_ROOT`` from ``run_upfront_gpqa.sh`` — the one holding
``server.log``. It also accepts the parent of several arm directories and picks the one being
written to.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

PROGRESS = re.compile(r"(\d+)/(\d+) \[([^\]]*)\]")
METRIC = re.compile(r"DG_VLLM_METRIC (\{.*\})\s*$")
DEGEN = re.compile(r"DG_DEGENERACY (\{.*\})\s*$")


def newest(paths):
    paths = [p for p in paths if p.exists()]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def resolve(run_dir: Path) -> Path:
    """Accept either an arm directory or a parent of arm directories."""
    if (run_dir / "server.log").exists():
        return run_dir
    candidates = [d for d in run_dir.iterdir() if d.is_dir() and (d / "server.log").exists()]
    if not candidates:
        return run_dir
    return max(candidates, key=lambda d: (d / "server.log").stat().st_mtime)


def _progress_in(log: Path):
    # tqdm rewrites one line with \r; take the last complete match in the tail.
    with log.open("rb") as fh:
        fh.seek(max(0, log.stat().st_size - 200_000))
        tail = fh.read().decode("utf-8", "replace").replace("\r", "\n")
    hits = PROGRESS.findall(tail)
    if not hits:
        return {"log": log.name, "done": 0, "total": None, "eta": "?"}
    done, total, timing = hits[-1]
    return {"log": log.name, "done": int(done), "total": int(total), "eta": timing}


def read_progress(run_dir: Path):
    """Progress across every eval stage, furthest-along first.

    NOT the newest log by mtime. A run whose smoke stage completed all 198 questions and whose full
    stage was then killed leaves a `full.log` that is newer but reads `0/198` -- picking by mtime
    reported that a finished run had done nothing, which is exactly the misreading this script is
    supposed to prevent. Both stages are returned so a disagreement is visible rather than resolved
    silently in favour of the wrong one.
    """
    stages = [_progress_in(p) for p in (run_dir / "full.log", run_dir / "smoke.log") if p.exists()]
    if not stages:
        return None
    stages.sort(key=lambda s: s["done"], reverse=True)
    return {**stages[0], "others": stages[1:]}


def read_server(run_dir: Path):
    log = run_dir / "server.log"
    if not log.exists():
        return None
    blocks, degen, guard = [], [], 0
    with log.open("r", errors="replace") as fh:
        for line in fh:
            if "ending request at block" in line:
                guard += 1
            m = METRIC.search(line)
            if m:
                try:
                    blocks.append(json.loads(m.group(1)))
                except ValueError:
                    pass
                continue
            m = DEGEN.search(line)
            if m:
                try:
                    degen.append(json.loads(m.group(1)))
                except ValueError:
                    pass
    return {"blocks": blocks, "degen": degen, "guard": guard, "mtime": log.stat().st_mtime}


def read_score(run_dir: Path):
    """Scores appear only at the end; smoke/ counts too when its stage ran the whole set."""
    for stage in ("full", "smoke"):
        files = sorted((run_dir / stage).rglob("results_*.json")) if (run_dir / stage).exists() else []
        if not files:
            continue
        blob = json.loads(files[-1].read_text())
        for task, m in blob.get("results", {}).items():
            out = {"stage": stage, "task": task}
            for k, v in m.items():
                if k.startswith("exact_match"):
                    out[k] = v
            out["n"] = blob.get("n-samples", {}).get(task, {}).get("effective")
            smp = sorted((run_dir / stage).rglob("samples_*.jsonl"))
            if smp:
                # ONE RECORD PER FILTER, so a 198-question run writes 396 records and every count
                # taken over the raw lines is doubled. Reading those raw counts is how a run with
                # 137 empty answers looked like 274, and how a median length of 0 looked like the
                # model had produced nothing at all. Key on doc_id.
                per_doc = {}
                for line in open(smp[-1]):
                    r = json.loads(line)
                    per_doc[r.get("doc_id")] = (r.get("resps") or [[""]])[0][0]
                texts = list(per_doc.values())
                out["questions"] = len(texts)
                out["empty"] = sum(1 for t in texts if not t.strip())
                out["boxed"] = sum(1 for t in texts if "\\boxed" in t)
                out["chars_median"] = int(statistics.median(len(t) for t in texts)) if texts else 0
            return out
    return None


def read_engine_health(run_dir: Path):
    """Whether the vLLM EngineCore died, and on what.

    This is the loudest thing the log can say and the script used to omit it entirely. A single
    unservable request raising out of ``prefill_forward`` is a FATAL engine error in vLLM V1: the
    EngineCore process exits and every request queued behind it returns an empty completion, so the
    eval still reports a score -- computed over mostly-empty responses. On 07-28 that turned a run
    that was scoring at the CUDA reference into a reported 23.74%.
    """
    log = run_dir / "server.log"
    if not log.exists():
        return None
    fatal, cause, served = 0, None, 0
    with log.open("r", errors="replace") as fh:
        for line in fh:
            if "EngineCore encountered a fatal error" in line:
                fatal += 1
            elif fatal and cause is None:
                # The exception line follows the traceback; keep the first concrete one.
                for marker in ("RuntimeError:", "NotImplementedError:", "AssertionError:", "TT_FATAL", "Error:"):
                    if marker in line:
                        cause = line.split(marker, 1)[1].strip()[:220] or None
                        break
            if '"event": "request_release"' in line:
                served += 1
    return {"fatal": fatal, "cause": cause, "served": served}


def fmt_pct(x):
    return "-" if x is None else f"{x * 100:.2f}%"


def render(run_dir: Path) -> str:
    L = []
    L.append(f"run: {run_dir}")

    score = read_score(run_dir)

    # An engine death invalidates every number below it, so it goes FIRST.
    health = read_engine_health(run_dir)
    if health and health["fatal"]:
        L.append("")
        L.append("  *** EngineCore DIED -- every request queued after it returned an empty answer.")
        L.append(f"  *** {health['served']} requests were actually served; the rest are empty by")
        L.append("  *** failure, not by the model. Any score below is meaningless as a quality number.")
        if health["cause"]:
            L.append(f"  *** cause: {health['cause']}")
        L.append("")

    prog = read_progress(run_dir)
    if prog and prog.get("total"):
        pct = 100.0 * prog["done"] / prog["total"]
        L.append(f"progress: {prog['done']}/{prog['total']} ({pct:.0f}%)  [{prog['eta']}]  via {prog['log']}")
        for other in prog.get("others", []):
            if other["done"] != prog["done"]:
                L.append(f"          ({other['log']}: {other['done']}/{other.get('total') or '?'} -- a")
                L.append("           second stage that was killed or has not started)")
    else:
        L.append("progress: no eval log yet")

    srv = read_server(run_dir)
    if srv is None:
        L.append("server: no server.log")
        return "\n".join(L)

    age = time.time() - srv["mtime"]
    # A finished run is quiet for the same reason a hung one is. Results on disk tell them apart.
    if score:
        state = "   (run is FINISHED -- results are written)"
    elif age > 300:
        state = "   <-- STALLED?"
    else:
        state = ""
    L.append(f"server.log last written {age:.0f}s ago{state}")

    blocks = srv["blocks"]
    if blocks:
        lat = [b["block_latency_s"] for b in blocks if "block_latency_s" in b]
        steps = [b["denoise_steps"] for b in blocks if "denoise_steps" in b]
        commit = [b["commit_latency_s"] for b in blocks if "commit_latency_s" in b]
        halted = sum(1 for b in blocks if b.get("halted"))
        toks = sum(b.get("committed_tokens", 0) for b in blocks)
        L.append(f"blocks: {len(blocks)}  committed_tokens: {toks}  halted: {halted}/{len(blocks)}")
        if lat:
            L.append(f"  block latency  median {statistics.median(lat):.2f}s  min {min(lat):.2f}  max {max(lat):.2f}")
        if steps:
            L.append(
                f"  denoise steps  median {statistics.median(steps):.0f}  min {min(steps)}  max {max(steps)}"
                + (
                    f"   -> {statistics.median(lat) and 256 / statistics.median(lat):.0f} tok/s per block"
                    if lat
                    else ""
                )
            )
        if commit:
            share = 100.0 * statistics.median(commit) / statistics.median(lat) if lat else 0
            L.append(f"  commit         median {statistics.median(commit):.2f}s ({share:.0f}% of a block)")
        dram = [b["dram"]["free_gib"] for b in blocks if isinstance(b.get("dram"), dict)]
        if dram:
            L.append(f"  DRAM free      {dram[-1]:.2f} GiB (min seen {min(dram):.2f})")

    # Degeneracy: with the content-region rule the interesting split is verdict, not raw top_frac.
    L.append(f"guard trips (requests ended early): {srv['guard']}")
    degen = srv["degen"]
    if degen:
        verdicts = {}
        for d in degen:
            verdicts[str(d.get("verdict", "?"))] = verdicts.get(str(d.get("verdict", "?")), 0) + 1
        L.append(f"degeneracy telemetry: {len(degen)} canvases  verdicts " + str(verdicts))
        tails = [d["stop_tail"] for d in degen if "stop_tail" in d]
        ctok = [d["content_tokens"] for d in degen if "content_tokens" in d]
        if tails:
            L.append(f"  stop_tail      median {statistics.median(tails):.0f}  max {max(tails)}")
        if ctok:
            L.append(f"  content_tokens median {statistics.median(ctok):.0f}  min {min(ctok)}")
        if not tails and not ctok:
            L.append("  (no content-region fields — this run predates the content-region guard fix)")

    if score:
        keys = [k for k in score if k.startswith("exact_match") and not k.endswith("_stderr")]
        L.append(f"SCORE ({score['stage']} stage, n={score.get('n')}):")
        for k in sorted(keys):
            L.append(f"  {k:34s} {fmt_pct(score[k])}")
        q, empty = score.get("questions"), score.get("empty")
        L.append(
            f"  answers: {score.get('boxed')} boxed, {empty} empty of {q} questions,"
            f" median {score.get('chars_median')} chars"
        )
        L.append("  bar: A100 CUDA reference 70.71% / 70.20% flexible-extract (thinking, 2 reps)")
        # A score computed over mostly-empty answers says nothing about quality, so refuse to let it
        # stand next to the bar unqualified: report what the model scored where it actually answered.
        if q and empty and empty > 0.05 * q:
            ans = q - empty
            got = score.get("exact_match,flexible-extract")
            if got is not None and ans:
                on_ans = 100.0 * got * q / ans
                L.append(
                    f"  !! {empty}/{q} answers are EMPTY. Over the {ans} the model did answer that is"
                    f" {on_ans:.1f}%,"
                )
                L.append("     and comparing THAT to the bar needs the reference restricted to the same")
                L.append("     questions -- the served set is a prefix, not a random sample.")
    else:
        L.append("SCORE: not written yet (lm_eval emits results only at the end)")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("-f", "--follow", action="store_true", help="refresh until interrupted")
    ap.add_argument("-i", "--interval", type=float, default=30.0)
    args = ap.parse_args()

    if not args.run_dir.exists():
        print(f"no such run dir: {args.run_dir}", file=sys.stderr)
        return 2
    run_dir = resolve(args.run_dir)

    if not args.follow:
        print(render(run_dir))
        return 0
    try:
        while True:
            out = render(run_dir)
            if sys.stdout.isatty():
                os.system("clear")
            else:
                print("=" * 72)
            print(time.strftime("%H:%M:%S"), "\n" + out, flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
