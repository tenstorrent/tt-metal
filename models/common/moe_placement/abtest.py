# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Counterbalanced on-device A/B of the original (O) and placed (B) caches, and the ship/no-ship gate.

Runs alternate O,B,B,O,O,B (by default three per side) per workload, so drift in temperature or clocks hits
both sides alike. Each run is a timed traced prefill reported by the adapter as per-iteration whole-prefill
seconds (iteration 0 excluded). A failed run gets one device reset and one retry. State is kept in
<workdir>/ab/state.json, so an interrupted A/B resumes where it stopped.

Gate (all must hold):
  * every B run is faster than every O run on every workload (no overlap)
  * on golden workloads, the runs passed the adapter's correctness check (PCC) and their cache digests are
    identical across all runs, O and B
  * the median latency change on every workload is at most ``max_latency_change_pct`` (default 0 = any gain)
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

ORDER = ("O", "B", "B", "O", "O", "B")


def plan(workloads: dict, order=ORDER) -> list:
    seen = {"O": 0, "B": 0}
    out = []
    for side in order:
        seen[side] += 1
        for w in workloads:
            out.append((f"{w}-{side}{seen[side]}", w, side))
    # group by workload, keeping the alternation within each
    return sorted(out, key=lambda x: list(workloads).index(x[1]))


class State:
    def __init__(self, path: Path, **init):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else dict(runs={}, **init)

    def save(self, **kw):
        self.data.update(kw)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2) + "\n")
        tmp.replace(self.path)


def run_ab(adapter, workdir: Path, caches: dict, workloads: dict, order=ORDER, log=print) -> dict:
    """caches: {'O': path, 'B': path}. workloads: name -> dict(sequence=Path, golden=bool). The adapter's
    ``run(mode, sequence, cache, out)`` returns 0 on success and writes measurement.json in ``out``."""
    ab = workdir / "ab"
    (ab / "runs").mkdir(parents=True, exist_ok=True)
    st = State(ab / "state.json", model=adapter.spec.name)
    for label, w, side in plan(workloads, order):
        if st.data["runs"].get(label, {}).get("status") == "complete":
            continue
        out = ab / "runs" / label
        for attempt in (1, 2):
            if out.exists():
                out.rename(out.with_name(f"{label}.failed-{int(time.time())}"))
            adapter.check_device_free()
            st.save(current=label)
            t0 = time.time()
            rc = adapter.run(
                "perf" if workloads[w].get("golden") else "nopcc",
                workloads[w]["sequence"],
                caches[side],
                out,
                ab / f"{label}.log",
            )
            if rc == 0:
                st.data["runs"][label] = dict(
                    status="complete", minutes=round((time.time() - t0) / 60, 1), attempt=attempt
                )
                st.save(current=None)
                log(f"{label}: done in {st.data['runs'][label]['minutes']} min")
                break
            st.data["runs"][label] = dict(status="failed", rc=rc, attempt=attempt)
            st.save(current=None)
            log(f"{label}: failed (rc {rc}), resetting device")
            adapter.reset_device(ab / f"reset-after-{label}-{attempt}.log")
        else:
            st.save(status="failed", error=f"{label} failed twice")
            raise RuntimeError(f"{label} failed twice")
    result = summarize(ab, workloads)
    st.save(status="done", gate=result["gate"])
    return result


def summarize(ab: Path, workloads: dict, max_latency_change_pct: float = 0.0) -> dict:
    import hashlib

    runs = {}
    for d in sorted((ab / "runs").iterdir()):
        m = d / "measurement.json"
        if "failed" in d.name or not m.exists():
            continue
        j = json.loads(m.read_text())
        if j.get("status") != "complete":
            continue
        whole = [x["whole_seconds"] for x in j["iteration_totals"] if x["iteration"] != 0]
        dig = d / "cache-digests.jsonl"
        runs[d.name] = dict(
            median_whole_s=statistics.median(whole),
            whole_s=whole,
            tokens_sha256=j.get("tokens_sha256"),
            digests=hashlib.sha256(dig.read_bytes()).hexdigest() if dig.exists() else None,
        )
    sides, reasons, ok = {}, [], True
    for w, cfg in workloads.items():
        o = [v["median_whole_s"] for n, v in runs.items() if n.startswith(f"{w}-O")]
        b = [v["median_whole_s"] for n, v in runs.items() if n.startswith(f"{w}-B")]
        if not o or not b:
            ok = False
            reasons.append(f"{w}: missing runs")
            continue
        change = 100 * (statistics.median(b) / statistics.median(o) - 1)
        sides[w] = dict(
            O_runs_s=o,
            B_runs_s=b,
            O_median_s=statistics.median(o),
            B_median_s=statistics.median(b),
            latency_change_pct=round(change, 3),
            throughput_gain_pct=round(100 * (statistics.median(o) / statistics.median(b) - 1), 3),
            separated=max(b) < min(o),
        )
        if not sides[w]["separated"]:
            ok = False
            reasons.append(f"{w}: O and B runs overlap")
        if change > max_latency_change_pct:
            ok = False
            reasons.append(f"{w}: latency change {change:+.2f}% above {max_latency_change_pct:+.2f}%")
        if cfg.get("golden"):
            digests = {v["digests"] for n, v in runs.items() if n.startswith(f"{w}-")}
            sides[w]["cache_digests_identical"] = len(digests) == 1 and None not in digests
            if not sides[w]["cache_digests_identical"]:
                ok = False
                reasons.append(f"{w}: cache digests differ between runs")
    gate = dict(passed=ok, reasons=reasons or ["placed cache faster on every workload, outputs identical"])
    summary = dict(runs=runs, sides=sides, gate=gate)
    (ab / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
