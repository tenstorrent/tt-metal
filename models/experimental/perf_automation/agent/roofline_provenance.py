# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where every number in the Roofline section came from, and whether it can move.

WHY THIS EXISTS. "Is the roofline right?" was answered four times in one session by reading code,
and the answer changed each time -- because reading a call chain cannot tell you what a value did,
only what it might do. Two of those answers were wrong in opposite directions. The chain is real but
it is long (census -> anchor -> facts -> stage_roots -> roofs -> row), it forks (two render paths,
two "baseline" files with the same name), and the interesting property -- does this move mid-run --
lives in whether a WRITE happens twice, which no amount of reading settles.

So this asks the running system instead:

  * every ceiling input, its value, its source, and whether an anchor holds it
  * for each stack: both roofs, which one binds, and the measurement beside it
  * the physics check the memory roof gets for free and the compute roof does not --
    measured must sit under the ceiling, and the implied rate under the hardware peak

It reads. It never anchors, never writes, never profiles. Rendering a report must not change what
the next report says, and neither may auditing one.

    python3 -m agent.roofline_provenance                      # newest run
    python3 -m agent.roofline_provenance --run <run-dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]
if str(_PA) not in sys.path:
    sys.path.insert(0, str(_PA))

_PINNED = "PINNED"
_FLOATS = "FLOATS"
_ABSENT = "absent"


def _read(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return None


def _newest_run(root: Path):
    cands = sorted(root.glob("runs/*/profiles/baseline_profile.json"), key=lambda q: q.stat().st_mtime)
    return cands[-1].parent.parent if cands else None


def _state_dir():
    d = (os.environ.get("PERF_MCP_STATE_DIR") or "").strip()
    return Path(d) if d else None


def collect(run_dir: Path, model: str = "", task: str = "main") -> dict:
    """Every roofline input with its provenance. Pure read."""
    import cc_optimize.summary as S

    out = {"run": str(run_dir), "model": model, "task": task, "inputs": [], "stages": [], "notes": []}
    prof = _read(run_dir / "profiles" / "baseline_profile.json")
    if prof is None:
        out["notes"].append("no baseline_profile.json in this run -- nothing to audit")
        return out

    # HOW MANY TIMES THE PICTURE WAS WRITTEN is the question a code read cannot answer. One write
    # means the ceiling could not have moved however it is wired; more than one means it could.
    bp = run_dir / "profiles" / "baseline_profile.json"
    sd = _state_dir()
    live = None
    if sd and model:
        live = sd / ("perf_mcp_baseline_%s_%s.json" % (model, task))
    out["picture"] = {
        "final_path": str(bp),
        "final_mtime": bp.stat().st_mtime if bp.exists() else None,
        "live_path": str(live) if live else "",
        "live_mtime": live.stat().st_mtime if (live and live.exists()) else None,
    }

    led = S._ledger()

    def anchored(kind, depth):
        try:
            return led.anchor_value(kind, depth=depth, model=model, task=task)
        except Exception:  # noqa: BLE001
            return None

    unit = ""
    thr = None
    if sd and model:
        thr = _read(sd / ("perf_mcp_throughput_%s_%s.json" % (model, task)))
    if isinstance(thr, dict):
        unit = str(thr.get("unit") or "token")

    ab = anchored(led.KIND_ACTIVE_BYTES, unit or "token")
    out["inputs"].append(
        {
            "name": "active_bytes (memory roof numerator)",
            "value": ("%.3f GB" % (ab * 1e6 / 1e9)) if ab else "-",
            "held_by": "ledger KIND_ACTIVE_BYTES @ before" if ab else _ABSENT,
            "status": _PINNED if ab else _FLOATS,
        }
    )
    pk = anchored(led.KIND_PEAK_FLOPS, unit or "token")
    out["inputs"].append(
        {
            "name": "peak_flops (compute roof denominator)",
            "value": ("%.1f TFLOPS" % (pk / 1e12)) if pk else "-",
            "held_by": "ledger KIND_PEAK_FLOPS @ before" if pk else _ABSENT,
            "status": _PINNED if pk else _FLOATS,
        }
    )
    fl = None
    for d in (str((thr or {}).get("perf_layers") or ""), "all", "unknown"):
        if d:
            fl = anchored(led.KIND_FLOOR, d)
            if fl:
                break
    out["inputs"].append(
        {
            "name": "modeled_floor_ms",
            "value": ("%.2f ms" % fl) if fl else "-",
            "held_by": "ledger KIND_FLOOR" if fl else _ABSENT,
            "status": _PINNED if fl else _FLOATS,
        }
    )

    stage_ms = {}
    try:
        stage_ms = S._measured_stage_ms(model, task) or {}
    except Exception:  # noqa: BLE001
        stage_ms = {}
    bw = float((thr or {}).get("peak_bw_gbps") or 0.0)
    if ab and bw:
        roofs = S._stage_roofs(
            int(round(ab * 1e6)),
            bw,
            int((thr or {}).get("tp_degree") or 1),
            unit or "token",
            prof,
            stage_ms,
            model=model,
            task=task,
        )
        for name, r in (roofs or {}).items():
            ms = stage_ms.get(name)
            binds = r.get("binds")
            roof = r.get("compute_ms") if binds == "compute" else r.get("memory_ms")
            out["stages"].append(
                {
                    "stage": name,
                    "binds": binds,
                    "memory_ms": r.get("memory_ms"),
                    "compute_ms": r.get("compute_ms"),
                    "measured_ms": ms,
                    "peak_flops": r.get("peak_flops"),
                    "peak_flops_now": r.get("peak_flops_now"),
                    # THE CHECK THE MEMORY ROOF GETS FOR FREE. A floor above the measurement claims a
                    # rate above the hardware's, which is the shape every roofline defect took.
                    "physical": (None if not (roof and ms) else bool(roof <= ms)),
                }
            )
    else:
        out["notes"].append("no anchored bytes or peak bandwidth -- stage roofs not reconstructible")
    return out


def render(a: dict) -> str:
    L = [
        "Roofline provenance",
        "=" * 100,
        "run   : %s" % a.get("run", ""),
        "model : %s / %s" % (a.get("model") or "?", a.get("task") or "?"),
    ]
    pic = a.get("picture") or {}
    if pic.get("final_mtime") and pic.get("live_mtime"):
        same = abs(pic["final_mtime"] - pic["live_mtime"]) < 2.0
        L.append(
            "picture: final and live baselines written %s -- the compute roof %s"
            % (
                "together (one profile)" if same else "APART (the live picture was refreshed)",
                "could not have moved" if same else "COULD have moved between them",
            )
        )
    L += ["", "%-42s %-16s %-34s %s" % ("ceiling input", "value", "held by", "status"), "-" * 100]
    for i in a.get("inputs", []):
        L.append("%-42s %-16s %-34s %s" % (i["name"], i["value"], i["held_by"], i["status"]))
    if a.get("stages"):
        L += [
            "",
            "%-10s %-8s %-12s %-12s %-12s %s" % ("stage", "binds", "memory_ms", "compute_ms", "measured", "physical?"),
            "-" * 100,
        ]
        for s in a["stages"]:
            f = lambda v: ("%.2f" % v) if isinstance(v, (int, float)) else "-"  # noqa: E731
            L.append(
                "%-10s %-8s %-12s %-12s %-12s %s"
                % (
                    s["stage"],
                    s.get("binds") or "-",
                    f(s.get("memory_ms")),
                    f(s.get("compute_ms")),
                    f(s.get("measured_ms")),
                    "-"
                    if s.get("physical") is None
                    else ("ok" if s["physical"] else "IMPOSSIBLE — floor above measurement"),
                )
            )
        for s in a["stages"]:
            p, n = s.get("peak_flops"), s.get("peak_flops_now")
            if p and n and abs(p - n) > 1e9:
                L.append(
                    "  note: %s roof pinned at %.1f TFLOPS while the current build runs %.1f -- a real "
                    "fidelity change, not a stale number" % (s["stage"], p / 1e12, n / 1e12)
                )
    for n in a.get("notes", []):
        L.append("note: %s" % n)
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="", help="run directory (default: newest under runs/)")
    ap.add_argument("--model", default=os.environ.get("PERF_MCP_MODEL", ""), help="model key for the ledger")
    ap.add_argument("--task", default=os.environ.get("PERF_MCP_TASK", "main"))
    ns = ap.parse_args(argv)
    run = Path(ns.run) if ns.run else _newest_run(_PA)
    if not run:
        print("no run with a baseline_profile.json under %s/runs" % _PA)
        return 1
    model = ns.model
    if not model:
        sd = _state_dir()
        if sd:
            for q in sd.glob("perf_mcp_baseline_*_%s.json" % ns.task):
                model = q.name[len("perf_mcp_baseline_") : -len("_%s.json" % ns.task)]
                break
    print(render(collect(run, model, ns.task)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
