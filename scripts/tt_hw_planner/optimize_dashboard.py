# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Live web dashboard for the optimize loop.

``optimize`` writes everything it does to files already: the run directory gets ``state.json``,
``events.jsonl``, ``manifest.json`` and ``profiles/``; the state dir gets the lever-attempt log
(``cc_kernlog_*``), the measurement ledger (``perf_measurements_*``), per-stage timings
(``perf_mcp_stage_ms_*``) and the full-pipeline baseline. This module reads those files — and ONLY
reads them — and serves them as one JSON snapshot plus a single-page dashboard that polls it, so an
operator can watch levers land live instead of scrolling the run log afterwards.

Read-only is a hard requirement: a dashboard that consumed ``hitl_proposal.json`` the way the
orchestrator does (read + unlink) would steal proposals from the terminal pause screen, so the
proposal is peeked at, never unlinked, and decisions stay in the terminal.

Nothing here names a model stage, op, or lever: every label rendered comes out of the files the run
wrote, so a model that reports different stages tomorrow still renders.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# The perf tree location has ONE owner (the optimize command); the dashboard reads the same tree.
from .commands.optimize import PERF_DIR

# A run counts as LIVE when one of its files moved this recently. Comfortably wider than the UI poll
# interval so an active run never flickers, narrower than a lever measurement (minutes), so a dead
# run does not look alive.
_LIVE_WINDOW_S = 45.0

# Which files prove a run is still working. The run dir alone is not enough: the orchestrator writes
# state.json/events.jsonl once per ITERATION, and the attempt log only when a lever RESOLVES, both of
# which are minutes apart — so a busy run measuring a candidate would read as dead. The agent's log
# is the actual heartbeat (it moves every few seconds while the agent works), and under --persist it
# is the ONLY thing moving, because the run dir then lives in a sandbox checkout the agent never
# touches while state is written back here.
_LIVE_STATE_GLOBS = ("cc_kernlog_%s_*.json", "cc_kernlog_%s_*.json.agent.log")
_EVENTS_TAIL = 60
_RUN_ID_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})-(.+)$")

# HITL handshake file names, mirrored from cc_optimize/hitl.py (loaded by path, not imported — the
# engine lives outside this package). Proposals are PEEKED at, never unlinked: the orchestrator's
# read CONSUMES, and a dashboard that consumed would steal proposals from the terminal pause screen.
_HITL_PROPOSAL = "hitl_proposal.json"
_HITL_DECISION = "hitl_decision.json"
_HITL_ACTIONS = ("commit", "revert")


# --------------------------------------------------------------------------- tolerant file reads


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001 -- a live run rewrites these files mid-read; partial = absent
        return None


def _read_jsonl_tail(path: Path, limit: int) -> list:
    try:
        lines = path.read_text().splitlines()
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for raw in lines[-limit:]:
        try:
            rows.append(json.loads(raw))
        except ValueError:
            continue
    return rows


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


# --------------------------------------------------------------------------- run / state discovery


def run_slug(run_dir: Path) -> str | None:
    """The model slug a run directory is named for (``<ts>-<slug>``), else None."""
    m = _RUN_ID_RE.match(run_dir.name)
    return m.group(2) if m else None


def find_run_dir(repo_root: Path, slug: str | None = None, run_ref: str | None = None) -> Path | None:
    """Resolve the run to show: an explicit id/path, else the newest run (for ``slug`` if given)."""
    runs_root = Path(repo_root) / PERF_DIR / "runs"
    if run_ref:
        p = Path(run_ref)
        if p.is_dir():
            return p.resolve()
        cand = runs_root / run_ref
        return cand.resolve() if cand.is_dir() else None
    if not runs_root.is_dir():
        return None
    cands = []
    for d in runs_root.iterdir():
        if not d.is_dir() or not _RUN_ID_RE.match(d.name):
            continue
        if slug and run_slug(d) != slug:
            continue
        mt = _mtime(d)
        cands.append((mt or 0.0, d))
    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][1]


def repo_root_for_run(run_dir: Path, default: Path) -> Path:
    """The checkout a run directory belongs to, derived from the path itself. A run lives at
    ``<checkout>/models/experimental/perf_automation/runs/<id>`` — when the operator points the
    dashboard at another checkout's run (or a leftover worktree's), the state files must come from
    THAT checkout's .state, not from wherever the command happens to run."""
    anchor = (Path(PERF_DIR) / "runs").parts
    parts = Path(run_dir).resolve().parts
    for i in range(len(parts) - len(anchor), -1, -1):
        if parts[i : i + len(anchor)] == anchor:
            return Path(*parts[:i])
    return Path(default)


def state_dir_candidates(repo_root: Path, slug: str | None) -> list:
    """Where run memory can live, in priority order: $PERF_MCP_STATE_DIR (what a --persist run sets),
    the repo's durable .state/<slug>, then the /tmp default. Resolved per call so a dashboard started
    before the run exports its state dir still picks it up on the next poll."""
    out = []
    env = os.environ.get("PERF_MCP_STATE_DIR")
    if env:
        out.append(Path(env))
        # A --persist run points this at .state/<slug>, but the board-level files (the thermal/power
        # profile) are shared across models and live one level up in .state itself. Without the parent
        # the Power Analysis tab reads empty while its data sits on disk, so search both.
        out.append(Path(env).parent)
    state_root = Path(repo_root) / PERF_DIR / ".state"
    if slug:
        out.append(state_root / slug)
    out.append(state_root)
    out.append(Path(tempfile.gettempdir()))
    seen, uniq = set(), []
    for d in out:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def _glob_first(dirs: list, pattern: str) -> list:
    """All matches across candidate dirs, highest-priority dir first, no duplicates by name."""
    seen, out = set(), []
    for d in dirs:
        try:
            matches = sorted(d.glob(pattern))
        except OSError:
            continue
        for m in matches:
            if m.name not in seen:
                seen.add(m.name)
                out.append(m)
    return out


# --------------------------------------------------------------------------- attempt normalization


def _attempt_status(rec: dict, rows: list | None = None) -> str:
    """One word the UI can colour. beat_baseline is written ONLY by the commit (perf_mcp.py), so it
    is the kept/reverted line; claimed_beat_baseline is what the agent believed before committing.

    WITH `rows`, AN ATTEMPT IS KEPT WHEN A COMMIT BANKED IT. The attempt row itself never carries
    beat_baseline -- the commit writes a separate row -- so every banked win read "reverted" beside a
    second "kept" row for its commit (Qwen-Image-Edit: six wins, six "reverted", six "kept"). The link
    is the report's own (measurements.banking_commit), and a candidate whose PCC failed is discarded
    however it measured (measurements.pcc_failed)."""
    if rec.get("wedged") or rec.get("measurement_failed"):
        return "wedged"
    if rec.get("beat_baseline"):
        return "kept"
    if rows is not None:
        from models.experimental.perf_automation.cc_optimize.measurements import banking_commit, pcc_failed

        if banking_commit(rec, rows) is not None:
            return "kept"
        if pcc_failed(rec):
            return "reverted"
    if rec.get("claimed_beat_baseline"):
        return "reverted"
    return "no-gain"


def _metric_now(metric, ledger: dict):
    """The run's metric with `current` taken from the ledger's latest committed reading of it.

    state.json is written by the FSM when the loop starts and the optimize loop never touches it
    again, so its `current` stays at the start-of-run value for the whole run even as the ledger
    records every committed win (voxtral: `device_ms` read 535.46 "current" after the device-time
    series had fallen to 397.64; qwen the same). The ledger's device-time readings are the SAME
    quantity -- linked by EVIDENCE, not by name: the metric's baseline IS one of this series'
    readings. The FSM may pin that baseline to a reading taken mid-run at the resident depth, not the
    series' first `before` (voxtral's 535.46 is a depth-26 `after` of a 609 -> 398 series), so match
    the reading the baseline equals, then `current` is the latest reading AT THAT SAME DEPTH and
    `scope` names that slice. A metric whose baseline matches no reading is a different quantity and is
    left untouched (its `current` is not borrowed from an unrelated series)."""
    if not isinstance(metric, dict):
        return metric
    from models.experimental.perf_automation.cc_optimize import measurements as _m

    rows = [r for r in (ledger or {}).get(_m.KIND_EAGER) or [] if isinstance(r.get("value_ms"), (int, float))]
    base = metric.get("baseline")
    if not isinstance(base, (int, float)) or not rows:
        return metric
    # The reading the pinned baseline equals (anywhere in the series, not just the first `before`):
    # that is the proof this series IS the headline metric, and it tells us which depth/slice it is.
    matched = next((r for r in rows if abs(r["value_ms"] - base) < 1e-6), None)
    if matched is None:
        return metric
    depth = matched.get("depth")
    afters = [r["value_ms"] for r in rows if r.get("phase") == _m.PHASE_AFTER and r.get("depth") == depth]
    out = {**metric, "scope": {"depth": depth, "mode": matched.get("mode")}}
    if afters:
        out["current"] = afters[-1]
    return out


def _stage_starts(stages: list, ledger: dict) -> None:
    """Give each stage its START: its share of the BEFORE end-to-end reading, as the ledger pinned it.

    `baseline_ms` used to come from the gate's best-so-far file, which every win overwrites, so a
    stage card read baseline == current and no gain -- vae_decode "722 -> 722" after going from
    ~3464 ms. With a pin, `start_ms` is that pin and `baseline_ms` follows it; without one (a run
    from before the pin existed) nothing changes."""
    from models.experimental.perf_automation.cc_optimize import measurements as _m

    pins = {
        str(r.get("depth") or "").strip().lower(): r["value_ms"]
        for r in (ledger or {}).get(_m.KIND_STAGE_E2E) or []
        if r.get("phase") == _m.PHASE_BEFORE and isinstance(r.get("value_ms"), (int, float))
    }
    for st in stages or []:
        v = pins.get(str(st.get("name") or "").strip().lower())
        if v is not None:
            st["start_ms"] = v
            st["baseline_ms"] = v


def _fullpipe_baseline(ledger: dict):
    """The end-to-end (all layers) reading the run started from, as the ledger pinned it, or None."""
    from models.experimental.perf_automation.cc_optimize import measurements as _m

    return next(
        (
            r["value_ms"]
            for r in (ledger or {}).get(_m.KIND_FULLPIPE) or []
            if r.get("phase") == _m.PHASE_BEFORE and isinstance(r.get("value_ms"), (int, float))
        ),
        None,
    )


def _load_attempts(dirs: list, slug: str | None) -> list:
    """Lever attempts for this model across tasks, archive (.cumulative) union live log — the same
    union _load_attempts_all reads, so the dashboard agrees with the engine about what was tried."""
    pattern = "cc_kernlog_%s_*.json" % slug if slug else "cc_kernlog_*.json"
    logs = _glob_first(dirs, pattern + ".cumulative") + _glob_first(dirs, pattern)
    recs, seen = [], set()
    for path in logs:
        data = _read_json(path)
        if not isinstance(data, list):
            continue
        task = ""
        m = re.match(r"cc_kernlog_(.+?)_([^_]+)\.json", path.name.removesuffix(".cumulative"))
        if m and slug and m.group(1) == slug:
            task = m.group(2)
        for rec in data:
            if not isinstance(rec, dict):
                continue
            # The engine's _fold_cumulative COPIES live rows into the archive, so the two logs overlap
            # by design and a plain concatenation lists every attempt twice. Collapse on the same
            # identity key _load_attempts_all uses -- op, rung, measurement, note -- so a duplicated
            # row folds while two genuinely different variants of one rung both survive.
            key = (
                rec.get("op_signature"),
                (rec.get("kernel_kind") or "").lower(),
                rec.get("measured_ms"),
                (rec.get("note") or "")[:400],
            )
            if key in seen:
                continue
            seen.add(key)
            recs.append((rec, task))
    from models.experimental.perf_automation.cc_optimize.measurements import banking_commit

    rows = [r for r, _ in recs]
    # A COMMIT ROW THAT BANKED AN ATTEMPT IS THAT ATTEMPT'S PROOF, NOT A SECOND HISTORY ENTRY: its sha
    # goes onto the attempt. A commit row nothing links to still stands on its own, as it always did.
    banked = {id(r): banking_commit(r, rows) for r in rows}
    folded = {id(c) for c in banked.values() if c is not None}
    out = []
    for rec, task in recs:
        if id(rec) in folded:
            continue
        _c = banked.get(id(rec))
        before = rec.get("fullpipe_best_ms")
        after = rec.get("fullpipe_ms")
        delta_pct = None
        if isinstance(before, (int, float)) and isinstance(after, (int, float)) and before:
            delta_pct = (after - before) / before * 100.0
        out.append(
            {
                "op": rec.get("op_signature") or "?",
                "lever": rec.get("kernel_kind") or "?",
                "task": task,
                "status": _attempt_status(rec, rows),
                "measured_ms": rec.get("measured_ms"),
                "fullpipe_ms": after,
                "fullpipe_delta_ms": rec.get("fullpipe_delta_ms"),
                "before_ms": before,
                "after_ms": after,
                "delta_pct": delta_pct,
                "commit": rec.get("commit") or (_c or {}).get("commit"),
                "pcc": rec.get("pcc"),
                "note": rec.get("note") or "",
                "stages": [s for s in (rec.get("stages") or []) if isinstance(s, dict)],
            }
        )
    return out


def _parse_batch(run_dir: Path, requested: int | None = None) -> int | None:
    """The resolved batch / concurrent-user count for the perf run.

    The perf harness reports the batch it actually ran (perf_adapter.batch_report_line, after
    resolve_batch) in its profile log. Surfacing it lets the dashboard show the batch on top and
    label throughput as per-user. Reads the newest profile log; returns None if not found."""
    from models.experimental.perf_automation.agent.perf_adapter import parse_batch_report

    def _report_batch():
        # Harnesses that don't print a PERF_BATCH_* token (e.g. diffusion pipelines that write CSV
        # profiles, not a .log) still record the batch in RUN_REPORT.md ('batch: N') / console.log.
        # Universal fallback so batch shows for ANY model, not just the LLM decode harness.
        import re as _rre

        for _f in ("RUN_REPORT.md", "console.log"):
            try:
                _t = (run_dir / _f).read_text(errors="replace")
            except Exception:
                continue
            for _a, _b in _rre.findall(r"PERF_BATCH_[A-Z]+=(\d+)|batch:\s*(\d+)", _t)[::-1]:
                if _a or _b:
                    return int(_a or _b)
        return None

    _pb = run_dir / ".requested_batch"
    if _pb.is_file():
        try:
            return int(_pb.read_text().strip())
        except Exception:
            pass
    prof = run_dir / "profiles"
    if not prof.is_dir():
        return _report_batch() or (int(requested) if requested else None)
    logs = sorted(prof.glob("*.log"), key=lambda p: (p.stat().st_mtime if p.exists() else 0.0), reverse=True)
    for lg in logs:
        try:
            txt = lg.read_text(errors="replace")
        except Exception:
            continue
        served = parse_batch_report(txt)
        if served is None:
            import re as _re

            _m = _re.findall(r"PERF_BATCH_[A-Z]+=(\d+)", txt)  # any batch-report token (STREAMS/ROWS/...)
            served = int(_m[-1]) if _m else None
        if served is not None:
            return served
    # No harness batch report (e.g. a non-decode model that never prints one): fall back to the batch
    # the run was ASKED to drive (--batch). Batch is a property of the run, not of the LLM decode path.
    return _report_batch() or (int(requested) if requested else None)


def _serving_metrics(
    stages: list, fullpipe: dict | None, stage_unit: str | None, throughput: dict | None = None
) -> dict | None:
    """The serving-style headline metrics (first-token / per-token / end-to-end / throughput),
    derived from stage VALUES alone. Which stage is the per-token one is read from the banked
    per-token pipeline time (the full-pipeline baseline declares unit="token"), and the first-token
    stage is the dominant one-shot stage — so nothing here keys on a stage's NAME."""
    cur = {s["name"]: s["ms"] for s in stages if s.get("ms") is not None}
    base = {s["name"]: s["baseline_ms"] for s in stages if s.get("baseline_ms") is not None}
    if not cur:
        return None
    out = {
        "e2e_latency": {"ms": sum(cur.values()), "baseline_ms": sum(base.values()) if base else None},
    }
    fp_ms = (fullpipe or {}).get("full_pipeline_ms")
    if stage_unit == "token" and fp_ms:
        tok_name = min(cur, key=lambda n: abs(cur[n] - fp_ms))
        tok_ms = cur[tok_name]
        # Prefer the top-level, ledger-based throughput: its baseline is the TRUE original reading.
        # Deriving the baseline from fp_ms (the CURRENT full-pipeline) pins baseline==current and
        # hides the real gain as a bogus "0.0% vs baseline". Throughput is PER USER (= 1/TPOT).
        tt = throughput or {}
        tp_cur, tp_base = tt.get("current"), tt.get("baseline")
        base_tok_ms = (1000.0 / tp_base) if tp_base else fp_ms  # per-token decode = 1/throughput
        out["per_token"] = {"ms": tok_ms, "baseline_ms": base_tok_ms, "stage": tok_name}
        out["throughput"] = {
            "per_s": tp_cur if tp_cur is not None else ((1000.0 / tok_ms) if tok_ms else None),
            "baseline": tp_base if tp_base is not None else (1000.0 / fp_ms),
            "unit": "tok/s/user",
        }
        oneshot = {n: v for n, v in cur.items() if n != tok_name}
        if oneshot:
            first = max(oneshot, key=lambda n: oneshot[n])
            out["first_token"] = {"ms": oneshot[first], "baseline_ms": base.get(first), "stage": first}
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _attempt_matches_bucket(op_sig: str, bucket: dict) -> bool:
    """Advisory match of an attempt's op_signature to a profile bucket (the engine's own _op_match is
    the authority for clearing rungs; the dashboard only needs 'has anyone touched this hot op')."""
    sig = _norm(op_sig)
    if not sig:
        return False
    if sig in _norm(str(bucket.get("id", ""))) or _norm(str(bucket.get("id", ""))) in sig:
        return True
    for top in bucket.get("top_ops") or []:
        code = _norm(str(top.get("op_code", "")))
        if code and (sig in code or code in sig):
            return True
    return False


# A top_op "shape" printed as an explicit contraction ("32x3072 @ 3072x131072") is the profiler's
# own marker for a matmul-class op — the "@" in the data, not a name matched in code. FLOPs for such
# an op are definitional: 2*M*K*N.
_CONTRACTION_RE = re.compile(r"(\d+)\s*x\s*(\d+)\s*@\s*(\d+)\s*x\s*(\d+)")


def _roofline_points(buckets: list) -> list:
    """One chart point per profiled contraction op: arithmetic intensity vs achieved TFLOP/s."""
    pts = []
    for b in buckets:
        for top in b.get("top_ops") or []:
            if not isinstance(top, dict):
                continue
            m = _CONTRACTION_RE.search(str(top.get("shape") or ""))
            ms, byts, count = top.get("device_ms"), top.get("bytes"), top.get("count") or 1
            if not (m and ms and byts):
                continue
            rows, k, _k2, cols = (int(g) for g in m.groups())
            flops = 2.0 * rows * k * cols * count
            pts.append(
                {
                    "op": top.get("op_code"),
                    "bucket": b.get("id"),
                    "ms": ms,
                    "bytes": byts,
                    "intensity": flops / byts,
                    "tflops": flops / (ms / 1000.0) / 1e12,
                }
            )
    return pts


# --------------------------------------------------------------------------- the snapshot


def _parallelism(ledger: dict, topology, batch) -> "dict | None":
    """Parallel-scaling facts for the Scaling view, read from the run itself (never a mesh string):
    tensor-parallel degree (ledger `tp_degree`), device count (board topology), the implied data-
    parallel degree, and the batch/users. None when the run declares nothing."""

    def _num(r):
        if not isinstance(r, dict):
            return None
        for k in ("value", "value_ms", "degree"):
            v = r.get(k)
            if isinstance(v, (int, float)) and v > 0:
                return int(v)
        return None

    tp = None
    for r in (ledger or {}).get("tp_degree") or []:
        n = _num(r)
        if n:
            tp = n
    devices = len(topology) if isinstance(topology, dict) and topology else None
    dp = (devices // tp) if (tp and devices and devices >= tp and devices % tp == 0) else None
    out = {"tp": tp, "dp": dp, "devices": devices, "batch": batch}
    return out if any(v is not None for v in out.values()) else None


def collect_state(run_dir: Path, state_dirs: list, slug: str | None = None, requested_batch: int | None = None) -> dict:
    """Assemble the one JSON snapshot the dashboard renders. Every section is best-effort: a file
    that does not exist yet (baseline still measuring) simply omits its section, never fails."""
    run_dir = Path(run_dir)
    slug = slug or run_slug(run_dir)
    now = time.time()

    manifest = _read_json(run_dir / "manifest.json") or {}
    config = manifest.get("config") or {}
    state = _read_json(run_dir / "state.json") or {}
    events = _read_jsonl_tail(run_dir / "events.jsonl", _EVENTS_TAIL)

    watched = [run_dir / "state.json", run_dir / "events.jsonl"]
    mtimes = [m for m in (_mtime(p) for p in watched) if m is not None]
    for pattern in _LIVE_STATE_GLOBS:
        for log in _glob_first(state_dirs, pattern % slug if slug else pattern % "*"):
            mt = _mtime(log)
            if mt is not None:
                mtimes.append(mt)
    age = (now - max(mtimes)) if mtimes else None

    # Per-stage timings + the full-pipeline baseline the per-lever gate banks (both keyed by the
    # names the MODEL reported, so the UI never assumes a stage set).
    stages_cur, stage_paths, stage_bytes, stage_unit = {}, {}, {}, None
    for f in _glob_first(state_dirs, "perf_mcp_stage_ms_%s_*.json" % slug if slug else "perf_mcp_stage_ms_*.json"):
        d = _read_json(f)
        if isinstance(d, dict) and isinstance(d.get("stages"), dict):
            stages_cur.update(d["stages"])
            stage_paths.update(d.get("paths") or {})
            stage_bytes.update(d.get("bytes") or {})
    fullpipe = None
    for f in _glob_first(
        state_dirs,
        (
            "perf_mcp_full_pipeline_baseline_1cq_%s_*.json" % slug
            if slug
            else "perf_mcp_full_pipeline_baseline_1cq_*.json"
        ),
    ):
        if f.name.endswith(".pending.json"):
            continue
        d = _read_json(f)
        if isinstance(d, dict) and d.get("full_pipeline_ms") is not None:
            fullpipe = d
            stage_unit = d.get("unit") or stage_unit
            break
    stages_base = (fullpipe or {}).get("stages") or {}
    # THE BAR'S OWN SPLIT WINS PER STAGE, same reasoning as summary._measured_stage_ms: it lives in
    # the SAME file as the fullpipe headline this dashboard already treats as ground truth (the
    # "current"/"baseline" throughput just below reads straight off it), so it cannot disagree with
    # that headline the way the per-run doc above can -- that one only updates when a measurement
    # happens to pass the optional stages_json argument to record_kernel_attempt, and was observed
    # showing a stage's timing from hours before the run's actual current committed-best. Falls back
    # to the per-run doc's own reading only for a stage the bar has not covered yet.
    stages_cur = {**stages_cur, **stages_base}

    stage_names = sorted(
        set(stages_cur) | set(stages_base), key=lambda n: -(stages_cur.get(n) or stages_base.get(n) or 0)
    )
    stages = [
        {
            "name": n,
            "ms": stages_cur.get(n),
            # The committed-best split is the CURRENT time, not the baseline -- using it for both
            # read baseline == current and hid every per-stage gain. The per-stage baseline is the
            # START, recorded only as a KIND_STAGE_E2E pin; _stage_starts fills it below where one
            # exists. No pin -> None (the UI shows current + the real end-to-end gain instead).
            "baseline_ms": None,
            "path": stage_paths.get(n),
            "bytes": stage_bytes.get(n),
        }
        for n in stage_names
    ]

    # The ledger, grouped by the kinds the run recorded (floors, roofs, anchors) — passed through,
    # not interpreted, so a new kind tomorrow renders without a dashboard change.
    ledger = {}
    for f in _glob_first(state_dirs, "perf_measurements_%s_*.jsonl" % slug if slug else "perf_measurements_*.jsonl"):
        for row in _read_jsonl_tail(f, 2000):
            if isinstance(row, dict) and row.get("kind"):
                ledger.setdefault(row["kind"], []).append(row)

    metric = _metric_now(state.get("metric"), ledger)
    _stage_starts(stages, ledger)

    # Throughput only when the run itself declared a per-token unit (the full-pipeline baseline's
    # "unit" field). Current comes from the ledger's committed after-rows — earliest-reading-wins
    # durability means the before row is the TRUE original, not this run's starting point.
    throughput = None
    if stage_unit == "token":
        fp = ledger.get("fullpipe_e2e") or []
        before = next((r.get("value_ms") for r in fp if r.get("phase") == "before"), None)
        afters = [r.get("value_ms") for r in fp if r.get("phase") == "after" and r.get("value_ms")]
        base_ms = before or (fullpipe or {}).get("full_pipeline_ms")
        cur_ms = afters[-1] if afters else None
        throughput = {
            "unit": "tok/s",
            "baseline": (1000.0 / base_ms) if base_ms else None,
            "current": (1000.0 / cur_ms) if cur_ms else None,
        }

    profile = _read_json(run_dir / "profiles" / "baseline_profile.json")
    buckets = []
    if isinstance(profile, dict):
        for b in profile.get("buckets") or []:
            if isinstance(b, dict):
                buckets.append(b)

    attempts = _load_attempts(state_dirs, slug)
    opportunities = []
    for b in sorted(buckets, key=lambda x: -(x.get("device_ms") or 0)):
        tried = sorted({a["lever"] for a in attempts if _attempt_matches_bucket(a["op"], b)})
        kept = any(a["status"] == "kept" for a in attempts if _attempt_matches_bucket(a["op"], b))
        opportunities.append(
            {
                "id": b.get("id"),
                "device_ms": b.get("device_ms"),
                "pct": b.get("pct"),
                "count": b.get("count"),
                "tags": b.get("tags") or {},
                "lever_state": b.get("lever_state") or {},
                "top_ops": (b.get("top_ops") or [])[:5],
                "tried_rungs": tried,
                "status": "cleared" if kept else ("touched" if tried else "open"),
            }
        )

    # Model-level headroom: the modeled floor is anchored in the ledger (kind "modeled_floor",
    # depth "all") — the one number that turns "current" into "how much is left on the table".
    headroom = None
    floors = [
        r.get("value_ms")
        for r in ledger.get("modeled_floor") or []
        if isinstance(r, dict) and r.get("depth") == "all" and r.get("value_ms")
    ]
    serving = _serving_metrics(stages, fullpipe, stage_unit, throughput)
    if floors:
        cur_total = (serving or {}).get("e2e_latency", {}).get("ms")
        if cur_total:
            headroom = {
                "floor_ms": floors[0],
                "current_ms": cur_total,
                "pct": max(0.0, (1.0 - floors[0] / cur_total) * 100.0),
            }

    # Roofline chart inputs: the roofs come from the run's OWN anchors — the manifest's env block
    # (DRAM bandwidth, per-core peaks) and the ledger's pinned peak_flops — and the points from the
    # profile's contraction ops. Nothing is restated here; a missing anchor just omits that roof.
    env = manifest.get("env") or {}
    peak_rows = [r.get("value_ms") for r in ledger.get("peak_flops") or [] if isinstance(r, dict) and r.get("value_ms")]
    roofline = {
        "points": _roofline_points(buckets),
        "bw_gbps": env.get("dram_bw_gbps"),
        "peak_tflops": (peak_rows[0] / 1e12) if peak_rows else None,
    }
    if not (roofline["points"] or roofline["bw_gbps"] or roofline["peak_tflops"]):
        roofline = None

    # HITL: PEEK ONLY. The orchestrator's read consumes the proposal; the dashboard must not.
    proposal = _read_json(run_dir / _HITL_PROPOSAL)

    thermal = None
    for f in _glob_first(state_dirs, "perf_mcp_thermal_profile.json"):
        thermal = _read_json(f)
        if thermal is not None:
            break
    topology = None
    for f in _glob_first(state_dirs, "perf_mcp_board_topology.json"):
        topology = _read_json(f)
        if topology is not None:
            break

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "run": {
            "id": run_dir.name,
            "dir": str(run_dir),
            "live": bool(age is not None and age <= _LIVE_WINDOW_S),
            "age_s": round(age, 1) if age is not None else None,
            "state": state.get("state"),
            "iteration": state.get("iteration"),
        },
        "model": {
            "slug": slug,
            "root": config.get("model_root"),
        },
        "config": {
            k: config.get(k)
            for k in ("metric", "devices", "pcc_test", "perf_test", "max_iter")
            if config.get(k) is not None
        },
        "metric": metric,
        "batch": _parse_batch(run_dir, requested_batch),
        "stages": stages,
        "serving": serving,
        "headroom": headroom,
        "throughput": throughput,
        "fullpipe_ms": (fullpipe or {}).get("full_pipeline_ms"),
        "fullpipe_baseline_ms": _fullpipe_baseline(ledger),
        "attempts": attempts,
        "opportunities": opportunities,
        "hitl_proposal": proposal if isinstance(proposal, dict) else None,
        "events": list(reversed(events)),
        "ledger": ledger,
        "roofline": roofline,
        "env": env or None,
        "thermal": thermal,
        "topology": topology,
        "parallelism": _parallelism(ledger, topology, _parse_batch(run_dir, requested_batch)),
    }


def post_hitl_decision(run_dir: Path, action: str) -> tuple:
    """Write the operator's HITL decision — the same file the orchestrator's terminal path posts
    (hitl.py post_decision). Refused when no proposal is pending, so a stale click cannot queue a
    decision for a FUTURE proposal; a race with the terminal is benign either way, because the next
    post_proposal deletes any unconsumed decision file."""
    if action not in _HITL_ACTIONS:
        return False, "unknown action %r (want one of %s)" % (action, ", ".join(_HITL_ACTIONS))
    run_dir = Path(run_dir)
    if not _read_json(run_dir / _HITL_PROPOSAL):
        return False, "no proposal pending — the run already moved on"
    try:
        (run_dir / _HITL_DECISION).write_text(json.dumps({"action": action, "note": "dashboard", "knob": ""}))
    except OSError as exc:
        return False, str(exc)
    return True, "posted"


# --------------------------------------------------------------------------- HTTP serving


def make_handler(collect_fn, decision_fn=None):
    class DashboardHandler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 -- BaseHTTPRequestHandler names it
            if self.path in ("/", "/index.html"):
                self._send(200, _HTML.encode(), "text/html; charset=utf-8")
            elif self.path.startswith("/api/state"):
                try:
                    payload = json.dumps(collect_fn(), default=str).encode()
                    self._send(200, payload, "application/json")
                except Exception as exc:  # noqa: BLE001 -- a bad poll must not kill the server
                    self._send(500, json.dumps({"error": str(exc)}).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            if self.path.startswith("/api/hitl-decision") and decision_fn is not None:
                try:
                    body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                    action = (json.loads(body or b"{}") or {}).get("action", "")
                except ValueError:
                    action = ""
                ok, msg = decision_fn(action)
                self._send(200 if ok else 409, json.dumps({"ok": ok, "message": msg}).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, *_args):  # keep the run's console clean
            pass

    return DashboardHandler


def make_server(host: str, port: int, collect_fn, decision_fn=None) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), make_handler(collect_fn, decision_fn))
    srv.daemon_threads = True
    return srv


def _reclaim_dashboard_port(port: int) -> bool:
    """Free ``port`` if a STALE planner dashboard/proxy is squatting it (a leftover from a prior run
    or a manual serve), so a new run binds its expected port instead of falling back to a random one
    the UI can't find. Terminates ONLY our own dashboard/proxy processes — identified by their command
    line — never an unrelated service. Returns True if it freed the port. Linux-only, best-effort."""
    import signal
    import subprocess

    pids: set[int] = set()
    try:
        out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return False
    for line in out.splitlines():
        if (":%d " % port) not in line:
            continue
        for m in re.findall(r"pid=(\d+)", line):
            pids.add(int(m))
    freed = False
    for pid in pids:
        try:
            cl = Path("/proc/%d/cmdline" % pid).read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
        except Exception:
            continue
        # only OUR own dashboard/proxy squatters — never anything else
        ours = ("optimize-dashboard" in cl) or ("dash_proxy" in cl) or ("tt_hw_planner" in cl and "dashboard" in cl)
        if ours:
            try:
                os.kill(pid, signal.SIGTERM)
                freed = True
            except Exception:
                pass
    if freed:
        time.sleep(1.0)  # let the socket release before rebinding
    return freed


def _make_server_or_reclaim(host: str, port: int, collect_fn, decision_fn=None):
    """make_server on the requested port; if it's busy because a stale planner dashboard/proxy holds
    it, free that (our process only) and retry so the run keeps its expected port — falling back to a
    free port only if the squatter isn't ours."""
    try:
        return make_server(host, port, collect_fn, decision_fn)
    except OSError as exc:
        if port == 0:
            raise
        if _reclaim_dashboard_port(port):
            try:
                srv = make_server(host, port, collect_fn, decision_fn)
                print("  [dashboard] reclaimed port %d from a stale planner dashboard" % port)
                return srv
            except OSError:
                pass
        print("  [dashboard] port %d unavailable (%s); using a free port instead" % (port, exc))
        return make_server(host, 0, collect_fn, decision_fn)


def serve(host: str, port: int, collect_fn, decision_fn=None) -> int:
    """Blocking serve (standalone command). Returns on Ctrl+C."""
    srv = _make_server_or_reclaim(host, port, collect_fn, decision_fn)
    url = "http://%s:%d/" % (host if host not in ("0.0.0.0", "::") else "127.0.0.1", srv.server_address[1])
    print(f"  [dashboard] live optimize view: {url}  (Ctrl+C to stop)")
    try:
        srv.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


def serve_in_thread(host: str, port: int, collect_fn, decision_fn=None):
    """Non-blocking serve for ``optimize --dashboard``: daemon thread, dies with the run process."""
    import threading

    srv = _make_server_or_reclaim(host, port, collect_fn, decision_fn)
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.5}, daemon=True)
    t.start()
    url = "http://%s:%d/" % (host if host not in ("0.0.0.0", "::") else "127.0.0.1", srv.server_address[1])
    return srv, t, url


# The single-page dashboard lives in its own module (it is ~300 lines of HTML/JS; keeping it here
# buries the collector). Imported, not inlined, so tests can read it off the same constant.
from ._optimize_dashboard_page import PAGE_HTML as _HTML  # noqa: E402
