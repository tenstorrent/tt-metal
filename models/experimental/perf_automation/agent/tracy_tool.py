"""M3 · tracy_tool — profile pipeline (PLAN section 7.4), mock stage-1 first.

Three deterministic stages, no LLM:
  1. RUN     (mocked in M3) a profiled pytest -> raw ops_perf_results_*.csv
  2. REFINE  tt-perf-report raw.csv --csv report.csv --start/--end-signpost
             (real CSV->CSV; inherits arch-peak Bound physics, section 4.3)
  3. TAG+BUCKET  group by OP_CLASS_MAP -> buckets; normalize tags (section 4.1);
             join ATTRIBUTES (-> lever_state) + MEMORY from the raw CSV.

Only stage 1 is swapped for real hardware in M8; control logic is unchanged.

Thresholds (PLAN section 4.3): grid tiny<10 / full=available cores; dispatch
gappy when median op-to-op gap > 6.5 microseconds (medians only, never sums).
"""

from __future__ import annotations

import csv
import math
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
import os
from typing import Any, Callable, Sequence

from .opclass import SIGNPOST_CODES, base_op_code, classify_op, is_layout_conversion

# ---- thresholds (PLAN section 4.3) ----
DISPATCH_GAP_NS = 6_500.0  # 6.5 us median op-to-op gap -> gappy
GRID_TINY = 10
DEFAULT_WORKER_CORES = 64  # WH default when CSV lacks AVAILABLE WORKER CORE COUNT
# rank=count cut (high call count + tiny us/call). TBD(count-thresh): provisional.
RANK_COUNT_MIN_CALLS = 32
RANK_COUNT_MAX_US_PER_CALL = 5.0

_FIDELITY_TOKENS = {"lofi": "lofi", "hifi2": "hifi2", "hifi3": "hifi3", "hifi4": "hifi4"}


def median(values: Sequence[float]) -> float:
    """Median of a non-empty sequence (noise floor, PLAN section 8.7)."""
    if not values:
        raise ValueError("median of empty sequence")
    return float(statistics.median(values))


def warm_wall_ms(walls: Sequence[float]) -> float:
    """Median wall with the cold (compile) run 0 dropped when more than one run exists."""
    warm = list(walls[1:]) if len(walls) > 1 else list(walls)
    return median(warm)


_FORWARD_WALL_RE = re.compile(r"FORWARD_WALL_MS=([0-9]+\.?[0-9]*)")
_PER_TOKEN_RE = re.compile(r"TRACE_PER_TOKEN_MS=([0-9]+\.?[0-9]*)")
_TRACE_SKIP_RE = re.compile(r"TRACE_REPLAY_SKIPPED=")


def _scan_log_sentinel(pattern: re.Pattern, profiles_dir: str | Path, runs: int) -> float | None:
    """Median (warm run) of the LAST match of `pattern` across each run's tracy log."""
    vals: list[float] = []
    for i in range(max(runs, 1)):
        lp = Path(profiles_dir) / ("run%d_tracy.log" % i)
        if not lp.is_file():
            continue
        try:
            hits = pattern.findall(lp.read_text(errors="ignore"))
        except OSError:
            continue
        if hits:
            vals.append(float(hits[-1]))
    if not vals:
        return None
    warm = vals[1:] if len(vals) > 1 else vals
    return round(median(warm), 4)


def forward_wall_ms(profiles_dir: str | Path, runs: int) -> float | None:
    return _scan_log_sentinel(_FORWARD_WALL_RE, profiles_dir, runs)


def per_token_ms(profiles_dir: str | Path, runs: int) -> float | None:
    """Clean per-token wall latency (ms) emitted by a trace-replay harness, or None if absent."""
    return _scan_log_sentinel(_PER_TOKEN_RE, profiles_dir, runs)


def decode_trace_status(profiles_dir: str | Path, runs: int) -> str:
    """Generic detector: read the perf run log and classify the pipeline's decode path.

    'traced'         -> a trace-capturable single-token step ran; per_token_ms is a clean number.
    'repeat_prefill' -> trace-replay was attempted but skipped (no cached step) -> structural decode
                        lever is the way to a clean per-token number for this model.
    'off'            -> neither sentinel present (trace disabled or non-generative pipeline).
    """
    for i in range(max(runs, 1)):
        lp = Path(profiles_dir) / ("run%d_tracy.log" % i)
        if not lp.is_file():
            continue
        try:
            text = lp.read_text(errors="ignore")
        except OSError:
            continue
        if _PER_TOKEN_RE.search(text):
            return "traced"
        if _TRACE_SKIP_RE.search(text):
            return "repeat_prefill"
    return "off"


def throughput(per_token_ms_val: float | None, batch_size: int) -> dict[str, float | None]:
    """Derive GPU-comparable T/S/U + T/S from a per-token wall latency (ms)."""
    if not per_token_ms_val or per_token_ms_val <= 0:
        return {"tokens_per_sec_per_user": None, "tokens_per_sec": None}
    tsu = round(1000.0 / per_token_ms_val, 4)
    return {"tokens_per_sec_per_user": tsu, "tokens_per_sec": round(tsu * max(batch_size, 1), 4)}


def host_overhead_bucket(buckets: Sequence[dict[str, Any]], device_ms: float) -> dict[str, Any]:
    """host_overhead = Σ device Op-to-Op Gap (dispatch idle). source=op_gap when real, else unavailable."""
    gaps = [b.get("dispatch_gap_ms") for b in buckets if b.get("dispatch_gap_ms") is not None]
    host_from_gaps = len(gaps) > 0
    host_ms = round(sum(gaps), 4) if host_from_gaps else 0.0
    return {
        "id": "host_overhead",
        "device_ms": host_ms,
        "pct": (host_ms / device_ms * 100.0) if device_ms else 0.0,
        "count": 0,
        "tags": {
            "op_class": "host_fallback",
            "bound": "host",
            "rank": "time",
            # NOT A STAGE. This is the gap between the device ops -- host time, belonging to no
            # phase of the model -- and it was tagged `decode`, which is a real stage name. Any
            # per-stage reader keying on `stage`/`regime` would then count a synthetic host row as a
            # decode measurement. "na" is in the router's vocabulary for exactly this, and it stays
            # correct whether or not such a reader currently exists.
            "regime": "na",
            "source": "op_gap" if host_from_gaps else "unavailable",
        },
        "lever_state": {},
        "top_ops": [],
    }


# ---------------------------------------------------------------------------
# tag normalizers (PLAN section 4.1) — pure functions over raw/report fields
# ---------------------------------------------------------------------------


def normalize_fidelity(math_fidelity: str) -> str:
    """'LoFi'/'HiFi2'.. -> lofi/hifi2..; blank/unknown -> na (PLAN section 4.1)."""
    token = (math_fidelity or "").strip().split()[0].lower() if math_fidelity else ""
    return _FIDELITY_TOKENS.get(token, "na")


def normalize_bound(bound: str) -> str:
    """tt-perf-report Bound -> dram/flop/both/host/slow (blank -> slow, section 4.1)."""
    b = (bound or "").strip().lower()
    if b in ("dram", "flop", "both", "host"):
        return b
    return "slow"


def normalize_memory(mem: str) -> str:
    """INPUT_0_MEMORY -> dram_interleaved/l1_interleaved/sharded (PLAN section 4.1)."""
    m = (mem or "").upper()
    if "SHARD" in m:
        return "sharded"
    if "L1" in m:
        return "l1_interleaved"
    if "DRAM" in m:
        return "dram_interleaved"
    return "unknown"


def normalize_grid(cores: float, available: int = DEFAULT_WORKER_CORES) -> str:
    """CORE COUNT vs available worker cores -> tiny/partial/full (PLAN section 4.1)."""
    c = int(round(cores))
    if c < GRID_TINY:
        return "tiny"
    if c >= available:
        return "full"
    return "partial"


def normalize_dispatch(gaps_ns: Sequence[float]) -> str:
    """gappy when the MEDIAN op-to-op gap > 6.5 us; never sum o2o (section 4.1)."""
    gaps = [g for g in gaps_ns if g is not None]
    if not gaps:
        return "ok"
    return "gappy" if median(gaps) > DISPATCH_GAP_NS else "ok"


# The per-core column the raw CSV carries, in preference order. MIN is deliberately not here: it is
# the fastest core, which is not the op's duration under any reading.
_PER_CORE_NS_COLS = ("DEVICE KERNEL DURATION PER CORE MAX [ns]",)


def device_time_source(members) -> str:
    """Which definition produced these durations: "per_core_max" or "cross_core".

    CARRY THE PROVENANCE, DO NOT ASSUME IT. Two ms figures may only be differenced when they measure
    the same thing, and this changed what device_ms MEANS on hardware whose cores do not share a
    clock. A baseline taken under the old definition and a reading taken under the new one look
    identical -- both are "device_ms" -- and subtracting them reports a gain or a regression that
    never happened. That is defect shape 5 in agent/integrity.py, and it has already produced four
    wrong headlines in this tool's history.

    Stamped onto the profile so the ledger and the report can refuse the comparison instead of
    silently making it, and so the first run on a new build states plainly which column it read."""
    for m in members or []:
        for col in _PER_CORE_NS_COLS:
            v = _to_float((m.get("raw") or {}).get(col))
            if v is not None and v > 0:
                return "per_core_max"
    return "cross_core"


def _member_device_us(m: dict) -> float:
    """One op's device time in MICROSECONDS, from the per-core column when the capture has it.

    See the call site for why the cross-core column is wrong on Blackhole. Returns 0.0 only when
    neither source parses, which is the same as the old behaviour for an unreadable row."""
    raw = m.get("raw") or {}
    for col in _PER_CORE_NS_COLS:
        v = _to_float(raw.get(col))
        if v is not None and v > 0:
            return v / 1e3  # ns -> us, matching tt-perf-report's unit
    return _to_float((m.get("report") or {}).get("Device Time")) or 0.0


def _rank(count: int, device_ms: float) -> str:
    """time -> tune the op; count -> remove/fuse. TBD(count-thresh) provisional."""
    if count <= 0:
        return "time"
    us_per_call = (device_ms * 1_000.0) / count
    if count >= RANK_COUNT_MIN_CALLS and us_per_call < RANK_COUNT_MAX_US_PER_CALL:
        return "count"
    return "time"


# ---------------------------------------------------------------------------
# lever_state — parsed from the raw ATTRIBUTES ComputeKernelConfig (section 4.4)
# ---------------------------------------------------------------------------

_CKC_RE = re.compile(r"ComputeKernelConfig\((?P<body>[^)]*)\)")
_LEVER_KEYS = ("math_fidelity", "fp32_dest_acc_en", "packer_l1_acc", "math_approx_mode")


def parse_lever_state(attributes: str) -> dict[str, str]:
    """Extract the already-pulled levers from ATTRIBUTES (PLAN section 4.4).

    Returns {math_fidelity, fp32_dest_acc_en, packer_l1_acc, math_approx_mode}
    for keys present in the ComputeKernelConfig; missing keys are omitted.
    """
    out: dict[str, str] = {}
    if not attributes:
        return out
    m = _CKC_RE.search(attributes)
    body = m.group("body") if m else attributes
    for key in _LEVER_KEYS:
        km = re.search(rf"{key}\s*=\s*([A-Za-z0-9_:]+)", body)
        if km:
            out[key] = km.group(1)
    return out


# ---------------------------------------------------------------------------
# stage 2 — REFINE (real tt-perf-report subprocess)
# ---------------------------------------------------------------------------


def refine(
    raw_csv: str | Path,
    out_csv: str | Path,
    start_signpost: str | None = None,
    end_signpost: str | None = None,
    id_range: str | None = None,
    arch: str | None = None,
) -> Path:
    """Run tt-perf-report raw.csv -> report.csv (CSV->CSV, no hardware needed)."""
    cmd = [
        "tt-perf-report",
        str(raw_csv),
        "--csv",
        str(out_csv),
        "--no-advice",
        "--no-color",
        "--no-stacked-report",
    ]
    if start_signpost:
        cmd += ["--start-signpost", start_signpost]
    if end_signpost:
        cmd += ["--end-signpost", end_signpost]
    if id_range:
        cmd += ["--id-range", id_range]
    if arch:
        cmd += ["--arch", arch]
    try:
        # tt-perf-report post-processing scales with the number of op rows, which scales with the
        # model -- so this is model-dependent, not a machine constant. Same adaptive chain.
        from .probes import adaptive_op_timeout

        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=adaptive_op_timeout("profile"))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        blob = (exc.stderr or "") + (exc.stdout or "")
        if "Unknown math fidelity" not in blob:
            raise
        # Upstream tt-perf-report (==1.2.2) gap: its fidelity_map knows only
        # HiFi4/HiFi2/LoFi and hard-errors on HiFi3, a legitimate ttnn.MathFidelity
        # value (LoFi<HiFi2<HiFi3<HiFi4) that the device runs and tracy reports
        # correctly. Any model that emits a HiFi3 matmul would otherwise be
        # un-profilable. Re-run the SAME invocation IN-PROCESS with tflops_per_core
        # patched to interpolate HiFi3 = HiFi4 * 4/3 (3 vs 4 fidelity phases ->
        # throughput ∝ 1/phases; lands between HiFi2 and HiFi4, matching the table's
        # own HiFi2 == HiFi4*2). Non-regressive: only the HiFi3-crash path takes this
        # branch; every normal report still uses the CLI above.
        _refine_in_process_hifi3(cmd)
    return Path(out_csv)


def _refine_in_process_hifi3(cmd: list[str]) -> None:
    """Run tt-perf-report's main() in-process with a HiFi3-aware tflops table.

    ``cmd`` is the exact ``["tt-perf-report", raw, "--csv", out, ...]`` argv that
    crashed on ``Unknown math fidelity: HiFi3``. We monkeypatch
    ``ArchitectureSpec.tflops_per_core`` to add HiFi3, invoke main(), then always
    restore the original method and argv so nothing leaks to later calls."""
    import tt_perf_report.perf_report as _pr

    spec_cls = _pr.ArchitectureSpec
    orig_tflops = spec_cls.tflops_per_core

    def _tflops_with_hifi3(self, math_fidelity: str) -> float:
        if math_fidelity == "HiFi3":
            return self.tflops_hifi4 * 4.0 / 3.0
        return orig_tflops(self, math_fidelity)

    old_argv = sys.argv
    spec_cls.tflops_per_core = _tflops_with_hifi3
    try:
        sys.argv = list(cmd)  # main() reads argv via argparse; cmd[0] is prog
        _pr.main()
    except SystemExit as exc:  # argparse/main may sys.exit(0) on success
        if exc.code not in (0, None):
            raise
    finally:
        spec_cls.tflops_per_core = orig_tflops
        sys.argv = old_argv


# ---------------------------------------------------------------------------
# stage 3 — TAG + BUCKET
# ---------------------------------------------------------------------------


def _to_float(value: str) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _raw_index(raw_csv: str | Path) -> dict[int, dict[str, str]]:
    """Index raw rows by GLOBAL CALL COUNT for the report->raw join."""
    out: dict[int, dict[str, str]] = {}
    with open(raw_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("OP CODE") in SIGNPOST_CODES:
                continue
            gcc = _to_float(row.get("GLOBAL CALL COUNT", ""))
            if gcc is not None:
                out[int(gcc)] = row
    return out


def _fingerprint(rep: dict, raw: dict) -> tuple:
    """The identity _top_ops groups by: (op code, shape, input memory). Shared so the neighbour pass
    keys on exactly the same thing the op rows do."""
    return (rep.get("OP Code", ""), _op_shape(raw), normalize_memory(raw.get("INPUT_0_MEMORY", "")))


def _neighbours(ordered: list) -> dict:
    """What ran immediately before and after each fingerprint, most common first.

    ADJACENCY IS IN THE CAPTURE AND WAS BEING THROWN AWAY. The report rows arrive in execution order
    (GLOBAL CALL COUNT, which _raw_index already joins on), and grouping by op_class discards it --
    so nothing downstream could ask "what does this op feed into", even though the playbook teaches
    that exact technique by hand (09 section 5, "Identify a mystery op by its neighbors").

    Most common, not all: an op appearing once per layer has the same neighbours every layer, so the
    mode is the structural answer and the occasional odd pairing at a stack boundary is noise.
    Returns {fingerprint: {"prev": op_code, "next": op_code}}.
    """
    from collections import Counter

    prev_c: dict[tuple, Counter] = {}
    next_c: dict[tuple, Counter] = {}
    for i, (fp, _code) in enumerate(ordered):
        if i > 0:
            prev_c.setdefault(fp, Counter())[ordered[i - 1][1]] += 1
        if i + 1 < len(ordered):
            next_c.setdefault(fp, Counter())[ordered[i + 1][1]] += 1
    out: dict[tuple, dict] = {}
    for fp in set(prev_c) | set(next_c):
        out[fp] = {
            "prev": (prev_c.get(fp) or Counter()).most_common(1)[0][0] if prev_c.get(fp) else "",
            "next": (next_c.get(fp) or Counter()).most_common(1)[0][0] if next_c.get(fp) else "",
        }
    return out


def stage_windows(raw_csv: str | Path) -> list:
    """[(stage, start, end)] the capture actually marked, in order. [] when unmarked.

    Read from the CAPTURE, not from the stage list the harness expects: a stage that failed to run
    leaves no marks, and pricing it from a window that is not there would invent a measurement.
    Signposts are identified exactly as tt-perf-report identifies them -- rows whose OP TYPE is
    "signpost" -- so the window list and the slicer agree by construction.
    """
    names = []
    try:
        with open(raw_csv, newline="") as f:
            for row in csv.DictReader(f):
                if str(row.get("OP TYPE") or "").strip().lower() != "signpost":
                    continue
                nm = str(row.get("OP CODE") or "").strip()
                if nm and nm not in names:
                    names.append(nm)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for nm in names:
        if nm.startswith("stage:") and not nm.endswith(":end"):
            end = "%s:end" % nm
            if end in names:
                out.append((nm.split(":", 1)[1], nm, end))
    return out


def _capture_truncated_reason(profiles_dir) -> str | None:
    """The phrase tracy used to say it stopped instrumenting, or None. Best-effort.

    Reads the run's own log rather than inferring: the capture still exits 0 and still writes a CSV
    when this happens, so nothing else distinguishes a truncated capture from a complete one.
    """
    try:
        from agent.probes import detect_capture_truncated
    except Exception:  # noqa: BLE001 -- a check that cannot load must not stop the profile
        return None
    try:
        for log in sorted(Path(profiles_dir).glob("*_tracy.log")):
            hit = detect_capture_truncated(log.read_text(errors="ignore"))
            if hit:
                return hit
    except OSError:
        return None
    return None


def _per_stage_buckets(raw_csv, profiles_dir, available_cores, arch) -> dict:
    """{stage: [buckets]} for every stage the capture marked. {} when it marked none.

    One refine() per window over the same raw rows -- no extra device work. Best-effort per stage: a
    window that fails to refine costs that stage its split, not the profile.
    """
    out = {}
    for stage, s0, s1 in stage_windows(raw_csv):
        try:
            rep = Path(profiles_dir) / ("iter_baseline_report_%s.csv" % stage)
            refine(raw_csv, rep, s0, s1, None, arch)
            bk = build_buckets(rep, raw_csv, available_cores)
            if bk:
                for b in bk:
                    b.setdefault("tags", {})["regime"] = stage
                out[stage] = bk
        except Exception:  # noqa: BLE001
            continue
    return out


def build_buckets(
    report_csv: str | Path,
    raw_csv: str | Path,
    available_cores: int = DEFAULT_WORKER_CORES,
) -> list[dict[str, Any]]:
    """Group refined ops by op_class into tagged buckets (PLAN section 7.4 stage 3)."""
    raw_by_gcc = _raw_index(raw_csv)
    with open(report_csv, newline="") as f:
        report_rows = list(csv.DictReader(f))

    groups: dict[str, list[dict[str, Any]]] = {}
    _ordered: list[tuple] = []  # (fingerprint, op_code) in execution order -- see _neighbours
    for rep in report_rows:
        op_code = rep.get("OP Code", "")
        if base_op_code(op_code) in SIGNPOST_CODES or not op_code:
            continue
        op_class = classify_op(op_code)
        gcc = _to_float(rep.get("Global Call Count", ""))
        raw = raw_by_gcc.get(int(gcc)) if gcc is not None else None
        groups.setdefault(op_class, []).append({"report": rep, "raw": raw or {}})
        _ordered.append((_fingerprint(rep, raw or {}), op_code))
    _nbrs = _neighbours(_ordered)

    buckets: list[dict[str, Any]] = []
    total_ms = 0.0
    for op_class, members in groups.items():
        # PER-CORE DURATION, NOT THE CROSS-CORE ONE. tt-perf-report's "Device Time" comes from
        # DEVICE KERNEL DURATION [ns], which process_device_log computes with `op_first_last`: the
        # FIRST start seen on any core to the LAST end seen on any core. Where cores do not share a
        # clock base -- Blackhole -- that span contains the inter-core offset as well as the op, so
        # the "duration" can be the offset rather than the work, and the error is unbounded.
        #
        # tt-metal already computes the correct figure and writes it out: the same post-processing
        # runs `op_core_first_last`, which pairs each core's own start and end and never crosses
        # cores, and emits DEVICE KERNEL DURATION PER CORE MIN/MAX/AVG [ns]. Nothing here read it.
        #
        # MAX, not AVG: an op is not finished until its SLOWEST core is, so the max across cores is
        # its duration -- averaging understates every multi-core op. On hardware where the cores do
        # share a clock the two agree, so this changes nothing there; it only removes an error that
        # exists on Blackhole.
        #
        # Falls back to Device Time when the per-core columns are absent (older captures), because a
        # missing column must not zero a bucket -- a zero device_ms reads as "infinitely fast".
        device_ms = sum(_member_device_us(m) for m in members) / 1e3
        total_ms += device_ms
        gaps = [_to_float(m["report"].get("Op-to-Op Gap")) for m in members]
        gaps = [g for g in gaps if g is not None]
        cores = [_to_float(m["report"].get("Cores")) or 0.0 for m in members]
        bounds = [normalize_bound(m["report"].get("Bound", "")) for m in members]
        fids = [normalize_fidelity(m["raw"].get("MATH FIDELITY", "")) for m in members]
        mems = [normalize_memory(m["raw"].get("INPUT_0_MEMORY", "")) for m in members]
        rep0 = members[0]
        buckets.append(
            {
                "id": op_class,
                "device_ms": device_ms,
                "count": len(members),
                "members": members,  # transient; pct filled below
                "_cores": cores,
                "_gaps": gaps,
                "_bounds": bounds,
                "_fids": fids,
                "_mems": mems,
                "_nbrs": _nbrs,
                "lever_state": parse_lever_state(rep0["raw"].get("ATTRIBUTES", "")),
            }
        )

    out: list[dict[str, Any]] = []
    for b in buckets:
        rep_cores = max(b["_cores"]) if b["_cores"] else 0.0
        churn_ms = 0.0
        churn_n = 0
        for m in b["members"]:
            rep, raw = m["report"], m["raw"]
            if is_layout_conversion(
                rep.get("OP Code", ""), raw.get("INPUT_0_LAYOUT", ""), raw.get("OUTPUT_0_LAYOUT", "")
            ):
                churn_ms += (_to_float(rep.get("Device Time")) or 0.0) / 1e3
                churn_n += 1
        tags = {
            "op_class": b["id"],
            "bound": _most_common(b["_bounds"]),
            "rank": _rank(b["count"], b["device_ms"]),
            "fidelity": _most_common(b["_fids"]),
            "grid": normalize_grid(rep_cores, available_cores),
            "dispatch": normalize_dispatch(b["_gaps"]),
            "memory": _most_common(b["_mems"]),
            "regime": "na",  # TBD(regime-source): M-dim source not wired yet
        }
        out.append(
            {
                "id": b["id"],
                "device_ms": b["device_ms"],
                "pct": (b["device_ms"] / total_ms * 100.0) if total_ms else 0.0,
                "count": b["count"],
                "tags": tags,
                "lever_state": b["lever_state"],
                # Σ Op-to-Op Gap (device idle between ops); gap is NS, Device Time is US -> /1e6.
                "dispatch_gap_ms": round(sum(b["_gaps"]) / 1e6, 4) if b["_gaps"] else 0.0,
                "layout_churn_ms": round(churn_ms, 4),
                "layout_churn_count": churn_n,
                "top_ops": _top_ops(b["members"], available_cores, neighbours=b.get("_nbrs")),
                # WHICH DEFINITION PRODUCED THESE MS. Two figures may only be differenced when they
                # measure the same thing; a per-core reading and a cross-core one both call
                # themselves device_ms. Carried per bucket so a later comparison can refuse rather
                # than assume -- see device_time_source.
                "device_time_source": device_time_source(b["members"]),
            }
        )
    # Stable, useful ordering: biggest device-time bucket first.
    out.sort(key=lambda x: x["device_ms"], reverse=True)
    _srcs = {b.get("device_time_source") for b in out}
    if _srcs:
        print(
            "  [tracy] device_ms from %s" % ("+".join(sorted(s for s in _srcs if s)) or "unknown"),
            file=sys.stderr,
            flush=True,
        )
    return out


def _pad(v: str) -> str:
    """'32[10]' -> '32' (the padded dim the kernel actually computes)."""
    s = str(v or "").split("[")[0].strip()
    return s or "?"


def _logical(v: str) -> str:
    """'32[10]' -> '10' (the rows the stage ASKED for); '32' -> '32' when nothing was padded.

    _pad keeps the other half -- the padded dim the kernel computes -- which is right for bytes and
    for a shape fingerprint, and wrong for counting items. A decode step retiring one row per user
    pads 8 rows to a 32-row tile, so every matmul in it reads 32 and the stage looks like it retires
    32 items. The logical count is what the model asked for, and it is already in the field.
    """
    s = str(v or "").strip()
    if "[" in s and "]" in s:
        inner = s.split("[", 1)[1].split("]", 1)[0].strip()
        if inner.isdigit():
            return inner
    return _pad(s)


def _op_rows(raw: dict) -> int:
    """LOGICAL rows of the left input -- how many items this op processed. 0 when unknown."""
    v = _logical(raw.get("INPUT_0_Y_PAD[LOGICAL]"))
    return int(v) if str(v).isdigit() else 0


def _op_shape(raw: dict) -> str:
    """Compact matmul shape fingerprint from the per-op input dims (e.g. '32x1024 @ 1024x1024')."""
    m, k0 = _pad(raw.get("INPUT_0_Y_PAD[LOGICAL]")), _pad(raw.get("INPUT_0_X_PAD[LOGICAL]"))
    k1, n = _pad(raw.get("INPUT_1_Y_PAD[LOGICAL]")), _pad(raw.get("INPUT_1_X_PAD[LOGICAL]"))
    return f"{m}x{k0} @ {k1}x{n}"


_DTYPE_BYTES = {
    "FLOAT32": 4.0,
    "FLOAT16": 2.0,
    "BFLOAT16": 2.0,
    "BFLOAT8_B": 1.0625,
    "BFLOAT4_B": 0.5625,
    "UINT32": 4.0,
    "INT32": 4.0,
    "UINT16": 2.0,
    "UINT8": 1.0,
    "INT8": 1.0,
}


def _tensor_bytes(raw: dict, prefix: str) -> float:
    """Bytes of one tensor from its padded dims × dtype-bytes (0 if absent/unknown)."""
    y, x = _pad(raw.get(f"{prefix}_Y_PAD[LOGICAL]")), _pad(raw.get(f"{prefix}_X_PAD[LOGICAL]"))
    if y == "?" or x == "?":
        return 0.0
    bpe = _DTYPE_BYTES.get(str(raw.get(f"{prefix}_DATATYPE", "")).strip().upper())
    return (int(y) * int(x) * bpe) if bpe else 0.0


def _op_bytes(raw: dict) -> float:
    """Bytes moved by one op = inputs read + output written (DRAM/L1 traffic)."""
    return sum(_tensor_bytes(raw, p) for p in ("INPUT_0", "INPUT_1", "INPUT_2", "OUTPUT_0"))


def _top_ops(
    members: list[dict[str, Any]], available_cores: int, k: int | None = None, neighbours: dict | None = None
) -> list[dict[str, Any]]:
    """EVERY distinct op in the bucket, by fingerprint (op + shape + memory), ranked by device-ms.

    This returned out[:6]. Everything past the sixth fingerprint was folded into the bucket total and
    never appeared as an op again -- not in the roofline, not in open_ops, not in blocking_ops, not in
    the report. The same display-limit-as-work-queue mistake as open_ops[:10], one layer upstream: the
    queue is FED from this list, so an op cut here can never be selected however wide the queue is.

    gemma-3-12b-it runs 8 buckets, so the whole model was describable in at most 48 op fingerprints
    regardless of how many it actually executes -- and PagedUpdateCache (4.18 ms gap, on ONE core) sat
    below its bucket's cut and was never optimized.

    Hotness still decides ORDER; it no longer decides EXISTENCE. PERF_MCP_TOP_OPS_MAX bounds the list
    for anyone who needs the old fixed size, so the limit is a decision someone makes rather than a
    constant nobody remembers choosing.
    """
    groups: dict[tuple, dict[str, Any]] = {}
    for m in members:
        rep, raw = m.get("report", {}), m.get("raw", {})
        shape = _op_shape(raw)
        mem = normalize_memory(raw.get("INPUT_0_MEMORY", ""))
        op = rep.get("OP Code", "")
        key = _fingerprint(rep, raw)
        _nb = (neighbours or {}).get(key) or {}
        g = groups.setdefault(
            key,
            {
                "op_code": op,
                "shape": shape,
                # The LOGICAL row count beside the padded fingerprint: `shape` is what the kernel
                # computed, this is what the model asked for. Item counting needs the second.
                "rows": _op_rows(raw),
                "memory": mem,
                "count": 0,
                "device_ms": 0.0,
                "bytes": 0.0,
                "cores": int(_to_float(rep.get("Cores")) or 0),
                "grid": normalize_grid(_to_float(rep.get("Cores")) or 0.0, available_cores),
                "fidelity": normalize_fidelity(raw.get("MATH FIDELITY", "")),
                # what ran either side of this op, most common -- see _neighbours
                "prev_op": _nb.get("prev", ""),
                "next_op": _nb.get("next", ""),
            },
        )
        g["count"] += 1
        g["device_ms"] += (_to_float(rep.get("Device Time")) or 0.0) / 1e3
        g["bytes"] += _op_bytes(raw)
    out = sorted(groups.values(), key=lambda x: x["device_ms"], reverse=True)
    for g in out:
        g["device_ms"] = round(g["device_ms"], 4)
    if k is None:
        _env = (os.environ.get("PERF_MCP_TOP_OPS_MAX") or "").strip()
        k = int(_env) if _env.isdigit() and int(_env) > 0 else None
    return out[:k] if k else out


def _most_common(values: Sequence[str]) -> str:
    if not values:
        return "na"
    return max(set(values), key=values.count)


def stack_report(buckets: list[dict[str, Any]], layout_churn: dict[str, Any] | None = None) -> str:
    """Human-readable stack the agent reads at SELECT (PLAN section 4.4)."""
    lines = [f"{'bucket':<12} {'ms':>8} {'pct':>6} {'count':>6}  tags"]
    for b in buckets:
        tag_str = " ".join(f"{k}={v}" for k, v in b["tags"].items())
        churn = (
            f"  [layout-churn {b['layout_churn_count']}× = {b['layout_churn_ms']:.3f}ms]"
            if b.get("layout_churn_count")
            else ""
        )
        lines.append(f"{b['id']:<12} {b['device_ms']:>8.3f} {b['pct']:>5.1f}% {b['count']:>6}  {tag_str}{churn}")
    if layout_churn and layout_churn.get("count"):
        lines.append(
            f"\nlayout coherence: {layout_churn['count']} pure layout-conversion ops "
            f"= {layout_churn['device_ms']:.3f}ms ({layout_churn['pct_device']:.1f}% of device time) -- "
            f"redundant if producers emit the consumer's layout (see #layout-coherence)."
        )
    return "\n".join(lines)


def tracy_tool(
    pcc_path: str,
    batch_size: int,
    seq_len: int,
    runs: int,
    profiles_dir: str | Path,
    start_signpost: str | None = None,
    end_signpost: str | None = None,
    id_range: str | None = None,
    arch: str | None = None,
    available_cores: int = DEFAULT_WORKER_CORES,
    run_profiled: Callable[..., tuple[Path, float]] | None = None,
) -> dict[str, Any]:
    """One profile call: RUN (mocked) -> REFINE -> TAG+BUCKET (PLAN section 7.4).

    `run_profiled(pcc_path, batch_size, seq_len, profiles_dir, i)` is the
    swappable stage-1: it returns (raw_csv_path, wall_ms) for run i. In M3 it is
    a mock that yields a fixture CSV; M8 swaps in the real `tracy -m pytest`.
    wall_ms is the MEDIAN across `runs` (noise floor, section 8.7).
    """
    if run_profiled is None:
        raise ValueError("run_profiled (stage-1) must be provided until M8 wires real Tracy")
    profiles_dir = Path(profiles_dir)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    walls: list[float] = []
    raw_csv: Path | None = None
    _per_run: list = []
    for i in range(runs):
        raw_csv, wall_ms = run_profiled(pcc_path, batch_size, seq_len, profiles_dir, i)
        walls.append(wall_ms)
        _per_run.append((wall_ms, raw_csv))
    # Pick the CSV of the run whose wall time IS the reported median, so device_ms and wall_ms
    # describe the SAME execution. Previously raw_csv was simply whatever the last iteration left,
    # while wall_ms was medianed -- two different runs reported as one profile.
    if len(_per_run) > 1:
        _median_wall = warm_wall_ms(walls)
        raw_csv = min(_per_run, key=lambda pair: abs(pair[0] - _median_wall))[1]

    raw_dest = profiles_dir / "iter_baseline_raw.csv"
    if Path(raw_csv) != raw_dest:
        shutil.copyfile(raw_csv, raw_dest)
    report_csv = profiles_dir / "iter_baseline_report.csv"
    refine(raw_dest, report_csv, start_signpost, end_signpost, id_range, arch)

    buckets = build_buckets(report_csv, raw_dest, available_cores)
    # PER-STAGE SLICES from the marks measure_adapter emits around each stage under a capture. Purely
    # additive: an unmarked capture yields no windows, no stage_buckets, and every consumer keeps the
    # whole-profile figure it already had.
    # READ THE LOG ONCE. Two consumers need this -- the warning below when there is no split, and the
    # field carried on the profile -- and reading it twice parses the same multi-thousand-line log
    # again for an answer that cannot have changed.
    _why = _capture_truncated_reason(profiles_dir)
    stage_buckets = _per_stage_buckets(raw_dest, profiles_dir, available_cores, arch)
    if not stage_buckets:
        # NAME THE CAUSE THAT ACTUALLY HAPPENED. This blamed the import unconditionally, and the
        # import was fine: tracy stops instrumenting after 32K source locations, saves what it has,
        # and records nothing further -- so marks emitted after that point are emitted (tracy's own
        # logger prints every one) into a capture that has already closed. Reading the log turns a
        # misleading hint into the reason, which is the difference between a one-line fix and a day.
        print(
            "  [tracy] no stage signposts in the capture -- the roofline falls back to whole-profile "
            "figures (one math-fidelity peak shared by every stack). %s"
            % (
                "Cause: tracy stopped instrumenting mid-run (%s), so marks emitted after that point "
                "had no capture to land in." % _why
                if _why
                else "Expected stage:<name> / stage:<name>:end from measure_adapter; check that "
                "`tracy` imports in the workload."
            ),
            file=sys.stderr,
            flush=True,
        )
    wall_ms = warm_wall_ms(walls)
    device_ms = round(sum(b["device_ms"] for b in buckets), 4)
    host = host_overhead_bucket(buckets, device_ms)
    host_ms = host["device_ms"]
    buckets.append(host)
    churn_ms = round(sum(b.get("layout_churn_ms", 0.0) for b in buckets), 4)
    churn_n = sum(b.get("layout_churn_count", 0) for b in buckets)
    layout_churn = {
        "device_ms": churn_ms,
        "count": churn_n,
        "pct_device": round(churn_ms / device_ms * 100.0, 1) if device_ms else 0.0,
    }
    pt_ms = per_token_ms(profiles_dir, runs)
    tput = throughput(pt_ms, batch_size)
    return {
        "wall_ms": wall_ms,
        "forward_wall_ms": forward_wall_ms(profiles_dir, runs),
        # The per-stage split when the capture carried marks; absent otherwise, and absence must read
        # as "no split available" rather than as an empty one.
        "stage_buckets": stage_buckets,
        # THE CAPTURE'S OWN HONESTY, carried with the numbers rather than left in a log nobody reads.
        # Tracy stops instrumenting after 32K source locations, saves what it has, and every later
        # record is lost -- roughly a third of the rows on a full-model forward. The run still exits 0
        # and still produces a CSV, so a truncated capture is indistinguishable from a complete one
        # downstream, and its op breakdown was rendered as if it described the whole run. Recorded
        # here, where the log is still in reach; None when the capture ran to completion.
        "capture_truncated": _why,
        "per_token_ms": pt_ms,
        "tokens_per_sec_per_user": tput["tokens_per_sec_per_user"],
        "tokens_per_sec": tput["tokens_per_sec"],
        "decode_status": decode_trace_status(profiles_dir, runs),
        "device_ms": device_ms,
        "host_ms": host_ms,
        "host_fraction": round(host_ms / wall_ms, 4) if wall_ms else 0.0,
        "layout_churn": layout_churn,
        "buckets": buckets,
        "stack_report": stack_report(buckets, layout_churn),
        "artifacts": {"raw_csv": str(raw_dest), "report_csv": str(report_csv)},
    }


def profile_model(*, perf_test, config, env, profiles_dir, run_profiled):
    """The SINGLE measurement path — used by before_loop (baseline) AND REMEASURE.

    Both callers go through here so they can never drift: identical signpost
    defaults, runs, and arch/cores resolution. This is the Before-Loop
    methodology verbatim; REMEASURE no longer reinvents the tracy_tool call.
    """
    return tracy_tool(
        pcc_path=perf_test,
        batch_size=config.get("batch_size", 1),
        seq_len=config.get("seq_len", 0),
        runs=config.get("runs", 1),
        profiles_dir=profiles_dir,
        start_signpost=config.get("start_signpost", "start"),
        end_signpost=config.get("end_signpost", "stop"),
        arch=env.get("arch"),
        available_cores=env.get("worker_cores", 64),
        run_profiled=run_profiled,
    )
