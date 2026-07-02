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
import re
import shutil
import statistics
import subprocess
from pathlib import Path
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


def forward_wall_ms(profiles_dir: str | Path, runs: int) -> float | None:
    vals: list[float] = []
    for i in range(max(runs, 1)):
        lp = Path(profiles_dir) / ("run%d_tracy.log" % i)
        if not lp.is_file():
            continue
        try:
            hits = _FORWARD_WALL_RE.findall(lp.read_text(errors="ignore"))
        except OSError:
            continue
        if hits:
            vals.append(float(hits[-1]))
    if not vals:
        return None
    warm = vals[1:] if len(vals) > 1 else vals
    return round(median(warm), 4)


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
            "regime": "decode",
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
    return "dram_interleaved"


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
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return Path(out_csv)


# ---------------------------------------------------------------------------
# stage 3 — TAG + BUCKET
# ---------------------------------------------------------------------------


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    for rep in report_rows:
        op_code = rep.get("OP Code", "")
        if base_op_code(op_code) in SIGNPOST_CODES or not op_code:
            continue
        op_class = classify_op(op_code)
        gcc = _to_float(rep.get("Global Call Count", ""))
        raw = raw_by_gcc.get(int(gcc)) if gcc is not None else None
        groups.setdefault(op_class, []).append({"report": rep, "raw": raw or {}})

    buckets: list[dict[str, Any]] = []
    total_ms = 0.0
    for op_class, members in groups.items():
        # tt-perf-report "Device Time" is in MICROSECONDS (verified against raw
        # DEVICE KERNEL DURATION [ns] on a real capture: ratio exactly 1000).
        device_ms = sum((_to_float(m["report"].get("Device Time")) or 0.0) for m in members) / 1e3
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
                "top_ops": _top_ops(b["members"], available_cores),
            }
        )
    # Stable, useful ordering: biggest device-time bucket first.
    out.sort(key=lambda x: x["device_ms"], reverse=True)
    return out


def _pad(v: str) -> str:
    """'32[10]' -> '32' (the padded dim the kernel actually computes)."""
    s = str(v or "").split("[")[0].strip()
    return s or "?"


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


def _top_ops(members: list[dict[str, Any]], available_cores: int, k: int = 6) -> list[dict[str, Any]]:
    """Rank the bucket's hot ops by fingerprint (op + shape + memory) by total device-ms, top k."""
    groups: dict[tuple, dict[str, Any]] = {}
    for m in members:
        rep, raw = m.get("report", {}), m.get("raw", {})
        shape = _op_shape(raw)
        mem = normalize_memory(raw.get("INPUT_0_MEMORY", ""))
        op = rep.get("OP Code", "")
        key = (op, shape, mem)
        g = groups.setdefault(
            key,
            {
                "op_code": op,
                "shape": shape,
                "memory": mem,
                "count": 0,
                "device_ms": 0.0,
                "bytes": 0.0,
                "cores": int(_to_float(rep.get("Cores")) or 0),
                "grid": normalize_grid(_to_float(rep.get("Cores")) or 0.0, available_cores),
                "fidelity": normalize_fidelity(raw.get("MATH FIDELITY", "")),
            },
        )
        g["count"] += 1
        g["device_ms"] += (_to_float(rep.get("Device Time")) or 0.0) / 1e3
        g["bytes"] += _op_bytes(raw)
    out = sorted(groups.values(), key=lambda x: x["device_ms"], reverse=True)
    for g in out:
        g["device_ms"] = round(g["device_ms"], 4)
    return out[:k]


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
    for i in range(runs):
        raw_csv, wall_ms = run_profiled(pcc_path, batch_size, seq_len, profiles_dir, i)
        walls.append(wall_ms)

    raw_dest = profiles_dir / "iter_baseline_raw.csv"
    if Path(raw_csv) != raw_dest:
        shutil.copyfile(raw_csv, raw_dest)
    report_csv = profiles_dir / "iter_baseline_report.csv"
    refine(raw_dest, report_csv, start_signpost, end_signpost, id_range, arch)

    buckets = build_buckets(report_csv, raw_dest, available_cores)
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
    return {
        "wall_ms": wall_ms,
        "forward_wall_ms": forward_wall_ms(profiles_dir, runs),
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
