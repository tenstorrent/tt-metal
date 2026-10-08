"""Turn one profiled test run (pytest log + ops perf CSV) into per-shape measurements."""

from __future__ import annotations

import glob
import re
import statistics
from pathlib import Path

KERNEL = "DEVICE KERNEL DURATION [ns]"
WIDTH = "INPUT_0_X_PAD[LOGICAL]"

_CASE = re.compile(r"case=(\S+) .*?max_abs=([0-9.eE+-]+) .*?PCC: ([0-9.eE+-]+|nan)")


def find_ops_csv(report_dir: Path) -> Path | None:
    hits = sorted(glob.glob(str(report_dir / "reports" / "*" / "ops_perf_results_*.csv")))
    return Path(hits[-1]) if hits else None


def parse_log(text: str, shapes: list[dict]) -> dict:
    out = {s["id"]: {"pytest": "missing", "pcc": None, "max_abs": None} for s in shapes}
    by_case = {s["case"]: s["id"] for s in shapes}
    for m in _CASE.finditer(text):
        sid = by_case.get(m.group(1))
        if sid:
            out[sid]["max_abs"] = float(m.group(2))
            out[sid]["pcc"] = float(m.group(3))
    for line in text.splitlines():
        for status in ("PASSED", "FAILED", "ERROR"):
            if line.startswith(status + " "):
                for s in shapes:
                    if f"-{s['id']}-" in line or f"[{s['id']}" in line or f"-{s['id']}]" in line:
                        if out[s["id"]]["pytest"] != "FAILED":
                            out[s["id"]]["pytest"] = status
    return out


def classify_failure(text: str) -> str:
    if re.search(r":\d+:\d+: error:", text) or re.search(r"(?i)failed to (compile|build)|compile failed", text):
        return "jit_compile_error"
    return "runtime_error"


def error_excerpt(text: str, n: int = 60) -> str:
    lines = text.splitlines()
    idx = [i for i, l in enumerate(lines) if re.search(r"error|Error|FAILED|TT_FATAL|TT_THROW|Traceback", l)]
    if idx:
        start = max(0, idx[0] - 5)
        return "\n".join(lines[start : start + n])
    return "\n".join(lines[-n:])


def parse_ops(csv_path: Path, cfg: dict) -> tuple[dict, "object"]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    d = df[df["OP CODE"] == cfg["op_code"]].copy()
    d["W"] = d[WIDTH].astype(str).str.split("[").str[0].astype(int)
    d = d.sort_values(["DEVICE ID", "GLOBAL CALL COUNT"])
    d["call"] = d.groupby(["W", "DEVICE ID"]).cumcount()
    warm = int(cfg["calls_per_shape"]["warmup"])
    out = {}
    for s in cfg["shapes"]:
        g = d[(d["W"] == s["local_width"]) & (d["call"] >= warm)]
        if g.empty:
            out[s["id"]] = None
            continue
        per_call_max = g.groupby("call")[KERNEL].max() / 1e3
        out[s["id"]] = {
            "us_chip_mean": round(float(g[KERNEL].mean() / 1e3), 3),
            "us_chip_max": round(float(per_call_max.mean()), 3),
            "us_chip_p50": round(float(g[KERNEL].median() / 1e3), 3),
            "us_min": round(float(g[KERNEL].min() / 1e3), 3),
            "us_max": round(float(g[KERNEL].max() / 1e3), 3),
            "n_rows": int(len(g)),
            "core_count": int(g["CORE COUNT"].iloc[0]) if "CORE COUNT" in g else None,
        }
    keep = [
        c
        for c in [
            "GLOBAL CALL COUNT",
            "DEVICE ID",
            "OP CODE",
            WIDTH,
            "CORE COUNT",
            KERNEL,
            "DEVICE FW DURATION [ns]",
            "DEVICE BRISC KERNEL DURATION [ns]",
            "DEVICE NCRISC KERNEL DURATION [ns]",
            "DEVICE TRISC0 KERNEL DURATION [ns]",
            "DEVICE TRISC1 KERNEL DURATION [ns]",
            "DEVICE TRISC2 KERNEL DURATION [ns]",
            "DEVICE ERISC KERNEL DURATION [ns]",
        ]
        if c in d.columns
    ]
    return out, d[keep]


def measure(cfg: dict, log_path: Path, report_dir: Path, pytest_rc: int) -> dict:
    """Measurements + correctness for one run. fail_class here is only about the test run."""
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    shapes = parse_log(text, cfg["shapes"])
    csv = find_ops_csv(report_dir)
    ops, rows = parse_ops(csv, cfg) if csv else ({}, None)
    expected = int(cfg["calls_per_shape"]["measured"]) * int(cfg.get("devices", 4))
    gate = cfg["accuracy_gate"]
    fail, error = "ok", None
    if pytest_rc != 0 or any(v["pytest"] != "PASSED" for v in shapes.values()):
        fail, error = classify_failure(text), error_excerpt(text)
    for sid, v in shapes.items():
        v.update(ops.get(sid) or {})
        if fail == "ok":
            if v["pcc"] is None or v["pcc"] < gate["pcc_min"] or (v["max_abs"] or 0) > gate["max_abs_max"]:
                fail, error = "accuracy_fail", f"{sid}: pcc={v['pcc']} max_abs={v['max_abs']} gate={gate}"
            elif not ops.get(sid) or ops[sid]["n_rows"] != expected:
                n = ops[sid]["n_rows"] if ops.get(sid) else 0
                fail, error = "infra", f"{sid}: {n} measured profiler rows, expected {expected} (csv={csv})"
    return {"fail_class": fail, "error": error, "shapes": shapes, "ops_csv": str(csv) if csv else None, "rows": rows}


def score_vs_baseline(shapes: dict, baseline: dict) -> float:
    from .policy_api import geomean

    ratios = []
    for sid, v in shapes.items():
        b = baseline["shapes"][sid]["us_chip_mean"]
        v["baseline_us"] = b
        v["speedup"] = round(b / v["us_chip_mean"], 4)
        ratios.append(v["speedup"])
    return round(geomean(ratios), 4)


def make_baseline(runs: list[dict], min_noise_pct: float) -> dict:
    shapes = {}
    noise = min_noise_pct
    for sid in runs[0]["shapes"]:
        vals = [r["shapes"][sid]["us_chip_mean"] for r in runs]
        med = statistics.median(vals)
        spread = (max(vals) - min(vals)) / med * 100
        noise = max(noise, spread)
        shapes[sid] = {
            "us_chip_mean": round(med, 3),
            "us_chip_max": round(statistics.median(r["shapes"][sid]["us_chip_max"] for r in runs), 3),
            "runs_us_chip_mean": vals,
            "spread_pct": round(spread, 2),
            "pcc": min(r["shapes"][sid]["pcc"] for r in runs),
            "max_abs": max(r["shapes"][sid]["max_abs"] for r in runs),
        }
    return {"shapes": shapes, "noise_pct": round(noise, 2), "runs": len(runs)}


def summary_md(node: str, res: dict) -> str:
    lines = [f"# {node}: {res['fail_class']}" + (f", score {res['score']:.4f}" if res.get("valid") else ""), ""]
    lines.append("| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for sid, v in res["shapes"].items():

        def f(k, fmt="{:.3f}"):
            return fmt.format(v[k]) if v.get(k) is not None else "-"

        lines.append(
            f"| {sid} | {f('us_chip_mean')} | {f('us_chip_max')} | {f('baseline_us')} | {f('speedup', '{:.4f}')} "
            f"| {f('pcc', '{:.7f}')} | {f('max_abs', '{:.4f}')} | {v.get('pytest', '-')} |"
        )
    if res.get("noise_pct") is not None:
        lines += ["", f"Noise band: ±{res['noise_pct']}% (from baseline.json)."]
    return "\n".join(lines) + "\n"
