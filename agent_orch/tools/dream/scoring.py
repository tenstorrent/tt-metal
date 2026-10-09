"""The evaluation contract: turn the JSON an eval command writes into a validated, scored attempt.

An eval command writes one JSON object to $DREAM_RESULT:

    {"valid": true,                       # the test's own correctness verdict
     "cases": {"<case id>": {"value": 11.19, "<extra>": 0.99999, ...}, ...},
     "error": null,                       # short message when valid is false
     "fail_class": null}                  # optional hint: jit_compile_error, runtime_error, accuracy_fail, infra

A single-case result may use a top-level "value" instead of "cases". The engine
applies the spec's gates to each case's extra fields, and scores a valid attempt
as the geometric mean over cases of its improvement ratio vs the baseline
(baseline = 1.0, higher is better, whatever eval.direction is).
"""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

from .campaign import parse_gate

FAIL_CLASSES = {
    "ok",
    "build_error",
    "jit_compile_error",
    "runtime_error",
    "hang",
    "accuracy_fail",
    "forbidden_edit",
    "infra",
    "lost",
    "isolation",
}
_OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
}


def geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


def log_excerpt(text: str, n: int = 60) -> str:
    lines = text.splitlines()
    idx = [i for i, l in enumerate(lines) if re.search(r"error|Error|FAILED|TT_FATAL|TT_THROW|Traceback", l)]
    if idx:
        start = max(0, idx[0] - 5)
        return "\n".join(lines[start : start + n])
    return "\n".join(lines[-n:])


def classify_log(text: str) -> str:
    if re.search(r":\d+:\d+: error:", text) or re.search(r"(?i)failed to (compile|build)|compile failed", text):
        return "jit_compile_error"
    return "runtime_error"


def read_result(path: Path, log_text: str = "") -> dict:
    """Normalize an eval command's result file. Never raises: problems become fail classes."""
    if not path.exists() or not path.read_text().strip():
        return {
            "valid": False,
            "fail_class": classify_log(log_text),
            "cases": {},
            "error": "eval command wrote no result JSON\n" + log_excerpt(log_text, 30),
        }
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return {"valid": False, "fail_class": "infra", "cases": {}, "error": f"result JSON does not parse: {e}"}
    cases = raw.get("cases")
    if cases is None and "value" in raw:
        cases = {"default": {"value": raw["value"], **raw.get("extra", {})}}
    cases = cases or {}
    out = {"valid": bool(raw.get("valid", False)), "cases": {}, "error": raw.get("error")}
    for cid, v in cases.items():
        v = dict(v) if isinstance(v, dict) else {"value": v}
        try:
            v["value"] = float(v["value"]) if v.get("value") is not None else None
        except (TypeError, ValueError):
            v["value"] = None
        out["cases"][str(cid)] = v
    if out["valid"]:
        bad = [cid for cid, v in out["cases"].items() if v["value"] is None or not math.isfinite(v["value"])]
        if not out["cases"] or bad:
            out.update(
                valid=False,
                fail_class="infra",
                error=f"valid result with missing/non-finite values: {bad or 'no cases'}",
            )
            return out
        out["fail_class"] = "ok"
    else:
        hint = raw.get("fail_class")
        out["fail_class"] = hint if hint in FAIL_CLASSES and hint != "ok" else "accuracy_fail"
        out["error"] = out["error"] or "eval command reported valid=false"
    return out


def apply_gates(res: dict, gates: list[str]) -> dict:
    if not res["valid"]:
        return res
    for cid, v in res["cases"].items():
        for g in gates:
            k, op, ref = parse_gate(g)
            x = v.get(k)
            try:
                ok = x is not None and _OPS[op](float(x), ref)
            except (TypeError, ValueError):
                ok = False
            if not ok:
                res.update(valid=False, fail_class="accuracy_fail", error=f"{cid}: gate '{g}' failed ({k}={x})")
                return res
    return res


def ratio(value: float, base: float, direction: str) -> float:
    return base / value if direction == "minimize" else value / base


def score_vs_baseline(res: dict, baseline: dict, direction: str) -> float | None:
    """Fill per-case baseline/ratio and return the geomean ratio, or None if the cases don't match the baseline."""
    if set(res["cases"]) != set(baseline["cases"]):
        return None
    ratios = []
    for cid, v in res["cases"].items():
        b = baseline["cases"][cid]["value"]
        v["baseline"] = b
        v["ratio"] = round(ratio(v["value"], b, direction), 4)
        ratios.append(v["ratio"])
    return round(geomean(ratios), 4)


def make_baseline(runs: list[dict], min_noise_pct: float, direction: str, unit: str) -> dict:
    cases = {}
    noise = float(min_noise_pct)
    for cid in runs[0]["cases"]:
        vals = [r["cases"][cid]["value"] for r in runs]
        med = statistics.median(vals)
        spread = (max(vals) - min(vals)) / abs(med) * 100 if med else 0.0
        noise = max(noise, spread)
        extras = {k: x for k, x in runs[0]["cases"][cid].items() if k != "value" and isinstance(x, (int, float))}
        cases[cid] = {"value": med, "runs": vals, "spread_pct": round(spread, 2), **extras}
    return {"cases": cases, "noise_pct": round(noise, 2), "runs": len(runs), "direction": direction, "unit": unit}


def drift(measure: dict, baseline: dict) -> list[tuple[str, float, float, float, bool]]:
    """Per case: (case, value, baseline, pct change, beyond noise)."""
    out = []
    for cid, v in measure["cases"].items():
        b = baseline["cases"].get(cid, {}).get("value")
        if b is None or v.get("value") is None:
            out.append((cid, v.get("value") or float("nan"), b or float("nan"), float("nan"), True))
            continue
        d = (v["value"] - b) / b * 100
        out.append((cid, v["value"], b, d, abs(d) > baseline["noise_pct"]))
    return out


def case_metrics(score: dict) -> dict:
    """Per-case {value, baseline, ratio, ...} of a score.json; also reads the legacy per-shape format."""
    if score.get("cases"):
        return score["cases"]
    return {
        sid: {
            "value": v.get("us_chip_mean"),
            "baseline": v.get("baseline_us"),
            "ratio": v.get("speedup"),
            "pcc": v.get("pcc"),
            "max_abs": v.get("max_abs"),
        }
        for sid, v in (score.get("shapes") or {}).items()
    }


def summary_md(node: str, res: dict, unit: str) -> str:
    head = f"# {node}: {res['fail_class']}" + (f", score {res['score']:.4f}" if res.get("valid") else "")
    lines = [head, ""]
    extras = sorted({k for v in res.get("cases", {}).values() for k in v} - {"value", "baseline", "ratio"})
    u = f" ({unit})" if unit else ""
    lines.append(f"| case | value{u} | baseline{u} | ratio | " + " | ".join(extras) + (" |" if extras else ""))
    lines.append("|---|---|---|---|" + "---|" * len(extras))
    for cid, v in res.get("cases", {}).items():

        def f(x):
            return f"{x:.6g}" if isinstance(x, (int, float)) else ("-" if x is None else str(x))

        lines.append(
            f"| {cid} | {f(v.get('value'))} | {f(v.get('baseline'))} | {f(v.get('ratio'))} | "
            + " | ".join(f(v.get(k)) for k in extras)
            + (" |" if extras else "")
        )
    if res.get("noise_pct") is not None:
        lines += ["", f"Noise band: ±{res['noise_pct']}% (from baseline.json). Ratios inside it are not improvements."]
    if res.get("error"):
        lines += ["", "```", str(res["error"])[:2000], "```"]
    return "\n".join(lines) + "\n"
