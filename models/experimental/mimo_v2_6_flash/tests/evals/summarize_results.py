# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""run_eval.sh's results step: one summary.json per run from the harness output (lmms-eval results JSON + the
mimo_openrouter journal stats, or the newest inspect / openbench .eval log). Also answers run_eval.sh's resume
question (--unfinished-log DIR prints the newest inspect log that did not finish, if any)."""

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time


def _newest(pattern):
    paths = glob.glob(pattern, recursive=True)
    return max(paths, key=os.path.getmtime) if paths else None


def _versions():
    out = {}
    for mod in ("lmms_eval", "openbench", "inspect_ai", "openai"):
        try:
            from importlib.metadata import version

            out[mod] = version(mod.replace("_", "-") if mod != "inspect_ai" else "inspect_ai")
        except Exception:
            out[mod] = None
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        out["scripts_git"] = subprocess.run(
            ["git", "-C", here, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        out["scripts_git"] = None
    return out


def unfinished_log(log_dir):
    path = _newest(os.path.join(log_dir, "*.eval"))
    if not path:
        return None
    from inspect_ai.log import read_eval_log

    log = read_eval_log(path, header_only=True)
    return None if log.status == "success" else path


def lmms_summary(out, task):
    path = _newest(os.path.join(out, "lmms", "**", "*results*.json"))
    summary = {"results_file": path}
    if path:
        with open(path) as f:
            res = json.load(f)
        metrics = res.get("results", {}).get(task, {})
        summary["metrics"] = {k: v for k, v in metrics.items() if not k.startswith("alias")}
        summary["n_samples"] = res.get("n-samples", {}).get(task)
        summary["limit"] = res.get("config", {}).get("limit")
    stats = os.path.join(out, "journal.stats.json")
    if os.path.exists(stats):
        with open(stats) as f:
            summary["requests"] = json.load(f)
    return summary


def inspect_summary(out):
    path = _newest(os.path.join(out, "logs", "*.eval"))
    summary = {"log_file": path}
    if not path:
        return summary
    from inspect_ai.log import read_eval_log

    log = read_eval_log(path, header_only=True)
    summary["status"] = log.status
    if log.error:
        summary["error"] = str(log.error.message)[:500]
    metrics = {}
    if log.results:
        for score in log.results.scores:
            for name, metric in score.metrics.items():
                metrics[f"{score.name}/{name}"] = metric.value
        summary["n_samples"] = log.results.total_samples
        summary["completed_samples"] = log.results.completed_samples
    summary["metrics"] = metrics
    summary["epochs"] = log.eval.config.epochs
    if log.stats and log.stats.model_usage:
        summary["model_usage"] = {m: u.model_dump(exclude_none=True) for m, u in log.stats.model_usage.items()}
    return summary


REFERENCE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aa_reference.json")
# Where each bench's headline score sits in its summary, and a cap on its item count for the interval (None: the
# scored items).
SCORE_KEYS = {
    "mmmu_pro_aa": ("aa_accuracy,none", None),
}


def wilson(p, n, z=1.96):
    """95% Wilson interval of a proportion p measured on n items."""
    if not n:
        return None
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(center - half, 4), round(center + half, 4)]


def compare(bench, metrics, n_samples, reference_file=REFERENCE_FILE):
    """TT score vs the Artificial Analysis reference: both scores, the delta, the TT 95% interval (Wilson) and the
    delta's 95% interval (normal approximation of two independent proportions, each on its own item count)."""
    if bench not in SCORE_KEYS or not metrics:
        return None
    key, n_items = SCORE_KEYS[bench]
    if key not in metrics:
        return None
    with open(reference_file) as f:
        ref = json.load(f).get(bench) or {}
    tt = float(metrics[key])
    n = n_samples.get("effective") if isinstance(n_samples, dict) else n_samples
    if n_items:  # a limited run scores fewer items; a full one never more than the distinct ones
        n = min(n_items, n) if n else n_items
    out = {"tt": round(tt, 4), "n_tt": n, "ci95_tt": wilson(tt, n), "reference": "Artificial Analysis"}
    aa, n_ref = ref.get("score"), ref.get("n")
    out.update(aa=aa, n_aa=n_ref, reference_note=ref.get("note") or ref.get("metric"), reference_url=ref.get("url"))
    if aa is not None and n:
        delta = tt - aa
        se = math.sqrt(tt * (1 - tt) / n + (aa * (1 - aa) / n_ref if n_ref else 0.0))
        out.update(delta=round(delta, 4), ci95_delta=[round(delta - 1.96 * se, 4), round(delta + 1.96 * se, 4)])
        out["within_95ci"] = abs(delta) <= 1.96 * se
    return out


def bar_check(bench, metrics, reference_file=REFERENCE_FILE):
    """A media bench against its acceptance bar (aa_reference.json "acceptance_bars": comparable models' published
    numbers, weaker evidence than a same-protocol reference)."""
    with open(reference_file) as f:
        bar = (json.load(f).get("acceptance_bars") or {}).get(bench)
    if not bar or not metrics or bar["metric"] not in metrics:
        return None
    value = float(metrics[bar["metric"]])
    passed = value >= bar["bar"] if bar["direction"] == "min" else value <= bar["bar"]
    sign = ">=" if bar["direction"] == "min" else "<="
    return {
        "value": value,
        "bar": bar["bar"],
        "rule": f"{sign} {bar['bar']} {bar['unit']}",
        "passed": passed,
        "evidence": "weaker: comparable models, other protocols",
        "basis": bar["basis"],
    }


def comparison_line(c):
    if c is None:
        return None
    pct = lambda v: "n/a" if v is None else f"{100 * v:.2f}%"  # noqa: E731
    line = f"TT {pct(c['tt'])} (95% CI {pct(c['ci95_tt'][0])}-{pct(c['ci95_tt'][1])}, n={c['n_tt']})"
    if c.get("aa") is None:
        return line + f" | AA: none published ({c.get('reference_note')})"
    lo, hi = c["ci95_delta"]
    return (
        line
        + f" | AA {pct(c['aa'])} (n={c['n_aa']}) | delta {100 * c['delta']:+.2f} pt (95% CI {100 * lo:+.2f}"
        + f" to {100 * hi:+.2f}) {'within' if c['within_95ci'] else 'OUTSIDE'} the interval"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unfinished-log")
    ap.add_argument("--harness", choices=["lmms", "inspect"])
    ap.add_argument("--bench")
    ap.add_argument("--task")
    ap.add_argument("--out")
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--target")
    ap.add_argument("--thinking")
    ap.add_argument("--start", type=float)
    ap.add_argument("--cmd", default="")
    args = ap.parse_args()

    if args.unfinished_log is not None:
        path = unfinished_log(args.unfinished_log) if os.path.isdir(args.unfinished_log) else None
        if path:
            print(path)
        return
    body = lmms_summary(args.out, args.task) if args.harness == "lmms" else inspect_summary(args.out)
    summary = {
        "bench": args.bench,
        "task": args.task,
        "harness": args.harness,
        "model": args.model,
        "base_url": args.base_url,
        "target": args.target,
        "thinking": args.thinking,
        "wall_s": round(time.time() - args.start, 1) if args.start else None,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": args.cmd,
        "versions": _versions(),
        **body,
    }
    summary["vs_artificial_analysis"] = compare(args.bench, summary.get("metrics"), summary.get("n_samples"))
    summary["acceptance_bar"] = bar_check(args.bench, summary.get("metrics"))
    path = os.path.join(args.out, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=1, default=str)
    json.dump({k: summary.get(k) for k in ("bench", "metrics", "n_samples", "wall_s")}, sys.stdout, default=str)
    print()
    line = comparison_line(summary["vs_artificial_analysis"])
    if line:
        print(f"{args.bench}: {line}")
    bar = summary["acceptance_bar"]
    if bar:
        verdict = "meets" if bar["passed"] else "MISSES"
        print(f"{args.bench}: TT {bar['value']:.2f} {verdict} the bar {bar['rule']} ({bar['evidence']})")
    if not summary.get("metrics"):
        sys.exit(f"no metrics in the harness output; see {args.out}/run.log")


if __name__ == "__main__":
    main()
