# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""run_eval.sh's results step: one summary.json per run from the harness output (lmms-eval results JSON + the
mimo_openrouter journal stats, or the newest inspect / openbench .eval log). Also answers run_eval.sh's resume
question (--unfinished-log DIR prints the newest inspect log that did not finish, if any)."""

import argparse
import glob
import json
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
    path = os.path.join(args.out, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=1, default=str)
    json.dump({k: summary.get(k) for k in ("bench", "metrics", "n_samples", "wall_s")}, sys.stdout, default=str)
    print()
    if not summary.get("metrics"):
        sys.exit(f"no metrics in the harness output; see {args.out}/run.log")


if __name__ == "__main__":
    main()
