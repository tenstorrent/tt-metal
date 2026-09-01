# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import argparse
import json
import os
import sys
from pathlib import Path

_FAMILIES = (
    ("chunk_time_ms", "chunk_time", "ms", True, 0),
    ("ttft_s", "ttft", "s", True, 3),
    ("throughput_tok_s", "throughput", "tok/s", False, 0),
)
_BASE, _REF = "sc1", "sc4"


def _tokens(label):
    text = label.lstrip("@")
    scale = 1000 if text.endswith("k") else 1
    try:
        return int(float(text.rstrip("k")) * scale)
    except ValueError:
        return None


def _ordered(labels):
    return sorted(labels, key=lambda l: (_tokens(l) is None, _tokens(l) or 0, l))


def _load(root):
    by_model = {}
    for path in sorted(Path(root).rglob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            print(f"::warning::unreadable metrics sidecar {path}: {exc}")
            continue
        model, config = rec.get("model"), rec.get("config")
        if not model or not config:
            continue
        by_model.setdefault(model, {})[config] = rec
    return by_model


def _ideal(family, label, base, ref):
    rb, rr = base.get("pipeline_ranks"), ref.get("pipeline_ranks")
    if not rb or not rr:
        return None
    if family == "chunk_time_ms":
        return 1.0
    if family == "throughput_tok_s":
        return rr / rb
    d = ref.get("chunk_index", {}).get(label)
    if d is None:
        return None
    c = d + 1
    return ((c + rb - 1) / rb) / ((c + rr - 1) / rr)


def _rows(base, ref, gain=True):
    rows = []
    for family, name, unit, lower_better, places in _FAMILIES:
        for label in _ordered(set(base.get(family, {})) | set(ref.get(family, {}))):
            b, r = base.get(family, {}).get(label), ref.get(family, {}).get(label)
            metric = f"{name} {label}"
            measured = None
            if gain and b is not None and r is not None and b > 0 and r > 0:
                measured = (b / r) if lower_better else (r / b)
            ideal = _ideal(family, label, base, ref) if gain else None
            fmt = lambda v: "-" if v is None else f"{v:,.{places}f} {unit}"  # noqa: E731
            factor = lambda v: "-" if v is None else f"{v:.2f}x"  # noqa: E731
            eff = "-" if not measured or not ideal else f"{measured / ideal * 100:.0f}%"
            rows.append([metric, fmt(b), fmt(r), factor(measured), factor(ideal), eff])
    return rows


def _table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    render = lambda vals: "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(vals)) + " |"  # noqa: E731
    return [sep, render(headers), sep, *[render(r) for r in rows], sep]


def _comparable(base, ref):
    for key in ("chunk_size", "num_chunks"):
        if base.get(key) != ref.get(key):
            return f"{key} differs ({_BASE}={base.get(key)}, {_REF}={ref.get(key)})"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", required=True, help="dir of *.json metrics sidecars (searched recursively)")
    ap.add_argument("--model", default=None, help="render only this model (default: every model found)")
    ap.add_argument("--gantt-url", default=None, help="link the pipeline gantt artifact under the table")
    args = ap.parse_args()

    lines = []
    by_model = _load(args.metrics_dir) if os.path.isdir(args.metrics_dir) else {}
    if args.model:
        by_model = {k: v for k, v in by_model.items() if k == args.model}
    if not by_model:
        which = f"for model {args.model} " if args.model else ""
        lines.append(f"No metrics sidecars found {which}under {args.metrics_dir}; nothing to compare.")
    for model in sorted(by_model):
        configs = by_model[model]
        base, ref = configs.get(_BASE) or {}, configs.get(_REF) or {}
        lines.append(f"#### {model}")
        lines.append("")
        note = None
        if not base or not ref:
            note = f"only {', '.join(sorted(configs))} ran this run; no gain to compute"
        elif reason := _comparable(base, ref):
            note = f"{_BASE} and {_REF} measured different requests ({reason}); gain withheld"
        if note:
            lines.append(note)
            lines.append("")
        shape = ref or base
        lines.append("```text")
        lines.append(f"request: {shape['num_chunks']} chunks x {shape['chunk_size']} tok = {shape['max_seq']} tok")
        lines.append(f"gain = how many times better {_REF} is than {_BASE}")
        if not note:
            rb, rr = base.get("pipeline_ranks"), ref.get("pipeline_ranks")
            lines.append(f"ideal = a perfect {rb}-rank -> {rr}-rank pipeline, C chunks deep:")
            lines.append("        chunk_time x1 (latency is work, not width), ttft xC*R/(C+R-1), throughput xR")
        lines += _table(["metric", _BASE, _REF, f"{_REF} gain", "ideal", "of ideal"], _rows(base, ref, gain=not note))
        lines.append("```")
        lines.append("")
        if args.gantt_url:
            lines.append(f"[4 Galaxy Pipeline gantt (PNG)]({args.gantt_url})")
            lines.append("")

    title = f"disaggregated prefill scaling -- {_REF} vs {_BASE}"
    block = "### {}\n\n{}\n".format(title, "\n".join(lines))
    print(block)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as fh:
            fh.write(block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
