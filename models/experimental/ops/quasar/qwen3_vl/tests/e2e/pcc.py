# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Per-stage PCC comparison, thresholds and the run verdict."""
import fnmatch
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch

THRESHOLDS_JSON = Path(__file__).with_name("thresholds.json")
_ORDER = [
    "vision.block",
    "vision.deepstack",
    "vision.merger",
    "text.layer",
    "text.norm",
    "text.logits.prefill",
    "text.logits.decode",
]


def _stage_sort_key(stage):
    m = re.search(r"(\d+)$", stage)
    prefix = stage[: m.start()] if m else stage
    return (_ORDER.index(prefix) if prefix in _ORDER else len(_ORDER), int(m.group(1)) if m else 0)


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    if a.std() == 0 or b.std() == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@dataclass
class StageResult:
    stage: str
    pcc: float
    max_abs: float
    threshold: float
    status: str
    seconds: float


def thresholds_for(preset, path=THRESHOLDS_JSON):
    data = json.loads(Path(path).read_text())
    table = {**data["default"], **data.get(preset, {})}

    def lookup(stage):
        hits = [p for p in table if fnmatch.fnmatchcase(stage, p)]
        if not hits:
            raise KeyError(f"no threshold for stage {stage}")
        return table[max(hits, key=len)]

    return lookup


def compare(goldens, actual, threshold, seconds):
    out = []
    for stage in sorted(goldens, key=_stage_sort_key):
        g, t, thr, sec = goldens[stage], actual.get(stage), threshold(stage), seconds.get(stage, 0.0)
        if t is None:
            out.append(StageResult(stage, float("nan"), float("nan"), thr, "MISSING", sec))
        elif tuple(t.shape) != tuple(g.shape):
            out.append(StageResult(stage, float("nan"), float("nan"), thr, "SHAPE", sec))
        elif not torch.isfinite(t).all():
            out.append(StageResult(stage, float("nan"), float("nan"), thr, "NONFINITE", sec))
        else:
            p = pcc(g, t)
            mx = float((g.float() - t.float()).abs().max())
            out.append(StageResult(stage, p, mx, thr, "PASS" if p >= thr else "FAIL", sec))
    return out


@dataclass
class Verdict:
    status: str
    first_failure: str | None
    markdown: str


def verdict(results, host_ops, override_hits, notes):
    failed = [r for r in results if r.status != "PASS"]
    status = "DIAGNOSTIC" if host_ops else ("FAIL" if failed else "PASS")
    first = failed[0].stage if failed else None
    lines = [f"# Verdict: {status}", ""]
    if first:
        lines += [f"**First failing stage:** `{first}`", ""]
    if host_ops:
        lines += [f"**Ops on host (not a pass):** {', '.join(host_ops)}", ""]
    lines += ["| stage | status | pcc | threshold | max abs | seconds |", "|---|---|---|---|---|---|"]
    lines += [
        f"| {r.stage} | {r.status} | {r.pcc:.5f} | {r.threshold} | {r.max_abs:.4g} | {r.seconds:.1f} |" for r in results
    ]
    if override_hits:
        lines += ["", "| override | hits |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sorted(override_hits.items())]
    if notes:
        lines += [""] + [f"- {n}" for n in notes]
    return Verdict(status, first, "\n".join(lines) + "\n")
