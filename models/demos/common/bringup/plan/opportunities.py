# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Performance review, list half: rank the profiled sections by measured device time and attach what the shared
knowledge says about each. Applies nothing; a person picks entries, and each pick becomes an implement task
(step: perf) that passes only if warm device time drops and every accuracy gate still passes.

    python -m models.demos.common.bringup.plan.opportunities --profile X.1 [--top 12]

Writes <bringup_dir>/opportunities.md; records opportunities_listed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.knowledge.check import HERE as KNOWLEDGE
from models.demos.common.bringup.knowledge.check import entries
from models.demos.common.bringup.reference.golden import load_spec

# section keyword -> words to look for in the repo map rows and known issues
KEYWORDS = {
    "sdpa": ["sdpa", "attention"],
    "attn": ["attention", "rope"],
    "rope": ["rope"],
    "moe": ["moe", "expert"],
    "experts": ["expert", "moe"],
    "router": ["router", "moe"],
    "dispatch": ["dispatch", "moe"],
    "combine": ["combine", "moe"],
    "all_reduce": ["collective", "all_reduce", "ccl"],
    "all_gather": ["collective", "all_gather", "ccl"],
    "mlp": ["mlp", "matmul"],
    "norm": ["norm"],
    "embed": ["embedding"],
    "kv": ["kv", "cache"],
}


def words_for(section: str) -> list[str]:
    toks = re.split(r"[._\s]+", section.lower())
    out = []
    for t in toks:
        out += KEYWORDS.get(t, [t])
    return sorted(set(out))


def repo_rows(words) -> list[str]:
    rows = []
    for line in (KNOWLEDGE / "repo_map.md").read_text().splitlines():
        if line.startswith("| ") and not line.startswith("| Need") and not line.startswith("|---"):
            if any(w in line.lower() for w in words):
                rows.append(line.split("|")[1].strip())
    return rows


def perf_issues(words) -> list[str]:
    out = []
    for it in entries((KNOWLEDGE / "known_issues.md").read_text()).get("Performance", []):
        if any(w in it.lower() for w in words):
            out.append(re.sub(r"^- \*\*(.+?)\*\*.*", r"\1", it))
    return out


def build(profile: dict, top: int) -> tuple[str, int]:
    total = sum(profile["sections_ms"].values())
    lines = [
        f"# Performance opportunities",
        "",
        f"Profile: rung {profile['rung']}, chunk {profile['chunk'][0]}->{profile['chunk'][1]}, layers "
        f"{len(profile['layers'])}. Wall {profile['wall_ms']:.0f} ms, device {total:.0f} ms (warm, slowest chip per section).",
        "",
        "Pick entries by marking `[x]`. Each picked entry becomes a perf task behind a switch, gated on warm device time",
        "and on every accuracy gate. Changes that trade accuracy for speed need an explicit decision here.",
        "",
        "| Pick | Rank | Section | Device ms | Share | Chip spread | Known issues | Repo map |",
        "|---|---|---|---|---|---|---|---|",
    ]
    n = 0
    for k, (sec, ms) in enumerate(sorted(profile["sections_ms"].items(), key=lambda x: -x[1])[:top], 1):
        chips = profile["sections_ms_per_chip"].get(sec, {})
        spread = (max(chips.values()) - min(chips.values())) if len(chips) > 1 else 0.0
        w = words_for(sec)
        lines.append(
            f"| [ ] | {k} | `{sec}` | {ms:.1f} | {100 * ms / total:.1f}% | {spread:.1f} ms | "
            f"{'; '.join(perf_issues(w)) or '-'} | {'; '.join(repo_rows(w)) or '-'} |"
        )
        n += 1
    return "\n".join(lines) + "\n", n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    ap.add_argument("--profile", required=True, help="task id whose <id>_profile.json to read")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    profile = json.loads((spec.bringup_dir / "results" / f"{a.profile}_profile.json").read_text())
    text, n = build(profile, a.top)
    out = spec.bringup_dir / "opportunities.md"
    if out.exists() and "[x]" in out.read_text():
        out = Path(str(out).replace(".md", ".new.md"))  # never overwrite a list a person already picked from
    out.write_text(text)
    print(text)
    metrics.record("opportunities_listed", n)


if __name__ == "__main__":
    main()
