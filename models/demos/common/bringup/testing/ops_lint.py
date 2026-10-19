# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up ops before stock TTNN (F113, owner). ``ttnn/ttnn/bringup/INDEX.md`` has one entry per op folder:

    - `ttnn.bringup.<op>` (<folder>): <what it does>.
      Consider instead of: <stock op>, <stock op>; <composition>. -> <folder>/README.md

and every op folder a README.md with SECTIONS. Checks:
  docs    an op folder has a README.md with every section and an INDEX.md entry; for an agent step, the folders it
          changed or created; ``python -m models.demos.common.bringup.testing.ops_lint`` checks all of them
  stock   a line a step adds to the model's code (tt/**.py, bringup/hooks.py) that calls a stock op an INDEX entry lists
          as a standalone "Consider instead of" item (``ttnn.all_gather``, not a composition ``ttnn.all_gather +
          ttnn.add`` or ``a per-expert ttnn.linear loop``) needs one line in the model's BREADCRUMBS.md:
          ``stock <ttnn.op> instead of <ttnn.bringup.op>: <reason>``. Soft: justified, never banned."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from models.demos.common.bringup.testing.settings_lint import _base_lines

FORKS = "ttnn/ttnn/bringup/"
SECTIONS = (
    "## What it does",
    "## Consider it instead of",
    "## Options",
    "## Measured",
    "## Limits",
    "## How to call",
    "## Used by",
)
NOT_OPS = {"__pycache__", "_testing"}
STOCK_ITEM = re.compile(r"`?(ttnn(?:\.\w+)+)`?")


def op_folders(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in NOT_OPS and not p.name.startswith("."))


def parse_index(text: str) -> dict[str, dict]:
    """folder -> {"op": its first listed callable, "consider": [standalone stock ops]}."""
    out, entry = {}, None
    for line in text.splitlines():
        if line.startswith("- `"):
            entry = {"op": line[3:].split("`", 1)[0], "consider": []}
        elif entry is not None and "Consider instead of:" in line:
            body, _, tail = line.split("Consider instead of:", 1)[1].partition("->")
            m = re.search(r"([\w.-]+)/README\.md", tail)
            if m:
                for item in re.split(r"[,;]", body.strip().rstrip(".")):
                    prefix = ""
                    for part in item.split(" / "):  # `ttnn.experimental.a / b`: b shares a's module path
                        part = part.strip().rstrip(".").strip("`")
                        if prefix and not part.startswith("ttnn."):
                            part = prefix + part
                        s = STOCK_ITEM.fullmatch(part)
                        if not s:
                            break
                        entry["consider"].append(s.group(1))
                        prefix = s.group(1).rsplit(".", 1)[0] + "."
                out[m.group(1)] = entry
            entry = None
    return out


def doc_problems(root: Path, folders: list[str] | None = None) -> list[str]:
    index = parse_index((root / "INDEX.md").read_text()) if (root / "INDEX.md").exists() else {}
    out = []
    for f in op_folders(root) if folders is None else folders:
        if not (root / f).is_dir() or f in NOT_OPS:
            continue
        readme = root / f / "README.md"
        if not readme.exists():
            out.append(f"{FORKS}{f}: no README.md (sections: {', '.join(SECTIONS)})")
        else:
            text = readme.read_text()
            missing = [s for s in SECTIONS if s not in text]
            if missing:
                out.append(f"{FORKS}{f}/README.md: missing sections {missing}")
        if f not in index:
            out.append(
                f"{FORKS}{f}: no INDEX.md entry (`- \\`ttnn.bringup.<op>\\` ({f}): ...` + `Consider instead of:`)"
            )
    return out


def stock_problems(repo: Path, model_dir: str, files: list[str], breadcrumbs: Path, base: str | None = "HEAD"):
    index = parse_index((repo / FORKS / "INDEX.md").read_text()) if (repo / FORKS / "INDEX.md").exists() else {}
    alt: dict[str, list[str]] = {}
    for e in index.values():
        for s in e["consider"]:
            alt.setdefault(s, []).append(e["op"])
    if not alt:
        return []
    pats = {s: re.compile(r"(?<![\w.])" + re.escape(s) + r"\s*\(") for s in alt}
    notes = breadcrumbs.read_text().lower() if breadcrumbs.exists() else ""
    model_dir = model_dir.rstrip("/") + "/"
    out = []
    for rel in files:
        p = repo / rel
        if not (rel.startswith(model_dir + "tt/") or rel == model_dir + "bringup/hooks.py") or not rel.endswith(".py"):
            continue
        if not p.is_file():
            continue
        had = _base_lines(repo, rel, base)
        for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if line.lstrip().startswith("#") or line.strip() in had:
                continue
            for s, pat in pats.items():
                if pat.search(line) and f"stock {s} instead of".lower() not in notes:
                    out.append(
                        f"{rel}:{n}: calls {s}; INDEX.md lists {' / '.join(alt[s])} to consider instead: use it, or "
                        f"justify in BREADCRUMBS.md with one line `stock {s} instead of {alt[s][0]}: <reason>`"
                    )
    return out


def violations(repo: Path, model_dir: str, changed: list[str], breadcrumbs: Path) -> list[str]:
    folders = sorted(
        {c[len(FORKS) :].split("/", 1)[0] for c in changed if c.startswith(FORKS) and "/" in c[len(FORKS) :]}
    )
    return doc_problems(repo / FORKS, folders) + stock_problems(repo, model_dir, changed, breadcrumbs)


def main(argv=None) -> int:
    root = Path(__file__).resolve().parents[5] / FORKS
    bad = doc_problems(root)
    for v in bad:
        print(f"FAIL {v}")
    print(f"ops lint: {len(op_folders(root))} op folders, {len(bad)} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
