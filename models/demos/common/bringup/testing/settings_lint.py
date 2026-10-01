# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""One place for a model's switches (core/model_settings.py). Checks:
  model code      `models/demos/<model>/tt/**.py` and `bringup/hooks.py` read no environment variable except in
                  `tt/settings.py` (which builds the Settings table)
  bring-up forks  `ttnn/ttnn/bringup/**` (not tests/): behaviour comes in as op arguments set from the model's
                  settings; an environment knob is allowed only for diagnostics, marked `diagnostic` on its line
The orchestrator runs it on every agent's changed files; `python -m models.demos.common.bringup.testing.settings_lint`
checks the whole model (the settings audit's gate, metric settings_violations)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ENV_READ = re.compile(r"os\.environ\b|os\.getenv\(|\bgetenv\s*\(|std::getenv")
FORKS = "ttnn/ttnn/bringup/"


def violations(repo: Path, model_dir: str, files: list[str]) -> list[str]:
    out = []
    model_dir = model_dir.rstrip("/") + "/"
    for rel in files:
        p = repo / rel
        if not p.is_file() or p.suffix not in (".py", ".cpp", ".hpp", ".h"):
            continue
        in_model = rel.startswith(model_dir + "tt/") or rel == model_dir + "bringup/hooks.py"
        in_fork = rel.startswith(FORKS) and "/tests/" not in rel
        if not (in_model or in_fork) or rel == model_dir + "tt/settings.py":
            continue
        for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if not ENV_READ.search(line) or line.lstrip().startswith(("#", "//", "*")):
                continue
            if in_model:
                out.append(f"{rel}:{n}: reads the environment; put the switch in {model_dir}tt/settings.py")
            elif "diagnostic" not in line.lower():
                out.append(
                    f"{rel}:{n}: a fork reads the environment; take it as an op argument (or mark a pure "
                    "diagnostic knob `diagnostic` on that line)"
                )
    return out


def model_files(repo: Path, model_dir: str) -> list[str]:
    base = repo / model_dir
    files = [str(p.relative_to(repo)) for p in (base / "tt").rglob("*.py")] if (base / "tt").is_dir() else []
    if (base / "bringup" / "hooks.py").exists():
        files.append(f"{model_dir}/bringup/hooks.py")
    return files


def main(argv=None) -> int:
    from models.demos.common.bringup.core import metrics
    from models.demos.common.bringup.reference.golden import load_spec

    spec = load_spec(None)
    repo = Path(spec.repo)
    md = str(Path(spec.model_dir).resolve().relative_to(repo.resolve()))
    files = model_files(repo, md)
    bad = violations(repo, md, files)
    for v in bad:
        print(f"FAIL {v}")
    has_settings = (repo / md / "tt" / "settings.py").exists()
    if not has_settings:
        print(f"FAIL {md}/tt/settings.py missing")
    metrics.record("settings_violations", len(bad) + (0 if has_settings else 1))
    print(f"settings lint: {len(files)} files, {len(bad)} violations")
    return 0 if not bad and has_settings else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
