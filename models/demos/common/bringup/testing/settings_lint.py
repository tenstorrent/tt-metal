# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""One place for a model's switches (core/model_settings.py). Checks:
  model code      `models/demos/<model>/tt/**.py` and `bringup/hooks.py` read no environment variable except in
                  `tt/settings.py` (which builds the Settings table)
  bring-up forks  `ttnn/ttnn/bringup/**` (not tests/): behaviour comes in as op arguments set from the model's
                  settings; an environment knob is allowed only for diagnostics, marked `diagnostic` on its line.
                  F106: only the lines a change adds count (vs ``base``, default HEAD): an imported fork's own knobs
                  (e.g. 170 MIMO_FL_* reads in flat_routed_expert) are not the editing agent's to convert; F231:
                  nor are the lines of a branch the task imports as is (tasks.yaml ``imports``)
  emulation       F137 (owner): no setting's default is an emulation (a choice that rounds to a format the device
                  neither computes nor stores in, agents' rule 17: Setting(emulation=...)); a choice whose name looks
                  like one (EMULATION_HINT) must be declared, so the check cannot be dodged by leaving it out
The orchestrator runs it on every agent's changed files; `python -m models.demos.common.bringup.testing.settings_lint`
checks the whole model (the settings audit's gate, metric settings_violations)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ENV_READ = re.compile(r"os\.environ\b|os\.getenv\(|\bgetenv\s*\(|std::getenv")
EMULATION_HINT = re.compile(r"qdq|tf32|emul|parity", re.I)  # names of choices that only mimic a reference format
FORKS = "ttnn/ttnn/bringup/"


def _base_lines(repo: Path, rel: str, base: str | None) -> set[str]:
    """The stripped lines of ``rel`` at ``base`` (empty: a new file, or no base)."""
    if not base:
        return set()
    r = subprocess.run(["git", "show", f"{base}:{rel}"], cwd=repo, capture_output=True, text=True)
    return {ln.strip() for ln in r.stdout.splitlines()} if r.returncode == 0 else set()


def violations(
    repo: Path, model_dir: str, files: list[str], base: str | None = "HEAD", imports: list[str] | None = None
) -> list[str]:
    """``imports`` (F231, tasks.yaml ``imports``): refs a task takes code from as is (a branch merge); a fork line that
    is verbatim in an import's version of the file is the import's own knob, as F106 treats a line already at ``base``.
    """
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
        had = _base_lines(repo, rel, base) if in_fork else set()
        for ref in (imports or []) if in_fork else []:
            had |= _base_lines(repo, rel, ref)
        for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if not ENV_READ.search(line) or line.lstrip().startswith(("#", "//", "*")):
                continue
            if line.strip() in had:  # F106: the fork had this knob before the change
                continue
            if in_model:
                out.append(f"{rel}:{n}: reads the environment; put the switch in {model_dir}tt/settings.py")
            elif "diagnostic" not in line.lower():
                out.append(
                    f"{rel}:{n}: a fork reads the environment; take it as an op argument (or mark a pure "
                    "diagnostic knob `diagnostic` on that line)"
                )
    return out


def emulation_violations(sets) -> list[str]:
    """F137: every Settings table's defaults and undeclared emulation-looking choices."""
    out = []
    for st in sets:
        for name, s in st.table.items():
            emu = {str(x) for x in s.emulation}
            if str(s.default) in emu:
                out.append(
                    f"{st.env_name(name)}: the default {s.default!r} is an emulation (rule 17); ship a device format"
                )
            for c in [s.default, *(s.choices or ())]:
                if isinstance(c, str) and EMULATION_HINT.search(c) and c not in emu:
                    out.append(
                        f"{st.env_name(name)}: choice {c!r} looks like an emulation; declare it in "
                        "Setting(emulation=...) (never a default) or name it after the device format it is"
                    )
    return sorted(set(out))


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
    from models.demos.common.bringup.testing.model_precision import model_settings

    bad = violations(repo, md, files) + emulation_violations(model_settings(spec))
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
