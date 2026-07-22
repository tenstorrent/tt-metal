"""Clone the FunAudioLLM/CosyVoice reference repo into ``model_data/CosyVoice_src/``.

Phase-0 setup step (BRINGUP_PLAN.md §7 Phase 0 Task 2). The reference repo is
read-only and is used only to extract op-inventory and to run
``example.py::cosyvoice2_example`` for golden fixtures.

Idempotent: if the target dir already exists at the pinned SHA, this is a no-op.
Otherwise it shallow-clones, then records the pinned commit SHA back into
``BRINGUP_PLAN.md`` §10 (References) so the next agent has a frozen reference.

Run inside the tt-metal env:
    source /root/tt-metal/python_env/bin/activate
    cd /root/tt-metal/models/demos/cosyvoice
    python scripts/clone_reference.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/FunAudioLLM/CosyVoice.git"
DEMO_ROOT = Path(__file__).resolve().parent.parent
TARGET = DEMO_ROOT / "model_data" / "CosyVoice_src"
PLAN = DEMO_ROOT / "BRINGUP_PLAN.md"


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> str:
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True).stdout.strip()


def get_sha(repo: Path) -> str:
    return run(["git", "rev-parse", "HEAD"], cwd=repo)


def record_sha_in_plan(sha: str) -> None:
    """Append/refresh a pinned-SHA marker line in BRINGUP_PLAN.md §10 references."""
    marker = "- Repo (CosyVoice pin): https://github.com/FunAudioLLM/CosyVoice @ commit"
    new_line = f"{marker} {sha}"
    text = PLAN.read_text()
    # Replace any existing pin line, else insert under the existing References repo entry.
    if marker in text:
        text = re.sub(rf"^{re.escape(marker)}.*$", new_line, text, flags=re.MULTILINE)
    else:
        anchor = "- Repo: https://github.com/FunAudioLLM/CosyVoice"
        replacement = f"{anchor}\n{new_line}"
        text = text.replace(anchor, replacement, 1)
    PLAN.write_text(text)


def main() -> int:
    if TARGET.exists() and (TARGET / ".git").exists():
        sha = get_sha(TARGET)
        print(f"[clone_reference] already present at {TARGET}  (SHA {sha[:12]}).")
        record_sha_in_plan(sha)
        return 0

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print(f"[clone_reference] cloning {REPO_URL} -> {TARGET}")
    run(["git", "clone", "--depth", "50", REPO_URL, str(TARGET)])
    # Initialize the vendored Matcha-TTS git submodule (third_party/Matcha-TTS).
    # The reference flow estimator imports `matcha.models.components.*`, so without
    # this submodule the reference path will fail `ModuleNotFoundError: No module named 'matcha'`.
    print("[clone_reference] initializing git submodules (Matcha-TTS)")
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=TARGET)
    sha = get_sha(TARGET)
    print(f"[clone_reference] HEAD = {sha}")
    gitlog = run(["git", "log", "-1", "--format=%ci  %s"], cwd=TARGET)
    print(f"[clone_reference] {gitlog}")
    record_sha_in_plan(sha)
    # Sanity check for required entry points referenced by the plan.
    for rel in ("example.py", "cosyvoice/cli/cosyvoice.py", "cosyvoice2.yaml"):
        p = TARGET / rel
        if not p.exists():
            # cosyvoice2.yaml is downloaded inside the HF snapshot, not the repo.
            if rel == "cosyvoice2.yaml":
                continue
            print(f"[clone_reference] WARNING: expected file missing: {rel}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
