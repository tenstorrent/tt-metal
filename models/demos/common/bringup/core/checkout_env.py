# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Pin the environment to the checkout the framework runs from (F68).

A shell profile may export TT_METAL_HOME / PYTHONPATH / TT_METAL_CACHE / PYTHON_ENV_DIR for another checkout; the JIT
kernel cache is then shared across trees (entries recompile on a mismatch, but two trees write the same entries), and
the profiler and venv tooling point elsewhere. The orchestrator and the CLI call ``pin()`` first, so every gate,
agent and test they start inherits this checkout's values (as eval/run_eval.py does per clone).
"""

from __future__ import annotations

import os
from pathlib import Path

from models.demos.common.bringup.core.spec import CODE_ROOT


def values(root: Path = CODE_ROOT) -> dict[str, str]:
    root = Path(root).resolve()
    return {
        "TT_METAL_HOME": str(root),
        "PYTHONPATH": str(root),
        "TT_METAL_CACHE": str(root / "built"),
        "PYTHON_ENV_DIR": str(root / "python_env"),
    }


def pin(root: Path = CODE_ROOT, env=os.environ, echo=print) -> dict[str, str]:
    """Set the variables in ``env``; returns {name: old value} for the ones that pointed elsewhere (and says so)."""
    changed = {}
    for k, v in values(root).items():
        if env.get(k) != v:
            changed[k] = env.get(k)
            env[k] = v
    moved = {k: old for k, old in changed.items() if old}
    if moved:
        echo(
            "checkout env pinned to "
            + str(Path(root).resolve())
            + ": "
            + ", ".join(f"{k} (was {v})" for k, v in moved.items())
        )
    return changed
