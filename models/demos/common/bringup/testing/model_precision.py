# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Framework rule: component and swap tests check each step at maximum precision (HiFi4, fp32 accumulation, a bf16
KV cache), whatever the model ships with. Performance work may lower precision; that is judged end to end by the
ladder and the contract tests on the shipped defaults, never by the component limits.

``apply(spec)`` sets the model's max-precision overrides (``tt/settings.py``: ``Settings(..., max_precision={...})``)
in the environment before a component or swap test builds its modules. A variable the caller already set wins.
Switch: ``tests.component_max_precision`` (defaults.yaml)."""

from __future__ import annotations

import importlib
import os
from pathlib import Path


def overrides(spec) -> dict[str, str]:
    from models.demos.common.bringup.core.model_settings import Settings

    md = Path(spec.model_dir).resolve()
    rel = md.relative_to(Path(spec.repo).resolve())
    try:
        mod = importlib.import_module(".".join(rel.parts) + ".tt.settings")
    except ModuleNotFoundError:
        return {}
    env = {}
    for v in vars(mod).values():
        if isinstance(v, Settings):
            env.update(v.max_precision_env())
    return env


def apply(spec) -> dict[str, str]:
    """Set the max-precision overrides; return the ones applied (printed by the tests)."""
    if not spec.get("tests.component_max_precision"):
        return {}
    applied = {}
    for k, v in overrides(spec).items():
        if k not in os.environ:
            os.environ[k] = v
            applied[k] = v
    if applied:
        print("component test at max precision: " + ", ".join(f"{k}={v}" for k, v in sorted(applied.items())))
    return applied
