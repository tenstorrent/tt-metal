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


def model_settings(spec, fresh: bool = False) -> list:
    """The model's ``Settings`` objects (tt/settings.py); ``fresh`` re-imports it (an agent changed it since)."""
    from models.demos.common.bringup.core.model_settings import Settings

    md = Path(spec.model_dir).resolve()
    rel = md.relative_to(Path(spec.repo).resolve())
    try:
        mod = importlib.import_module(".".join(rel.parts) + ".tt.settings")
        if fresh:
            mod = importlib.reload(mod)
    except ModuleNotFoundError:
        return []
    return [v for v in vars(mod).values() if isinstance(v, Settings)]


def overrides(spec) -> dict[str, str]:
    env = {}
    for v in model_settings(spec):
        env.update(v.max_precision_env())
    return env


def switch_env(spec, switch: dict) -> tuple[dict[str, str], list[str]]:
    """F103: a tasks.yaml ``ab`` switch {name: value} as environment variables. A model setting's name becomes its
    variable (the Settings prefix + name, e.g. EXPERTS_IMPL -> DS41_EXPERTS_IMPL) and its value must be one of the
    setting's choices (ValueError otherwise); a setting's variable name stays. Returns (env, the names that are no
    setting: plain environment variables the code must read itself)."""
    sets = model_settings(spec, fresh=True)
    env, plain = {}, []
    for name, value in (switch or {}).items():
        hit = next(((st, n) for st in sets for n in st.table if name in (n, st.env_name(n))), None)
        if hit is None:
            env[name] = str(value)
            plain.append(name)
            continue
        st, n = hit
        choices = st.table[n].choices
        if choices is not None and str(value) not in {str(c) for c in choices}:
            raise ValueError(f"ab switch {name}={value}: setting {n} takes one of {list(choices)}")
        env[st.env_name(n)] = str(value)
    return env, plain


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
