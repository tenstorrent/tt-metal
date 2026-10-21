# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""A model's settings (tt/settings.py) as the framework uses them: an A/B switch as environment variables (F103) and
an accepted perf pick shipped as the new default (F104).

F137 (owner): every gated device test runs the shipped defaults; there is no max-precision device run, not even as a
diagnostic (a failing per-step test is diagnosed on the CPU: testing/devfmt.py)."""

from __future__ import annotations

import importlib
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


def set_defaults(spec, switch: dict) -> list[str]:
    """F104: an accepted perf pick ships: every model setting ``switch`` names (tasks.yaml ab.change) gets the value as
    its default in tt/settings.py (the default argument of its Setting(...) entry rewritten in place). Returns
    "NAME: old -> new" per changed default; names that are no setting are left (the code reads them itself)."""
    import ast
    import json

    sets = model_settings(spec, fresh=True)
    path = Path(spec.model_dir) / "tt" / "settings.py"
    if not sets or not path.exists():
        return []
    src = path.read_text()
    lines = src.splitlines(keepends=True)
    offs = [0]
    for ln in lines:
        offs.append(offs[-1] + len(ln))
    edits, done = [], []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if not (isinstance(k, ast.Constant) and isinstance(v, ast.Call) and v.args):
                continue
            hit = next(
                (
                    (st, n)
                    for st in sets
                    for n in st.table
                    if n == k.value and any(x in switch for x in (n, st.env_name(n)))
                ),
                None,
            )
            if hit is None:
                continue
            st, n = hit
            old = st.table[n].default
            raw = switch.get(n, switch.get(st.env_name(n)))
            new = raw.lower() in ("1", "true", "yes") if isinstance(old, bool) else type(old)(raw)
            if new == old:
                continue
            a = v.args[0]
            start, end = offs[a.lineno - 1] + a.col_offset, offs[a.end_lineno - 1] + a.end_col_offset
            edits.append((start, end, json.dumps(new) if isinstance(new, str) else repr(new)))
            done.append(f"{n}: {old!r} -> {new!r}")
    for start, end, text in sorted(edits, reverse=True):
        src = src[:start] + text + src[end:]
    if edits:
        path.write_text(src)
    return done
