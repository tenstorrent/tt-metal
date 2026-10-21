# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F112 / F128 / F132: every check's target is the full-precision (native) reference, reached only through
reference.golden.reference_at; no spec key switches it."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.demos.common.bringup.core.spec import Spec
from models.demos.common.bringup.reference.golden import reference_at
from models.demos.common.bringup.reference.interface import step_reference

ROOT = Path(__file__).resolve().parents[1]


def _spec(reference, **keys):
    hooks = SimpleNamespace(reference=reference)
    return SimpleNamespace(hooks=lambda: hooks, get=lambda key, d=None: keys.get(key, d))


def test_per_step_checks_always_get_the_native_reference():
    def reference(spec, layers=None, dtype=None, numerics="official"):
        return (layers, dtype, numerics)

    assert step_reference(_spec(reference), [3], "f32") == ([3], "f32", "native")
    assert step_reference(_spec(reference, **{"tests.step_numerics": "official"}), [3], "f32")[2] == "native"


def test_a_kwargs_only_hook_stops():
    """A hook that takes **kw would accept numerics and may ignore it: a rounded golden labelled native (V12)."""

    def reference(spec, layers=None, dtype=None, **kw):
        return kw

    with pytest.raises(SystemExit, match="numerics"):
        reference_at(_spec(reference), "native")


def test_a_hook_without_the_switch_never_falls_back_to_its_rounded_arithmetic():
    """F128 (owner): a hook that cannot run at full precision stops, for either numerics, unless the spec declares its
    only arithmetic free of roundings (reference.no_rounding)."""

    def reference(spec, layers=None, dtype=None):
        return "official-only"

    for numerics in ("native", "official"):
        with pytest.raises(SystemExit, match="numerics"):
            reference_at(_spec(reference), numerics)
    assert step_reference(_spec(reference, **{"reference.no_rounding": True}), [0], None) == "official-only"
    with pytest.raises(ValueError):
        reference_at(_spec(reference, **{"reference.no_rounding": True}), "rounded")


@pytest.mark.parametrize("key", ["golden.numerics", "tests.step_numerics"])
def test_the_spec_cannot_switch_the_target(fx, key):
    a, b = key.split(".")
    errs = Spec.load(fx(**{a: {b: "official"}})).validate()
    assert any(key in e and "F132" in e for e in errs), errs


def test_nothing_calls_the_reference_hook_but_reference_at():
    """F132: one entry point, so every caller names its numerics."""
    bad = []
    for f in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        for n in ast.walk(ast.parse(f.read_text())):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "reference":
                bad.append(f"{f.relative_to(ROOT)}:{n.lineno}")
    assert not bad, bad
