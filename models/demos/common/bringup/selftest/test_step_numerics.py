# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F112 / F128: per-step checks ask the model's reference for its full-precision (native) numerics."""

from types import SimpleNamespace

from models.demos.common.bringup.reference.interface import step_reference


def _spec(reference, numerics="native"):
    hooks = SimpleNamespace(reference=reference)
    return SimpleNamespace(hooks=lambda: hooks, get=lambda key, d=None: numerics if key == "tests.step_numerics" else d)


def test_hook_with_numerics_gets_the_setting():
    def reference(spec, layers=None, dtype=None, numerics="official"):
        return (layers, dtype, numerics)

    assert step_reference(_spec(reference), [3], "f32") == ([3], "f32", "native")
    assert step_reference(_spec(reference, "official"), [3], "f32")[2] == "official"


def test_hook_with_kwargs_gets_the_setting():
    def reference(spec, layers=None, dtype=None, **kw):
        return kw

    assert step_reference(_spec(reference), [0], None) == {"numerics": "native"}


def test_a_hook_without_the_switch_never_falls_back_to_its_rounded_arithmetic():
    """F128 (owner): the per-step target is the full-precision reference; a hook that cannot run it stops, unless the
    spec declares its only arithmetic free of roundings (reference.no_rounding)."""
    import pytest

    def reference(spec, layers=None, dtype=None):
        return "official-only"

    with pytest.raises(SystemExit, match="numerics"):
        step_reference(_spec(reference), [0], None)
    hooks = SimpleNamespace(reference=reference)
    exact = SimpleNamespace(hooks=lambda: hooks, get=lambda key, d=None: {"reference.no_rounding": True}.get(key, d))
    assert step_reference(exact, [0], None) == "official-only"
