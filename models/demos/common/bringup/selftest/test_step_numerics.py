# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F112: per-step checks ask the model's reference for its native numerics when its hook takes `numerics`."""

from types import SimpleNamespace

from models.demos.common.bringup.reference.interface import step_reference


def _spec(reference, numerics="native"):
    hooks = SimpleNamespace(reference=reference)
    return SimpleNamespace(hooks=lambda: hooks, get=lambda key: numerics if key == "tests.step_numerics" else None)


def test_hook_with_numerics_gets_the_setting():
    def reference(spec, layers=None, dtype=None, numerics="official"):
        return (layers, dtype, numerics)

    assert step_reference(_spec(reference), [3], "f32") == ([3], "f32", "native")
    assert step_reference(_spec(reference, "official"), [3], "f32")[2] == "official"


def test_hook_with_kwargs_gets_the_setting():
    def reference(spec, layers=None, dtype=None, **kw):
        return kw

    assert step_reference(_spec(reference), [0], None) == {"numerics": "native"}


def test_old_hook_keeps_its_only_arithmetic():
    def reference(spec, layers=None, dtype=None):
        return "official-only"

    assert step_reference(_spec(reference), [0], None) == "official-only"
