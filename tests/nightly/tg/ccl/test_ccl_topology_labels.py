# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The galaxy (8x4) cases of ``tests/nightly/t3000/ccl/test_ccl_topology_labels.py``.

That module parametrises every test over the meshes it applies to (1x2, 1x8, 2x4, 8x4) and the ``mesh_device``
fixture skips the 8x4 cases on an 8-device box, so the T3K nightly (``pytest tests/nightly/t3000/ccl``) runs the first
three and nothing runs the fourth. The Galaxy CCL job runs this directory wholesale (``pytest tests/nightly/tg/ccl``),
so this module is the wiring: it re-collects every test of the t3000 module that carries an 8x4 parametrisation, with
its ``mesh_device`` parametrisation narrowed to ``(8, 4)`` and every other mark kept, and re-exports the t3000
module's own fixtures (``strict_ccl_topology`` / ``warn_only_ccl_topology``; pytest resolves a module-level fixture
from the module the test is collected in). Tests that apply to line meshes only are left out. Nothing is duplicated:
the cases, their docstrings and their negative controls live in the t3000 module; a test added there with ``(8, 4)``
among its meshes runs here too.
"""

import types

import pytest

from tests.nightly.t3000.ccl import test_ccl_topology_labels as _t3000

GALAXY_MESH = (8, 4)


def _is_fixture(obj):
    # pytest < 8.4 marks the decorated function; 8.4+ wraps it in a FixtureFunctionDefinition.
    return hasattr(obj, "_pytestfixturefunction") or type(obj).__name__ == "FixtureFunctionDefinition"


def _is_mesh_parametrisation(mark):
    return mark.name == "parametrize" and mark.args and mark.args[0] == "mesh_device"


def _galaxy_variant(test):
    """A copy of `test` whose `mesh_device` parametrisation is the galaxy mesh alone, or None when the test does not
    apply to it (its meshes are lines only)."""
    marks = list(getattr(test, "pytestmark", []))
    mesh_marks = [mark for mark in marks if _is_mesh_parametrisation(mark)]
    if not mesh_marks or not any(GALAXY_MESH in mark.args[1] for mark in mesh_marks):
        return None
    variant = types.FunctionType(test.__code__, test.__globals__, test.__name__, test.__defaults__, test.__closure__)
    variant.__dict__.update({key: value for key, value in test.__dict__.items() if key != "pytestmark"})
    variant.__doc__ = test.__doc__
    variant.__module__ = __name__
    variant.pytestmark = [mark for mark in marks if not _is_mesh_parametrisation(mark)] + [
        pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True, ids=["8x4"]).mark
    ]
    return variant


for _name in dir(_t3000):
    _obj = getattr(_t3000, _name)
    if _is_fixture(_obj):
        globals()[_name] = _obj
    elif _name.startswith("test_"):
        _variant = _galaxy_variant(_obj)
        if _variant is not None:
            globals()[_name] = _variant
del _name, _obj
