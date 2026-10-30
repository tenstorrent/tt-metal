"""Every emitted PCC test must define the module-level helpers it calls.

Regression guard for the `NameError` that killed *every* generated
`tests/pcc/test_<comp>.py` before a single PCC comparison could run:
`_PCC_TEST_TEMPLATE` called `_captured_submodule_path(COMPONENT_NAME)` inside
`_build_torch_reference` but never emitted its definition.

This was previously worked around per-model, by hand-pasting the definition into
each generated test file, so it silently returned on every new model. It cannot
be fixed from a `conftest.py`: the sharded test imports its sibling by path, so
conftest injection never reaches it. The definition has to be IN the file.

The undefined-name scan below is deliberately generic -- it fails on ANY
module-level helper the template calls without defining, not just the one name
that caused this outage.
"""

import ast
import builtins
from pathlib import Path

import pytest

from scripts.tt_hw_planner.bringup_loop import _emit_pcc_template


@pytest.fixture()
def emitted(tmp_path: Path) -> str:
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    test_path, created, existed = _emit_pcc_template(
        demo_dir=demo_dir,
        component_name="some_component",
        model_id="org/model",
        hf_reference="org/model",
        new_shape={"batch": 1, "seq": 16, "hidden": 64},
        repo_root=tmp_path,
    )
    assert created and not existed
    return test_path.read_text()


def test_emitted_test_is_syntactically_valid(emitted: str):
    compile(emitted, "test_some_component.py", "exec")


def test_emitted_test_defines_captured_submodule_path(emitted: str):
    """The specific regression: called AND defined, in the same file."""
    assert "_captured_submodule_path(COMPONENT_NAME)" in emitted
    assert "def _captured_submodule_path(" in emitted


def test_emitted_helper_is_the_shared_constant_not_a_copy(emitted: str):
    """The inlined helpers must come from `CAPTURE_LOADER_SOURCE`, not a re-type.

    `capture_inputs` both WRITES the manifest and owns the source that reads it
    back. A second, hand-copied implementation would be free to drift from it.
    """
    from scripts.tt_hw_planner.capture_inputs import CAPTURE_LOADER_SOURCE

    assert CAPTURE_LOADER_SOURCE.strip() in emitted


def test_injection_is_idempotent(emitted: str):
    """Emit-time and upgrade-time injection must not double-define."""
    from scripts.tt_hw_planner.capture_inputs import inject_capture_loader

    assert emitted.count("def _captured_submodule_path(") == 1
    once = inject_capture_loader(emitted)
    assert once == emitted
    assert once.count("def _captured_submodule_path(") == 1


def _module_level_called_names(tree: ast.Module) -> set:
    return {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}


def _bound_names(tree: ast.Module) -> set:
    bound = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return bound


def test_emitted_test_calls_no_undefined_helper(emitted: str):
    """Generic guard: no call to a name the emitted file never binds."""
    tree = ast.parse(emitted)
    undefined = _module_level_called_names(tree) - _bound_names(tree)
    assert not undefined, f"emitted PCC test calls undefined name(s): {sorted(undefined)}"
