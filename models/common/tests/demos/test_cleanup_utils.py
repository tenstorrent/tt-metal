# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.common.tests.demos import cleanup_utils


def test_cleanup_dp_model_case_orders_owners_before_parent_and_children(monkeypatch):
    calls = []
    parent = SimpleNamespace(quiesce_devices=lambda: calls.append("parent-quiesce"))
    submeshes = [object(), object()]
    group = SimpleNamespace(cleanup=lambda: calls.append("group-cleanup"))
    models = [("model-0", submeshes[0]), ("model-1", submeshes[1])]

    monkeypatch.setattr(
        cleanup_utils,
        "cleanup_model_case",
        lambda model, submesh: calls.append(("model-cleanup", model, submesh)),
    )
    monkeypatch.setattr(
        cleanup_utils.ttnn,
        "close_mesh_device",
        lambda submesh: calls.append(("child-close", submesh)),
    )

    cleanup_utils.cleanup_dp_model_case(group, [], models, parent, submeshes)

    assert calls == [
        "group-cleanup",
        ("model-cleanup", "model-0", submeshes[0]),
        ("model-cleanup", "model-1", submeshes[1]),
        "parent-quiesce",
        ("child-close", submeshes[0]),
        ("child-close", submeshes[1]),
    ]


def test_cleanup_dp_model_case_closes_every_carved_child_after_partial_build(monkeypatch):
    calls = []
    submeshes = [object(), object(), object(), object()]
    lanes = [SimpleNamespace(cleanup=lambda: calls.append("lane-cleanup"))]
    parent = SimpleNamespace(quiesce_devices=lambda: calls.append("parent-quiesce"))

    monkeypatch.setattr(
        cleanup_utils,
        "cleanup_model_case",
        lambda model, submesh: calls.append(("model-cleanup", model, submesh)),
    )
    monkeypatch.setattr(
        cleanup_utils.ttnn,
        "close_mesh_device",
        lambda submesh: calls.append(("child-close", submesh)),
    )

    cleanup_utils.cleanup_dp_model_case(None, lanes, [("model-0", submeshes[0])], parent, submeshes)

    assert calls[:3] == ["lane-cleanup", ("model-cleanup", "model-0", submeshes[0]), "parent-quiesce"]
    assert calls[3:] == [("child-close", submesh) for submesh in submeshes]


def test_cleanup_dp_model_case_attempts_all_children_after_cleanup_failure(monkeypatch, expect_error):
    closed = []
    submeshes = [object(), object(), object()]

    def fail_group_cleanup():
        raise RuntimeError("group cleanup failed")

    monkeypatch.setattr(cleanup_utils, "cleanup_model_case", lambda *_: None)
    monkeypatch.setattr(cleanup_utils.ttnn, "close_mesh_device", closed.append)

    with expect_error(RuntimeError, "group cleanup failed"):
        cleanup_utils.cleanup_dp_model_case(
            SimpleNamespace(cleanup=fail_group_cleanup),
            [],
            [],
            SimpleNamespace(quiesce_devices=lambda: None),
            submeshes,
        )

    assert closed == submeshes


def test_two_sequential_dp_profiles_release_children_between_carves(monkeypatch):
    class Child:
        def __init__(self, generation):
            self.generation = generation
            self.closed = False

    class Parent:
        def __init__(self):
            self.children = []
            self.quiesce_count = 0

        def quiesce_devices(self):
            self.quiesce_count += 1

        def carve(self, generation):
            assert all(child.closed for child in self.children)
            children = [Child(generation), Child(generation)]
            self.children.extend(children)
            return children

    parent = Parent()
    monkeypatch.setattr(cleanup_utils.ttnn, "close_mesh_device", lambda child: setattr(child, "closed", True))

    for generation in ("performance", "accuracy"):
        parent.quiesce_devices()
        children = parent.carve(generation)
        cleanup_utils.cleanup_dp_model_case(None, [], [], parent, children)

    assert parent.quiesce_count == 4
    assert all(child.closed for child in parent.children)


@pytest.mark.parametrize(
    ("demo_path", "create_name"),
    [
        ("models/common/tests/demos/qwen2_7b/demo.py", "_create_dp_submeshes"),
        ("models/common/tests/demos/qwen25_7b/demo.py", "_create_dp_submeshes"),
        ("models/common/tests/demos/deepseek_r1_distill_qwen_14b/demo.py", "_create_dp_submeshes"),
        ("models/common/tests/demos/llama32_1b/demo.py", "create_dp_submeshes"),
        ("models/common/tests/demos/llama32_3b/demo.py", "create_dp_submeshes"),
    ],
)
def test_dp_demos_quiesce_before_carving_and_use_shared_teardown(demo_path, create_name):
    tree = ast.parse(Path(demo_path).read_text(encoding="utf-8"), filename=demo_path)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_dp_smoke")
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    create_call = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == create_name)
    parent_quiesce = next(
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "mesh_device"
        and node.func.attr == "quiesce_devices"
    )
    teardown_call = next(
        node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "cleanup_dp_model_case"
    )

    assert parent_quiesce.lineno < create_call.lineno < teardown_call.lineno
    assert [ast.unparse(argument) for argument in teardown_call.args] == [
        "group",
        "lanes",
        "models",
        "mesh_device",
        "submeshes",
    ]
