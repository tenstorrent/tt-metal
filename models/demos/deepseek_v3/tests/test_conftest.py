# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

import pytest

from models.demos.deepseek_v3.conftest import _clear_requested_state_dict_cache, clear_state_dict_cache

_DEEPSEEK_TEST_ROOT = Path(__file__).parent / "fused_op_unit_tests"


def _entrypoint(relative_path, function_name):
    module = ast.parse((_DEEPSEEK_TEST_ROOT / relative_path).read_text())
    return next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == function_name)


def _decorator_name(decorator):
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    parts = []
    while isinstance(decorator, ast.Attribute):
        parts.append(decorator.attr)
        decorator = decorator.value
    if isinstance(decorator, ast.Name):
        parts.append(decorator.id)
    return ".".join(reversed(parts))


class _StateDict:
    def __init__(self):
        self.cache_clear_count = 0

    def clear_cache(self):
        self.cache_clear_count += 1


class _Request:
    def __init__(self, state_dict):
        self.fixturenames = []
        self.state_dict = state_dict
        self.request_count = 0

    def getfixturevalue(self, name):
        assert name == "state_dict"
        self.request_count += 1
        return self.state_dict


def test_state_dict_cache_cleanup_detects_dynamic_fixture_request():
    state_dict = _StateDict()
    request = _Request(state_dict)

    # Dynamic fixture requests become visible only after autouse setup has run.
    request.fixturenames.append("state_dict")
    _clear_requested_state_dict_cache(request)

    assert request.request_count == 1
    assert state_dict.cache_clear_count == 1


def test_state_dict_cache_cleanup_checks_dynamic_request_after_yield():
    state_dict = _StateDict()
    request = _Request(state_dict)
    fixture = clear_state_dict_cache.__wrapped__(request)

    assert next(fixture) is None
    assert request.request_count == 0

    request.fixturenames.append("state_dict")
    sentinel = object()
    assert next(fixture, sentinel) is sentinel
    assert request.request_count == 1
    assert state_dict.cache_clear_count == 1


def test_state_dict_cache_cleanup_keeps_random_tests_lazy():
    state_dict = _StateDict()
    request = _Request(state_dict)

    _clear_requested_state_dict_cache(request)

    assert request.request_count == 0
    assert state_dict.cache_clear_count == 0


def test_single_device_embedding_requests_real_weights_inside_the_real_weight_branch():
    entrypoint = _entrypoint("embedding/test_ds_embedding.py", "test_ds_embedding_single_device")
    argument_names = {argument.arg for argument in entrypoint.args.args}

    assert "state_dict" not in argument_names
    assert "request" in argument_names

    real_weight_branch = next(
        node for node in entrypoint.body if isinstance(node, ast.If) and ast.unparse(node.test) == "use_real_weights"
    )
    lazy_requests = [
        node
        for node in ast.walk(real_weight_branch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "getfixturevalue"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "state_dict"
    ]
    assert len(lazy_requests) == 1


@pytest.mark.parametrize(
    "relative_path,function_name",
    [
        ("lm_head/test_ds_lm_head.py", "test_ds_lm_head_single_device"),
        ("rms_norm/test_ds_rms_norm.py", "test_ds_rms_norm_single_device"),
    ],
)
def test_always_skipped_single_device_entrypoints_do_not_request_real_weights(
    relative_path,
    function_name,
):
    entrypoint = _entrypoint(relative_path, function_name)
    argument_names = {argument.arg for argument in entrypoint.args.args}
    decorator_names = {_decorator_name(decorator) for decorator in entrypoint.decorator_list}

    assert "state_dict" not in argument_names
    assert "pytest.mark.skip" in decorator_names
