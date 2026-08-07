# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.common.models.qwen3_32b import executor as qwen3_executor

_DEMO_PATH = "models/common/tests/demos/qwen3_32b/demo.py"
_DEMO_SOURCE = Path(_DEMO_PATH).read_text(encoding="utf-8")
_DEMO_TREE = ast.parse(_DEMO_SOURCE, filename=_DEMO_PATH)


def _function(name):
    return next(node for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _calls(function_name, called_name):
    return [
        node
        for node in ast.walk(_function(function_name))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == called_name
    ]


def _has_trace_surface(executor) -> bool:
    return hasattr(executor, "trace_id_prefill") and hasattr(executor, "trace_ids_decode")


def test_demo_case_manifest_is_preserved():
    decorators = [node for node in _function("test_qwen3_32b").decorator_list if isinstance(node, ast.Call)]
    test_config = next(node for node in decorators if ast.literal_eval(node.args[0]) == "test_config")
    optimizations = next(node for node in decorators if ast.literal_eval(node.args[0]) == "optimizations")
    assert [ast.literal_eval(element.keywords[0].value) for element in test_config.args[1].elts] == [
        "token-accuracy",
        "batch-1",
        "batch-32",
        "batch-32-ci",
        "eval-32",
        "ci-b1-DP-2",
        "ci-b1-DP-4",
        "ci-b1-DP-8",
        "ci-b1-DP-16",
        "ci-b1-DP-32",
    ]
    assert ast.literal_eval(optimizations.args[1]) == ["performance", "accuracy"]


def test_demo_keeps_qwen3_trace_region_and_fabric():
    assert '"trace_region_size": 50_000_000' in _DEMO_SOURCE
    assert "ttnn.FabricConfig.FABRIC_1D" in _DEMO_SOURCE


def test_demo_uses_model_owned_runtime_compatibility_wrappers():
    imports = [ast.unparse(node) for node in _DEMO_TREE.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert any("models.common.models.qwen3_32b.executor" in statement for statement in imports)
    assert "EagerQwen3_32BExecutor" in _DEMO_SOURCE
    assert "TracedQwen3_32BExecutor" in _DEMO_SOURCE
    assert "Qwen3_32B.from_pretrained" in _DEMO_SOURCE


@pytest.mark.parametrize("data_parallel", [2, 4, 8, 16, 32])
def test_every_dp_case_skips_before_submesh_or_model_construction(data_parallel, expect_error):
    namespace = {"pytest": pytest, "ttnn": SimpleNamespace(MeshDevice=object), "_MIN_TP_DEVICES": 8}
    function = _function("_dp_or_skip")
    exec(compile(ast.Module(body=[function], type_ignores=[]), _DEMO_PATH, "exec"), namespace)
    mesh = SimpleNamespace(get_num_devices=lambda: 8)
    if data_parallel == 8:
        namespace["_dp_or_skip"](mesh, data_parallel)
    else:
        with expect_error(pytest.skip.Exception, f"DP-{data_parallel}"):
            namespace["_dp_or_skip"](mesh, data_parallel)
    run_dp = _function("_run_dp_smoke")
    calls = [
        node.func.id for node in ast.walk(run_dp) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls[0] == "_dp_or_skip"
    assert "create_dp_submeshes" not in calls[: calls.index("_skip_below_min_tp_devices")]


def test_demo_allocates_kv_cache_with_vllm_shape_compatibility_arguments():
    for function_name in ("_run_token_accuracy", "_run_perf_benchmark", "_run_eval_repeat_batch32"):
        allocations = [
            node
            for node in ast.walk(_function(function_name))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "allocate_kv_cache"
        ]
        assert allocations
        assert all(call.args or call.keywords for call in allocations)


def test_perf_and_eval_use_traced_model_owned_wrapper():
    assert _calls("_run_perf_benchmark", "TracedQwen3_32BExecutor")
    assert _calls("_run_eval_repeat_batch32", "TracedQwen3_32BExecutor")


def test_eval_repeat_threads_sampling_mode_to_traced_executor():
    call = _calls("_run_eval_repeat_batch32", "TracedQwen3_32BExecutor")[0]
    ondevice_decode_loop = next(keyword for keyword in call.keywords if keyword.arg == "ondevice_decode_loop")

    assert ast.unparse(ondevice_decode_loop.value) == "sampling_params is not None"


def test_eval_repeat_uses_decode_only_trace_mode():
    call = _calls("_run_eval_repeat_batch32", "TracedQwen3_32BExecutor")[0]
    trace_mode = next(keyword for keyword in call.keywords if keyword.arg == "trace_mode")

    assert ast.literal_eval(trace_mode.value) == "decode_only"


def test_eval_repeat_warms_executor_before_shared_perf_runner_replay():
    assert _calls("_run_eval_repeat_batch32", "_warmup_demo_executor")

    helper_source = ast.unparse(_function("_warmup_demo_executor"))
    assert "executor.warmup_model_decode" in helper_source
    assert "executor.warmup_model_prefill" in helper_source


def test_traced_compatibility_wrapper_is_accepted_by_transition_perf_helper(monkeypatch):
    def fake_init(self, model, runtime_config, config):
        self.model = model
        self.runtime_config = runtime_config
        self.config = config

    monkeypatch.setattr(qwen3_executor.Qwen3_32BExecutor, "__init__", fake_init)

    model = SimpleNamespace(model_args=SimpleNamespace(), config=SimpleNamespace(max_seq_len=4096, max_batch_size=32))
    traced = qwen3_executor.TracedQwen3_32BExecutor(model, mesh_device=object(), ondevice_decode_loop=True)

    assert _has_trace_surface(traced)
