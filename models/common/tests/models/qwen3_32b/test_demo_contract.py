# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.common.models.qwen3_32b import executor as qwen3_executor
from models.demos.utils.model_targets import resolve_metric_tolerance
from models.demos.utils.trace_region_sizes import resolve_trace_region_size

_DEMO_PATH = "models/common/tests/demos/qwen3_32b/demo.py"
_DEMO_SOURCE = Path(_DEMO_PATH).read_text(encoding="utf-8")
_DEMO_TREE = ast.parse(_DEMO_SOURCE, filename=_DEMO_PATH)
_COMMON_CONFTEST_SOURCE = Path("models/common/tests/conftest.py").read_text(encoding="utf-8")


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
        "eval-32-perf-report",
        "ci-b1-DP-2",
        "ci-b1-DP-4",
        "ci-b1-DP-8",
        "ci-b1-DP-16",
        "ci-b1-DP-32",
    ]
    assert ast.literal_eval(optimizations.args[1]) == ["performance", "accuracy"]


def test_demo_resolves_qwen3_trace_region_and_matches_ring_fabric():
    assert 'resolve_trace_region_size("qwen3-32b", env)' in _DEMO_SOURCE
    assert '"trace_region_size": 50_000_000' not in _DEMO_SOURCE
    assert "ttnn.FabricConfig.FABRIC_1D_RING" in _DEMO_SOURCE
    assert resolve_trace_region_size("qwen3-32b", "T3K") == 90_000_000
    assert resolve_trace_region_size("qwen3-32b", "P150x4") == 90_000_000


def test_demo_exposes_p150x4_and_uses_canonical_device_naming():
    assert '"P150x4": (1, 4)' in _DEMO_SOURCE
    assert "bh_hardware" not in _DEMO_SOURCE
    assert not any(isinstance(node, ast.FunctionDef) and node.name == "get_device_name" for node in _DEMO_TREE.body)
    imports = [ast.unparse(node) for node in _DEMO_TREE.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert any("models.common.device_utils import get_device_name" in statement for statement in imports)


def test_required_bh_gate_failures_are_not_converted_to_fixture_skips():
    assert 'mesh_device_name in {"P150", "P150X4"}' in _COMMON_CONFTEST_SOURCE
    assert "if blackhole_selected:\n                raise" in _COMMON_CONFTEST_SOURCE


def test_demo_uses_model_owned_runtime_compatibility_wrappers():
    imports = [ast.unparse(node) for node in _DEMO_TREE.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert any("models.common.models.qwen3_32b.executor" in statement for statement in imports)
    assert "EagerQwen3_32BExecutor" in _DEMO_SOURCE
    assert "TracedQwen3_32BExecutor" in _DEMO_SOURCE
    assert "Qwen3_32B.from_pretrained" in _DEMO_SOURCE


@pytest.mark.parametrize("data_parallel", [2, 4, 8, 16, 32])
def test_every_dp_case_skips_before_submesh_or_model_construction(data_parallel, expect_error):
    namespace = {"pytest": pytest, "ttnn": SimpleNamespace(MeshDevice=object), "_MIN_TP_DEVICES": 4}
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


def test_eval_perf_report_reuses_three_repeat_geometry_and_first_repeat_profiler():
    source = ast.unparse(_function("_run_eval_repeat_batch32"))
    assert "repeat_batches=_EVAL_REPEAT_BATCHES" in source
    assert "first_repeat_profiler=profiler" in source
    assert "_assert_eval32_perf_target(first_result, expected" in source
    assert "'on_device_topk' if perf_report else 'host'" in source
    assert "eval-32-perf-report" in ast.unparse(_function("test_qwen3_32b"))


def test_eval_perf_targets_fail_closed_when_missing_or_failed(expect_error):
    resolve_namespace = {
        "resolve_perf_targets": lambda *args, **kwargs: None,
        "_EVAL32_TARGET_SEQ_LEN": 686,
    }
    resolve_function = _function("_resolve_eval32_perf_targets")
    exec(compile(ast.Module(body=[resolve_function], type_ignores=[]), _DEMO_PATH, "exec"), resolve_namespace)
    with expect_error(ValueError, "qualification gates fail closed"):
        resolve_namespace["_resolve_eval32_perf_targets"]("Qwen/Qwen3-32B", "P150x4")

    assert_namespace = {
        "resolve_metric_tolerance": resolve_metric_tolerance,
        "PERF_TOLERANCE": 0.05,
    }
    assert_function = _function("_assert_eval32_perf_target")
    exec(compile(ast.Module(body=[assert_function], type_ignores=[]), _DEMO_PATH, "exec"), assert_namespace)
    result = SimpleNamespace(tok_s_u=1.0, ttft_ms=1_000.0)
    expected = {"decode_t/s/u": 21.6, "prefill_time_to_first_token": 87}
    with expect_error(AssertionError, "tok/s/u.*ttft_ms"):
        assert_namespace["_assert_eval32_perf_target"](result, expected, case_name="BH/eval")


def test_other_declared_p150x4_perf_nodes_fail_closed_on_missing_targets(expect_error):
    namespace = {}
    function = _function("_require_p150x4_local_perf_target")
    exec(compile(ast.Module(body=[function], type_ignores=[]), _DEMO_PATH, "exec"), namespace)
    namespace["_require_p150x4_local_perf_target"]("T3K", {}, case_name="WH")
    with expect_error(ValueError, "missing frozen P150x4 perf target"):
        namespace["_require_p150x4_local_perf_target"]("P150x4", {}, case_name="BH/batch-32-ci")


def test_traced_compatibility_wrapper_is_accepted_by_transition_perf_helper(monkeypatch):
    def fake_init(self, model, runtime_config, config):
        self.model = model
        self.runtime_config = runtime_config
        self.config = config

    monkeypatch.setattr(qwen3_executor.Qwen3_32BExecutor, "__init__", fake_init)

    model = SimpleNamespace(model_args=SimpleNamespace(), config=SimpleNamespace(max_seq_len=4096, max_batch_size=32))
    traced = qwen3_executor.TracedQwen3_32BExecutor(model, mesh_device=object(), ondevice_decode_loop=True)

    assert _has_trace_surface(traced)
