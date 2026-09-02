# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.common.llm_runtime.config import TraceConfig

_DEMO_PATH = "models/common/tests/demos/llama32_1b/demo.py"
_DEMO_TREE = ast.parse(Path(_DEMO_PATH).read_text(encoding="utf-8"), filename=_DEMO_PATH)


def _demo_function(name, namespace=None):
    function = next(node for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {} if namespace is None else namespace
    exec(compile(ast.Module(body=[function], type_ignores=[]), _DEMO_PATH, "exec"), namespace)
    return namespace[name]


_warmup_demo_executor = _demo_function("_warmup_demo_executor")


@pytest.mark.parametrize("lane_group", [False, True])
def test_demo_warmup_compiles_eager_programs_before_trace_capture(lane_group):
    calls = []
    config = SimpleNamespace(trace=TraceConfig("all"), device_sampling_enabled=True)

    def warmup_prefill(**kwargs):
        calls.append(("prefill", kwargs))

    def warmup_decode(**kwargs):
        calls.append(("decode", kwargs))

    executor = SimpleNamespace(
        warmup_model_prefill=warmup_prefill,
        warmup_model_decode=warmup_decode,
        max_batch_size=4,
    )
    if lane_group:
        executor.lanes = [SimpleNamespace(config=config)]
    else:
        executor.config = config
        executor.model = SimpleNamespace(config=SimpleNamespace(max_batch_size=4))

    kv_cache = object()
    page_table = SimpleNamespace(shape=(4, 8))
    _warmup_demo_executor(executor, kv_cache=kv_cache, page_table=page_table)

    assert [(kind, kwargs["enable_trace"]) for kind, kwargs in calls] == [
        ("decode", False),
        ("prefill", False),
        ("prefill", True),
        ("decode", True),
    ]
    for _, kwargs in calls:
        assert kwargs["kv_cache"] is kv_cache
        assert kwargs["can_sample_on_device"] is True
    for kind, kwargs in calls:
        if kind == "decode":
            assert kwargs["max_batch_size"] == 4
            assert kwargs["num_blocks"] == 8


def _called_names(function_name):
    function = next(
        node for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    return [
        node.func.id for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]


@pytest.mark.parametrize(("data_parallel", "expected_tp_devices"), [(4, 2), (8, 1)])
def test_t3k_dp_topology_preserves_supported_tp_lanes(data_parallel, expected_tp_devices):
    helper = _demo_function("_dp_tp_devices_or_skip", {"pytest": pytest, "ttnn": SimpleNamespace(MeshDevice=object)})
    mesh = SimpleNamespace(get_num_devices=lambda: 8)

    assert helper(mesh, data_parallel) == expected_tp_devices


def test_t3k_dp2_skips_unsupported_tp4_lanes(expect_error):
    helper = _demo_function("_dp_tp_devices_or_skip", {"pytest": pytest, "ttnn": SimpleNamespace(MeshDevice=object)})
    mesh = SimpleNamespace(get_num_devices=lambda: 8)

    with expect_error(pytest.skip.Exception, "creates TP4 lanes"):
        helper(mesh, 2)


def test_dp_build_validates_and_resolves_cache_from_each_lane_submesh():
    function = next(
        node for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef) and node.name == "_run_dp_smoke"
    )
    lane_loop = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "sm"
    )
    calls = [node for node in ast.walk(lane_loop) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    call_names = [node.func.id for node in calls]
    assert "_skip_unless_heads_divide_mesh" in call_names
    assert "lazy_weight_cache_dir_for_demo" in call_names

    from_pretrained_call = next(node for node in calls if node.func.id == "from_pretrained")
    cache_dir = next(keyword.value for keyword in from_pretrained_call.keywords if keyword.arg == "cache_dir")
    assert isinstance(cache_dir, ast.Name)
    assert cache_dir.id == "lane_cache_dir"


@pytest.mark.parametrize("function_name", ["_run_perf_benchmark", "_run_dp_smoke"])
def test_traced_demo_paths_warm_up_before_benchmark(function_name):
    calls = _called_names(function_name)
    assert calls.index("_warmup_demo_executor") < calls.index("run_perf_benchmark")


def test_eval_repeat_warms_each_fresh_executor():
    calls = _called_names("_run_eval_repeat_batch32")
    assert "_warmup_demo_executor" in calls


def test_perf_path_enables_pipeline_readback_by_default():
    function = next(
        node for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef) and node.name == "_run_perf_benchmark"
    )
    benchmark_call = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_perf_benchmark"
    )
    keywords = {keyword.arg: keyword.value for keyword in benchmark_call.keywords}
    assert isinstance(keywords["pipeline_readback"], ast.Name)
    assert keywords["pipeline_readback"].id == "pipeline_readback"
