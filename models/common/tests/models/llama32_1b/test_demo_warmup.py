# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.common.llm_runtime.config import TraceConfig

_DEMO_PATH = "models/common/tests/demos/llama32_1b/demo.py"
_DEMO_TREE = ast.parse(Path(_DEMO_PATH).read_text(encoding="utf-8"), filename=_DEMO_PATH)


def _demo_function(name):
    function = next(node for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {}
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
