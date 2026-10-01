# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.common.llm_runtime.config import TraceConfig

_DEMO_PATH = "models/common/tests/demos/llama32_3b/demo.py"
_DEMO_TREE = ast.parse(Path(_DEMO_PATH).read_text(encoding="utf-8"), filename=_DEMO_PATH)


def _demo_function(name, namespace=None):
    function = next(node for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {} if namespace is None else namespace
    exec(compile(ast.Module(body=[function], type_ignores=[]), _DEMO_PATH, "exec"), namespace)
    return namespace[name]


_warmup_demo_executor = _demo_function("_warmup_demo_executor")


@pytest.mark.parametrize("lane_group", [False, True])
@pytest.mark.parametrize(
    ("trace_mode", "expected_trace_calls"),
    [
        ("all", [("prefill", True), ("decode", True)]),
        ("decode_only", [("decode", True)]),
    ],
)
def test_demo_warmup_compiles_eager_programs_before_enabled_trace_capture(lane_group, trace_mode, expected_trace_calls):
    calls = []
    config = SimpleNamespace(trace=TraceConfig(trace_mode), device_sampling_enabled=True)

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

    eager_calls = [("decode", False), ("prefill", False)]
    assert [(kind, kwargs["enable_trace"]) for kind, kwargs in calls] == eager_calls + expected_trace_calls
    for _, kwargs in calls:
        assert kwargs["kv_cache"] is kv_cache
        assert kwargs["can_sample_on_device"] is True
    for kind, kwargs in calls:
        if kind == "decode":
            assert kwargs["max_batch_size"] == 4
            assert kwargs["num_blocks"] == 8


@pytest.mark.parametrize(
    ("num_devices", "traced", "expected_mode"),
    [(1, True, "decode_only"), (2, True, "all"), (8, True, "all"), (1, False, "none")],
)
def test_create_executor_preserves_3b_trace_device_matrix(num_devices, traced, expected_mode):
    captured = {}

    def executor_config(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    namespace = {
        "Llama32_3BTransformer1D": object,
        "Llama32_3BExecutor": lambda model, model_args, config: config,
        "Llama32_3BExecutorConfig": executor_config,
        "PagedKVCacheConfig": lambda **kwargs: SimpleNamespace(**kwargs),
        "TraceConfig": TraceConfig,
        "WarmupConfig": lambda: object(),
    }
    create_executor = _demo_function("create_executor", namespace)
    model = SimpleNamespace(
        model_args=object(),
        config=SimpleNamespace(
            max_seq_len=4096,
            max_batch_size=32,
            num_devices=num_devices,
            block_configs=[SimpleNamespace(attention_config=SimpleNamespace(kv_cache_dtype=object()))],
        ),
    )

    result = create_executor(model, traced=traced, device_sampling_enabled=True)

    assert result.trace.mode == expected_mode
    assert captured["device_sampling_enabled"] is True
    assert captured["paged_kv_cache"].num_blocks == 4096


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


def test_create_model_preserves_reduced_layer_diagnostic_override(monkeypatch):
    captured = {}
    model = SimpleNamespace()

    def from_pretrained(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(model=model, tokenizer=object())

    namespace = {
        "Path": Path,
        "Llama32_3BTransformer1D": object,
        "LLAMA32_3B_ACCURACY": object(),
        "LLAMA32_3B_PERFORMANCE": object(),
        "_skip_unless_heads_divide_mesh": lambda *_: None,
        "from_pretrained": from_pretrained,
        "os": os,
        "pytest": pytest,
        "ttnn": SimpleNamespace(MeshDevice=object),
    }
    create_model = _demo_function("create_model", namespace)
    monkeypatch.setenv("LLAMA32_3B_DEMO_NUM_LAYERS", "3")

    assert create_model(object(), "performance", Path("cache")) is model
    assert captured["n_layers"] == 3


def test_token_accuracy_cleans_up_executor_in_finally():
    function = next(
        node for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef) and node.name == "_run_token_accuracy"
    )
    cleanup_finally = [
        statement
        for node in ast.walk(function)
        if isinstance(node, ast.Try)
        for statement in node.finalbody
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and statement.value.func.attr == "cleanup"
    ]
    assert len(cleanup_finally) == 1
