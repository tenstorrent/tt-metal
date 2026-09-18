# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.common.llm_runtime.config import TraceConfig

_DEMO_PATH = "models/common/tests/demos/mistral_7b/demo.py"
_DEMO_TREE = ast.parse(Path(_DEMO_PATH).read_text(encoding="utf-8"), filename=_DEMO_PATH)


def _demo_function(name, namespace=None):
    function = next(node for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {} if namespace is None else namespace
    exec(compile(ast.Module(body=[function], type_ignores=[]), _DEMO_PATH, "exec"), namespace)
    return namespace[name]


def _called_names(function_name):
    function = next(
        node for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    return [
        node.func.id for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]


def test_demo_case_manifest_and_optimization_profiles_are_preserved():
    test_function = next(
        node for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef) and node.name == "test_mistral_7b"
    )
    decorators = [node for node in test_function.decorator_list if isinstance(node, ast.Call)]
    test_config = next(node for node in decorators if ast.literal_eval(node.args[0]) == "test_config")
    optimizations = next(node for node in decorators if ast.literal_eval(node.args[0]) == "optimizations")
    case_ids = [ast.literal_eval(element.keywords[0].value) for element in test_config.args[1].elts]
    assert case_ids == [
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


@pytest.mark.parametrize(
    "devices,data_parallel,skips",
    [
        (1, 2, True),
        (2, 2, False),
        (2, 8, True),
        (8, 2, True),
        (8, 4, True),
        (8, 8, False),
        (8, 16, True),
    ],
)
def test_dp_manifest_runs_only_single_device_lanes(expect_error, devices, data_parallel, skips):
    check = _demo_function("_dp_or_skip", {"pytest": pytest, "ttnn": SimpleNamespace(MeshDevice=object)})
    mesh = SimpleNamespace(get_num_devices=lambda: devices)
    if skips:
        with expect_error(pytest.skip.Exception, "single-device groups"):
            check(mesh, data_parallel)
    else:
        check(mesh, data_parallel)


def test_demo_reserves_trace_space_by_mesh(monkeypatch):
    fabric_1d = object()
    mesh_shapes = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}
    resolve = _demo_function(
        "_ttnn_mesh_device_param_from_env",
        {
            "os": os,
            "pytest": pytest,
            "_MESH_DEVICE_TO_SHAPE": mesh_shapes,
            "ttnn": SimpleNamespace(FabricConfig=SimpleNamespace(FABRIC_1D=fabric_1d)),
        },
    )

    for mesh_name, expected_trace_region_size in (("N150", 50_000_000), ("N300", 50_000_000), ("T3K", 100_000_000)):
        monkeypatch.setenv("MESH_DEVICE", mesh_name)
        param = resolve()
        assert param["mesh_shape"] == mesh_shapes[mesh_name]
        assert param["trace_region_size"] == expected_trace_region_size


def test_demo_imports_promoted_runner_helpers_and_model_owned_executor():
    imported = {
        (node.module, alias.name)
        for node in _DEMO_TREE.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    for helper in (
        "load_eval_repeat_prompts_batch32",
        "make_contiguous_page_table",
        "run_eval_repeat_batch32",
        "run_perf_benchmark",
        "run_teacher_forcing",
    ):
        assert ("models.common.tests.demos.run_helpers", helper) in imported
    assert ("models.common.models.mistral_7b.executor", "Mistral7BExecutor") in imported
    assert not any(module == "models.common.models.executor" for module, _ in imported)


def test_demo_warmup_compiles_eager_programs_before_trace_capture():
    calls = []
    config = SimpleNamespace(trace=TraceConfig("all"), device_sampling_enabled=True)
    executor = SimpleNamespace(
        config=config,
        model=SimpleNamespace(config=SimpleNamespace(max_batch_size=8)),
        warmup_model_prefill=lambda **kwargs: calls.append(("prefill", kwargs)),
        warmup_model_decode=lambda **kwargs: calls.append(("decode", kwargs)),
    )
    warmup = _demo_function("_warmup_demo_executor")
    kv_cache = object()
    warmup(executor, kv_cache=kv_cache, page_table=SimpleNamespace(shape=(8, 32)))

    assert [(kind, kwargs["enable_trace"]) for kind, kwargs in calls] == [
        ("decode", False),
        ("prefill", False),
        ("prefill", True),
        ("decode", True),
    ]
    assert all(kwargs["kv_cache"] is kv_cache for _, kwargs in calls)


def test_demo_warmup_registers_representative_prefill_before_trace_capture():
    calls = []
    eager_execution = object()
    executor = SimpleNamespace(
        config=SimpleNamespace(trace=TraceConfig("all"), device_sampling_enabled=False),
        eager_execution=eager_execution,
        model=SimpleNamespace(config=SimpleNamespace(max_batch_size=32)),
        warmup_model_prefill=lambda **kwargs: calls.append(("prefill", kwargs)),
        warmup_model_decode=lambda **kwargs: calls.append(("decode", kwargs)),
        compile_prefill=lambda **kwargs: calls.append(("compile_prefill", kwargs)),
    )
    tokens = torch.zeros((32, 700), dtype=torch.long)
    prompt_lens = torch.tensor([64] * 30 + [400, 700])
    page_table = torch.zeros((32, 64), dtype=torch.int32)
    kv_cache = object()

    _demo_function("_warmup_demo_executor")(
        executor,
        kv_cache=kv_cache,
        page_table=page_table,
        prefill_compile_case=(tokens, prompt_lens),
    )

    assert [kind for kind, _ in calls] == ["decode", "prefill", "compile_prefill", "prefill", "decode"]
    compile_kwargs = calls[2][1]
    assert compile_kwargs["tokens"] is tokens
    assert compile_kwargs["prompt_lens"] is prompt_lens
    assert compile_kwargs["execution"] is eager_execution


@pytest.mark.parametrize("function_name", ["_run_perf_benchmark", "_run_eval_repeat_batch32"])
def test_traced_demo_paths_warm_up_fresh_executor(function_name):
    assert "_warmup_demo_executor" in _called_names(function_name)


def test_dp_warmup_compiles_the_tokenized_prefill_signature_before_trace_capture():
    function = next(
        node for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef) and node.name == "_run_dp_smoke"
    )
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    tokenization = next(node for node in calls if node.func.id == "tokenize_prompts")
    warmup = next(node for node in calls if node.func.id == "_warmup_demo_executor")
    assert tokenization.lineno < warmup.lineno

    keywords = {keyword.arg: keyword.value for keyword in warmup.keywords}
    compile_case = keywords["prefill_compile_case"]
    assert isinstance(compile_case, ast.Tuple)
    assert [element.id for element in compile_case.elts] == ["input_tokens", "prompt_lens"]
    assert isinstance(keywords["prefill_sampling_params"], ast.Name)
    assert keywords["prefill_sampling_params"].id == "sampling_params"
    assert isinstance(keywords["prefill_compile_execution"], ast.Attribute)
    assert keywords["prefill_compile_execution"].attr == "traced_prefill_execution"


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


def test_strict_special_token_guard_delegates_after_eos_truncation():
    captured = {}

    def shared(outputs, tokenizer, **kwargs):
        captured.update(outputs=outputs, tokenizer=tokenizer, kwargs=kwargs)

    guard = _demo_function("assert_no_special_tokens", {"assert_no_special_tokens_shared": shared})
    tokenizer = SimpleNamespace(eos_token_id=2)
    guard([[10, 2, 99], [20]], tokenizer, case_name="case", is_ci_env=True)

    assert captured["outputs"] == [[10], [20]]
    assert captured["kwargs"] == {"case_name": "case", "is_ci_env": True}


def test_create_executor_uses_model_owned_runtime_and_resolved_cache():
    captured = {}

    def executor_config(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    create_executor = _demo_function(
        "create_executor",
        {
            "Mistral7B": object,
            "Mistral7BExecutor": lambda model, runtime_config, config: config,
            "Mistral7BExecutorConfig": executor_config,
            "PagedKVCacheConfig": lambda **kwargs: SimpleNamespace(**kwargs),
            "TraceConfig": TraceConfig,
            "WarmupConfig": lambda: object(),
        },
    )
    model = SimpleNamespace(
        model_args=object(),
        config=SimpleNamespace(
            max_seq_len=2048,
            max_batch_size=8,
            block_configs=[SimpleNamespace(attention_config=SimpleNamespace(kv_cache_dtype=object()))],
        ),
    )

    result = create_executor(model, traced=True, device_sampling_enabled=True)

    assert result.trace.mode == "all"
    assert result.device_sampling_enabled is True
    assert captured["paged_kv_cache"].num_blocks == 512
