# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.common.llm_runtime.config import TraceConfig
from models.common.llm_runtime.prefill.plan import _plan_prefill_requests

_DEMO_PATH = "models/common/tests/demos/deepseek_r1_distill_qwen_14b/demo.py"
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


def test_demo_case_manifest_is_preserved():
    test_function = next(
        node
        for node in _DEMO_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "test_deepseek_r1_qwen_14b"
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


def test_demo_reserves_trace_space_by_mesh(monkeypatch):
    for mesh_name, mesh_shape, trace_region_size in (
        ("N300", (1, 2), 50_000_000),
        ("T3K", (1, 8), 100_000_000),
    ):
        monkeypatch.setenv("MESH_DEVICE", mesh_name)
        device_params = _demo_function(
            "_ttnn_mesh_device_param_from_env",
            {
                "os": os,
                "pytest": pytest,
                "_MESH_DEVICE_TO_SHAPE": {mesh_name: mesh_shape},
                "ttnn": SimpleNamespace(FabricConfig=SimpleNamespace(FABRIC_1D=object())),
            },
        )()

        assert device_params["mesh_shape"] == mesh_shape
        assert device_params["trace_region_size"] == trace_region_size


def test_demo_warmup_compiles_eager_programs_before_trace_capture():
    calls = []
    config = SimpleNamespace(trace=TraceConfig("all"), device_sampling_enabled=True)
    executor = SimpleNamespace(
        config=config,
        model=SimpleNamespace(config=SimpleNamespace(max_batch_size=4)),
        warmup_model_prefill=lambda **kwargs: calls.append(("prefill", kwargs)),
        warmup_model_decode=lambda **kwargs: calls.append(("decode", kwargs)),
    )
    warmup = _demo_function("_warmup_demo_executor")
    kv_cache = object()
    warmup(executor, kv_cache=kv_cache, page_table=SimpleNamespace(shape=(4, 8)))

    assert [(kind, kwargs["enable_trace"]) for kind, kwargs in calls] == [
        ("decode", False),
        ("prefill", False),
        ("prefill", True),
        ("decode", True),
    ]
    assert all(kwargs["kv_cache"] is kv_cache for _, kwargs in calls)


def test_demo_warmup_registers_concrete_prefill_before_trace_capture():
    calls = []
    config = SimpleNamespace(trace=TraceConfig("all"), device_sampling_enabled=False)
    eager_execution = object()
    executor = SimpleNamespace(
        config=config,
        eager_execution=eager_execution,
        model=SimpleNamespace(config=SimpleNamespace(max_batch_size=32)),
        warmup_model_prefill=lambda **kwargs: calls.append(("prefill", kwargs)),
        warmup_model_decode=lambda **kwargs: calls.append(("decode", kwargs)),
        compile_prefill=lambda **kwargs: calls.append(("compile_prefill", kwargs)),
    )
    warmup = _demo_function("_warmup_demo_executor")
    tokens = torch.zeros((32, 700), dtype=torch.long)
    prompt_lens = torch.tensor([64] * 30 + [400, 700])
    page_table = torch.zeros((32, 64), dtype=torch.int32)
    kv_cache = object()

    warmup(
        executor,
        kv_cache=kv_cache,
        page_table=page_table,
        prefill_compile_case=(tokens, prompt_lens),
    )

    assert [(kind, kwargs.get("enable_trace")) for kind, kwargs in calls] == [
        ("decode", False),
        ("prefill", False),
        ("compile_prefill", None),
        ("prefill", True),
        ("decode", True),
    ]
    compile_kwargs = calls[2][1]
    assert compile_kwargs["tokens"] is tokens
    assert compile_kwargs["prompt_lens"] is prompt_lens
    assert compile_kwargs["page_table"] is page_table
    assert compile_kwargs["kv_cache"] is kv_cache
    assert compile_kwargs["empty_slots"] == list(range(32))
    assert compile_kwargs["execution"] is eager_execution


def test_demo_warmup_uses_lane_group_capacity_and_lane_trace_policy():
    calls = []
    lane_config = SimpleNamespace(trace=TraceConfig("all"), device_sampling_enabled=True)
    group = SimpleNamespace(
        lanes=[SimpleNamespace(config=lane_config) for _ in range(4)],
        max_batch_size=4,
        warmup_model_prefill=lambda **kwargs: calls.append(("prefill", kwargs)),
        warmup_model_decode=lambda **kwargs: calls.append(("decode", kwargs)),
    )
    warmup = _demo_function("_warmup_demo_executor")
    kv_cache = [object() for _ in range(4)]
    warmup(group, kv_cache=kv_cache, page_table=SimpleNamespace(shape=(4, 128)))

    decode_calls = [kwargs for kind, kwargs in calls if kind == "decode"]
    assert len(decode_calls) == 2
    assert all(kwargs["max_batch_size"] == 4 for kwargs in decode_calls)
    assert all(kwargs["num_blocks"] == 128 for kwargs in decode_calls)
    assert all(kwargs["kv_cache"] is kv_cache for _, kwargs in calls)


def test_eval_prefill_signature_multiset_is_rotation_invariant_and_not_static_warmup_shaped():
    tokens = torch.zeros((32, 700), dtype=torch.long)
    prompt_lens = torch.tensor([64] * 30 + [400, 700])
    page_table = torch.zeros((32, 64), dtype=torch.int32)

    def planned_shapes(offset):
        rotated_tokens = torch.roll(tokens, shifts=-offset, dims=0)
        rotated_lens = torch.roll(prompt_lens, shifts=-offset, dims=0)
        requests = _plan_prefill_requests(
            tokens=rotated_tokens,
            page_table=page_table,
            prompt_lens=rotated_lens,
            empty_slots=list(range(32)),
            start_pos=None,
            block_size=32,
            max_batch_size=32,
            max_prefill_chunk_size=1024,
            supports_batched_prefill=True,
            max_prefill_batch_size=8,
            max_actual_page_table_width=32,
            canonical_page_table_width=64,
        )
        return sorted(
            (request.padded_sequence_length, request.padded_batch_size, len(request.source_rows))
            for request in requests
        )

    expected = [(128, 8, 6), (128, 8, 8), (128, 8, 8), (128, 8, 8), (1024, 2, 2)]
    assert planned_shapes(0) == expected
    assert planned_shapes(1) == expected
    assert planned_shapes(2) == expected


@pytest.mark.parametrize("function_name", ["_run_perf_benchmark", "_run_eval_repeat_batch32"])
def test_traced_demo_paths_warm_up_fresh_executor(function_name):
    assert "_warmup_demo_executor" in _called_names(function_name)


def test_create_executor_uses_model_owned_executor_and_resolved_cache():
    captured = {}

    def executor_config(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    namespace = {
        "DeepSeekR1Qwen14B": object,
        "DeepSeekR1Qwen14BExecutor": lambda model, runtime_config, config: config,
        "DeepSeekR1Qwen14BExecutorConfig": executor_config,
        "PagedKVCacheConfig": lambda **kwargs: SimpleNamespace(**kwargs),
        "TraceConfig": TraceConfig,
        "WarmupConfig": lambda: object(),
    }
    create_executor = _demo_function("create_executor", namespace)
    model = SimpleNamespace(
        model_args=object(),
        config=SimpleNamespace(
            max_seq_len=2048,
            max_batch_size=32,
            block_configs=[SimpleNamespace(attention_config=SimpleNamespace(kv_cache_dtype=object()))],
        ),
    )

    result = create_executor(model, traced=True, device_sampling_enabled=True)

    assert result.trace.mode == "all"
    assert result.device_sampling_enabled is True
    assert captured["paged_kv_cache"].num_blocks == 2048


def test_eval_uses_decode_only_trace_while_ordinary_traced_executor_uses_all():
    create_executor = next(
        node for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef) and node.name == "create_executor"
    )
    trace_config = next(
        node
        for node in ast.walk(create_executor)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "TraceConfig"
    )
    assert isinstance(trace_config.keywords[0].value, ast.Name)
    assert trace_config.keywords[0].value.id == "trace_mode"
    derived_mode = next(
        node
        for node in ast.walk(create_executor)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "trace_mode" for target in node.targets)
    )
    assert ast.unparse(derived_mode.value) == "'all' if traced else 'none'"

    eval_function = next(
        node
        for node in _DEMO_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_eval_repeat_batch32"
    )
    eval_create = next(
        node
        for node in ast.walk(eval_function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "create_executor"
    )
    keywords = {keyword.arg: keyword.value for keyword in eval_create.keywords}
    assert ast.literal_eval(keywords["traced"]) is True
    assert ast.literal_eval(keywords["trace_mode"]) == "decode_only"


def test_deepseek_stop_guard_truncates_eos_but_not_ordinary_reasoning_tokens(expect_error, monkeypatch):
    shared_calls = []

    def shared_guard(generated_token_ids, tokenizer, **kwargs):
        shared_calls.append((generated_token_ids, kwargs))
        if kwargs["is_ci_env"] is None and os.environ.get("TT_DEMO_STRICT_SPECIAL_TOKENS") == "1":
            outputs_before_eos = [
                output[: output.index(tokenizer.eos_token_id)] if tokenizer.eos_token_id in output else output
                for output in generated_token_ids
            ]
            if any(99 in output for output in outputs_before_eos):
                raise AssertionError("model produced special tokens")

    guard = _demo_function("assert_no_special_tokens", {"assert_no_special_tokens_shared": shared_guard})
    tokenizer = SimpleNamespace(
        all_special_ids=[10, 99],
        eos_token_id=10,
    )

    monkeypatch.setenv("TT_DEMO_STRICT_SPECIAL_TOKENS", "1")
    guard([[1, 10, 99], [2, 3, 4]], tokenizer)
    assert shared_calls[-1][0] == [[1], [2, 3, 4]]
    with expect_error(AssertionError, "model produced special tokens"):
        guard([[1, 99]], tokenizer)


def test_dp_smoke_uses_model_owned_lane_group_execution():
    calls = _called_names("_run_dp_smoke")
    assert "_dp_lane_tp_or_skip" in calls
    assert "_create_dp_submeshes" in calls
    assert "create_executor" in calls
    assert "LaneGroupExecutor" in calls
    assert "run_perf_benchmark" in calls
    assert "cleanup_dp_model_case" in calls
    assert "_skip_below_min_tp_devices" not in calls


def test_runnable_dp_lane_build_errors_are_not_converted_to_topology_skips():
    function = next(
        node for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef) and node.name == "_run_dp_smoke"
    )
    pytest_skip_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pytest"
        and node.func.attr == "skip"
    ]
    assert pytest_skip_calls == []


def test_deepseek_dp_topology_accepts_t3k_dp2_tp4_and_dp4_tp2(expect_error):
    topology = _demo_function(
        "_dp_lane_tp_or_skip",
        {"ttnn": SimpleNamespace(MeshDevice=object), "pytest": pytest, "_MIN_TP_DEVICES": 2},
    )
    t3k = SimpleNamespace(get_num_devices=lambda: 8)

    assert topology(t3k, 2) == 4
    assert topology(t3k, 4) == 2
    with expect_error(pytest.skip.Exception, "DP-8 on 8 devices creates TP1 lanes"):
        topology(t3k, 8)
    with expect_error(pytest.skip.Exception, "DP-16 cannot partition 8 devices"):
        topology(t3k, 16)


def test_deepseek_dp4_partitions_four_tp2_submeshes():
    calls = []
    submeshes = [object() for _ in range(4)]
    parent = SimpleNamespace(
        create_submeshes=lambda shape: calls.append(shape) or submeshes,
    )
    fake_ttnn = SimpleNamespace(MeshDevice=object, MeshShape=lambda rows, columns: (rows, columns))
    create_submeshes = _demo_function("_create_dp_submeshes", {"ttnn": fake_ttnn})

    assert create_submeshes(parent, 4, 2) == submeshes
    assert calls == [(1, 2)]


def test_deepseek_dp_lane_cache_reuses_lane_topology(tmp_path):
    cache_dir = tmp_path / "DeepSeek-R1-Distill-Qwen-14B" / "T3K"
    cache_dir.mkdir(parents=True)
    lane_cache_dir = _demo_function("_dp_lane_cache_dir", {"Path": Path})(cache_dir, 2)

    assert lane_cache_dir == cache_dir.parent / "N300"
    assert lane_cache_dir.is_dir()
    assert _demo_function("_dp_lane_cache_dir", {"Path": Path})(cache_dir, 4) == cache_dir.parent / "N150x4"


def test_deepseek_dp_lane_contract_checks_heads_capacity_and_cache(expect_error):
    validate = _demo_function(
        "_validate_dp_lane",
        {
            "DeepSeekR1Qwen14B": object,
            "DeepSeekR1Qwen14BExecutor": object,
            "math": __import__("math"),
        },
    )
    attention = SimpleNamespace(n_heads=40, n_kv_heads=8)
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_devices=2,
            max_batch_size=1,
            block_configs=[SimpleNamespace(attention_config=attention)],
        )
    )
    cache = SimpleNamespace(max_num_blocks=128, num_blocks=128)
    lane = SimpleNamespace(config=SimpleNamespace(paged_kv_cache=cache))

    validate(model, lane, 2, 4096)
    model.config.num_devices = 4
    with expect_error(ValueError, "expected TP2, model uses TP4"):
        validate(model, lane, 2, 4096)
    model.config.num_devices = 2
    model.config.max_batch_size = 2
    with expect_error(ValueError, "capacity 1"):
        validate(model, lane, 2, 4096)
    model.config.max_batch_size = 1
    cache.num_blocks = None
    with expect_error(ValueError, "cache must contain 128 blocks"):
        validate(model, lane, 2, 4096)


def test_token_accuracy_cleans_up_executor_in_finally():
    function = next(
        node for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef) and node.name == "_run_token_accuracy"
    )
    cleanup_calls = [
        statement
        for node in ast.walk(function)
        if isinstance(node, ast.Try)
        for statement in node.finalbody
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and statement.value.func.attr == "cleanup"
    ]
    assert len(cleanup_calls) == 1


def test_main_demo_does_not_synchronize_parent_mesh_after_prebuild_skip():
    function = next(
        node
        for node in _DEMO_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "test_deepseek_r1_qwen_14b"
    )
    try_node = next(node for node in function.body if isinstance(node, ast.Try))

    assert len(try_node.finalbody) == 1
    guard = try_node.finalbody[0]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == "model is not None"
    assert any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "cleanup_model_case"
        for node in ast.walk(guard)
    )


@pytest.mark.parametrize("function_name", ["_run_token_accuracy", "_run_perf_benchmark", "_run_eval_repeat_batch32"])
def test_demo_reads_model_geometry_from_model_config(function_name):
    function = next(
        node for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    model_args_aliases = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "model"
        and node.value.attr == "model_args"
    ]
    config_fields = {
        node.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "model"
        and node.value.attr == "config"
    }

    assert model_args_aliases == []
    assert {"max_batch_size", "max_seq_len"} <= config_fields
