# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.common.llm_runtime.config import TraceConfig
from models.common.llm_runtime.prefill.plan import _plan_prefill_requests

_DEMO_PATH = "models/common/tests/demos/phi4/demo.py"
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
        node for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef) and node.name == "test_phi4"
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


def test_phi4_trace_region_covers_measured_representative_trace_set():
    source = Path(_DEMO_PATH).read_text(encoding="utf-8")
    assert '"trace_region_size": 60_000_000' in source


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


def test_demo_registers_representative_prefill_before_trace_activation():
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
    sampling_params = object()
    traced_execution = object()
    _demo_function("_warmup_demo_executor")(
        executor,
        kv_cache=kv_cache,
        page_table=page_table,
        prefill_compile_case=(tokens, prompt_lens),
        prefill_sampling_params=sampling_params,
        prefill_compile_execution=traced_execution,
    )
    assert [kind for kind, _ in calls] == ["decode", "prefill", "compile_prefill", "prefill", "decode"]
    compile_kwargs = calls[2][1]
    assert compile_kwargs["sampling_params"] is sampling_params
    assert compile_kwargs["execution"] is traced_execution
    assert compile_kwargs["empty_slots"] == list(range(32))


def test_eval_representative_prefill_keeps_eager_execution():
    function = next(
        node
        for node in _DEMO_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_eval_repeat_batch32"
    )
    warmup_call = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_warmup_demo_executor"
    )
    keywords = {keyword.arg: keyword.value for keyword in warmup_call.keywords}
    assert ast.unparse(keywords["prefill_compile_case"]) == "representative_prefill"
    assert "prefill_compile_execution" not in keywords


@pytest.mark.parametrize(
    ("function_name", "execution_expression"),
    [
        ("_run_perf_benchmark", "traced_executor.traced_prefill_execution"),
        ("_run_dp_smoke", "group.traced_prefill_execution"),
    ],
)
def test_real_prompts_are_tokenized_before_traced_prefill_registration(function_name, execution_expression):
    function = next(
        node for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    tokenization = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "tokenize_prompts"
    )
    warmup_call = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_warmup_demo_executor"
    )
    benchmark_call = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_perf_benchmark"
    )
    assert tokenization.lineno < warmup_call.lineno < benchmark_call.lineno
    keywords = {keyword.arg: keyword.value for keyword in warmup_call.keywords}
    assert ast.unparse(keywords["prefill_compile_case"]) == "(input_tokens, prompt_lens)"
    assert ast.unparse(keywords["prefill_sampling_params"]) in {"prefill_sampling_params", "sampling_params"}
    assert ast.unparse(keywords["prefill_compile_execution"]) == execution_expression


def test_demo_uses_frozen_phi_geometry_without_auto_config():
    imports = [ast.unparse(node) for node in _DEMO_TREE.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all("AutoConfig" not in statement for statement in imports)
    assignments = {
        target.id: ast.literal_eval(node.value)
        for node in _DEMO_TREE.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id.startswith("_PHI4_NUM_")
    }
    assert assignments == {"_PHI4_NUM_ATTENTION_HEADS": 40, "_PHI4_NUM_KV_HEADS": 10}


def test_eval_prefill_signature_multiset_is_rotation_invariant():
    tokens = torch.zeros((32, 700), dtype=torch.long)
    prompt_lens = torch.tensor([64] * 30 + [400, 700])
    page_table = torch.zeros((32, 64), dtype=torch.int32)

    def planned_shapes(offset):
        requests = _plan_prefill_requests(
            tokens=torch.roll(tokens, shifts=-offset, dims=0),
            page_table=page_table,
            prompt_lens=torch.roll(prompt_lens, shifts=-offset, dims=0),
            empty_slots=list(range(32)),
            start_pos=None,
            block_size=32,
            max_batch_size=32,
            max_prefill_chunk_size=2048,
            supports_batched_prefill=True,
            max_prefill_batch_size=8,
            max_actual_page_table_width=32,
            canonical_page_table_width=64,
        )
        return sorted(
            (request.padded_sequence_length, request.padded_batch_size, len(request.source_rows))
            for request in requests
        )

    assert planned_shapes(0) == planned_shapes(1) == planned_shapes(2)


def test_eval_uses_decode_only_trace_and_all_paths_warm_up():
    assert "_warmup_demo_executor" in _called_names("_run_perf_benchmark")
    assert "_warmup_demo_executor" in _called_names("_run_eval_repeat_batch32")
    function = next(
        node
        for node in _DEMO_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_eval_repeat_batch32"
    )
    call = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "create_executor"
    )
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    assert ast.literal_eval(keywords["trace_mode"]) == "decode_only"


def test_phi4_stop_guard_preserves_chatml_turn_semantics(expect_error, monkeypatch):
    shared_calls = []

    def shared_guard(outputs, tokenizer, **kwargs):
        shared_calls.append(outputs)
        if kwargs["is_ci_env"] is None and os.environ.get("TT_DEMO_STRICT_SPECIAL_TOKENS") == "1":
            if any(99 in output for output in outputs):
                raise AssertionError("model produced special tokens")

    guard = _demo_function("assert_no_special_tokens", {"assert_no_special_tokens_shared": shared_guard})
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: {"<|im_end|>": 11, "<|im_start|>": 12}[token])
    monkeypatch.setenv("TT_DEMO_STRICT_SPECIAL_TOKENS", "1")
    guard([[1, 12, 99], [2, 11, 99]], tokenizer)
    assert shared_calls[-1] == [[1], [2]]
    with expect_error(AssertionError, "model produced special tokens"):
        guard([[1, 99, 12]], tokenizer)


def test_phi4_dp_topology_accepts_only_t3k_dp4_tp2(expect_error):
    topology = _demo_function(
        "_dp_lane_tp_or_skip",
        {"ttnn": SimpleNamespace(MeshDevice=object), "pytest": pytest, "_MIN_TP_DEVICES": 2},
    )
    t3k = SimpleNamespace(get_num_devices=lambda: 8)
    assert topology(t3k, 4) == 2
    with expect_error(pytest.skip.Exception, "DP-2 on 8 devices creates TP4 lanes"):
        topology(t3k, 2)
    with expect_error(pytest.skip.Exception, "DP-8 on 8 devices creates TP1 lanes"):
        topology(t3k, 8)
    with expect_error(pytest.skip.Exception, "DP-16 cannot partition 8 devices"):
        topology(t3k, 16)


def test_t3k_policy_is_two_dp4_runnable_and_eighteen_intentional_skips(expect_error):
    topology = _demo_function(
        "_dp_lane_tp_or_skip",
        {"ttnn": SimpleNamespace(MeshDevice=object), "pytest": pytest, "_MIN_TP_DEVICES": 2},
    )
    ordinary_guard = _demo_function(
        "_skip_unless_heads_divide_mesh",
        {
            "ttnn": SimpleNamespace(MeshDevice=object),
            "pytest": pytest,
            "_PHI4_NUM_ATTENTION_HEADS": 40,
            "_PHI4_NUM_KV_HEADS": 10,
        },
    )
    t3k = SimpleNamespace(get_num_devices=lambda: 8)
    runnable = 0
    skipped = 0

    for _optimization in ("performance", "accuracy"):
        for _ordinary_case in ("token-accuracy", "batch-1", "batch-32", "batch-32-ci", "eval-32"):
            with expect_error(pytest.skip.Exception, "Incompatible mesh for Phi-4"):
                ordinary_guard(t3k)
            skipped += 1
        for data_parallel in (2, 4, 8, 16, 32):
            if data_parallel == 4:
                assert topology(t3k, data_parallel) == 2
                runnable += 1
            else:
                with expect_error(pytest.skip.Exception, f"DP-{data_parallel}"):
                    topology(t3k, data_parallel)
                skipped += 1

    assert (runnable, skipped) == (2, 18)


def test_demo_uses_phi_provider_prompt_encoding_only():
    source = Path(_DEMO_PATH).read_text(encoding="utf-8")
    assert "models.tt_transformers.tt.common" not in source
    assert "encode_prompt_hf" not in source
    assert "encode_prompt(tokenizer, p)" in source


def test_n300_policy_remains_nine_passes_and_eleven_skips(expect_error):
    topology = _demo_function(
        "_dp_lane_tp_or_skip",
        {"ttnn": SimpleNamespace(MeshDevice=object), "pytest": pytest, "_MIN_TP_DEVICES": 2},
    )
    n300 = SimpleNamespace(get_num_devices=lambda: 2)
    skipped_dp_nodes = 0
    skip_messages = {
        2: "DP-2 on 2 devices creates TP1 lanes",
        4: "DP-4 cannot partition 2 devices",
        8: "DP-8 cannot partition 2 devices",
        16: "DP-16 cannot partition 2 devices",
        32: "DP-32 cannot partition 2 devices",
    }
    for data_parallel, message in skip_messages.items():
        with expect_error(pytest.skip.Exception, message):
            topology(n300, data_parallel)
        skipped_dp_nodes += 2
    skipped_accuracy_eval = 1
    assert skipped_dp_nodes + skipped_accuracy_eval == 11
    assert 20 - skipped_dp_nodes - skipped_accuracy_eval == 9


def test_phi4_dp_lane_cache_reuses_n300_topology(tmp_path):
    cache_dir = tmp_path / "phi-4" / "T3K"
    cache_dir.mkdir(parents=True)
    lane_cache_dir = _demo_function("_dp_lane_cache_dir", {"Path": Path})(cache_dir, 2)
    assert lane_cache_dir == cache_dir.parent / "N300"


def test_dp_smoke_uses_lane_group_and_does_not_skip_build_failures():
    calls = _called_names("_run_dp_smoke")
    assert {"_dp_lane_tp_or_skip", "_create_dp_submeshes", "create_executor", "LaneGroupExecutor"} <= set(calls)
    function = next(
        node for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef) and node.name == "_run_dp_smoke"
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pytest"
        and node.func.attr == "skip"
        for node in ast.walk(function)
    )
    source = ast.unparse(function)
    assert "make_contiguous_page_table(1, max_seq_len, 32).repeat(data_parallel, 1)" in source


def test_supported_ordinary_model_build_errors_are_not_skipped():
    function = next(
        node for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef) and node.name == "create_model"
    )
    assert not any(isinstance(node, ast.Try) for node in ast.walk(function))


def test_main_cleanup_is_guarded_after_prebuild_skip():
    function = next(node for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef) and node.name == "test_phi4")
    try_node = next(node for node in function.body if isinstance(node, ast.Try))
    assert len(try_node.finalbody) == 1
    assert ast.unparse(try_node.finalbody[0].test) == "model is not None"
