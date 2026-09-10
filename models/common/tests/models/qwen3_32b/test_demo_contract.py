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


def test_demo_imports_every_called_shared_run_helper():
    imported = {
        alias.name
        for node in _DEMO_TREE.body
        if isinstance(node, ast.ImportFrom) and node.module == "models.common.tests.demos.run_helpers"
        for alias in node.names
    }
    assert {
        "eval_decode_trace_mode",
        "load_eval_repeat_prompts_batch32",
        "require_canonical_eval_modes_in_ci",
        "run_eval_repeat_batch32",
        "run_perf_benchmark",
        "run_teacher_forcing",
    } <= imported


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


def _cross_cardinality_namespace():
    names = {
        "_compare_cross_cardinality_token_ids",
        "_require_cross_cardinality_prefill_geometry",
    }
    nodes = [
        node for node in _DEMO_TREE.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    namespace = {
        "_CROSS_CARDINALITY_REQUEST_IDS": tuple(f"request-{index}" for index in range(32)),
        "_CROSS_CARDINALITIES": (1, 2, 4, 32),
        "_CROSS_CARDINALITY_DECODE_TOKENS": 1,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), _DEMO_PATH, "exec"), namespace)
    return namespace


def test_cross_cardinality_experiment_is_one_canonical_exact_token_node():
    function = _function("test_qwen3_32b_p150x4_seeded_cross_cardinality")
    source = ast.unparse(function)
    assert "get_device_name(mesh_device) != 'P150x4'" in source
    assert "_require_cross_cardinality_environment()" in source
    assert "ma.disable_batched_prefill is True" in source
    assert "ma.batched_prefill_batched_extract is True" in source
    assert "sampling_params=sampling_params" in source
    assert "prefill_sampling_params=None" in source
    assert "ondevice_decode_loop=True" in source
    assert "trace_mode=eval_decode_trace_mode('traced')" in source
    assert "model.sampling.config.seeds" not in source
    assert "_snapshot_cross_cardinality_prefill" in source
    assert "_require_cross_cardinality_prefill_geometry" in source
    assert "_compare_cross_cardinality_token_ids(controls, prefixes)" in source
    assert "QWEN3_32B_CROSS_CARDINALITY_VERDICT=" in source
    assert "decode_eval_output" not in source
    assert "assert_cross_cardinality_consistency" not in source
    assert "ma.disable_batched_prefill = False" in source
    assert "ma.disable_batched_prefill = True" in source
    control_cases = next(
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "control_cases" for target in node.targets)
    )
    assert isinstance(control_cases, ast.ListComp)
    assert ast.unparse(control_cases.elt).startswith(
        "prepare_requests(sequential_executor, [prompt], [seed], batched_candidate=False)"
    )
    assert len(control_cases.generators) == 1
    generator = control_cases.generators[0]
    assert isinstance(generator.target, ast.Tuple)
    assert tuple(element.id for element in generator.target.elts if isinstance(element, ast.Name)) == (
        "prompt",
        "seed",
    )
    assert ast.unparse(generator.iter) == "zip(prompts, _CROSS_CARDINALITY_SEEDS, strict=True)"
    assert "first 2/4 requests in one Q128 batch" in source
    assert ast.literal_eval(
        next(
            node.value
            for node in _DEMO_TREE.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "_CROSS_CARDINALITIES" for target in node.targets)
        )
    ) == (1, 2, 4, 32)
    assert "qwen3-32b-request-{index:02d}" in _DEMO_SOURCE
    seed_assignment = next(
        node
        for node in _DEMO_TREE.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_CROSS_CARDINALITY_SEEDS" for target in node.targets)
    )
    namespace = {}
    exec(compile(ast.Module(body=[seed_assignment], type_ignores=[]), _DEMO_PATH, "exec"), namespace)
    seeds = namespace["_CROSS_CARDINALITY_SEEDS"]
    assert len(seeds) == len(set(seeds)) == 32
    prompt_order = next(
        node
        for node in _DEMO_TREE.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_CROSS_CARDINALITY_PROMPT_ORDER" for target in node.targets
        )
    )
    order_namespace = {}
    exec(compile(ast.Module(body=[prompt_order], type_ignores=[]), _DEMO_PATH, "exec"), order_namespace)
    assert tuple(order_namespace["_CROSS_CARDINALITY_PROMPT_ORDER"]) == (*range(2, 32), 0, 1)
    assert len(_calls("test_qwen3_32b_p150x4_seeded_cross_cardinality", "make_executor")) == 2
    make_calls = _calls("test_qwen3_32b_p150x4_seeded_cross_cardinality", "make_executor")
    expected_policies = {
        ast.literal_eval(
            next(keyword.value for keyword in call.keywords if keyword.arg == "expected_disable_batched_prefill")
        )
        for call in make_calls
    }
    assert expected_policies == {True, False}
    assert "executor.prefill_runtime.config.disable_batched_prefill" in source
    assert "compile_prefill_case" in source
    assert "executor.warmup_model_decode(enable_trace=False, **decode_kwargs)" in source
    assert "executor.warmup_model_decode(enable_trace=True, **decode_kwargs)" in source
    assert source.index("compile_prefill_case(sequential_executor") < source.index(
        "activate_decode_trace(sequential_executor"
    )
    assert source.index("compile_prefill_case(candidate_executor") < source.index(
        "activate_decode_trace(candidate_executor"
    )
    assert "compiler.trace_count == len(coverage) >= 1" in source
    assert "signature.sampling_path == 'topk'" in source
    assert "len(topk_coverage) == 1" in source
    assert "compiler.trace_key_for_program(decode_program_key) == expected_topk_trace_key" in source
    assert "compiler.trace_count == expected_semantic_trace_count and compiler.trace_active" in source
    assert "compiler.trace_count == 1" not in source
    assert "record is not None and record.artifact is not None" in source
    assert "replay_delta != _CROSS_CARDINALITY_DECODE_TOKENS" in source
    assert "post_activation_compile_rejections == 0" in source
    assert "control_trace_lifecycle" in source
    assert "candidate_trace_lifecycle" in source
    assert "control_replay_evidence" in source
    assert "candidate_replay_evidence" in source
    assert "control_prefill_geometry" in source
    assert "candidate_prefill_geometry" in source
    assert "eager_prefill_decode_traced" in source


def test_cross_cardinality_geometry_requires_real_batched_requests_and_source_rows(expect_error):
    namespace = _cross_cardinality_namespace()
    require = namespace["_require_cross_cardinality_prefill_geometry"]
    batched_2 = (
        {
            "kind": "batched",
            "source_rows": (0, 1),
            "active_batch_size": 2,
            "padded_batch_size": 2,
            "padded_sequence_length": 128,
            "operation_variants": ("regular-batched",),
        },
    )
    require(batched_2, cardinality=2, batched_candidate=True)

    sequential_2 = tuple(
        {
            "kind": "single",
            "source_rows": (row,),
            "active_batch_size": 1,
            "padded_batch_size": 1,
            "padded_sequence_length": 128,
            "operation_variants": ("regular-single",),
        }
        for row in range(2)
    )
    with expect_error(AssertionError, "cardinality 2 prepared-prefill geometry disagrees"):
        require(sequential_2, cardinality=2, batched_candidate=True)

    batched_32 = (
        {
            "kind": "batched",
            "source_rows": tuple(range(30)),
            "active_batch_size": 30,
            "padded_batch_size": 32,
            "padded_sequence_length": 128,
            "operation_variants": ("regular-batched",),
        },
        {
            "kind": "batched",
            "source_rows": (30, 31),
            "active_batch_size": 2,
            "padded_batch_size": 2,
            "padded_sequence_length": 1024,
            "operation_variants": ("regular-batched",),
        },
    )
    require(batched_32, cardinality=32, batched_candidate=True)

    stale_31_plus_1 = (
        {
            "kind": "batched",
            "source_rows": tuple(range(31)),
            "active_batch_size": 31,
            "padded_batch_size": 32,
            "padded_sequence_length": 128,
            "operation_variants": ("regular-batched",),
        },
        {
            "kind": "single",
            "source_rows": (31,),
            "active_batch_size": 1,
            "padded_batch_size": 1,
            "padded_sequence_length": 1024,
            "operation_variants": ("regular-single",),
        },
    )
    with expect_error(AssertionError, "cardinality 32 prepared-prefill geometry disagrees"):
        require(stale_31_plus_1, cardinality=32, batched_candidate=True)


def test_cross_cardinality_verdict_compares_exact_tokens_and_accepts_negative_execution(expect_error):
    namespace = _cross_cardinality_namespace()
    request_ids = namespace["_CROSS_CARDINALITY_REQUEST_IDS"]
    controls = {request_id: (index, index + 1) for index, request_id in enumerate(request_ids)}
    prefixes = {
        cardinality: {request_id: controls[request_id] for request_id in request_ids[:cardinality]}
        for cardinality in (1, 2, 4, 32)
    }

    verdict, mismatches = namespace["_compare_cross_cardinality_token_ids"](controls, prefixes)
    assert verdict == "INVARIANT"
    assert mismatches == ()

    prefixes[4][request_ids[2]] = (2, 999)
    verdict, mismatches = namespace["_compare_cross_cardinality_token_ids"](controls, prefixes)
    assert verdict == "BATCHED_PREFILL_REJECTED"
    assert mismatches == (
        {
            "cardinality": 4,
            "request_id": request_ids[2],
            "first_token_difference": 1,
            "control_token_count": 2,
            "batched_token_count": 2,
        },
    )

    with expect_error(AssertionError, "must contain all 32 fixed request IDs"):
        namespace["_compare_cross_cardinality_token_ids"]({request_ids[0]: (0,)}, prefixes)

    prefixes = {
        cardinality: {request_id: controls[request_id] for request_id in request_ids[:cardinality]}
        for cardinality in (1, 2, 4, 32)
    }
    prefixes[4][request_ids[2]] = (2,)
    with expect_error(AssertionError, "candidates must each return 2 generated tokens"):
        namespace["_compare_cross_cardinality_token_ids"](controls, prefixes)


def test_cross_cardinality_environment_and_checked_in_policy_fail_closed():
    source = ast.unparse(_function("_require_cross_cardinality_environment"))
    assert "DISABLE_BATCHED_PREFILL" in source
    assert "DISABLE_BATCHED_EXTRACT" in source
    create_call = _calls("test_qwen3_32b_p150x4_seeded_cross_cardinality", "create_model")[0]
    assert not any(keyword.arg == "disable_batched_prefill" for keyword in create_call.keywords)
    canonical_calls = _calls("test_qwen3_32b", "create_model")
    assert canonical_calls
    assert all(
        not any(keyword.arg == "disable_batched_prefill" for keyword in call.keywords) for call in canonical_calls
    )


def test_demo_resolves_qwen3_trace_region_and_matches_ring_fabric():
    assert 'resolve_trace_region_size("qwen3-32b", env)' in _DEMO_SOURCE
    assert '"trace_region_size": 50_000_000' not in _DEMO_SOURCE
    assert "ttnn.FabricConfig.FABRIC_1D_RING" in _DEMO_SOURCE
    assert resolve_trace_region_size("qwen3-32b", "T3K") == 90_000_000
    assert resolve_trace_region_size("qwen3-32b", "P150x4") == 100_000_000


def test_demo_exposes_p150x4_and_uses_canonical_device_naming():
    assert '"P150x4": (1, 4)' in _DEMO_SOURCE
    assert "bh_hardware" not in _DEMO_SOURCE
    assert not any(isinstance(node, ast.FunctionDef) and node.name == "get_device_name" for node in _DEMO_TREE.body)
    imports = [ast.unparse(node) for node in _DEMO_TREE.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert any("models.common.device_utils import get_device_name" in statement for statement in imports)


def test_required_bh_gate_failures_are_not_converted_to_fixture_skips():
    assert 'mesh_device_name in {"P150", "P300", "P150X4"}' in _COMMON_CONFTEST_SOURCE
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


def test_perf_warms_executor_before_shared_perf_runner_replay():
    function = _function("_run_perf_benchmark")
    tokenize_call = _calls("_run_perf_benchmark", "tokenize_prompts")[0]
    warmup_call = _calls("_run_perf_benchmark", "_warmup_demo_executor")[0]
    runner_call = _calls("_run_perf_benchmark", "run_perf_benchmark")[0]
    profiler_start = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and ast.unparse(node.func) == "profiler.start"
    )
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in warmup_call.keywords}

    assert ast.unparse(warmup_call.args[0]) == "traced_executor"
    assert keywords["kv_cache"] == "kv_cache"
    assert keywords["page_table"] == "page_table"
    assert keywords["prefill_compile_case"] == "(input_tokens, prompt_lens)"
    assert keywords["prefill_sampling_params"] == "sampling_params"
    assert keywords["prefill_compile_execution"] == "traced_executor.traced_prefill_execution"
    assert tokenize_call.lineno < warmup_call.lineno < profiler_start.lineno < runner_call.lineno


def test_eval_repeat_threads_sampling_mode_to_traced_executor():
    call = _calls("_run_eval_repeat_batch32", "TracedQwen3_32BExecutor")[0]
    ondevice_decode_loop = next(keyword for keyword in call.keywords if keyword.arg == "ondevice_decode_loop")

    assert ast.unparse(ondevice_decode_loop.value) == "sampling_params is not None"


def test_eval_repeat_preserves_decode_only_determinism_and_uses_full_trace_for_perf_report():
    function = _function("_run_eval_repeat_batch32")
    call = _calls("_run_eval_repeat_batch32", "TracedQwen3_32BExecutor")[0]
    trace_mode = next(keyword for keyword in call.keywords if keyword.arg == "trace_mode")
    configure_call = _calls("_run_eval_repeat_batch32", "_require_eval_perf_prefill_trace_parity")[0]

    assert ast.unparse(trace_mode.value) == (
        "'all' if perf_report else eval_decode_trace_mode(os.environ.get('EVAL_DECODE_MODE', 'traced'))"
    )
    assert configure_call.lineno < call.lineno
    assert "if perf_report:\n        _require_eval_perf_prefill_trace_parity(ma)" in ast.unparse(function)


def test_eval_perf_report_validates_bh_sequential_policy_and_model_owned_trace_coverage(expect_error):
    namespace = {"_EVAL_PERF_TRACE_PREFILL_BUCKETS": (128, 1024)}
    function = _function("_require_eval_perf_prefill_trace_parity")
    exec(compile(ast.Module(body=[function], type_ignores=[]), _DEMO_PATH, "exec"), namespace)

    model_args = SimpleNamespace(
        max_prefill_chunk_size=4096,
        max_seq_len=1024,
        cluster_shape=(1, 4),
        disable_batched_prefill=True,
        trace_prefill_supported_seq_lens=(128, 1024),
        can_enable_trace=lambda seq_len, num_cached_tokens=0: num_cached_tokens == 0 and seq_len in (128, 1024),
    )
    namespace["_require_eval_perf_prefill_trace_parity"](model_args)
    assert model_args.disable_batched_prefill is True
    assert model_args.trace_prefill_supported_seq_lens == (128, 1024)
    assert model_args.can_enable_trace(128)
    assert model_args.can_enable_trace(1024)
    assert not model_args.can_enable_trace(2048)
    assert not model_args.can_enable_trace(128, num_cached_tokens=32)

    t3k = SimpleNamespace(
        max_prefill_chunk_size=4096,
        max_seq_len=1024,
        cluster_shape=(1, 8),
        disable_batched_prefill=False,
        trace_prefill_supported_seq_lens=(128, 1024),
        can_enable_trace=lambda seq_len, num_cached_tokens=0: num_cached_tokens == 0 and seq_len in (128, 1024),
    )
    namespace["_require_eval_perf_prefill_trace_parity"](t3k)
    assert t3k.disable_batched_prefill is False

    insufficient = SimpleNamespace(max_prefill_chunk_size=512, max_seq_len=1024, cluster_shape=(1, 4))
    with expect_error(ValueError, "requires 128/1024 prefill trace coverage"):
        namespace["_require_eval_perf_prefill_trace_parity"](insufficient)

    bh_batched = SimpleNamespace(
        max_prefill_chunk_size=4096,
        max_seq_len=1024,
        cluster_shape=(1, 4),
        disable_batched_prefill=False,
    )
    with expect_error(RuntimeError, "requires model-owned sequential prefill on P150x4"):
        namespace["_require_eval_perf_prefill_trace_parity"](bh_batched)

    missing_bucket = SimpleNamespace(
        max_prefill_chunk_size=4096,
        max_seq_len=1024,
        cluster_shape=(1, 4),
        disable_batched_prefill=True,
        trace_prefill_supported_seq_lens=(128,),
        can_enable_trace=lambda seq_len, num_cached_tokens=0: seq_len == 128,
    )
    with expect_error(ValueError, "requires model-owned prefill trace buckets"):
        namespace["_require_eval_perf_prefill_trace_parity"](missing_bucket)

    rejecting_predicate = SimpleNamespace(
        max_prefill_chunk_size=4096,
        max_seq_len=1024,
        cluster_shape=(1, 4),
        disable_batched_prefill=True,
        trace_prefill_supported_seq_lens=(128, 1024),
        can_enable_trace=lambda seq_len, num_cached_tokens=0: seq_len == 128,
    )
    with expect_error(RuntimeError, "model predicate rejects required prefill trace coverage"):
        namespace["_require_eval_perf_prefill_trace_parity"](rejecting_predicate)


def test_eval_repeat_defaults_to_tttv1_slot_stable_page_table_with_diagnostic_override():
    source = ast.unparse(_function("_run_eval_repeat_batch32"))
    assert "page_table_mode=os.environ.get('EVAL_PAGE_TABLE_MODE', 'slot-stable')" in source


def test_eval_repeat_warms_executor_before_shared_perf_runner_replay():
    warmup_call = _calls("_run_eval_repeat_batch32", "_warmup_demo_executor")[0]
    runner_call = _calls("_run_eval_repeat_batch32", "run_eval_repeat_batch32")[0]
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in warmup_call.keywords}

    assert keywords["prefill_compile_case"] == "representative_prefill"
    assert keywords["prefill_sampling_params"] == "sampling_params"
    assert keywords["prefill_compile_execution"] == ("executor.traced_prefill_execution if perf_report else None")
    assert warmup_call.lineno < runner_call.lineno

    helper_source = ast.unparse(_function("_warmup_demo_executor"))
    assert helper_source.index("executor.compile_prefill") < helper_source.index(
        "executor.warmup_model_prefill(enable_trace=True"
    )
    assert "executor.warmup_model_decode" in helper_source
    assert "executor.warmup_model_prefill" in helper_source


def test_eval_perf_report_reuses_three_repeat_geometry_and_first_repeat_profiler():
    source = ast.unparse(_function("_run_eval_repeat_batch32"))
    assert "repeat_batches=_EVAL_REPEAT_BATCHES" in source
    assert "first_repeat_profiler=profiler" in source
    assert "if expected is None" in source
    assert "_assert_eval32_perf_target(first_result, expected" in source
    assert "'on_device_topk' if perf_report else 'host'" in source
    assert "eval-32-perf-report" in ast.unparse(_function("test_qwen3_32b"))


def test_eval_perf_targets_run_observationally_when_profile_floor_is_missing_but_failed_floor_fails(expect_error):
    warnings = []
    resolve_namespace = {
        "resolve_perf_targets": lambda *args, **kwargs: {
            "decode_t/s/u": 21.6,
            "prefill_time_to_first_token": 87,
        },
        "_EVAL32_TARGET_SEQ_LEN": 686,
        "logger": SimpleNamespace(warning=warnings.append),
    }
    resolve_function = _function("_resolve_eval32_perf_targets")
    exec(compile(ast.Module(body=[resolve_function], type_ignores=[]), _DEMO_PATH, "exec"), resolve_namespace)
    assert resolve_namespace["_resolve_eval32_perf_targets"]("Qwen/Qwen3-32B", "P150x4", "accuracy") is None
    assert resolve_namespace["_resolve_eval32_perf_targets"]("Qwen/Qwen3-32B", "P150x4", "performance") == {
        "decode_t/s/u": 21.6,
        "prefill_time_to_first_token": 87,
    }
    assert "observationally" in warnings[0]

    resolve_namespace["resolve_perf_targets"] = lambda *args, **kwargs: None
    assert resolve_namespace["_resolve_eval32_perf_targets"]("Qwen/Qwen3-32B", "P150x4", "performance") is None
    with expect_error(ValueError, "qualification gates fail closed"):
        resolve_namespace["_resolve_eval32_perf_targets"]("Qwen/Qwen3-32B", "T3K", "performance")

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


def test_other_declared_p150x4_perf_nodes_run_observationally_without_floor_and_preserve_complete_floor():
    warnings = []
    namespace = {"logger": SimpleNamespace(warning=warnings.append)}
    function = _function("_resolve_local_perf_floor")
    exec(compile(ast.Module(body=[function], type_ignores=[]), _DEMO_PATH, "exec"), namespace)
    assert namespace["_resolve_local_perf_floor"]("T3K", {}, case_name="WH") == {}
    assert namespace["_resolve_local_perf_floor"]("P150x4", {}, case_name="BH/batch-32-ci") is None
    complete = {"tok_s_u": 20.0, "ttft_ms": 120.0}
    assert namespace["_resolve_local_perf_floor"]("P150x4", complete, case_name="BH/batch-32-ci") == complete
    assert "observationally" in warnings[0]


def test_complete_local_perf_floor_still_fails_both_missed_targets(expect_error):
    namespace = {"PERF_TOLERANCE": 0.05}
    function = _function("_assert_local_perf_target")
    exec(compile(ast.Module(body=[function], type_ignores=[]), _DEMO_PATH, "exec"), namespace)
    result = SimpleNamespace(tok_s_u=10.0, ttft_ms=200.0)
    expected = {"tok_s_u": 20.0, "ttft_ms": 100.0}

    with expect_error(AssertionError, "tok/s/u.*ttft_ms"):
        namespace["_assert_local_perf_target"](result, expected, case_name="BH/batch-32-ci")


def test_main_demo_resolves_eval_floor_by_optimization_profile_and_local_perf_only_gates_complete_floor():
    main_source = ast.unparse(_function("test_qwen3_32b"))
    perf_source = ast.unparse(_function("_run_perf_benchmark"))
    eval_source = ast.unparse(_function("_run_eval_repeat_batch32"))

    assert "_resolve_eval32_perf_targets(hf_model, device_name, optimizations)" in main_source
    assert "expected = _resolve_local_perf_floor" in perf_source
    assert perf_source.index("Performance [{case_name}]") < perf_source.index("_resolve_local_perf_floor")
    assert "if expected:" in perf_source
    assert "_assert_local_perf_target(result, expected" in perf_source
    assert "config_params={'optimization_profile': case_name.split('/', 1)[0]}" in eval_source


def test_traced_compatibility_wrapper_is_accepted_by_transition_perf_helper(monkeypatch):
    def fake_init(self, model, runtime_config, config):
        self.model = model
        self.runtime_config = runtime_config
        self.config = config

    monkeypatch.setattr(qwen3_executor.Qwen3_32BExecutor, "__init__", fake_init)

    model = SimpleNamespace(model_args=SimpleNamespace(), config=SimpleNamespace(max_seq_len=4096, max_batch_size=32))
    traced = qwen3_executor.TracedQwen3_32BExecutor(model, mesh_device=object(), ondevice_decode_loop=True)

    assert _has_trace_surface(traced)
