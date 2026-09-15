# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.demos.utils.model_targets import resolve_accuracy_targets, resolve_metric_tolerance
from models.demos.utils.trace_region_sizes import resolve_trace_region_size

_DEMO_PATH = "models/common/tests/demos/llama33_70b/demo.py"
_DEMO_SOURCE = Path(_DEMO_PATH).read_text(encoding="utf-8")
_DEMO_TREE = ast.parse(_DEMO_SOURCE, filename=_DEMO_PATH)
_SMOKE_PATH = "models/common/tests/models/llama33_70b/test_p150x4_smoke.py"
_SMOKE_SOURCE = Path(_SMOKE_PATH).read_text(encoding="utf-8")
_SMOKE_TREE = ast.parse(_SMOKE_SOURCE, filename=_SMOKE_PATH)
_REQUIRED_CAPABILITIES_PATH = "models/tttv2_llama33_70b_bh_required_capabilities.json"


def _function(name):
    return next(node for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _calls(function_name, called_name):
    return [
        node
        for node in ast.walk(_function(function_name))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == called_name
    ]


def test_demo_case_manifest_is_preserved():
    decorators = [node for node in _function("test_llama33_70b").decorator_list if isinstance(node, ast.Call)]
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


def test_demo_resolves_central_trace_region_size_for_each_supported_sku():
    source = ast.unparse(_function("_ttnn_mesh_device_param_from_env"))
    assert "resolve_trace_region_size('llama3.3-70b', env)" in source
    assert '"trace_region_size": 50_000_000' not in _DEMO_SOURCE
    assert resolve_trace_region_size("llama3.3-70b", "T3K") == 224_000_000
    assert resolve_trace_region_size("llama3.3-70b", "P150x4") == 224_000_000


def test_demo_collects_physical_p150x4_without_adding_unmeasured_perf_targets():
    assignment = next(
        node
        for node in _DEMO_TREE.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_MESH_DEVICE_TO_SHAPE"
    )
    mesh_map = ast.literal_eval(assignment.value)
    assert mesh_map == {"T3K": (1, 8), "P150x4": (1, 4)}
    assert "bh_hardware" not in _DEMO_SOURCE
    assert '"P150x4": {"tok_s_u"' not in _DEMO_SOURCE


def test_p150x4_token_accuracy_uses_independently_existing_central_floor():
    source = ast.unparse(_function("_run_token_accuracy"))
    assert "is_ci_env or device_name == 'P150x4'" in source
    assert "token accuracy is observational" not in source
    assert _calls("_run_token_accuracy", "resolve_accuracy_targets")
    assert resolve_accuracy_targets("meta-llama/Llama-3.3-70B-Instruct", "P150x4", batch_size=1, seq_len=512) == {
        "top1": 96,
        "top5": 100,
    }


def test_p150x4_eval_perf_has_no_independent_floor_to_copy_or_invent():
    provenance = next(
        node
        for node in _DEMO_TREE.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_EVAL32_TARGET_PROVENANCE"
    )
    assert ast.literal_eval(provenance.value) == {}


def test_required_capability_policy_allows_observation_but_never_acceptance_without_floor():
    contract = json.loads(Path(_REQUIRED_CAPABILITIES_PATH).read_text(encoding="utf-8"))
    policy = next(row for row in contract["cross_cutting_requirements"] if row["id"] == "fail_closed_performance")
    policy_text = f"{policy['capability']} {policy['acceptance_condition']}"
    for phrase in (
        "observational",
        "must not claim acceptance",
        "complete independently frozen floor",
        "target miss fails",
        "TTFT",
        "decode tokens/s/user",
        "aggregate tokens/s",
    ):
        assert phrase in policy_text


def test_demo_uses_model_owned_runtime_provider_and_shared_helpers():
    imports = [ast.unparse(node) for node in _DEMO_TREE.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert any("models.common.models.llama33_70b.executor" in statement for statement in imports)
    assert any("models.common.models.llama33_70b.hf_adaptor" in statement for statement in imports)
    assert any("models.common.tests.demos.run_helpers" in statement for statement in imports)
    assert any("models.common.device_utils import get_device_name" in statement for statement in imports)
    assert not any(node.name == "get_device_name" for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef))
    assert all("models.common.models.executor" not in statement for statement in imports)
    assert all("AutoConfig" not in statement and "AutoTokenizer" not in statement for statement in imports)


def test_blackhole_tp4_smoke_uses_product_admission_and_exact_ring_geometry():
    admission = next(
        node
        for node in _SMOKE_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "_assert_physical_bh_tp4"
    )
    source = ast.unparse(admission)

    assert "ttnn.cluster.get_cluster_type() in LLAMA33_70B_BH_TP4_CLUSTER_TYPES" in source
    assert "mesh_device.get_num_devices() == 4" in source
    assert "tuple(mesh_device.shape) == (1, 4)" in source
    assert "ttnn.FabricConfig.FABRIC_1D_RING" in _SMOKE_SOURCE
    assert 'ids=["physical-BH-TP4-ring"]' in _SMOKE_SOURCE


def test_supported_tp8_model_build_failures_are_not_converted_to_skips():
    create_model = _function("create_model")
    assert not any(isinstance(node, ast.Try) for node in ast.walk(create_model))
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pytest"
        and node.func.attr == "skip"
        for node in ast.walk(create_model)
    )


@pytest.mark.parametrize("data_parallel", [2, 4, 8, 16, 32])
def test_every_dp_case_skips_before_submesh_or_model_construction(data_parallel, expect_error):
    namespace = {"pytest": pytest, "ttnn": SimpleNamespace(MeshDevice=object)}
    function = _function("_dp_or_skip")
    exec(compile(ast.Module(body=[function], type_ignores=[]), _DEMO_PATH, "exec"), namespace)
    mesh = SimpleNamespace(get_num_devices=lambda: 8)
    with expect_error(pytest.skip.Exception, f"DP-{data_parallel}"):
        namespace["_dp_or_skip"](mesh, data_parallel)
    run_dp = _function("_run_dp_smoke")
    calls = [
        node.func.id for node in ast.walk(run_dp) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls == ["_dp_or_skip"]


def test_demo_allocates_kv_cache_without_model_shape_arguments():
    for function_name in ("_run_token_accuracy", "_run_perf_benchmark", "_run_eval_repeat_batch32"):
        allocations = [
            node
            for node in ast.walk(_function(function_name))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "allocate_kv_cache"
        ]
        assert allocations
        assert all(not call.args and not call.keywords for call in allocations)


def test_perf_registers_actual_prefill_before_closed_world_trace_activation():
    function = _function("_run_perf_benchmark")
    tokenization = _calls("_run_perf_benchmark", "tokenize_prompts")[0]
    warmup = _calls("_run_perf_benchmark", "_warmup_demo_executor")[0]
    benchmark = _calls("_run_perf_benchmark", "run_perf_benchmark")[0]
    assert tokenization.lineno < warmup.lineno < benchmark.lineno
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in warmup.keywords}
    assert keywords["prefill_compile_case"] == "(input_tokens, prompt_lens)"
    assert keywords["prefill_compile_execution"] == "traced_executor.traced_prefill_execution"
    assert any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "compile_prefill"
        for node in ast.walk(_function("_warmup_demo_executor"))
    )


def test_eval_and_perf_report_preserve_decode_only_trace_with_eager_prefill():
    create = _calls("_run_eval_repeat_batch32", "create_executor")[0]
    create_keywords = {keyword.arg: keyword.value for keyword in create.keywords}
    assert (
        ast.unparse(create_keywords["trace_mode"])
        == "eval_decode_trace_mode(os.environ.get('EVAL_DECODE_MODE', 'traced'))"
    )
    warmup = _calls("_run_eval_repeat_batch32", "_warmup_demo_executor")[0]
    warmup_keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in warmup.keywords}
    assert warmup_keywords["prefill_compile_case"] == "representative_prefill"
    assert "prefill_compile_execution" not in warmup_keywords
    source = ast.unparse(_function("_run_eval_repeat_batch32"))
    assert "page_table_mode=os.environ.get('EVAL_PAGE_TABLE_MODE', 'slot-stable')" in source
    assert "'EVAL_IDENTICAL_PROMPT_INDEX'" in source
    assert "'EVAL_ACTIVE_BATCH_SIZE'" in source
    assert "trace_mode='all'" not in source
    assert "traced_prefill_execution" not in source


def test_eval_perf_report_reuses_three_repeat_geometry_and_first_repeat_telemetry():
    source = ast.unparse(_function("_run_eval_repeat_batch32"))
    assert "_EVAL_REPEAT_BATCHES if perf_report" in source
    assert "first_repeat_profiler=profiler" in source
    assert "'on_device_topk' if perf_report else 'host'" in source
    assert "_assert_eval32_perf_target(first_result, expected" in source
    assert "config_params={'optimization_profile': case_name.split('/', 1)[0]}" in source
    assert "if expected is not None" in source
    assert "run_type='demo_perf'" in source


def test_eval_perf_report_is_dispatched_for_both_profiles_and_resolves_target():
    source = ast.unparse(_function("test_llama33_70b"))
    assert "test_config in ('eval-32', 'eval-32-perf-report')" in source
    assert "_preflight_perf_target" in source
    assert "perf_report=perf_report" in source
    assert "perf_expected = resolved_perf_expected" in source
    assert "eval_expected = resolved_perf_expected" in source
    preflight = _calls("test_llama33_70b", "_preflight_perf_target")[0]
    create = _calls("test_llama33_70b", "create_model")[0]
    assert preflight.lineno < create.lineno


def test_eval_perf_targets_observe_when_missing_but_enforce_complete_floor(expect_error):
    resolve_function = _function("_resolve_eval32_perf_targets")
    logger = SimpleNamespace(warning=lambda message: None)

    missing_namespace = {
        "resolve_perf_targets": lambda *args, **kwargs: None,
        "_EVAL32_TARGET_PROVENANCE": {},
        "_EVAL32_FIXED_PROVENANCE": {},
        "logger": logger,
    }
    exec(compile(ast.Module(body=[resolve_function], type_ignores=[]), _DEMO_PATH, "exec"), missing_namespace)
    assert (
        missing_namespace["_resolve_eval32_perf_targets"]("meta-llama/Llama-3.3-70B-Instruct", "P150x4", "performance")
        is None
    )

    incomplete_namespace = {
        "resolve_perf_targets": lambda *args, **kwargs: {"decode_t/s/u": 10.0},
        "_EVAL32_FIXED_PROVENANCE": {
            "batch_size": 32,
            "decode_tokens": 200,
            "repeat_batches": 3,
            "sampling_mode": "on_device_topk",
            "trace_mode": "decode_only",
            "prefill_trace_mode": "eager",
        },
        "_EVAL32_TARGET_PROVENANCE": {
            "performance": {
                "P150x4": {
                    "batch_size": 32,
                    "seq_len": 512,
                    "decode_tokens": 200,
                    "repeat_batches": 3,
                    "sampling_mode": "on_device_topk",
                    "trace_mode": "decode_only",
                    "prefill_trace_mode": "eager",
                    "source": "reviewed-test-artifact",
                }
            }
        },
        "logger": logger,
    }
    exec(compile(ast.Module(body=[resolve_function], type_ignores=[]), _DEMO_PATH, "exec"), incomplete_namespace)
    assert (
        incomplete_namespace["_resolve_eval32_perf_targets"](
            "meta-llama/Llama-3.3-70B-Instruct", "P150x4", "performance"
        )
        is None
    )

    bad_provenance_namespace = {
        "resolve_perf_targets": lambda *args, **kwargs: {
            "decode_t/s/u": 10.0,
            "prefill_time_to_first_token": 100.0,
        },
        "_EVAL32_FIXED_PROVENANCE": incomplete_namespace["_EVAL32_FIXED_PROVENANCE"],
        "_EVAL32_TARGET_PROVENANCE": {
            "accuracy": {
                "P150x4": {
                    "batch_size": 32,
                    "seq_len": 512,
                    "decode_tokens": 200,
                    "repeat_batches": 3,
                    "sampling_mode": "host",
                    "trace_mode": "decode_only",
                    "prefill_trace_mode": "eager",
                    "source": "reviewed-test-artifact",
                }
            }
        },
        "logger": logger,
    }
    exec(
        compile(ast.Module(body=[resolve_function], type_ignores=[]), _DEMO_PATH, "exec"),
        bad_provenance_namespace,
    )
    with expect_error(ValueError, "Invalid accuracy eval-32 perf provenance.*sampling_mode"):
        bad_provenance_namespace["_resolve_eval32_perf_targets"](
            "meta-llama/Llama-3.3-70B-Instruct", "P150x4", "accuracy"
        )

    resolver_calls = []
    good_namespace = {
        "resolve_perf_targets": lambda *args, **kwargs: (
            resolver_calls.append((args, kwargs)) or {"decode_t/s/u": 10.0, "prefill_time_to_first_token": 100.0}
        ),
        "_EVAL32_FIXED_PROVENANCE": incomplete_namespace["_EVAL32_FIXED_PROVENANCE"],
        "_EVAL32_TARGET_PROVENANCE": incomplete_namespace["_EVAL32_TARGET_PROVENANCE"],
        "logger": logger,
    }
    exec(compile(ast.Module(body=[resolve_function], type_ignores=[]), _DEMO_PATH, "exec"), good_namespace)
    assert good_namespace["_resolve_eval32_perf_targets"](
        "meta-llama/Llama-3.3-70B-Instruct", "P150x4", "performance"
    ) == {"decode_t/s/u": 10.0, "prefill_time_to_first_token": 100.0}
    assert resolver_calls == [
        (
            ("meta-llama/Llama-3.3-70B-Instruct", "P150x4"),
            {"batch_size": 32, "seq_len": 512},
        )
    ]

    assert_namespace = {
        "resolve_metric_tolerance": resolve_metric_tolerance,
        "PERF_TOLERANCE": 0.05,
    }
    assert_function = _function("_assert_eval32_perf_target")
    exec(compile(ast.Module(body=[assert_function], type_ignores=[]), _DEMO_PATH, "exec"), assert_namespace)
    result = SimpleNamespace(tok_s_u=1.0, ttft_ms=1_000.0)
    expected = {"decode_t/s/u": 10.0, "prefill_time_to_first_token": 100.0}
    with expect_error(AssertionError, "tok/s/u.*ttft_ms"):
        assert_namespace["_assert_eval32_perf_target"](result, expected, case_name="BH/eval")


def test_local_perf_nodes_observe_without_floor_and_enforce_complete_floor():
    warnings = []
    namespace = {"logger": SimpleNamespace(warning=warnings.append)}
    function = _function("_resolve_local_perf_target")
    exec(compile(ast.Module(body=[function], type_ignores=[]), _DEMO_PATH, "exec"), namespace)
    assert namespace["_resolve_local_perf_target"]({}, case_name="BH/batch-32-ci") == {}
    assert "observationally without an acceptance claim" in warnings[-1]
    complete = {"tok_s_u": 10.0, "ttft_ms": 100.0}
    assert namespace["_resolve_local_perf_target"](complete, case_name="WH/batch-32") is complete

    perf_source = ast.unparse(_function("_run_perf_benchmark"))
    assert "if expected" in perf_source
    assert "assert not failures" in perf_source


def test_eval_perf_preflight_applies_to_every_sku_and_canonical_sampling_is_early(expect_error):
    preflight_source = ast.unparse(_function("_preflight_perf_target"))
    assert "if test_config == 'eval-32-perf-report'" in preflight_source
    assert "return _resolve_local_perf_target(expected, case_name=case_name)" in preflight_source

    helper = _function("_run_eval_repeat_batch32")
    config_guard = _calls("_run_eval_repeat_batch32", "_require_eval_perf_report_configuration")[0]
    tokenizer = next(
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Assign) and ast.unparse(node.value) == "model.demo_tokenizer"
    )
    assert config_guard.lineno < tokenizer.lineno
    config_source = ast.unparse(_function("_require_eval_perf_report_configuration"))
    assert "sampling_mode != 'on_device_topk'" in config_source
    assert "decode_tokens != _EVAL32_FIXED_PROVENANCE['decode_tokens']" in config_source

    config_namespace = {
        "require_canonical_eval_modes_in_ci": lambda environ: None,
        "_EVAL32_FIXED_PROVENANCE": {"decode_tokens": 200},
    }
    exec(
        compile(
            ast.Module(body=[_function("_require_eval_perf_report_configuration")], type_ignores=[]),
            _DEMO_PATH,
            "exec",
        ),
        config_namespace,
    )
    config_namespace["_require_eval_perf_report_configuration"]({})
    with expect_error(ValueError, "SAMPLING_MODE=on_device_topk"):
        config_namespace["_require_eval_perf_report_configuration"]({"SAMPLING_MODE": "host"})
    with expect_error(ValueError, "PERF_NUM_DECODE_TOKENS=200"):
        config_namespace["_require_eval_perf_report_configuration"]({"PERF_NUM_DECODE_TOKENS": "64"})

    calls = []
    preflight_namespace = {
        "os": SimpleNamespace(environ={}),
        "_require_eval_perf_report_configuration": lambda environ: calls.append(("configuration", environ)),
        "_resolve_eval32_perf_targets": lambda model, device, profile: calls.append(
            ("eval_target", model, device, profile)
        )
        or {"floor": True},
        "_resolve_local_perf_target": lambda expected, case_name: calls.append(("local_target", expected, case_name))
        or expected,
    }
    exec(
        compile(ast.Module(body=[_function("_preflight_perf_target")], type_ignores=[]), _DEMO_PATH, "exec"),
        preflight_namespace,
    )
    assert preflight_namespace["_preflight_perf_target"](
        test_config="eval-32-perf-report",
        optimization_profile="performance",
        device_name="T3K",
        hf_model="llama",
        expected={},
    ) == {"floor": True}
    assert calls[:2] == [("configuration", {}), ("eval_target", "llama", "T3K", "performance")]
    assert preflight_namespace["_preflight_perf_target"](
        test_config="batch-32-ci",
        optimization_profile="accuracy",
        device_name="P150x4",
        hf_model="llama",
        expected={"tok_s_u": 1.0, "ttft_ms": 2.0},
    ) == {"tok_s_u": 1.0, "ttft_ms": 2.0}
    assert calls[-1] == (
        "local_target",
        {"tok_s_u": 1.0, "ttft_ms": 2.0},
        "accuracy/batch-32-ci",
    )


def test_prefill_ab_override_does_not_mutate_frozen_model_args():
    assert "model.model_args.disable_batched_prefill = True" not in _DEMO_SOURCE
    assert _DEMO_SOURCE.count("shared prefill runtime reads DISABLE_BATCHED_PREFILL") == 2


def test_shared_special_token_guard_is_used_on_free_running_output():
    assert not any(
        node.name == "assert_no_special_tokens" for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef)
    )
    assert _calls("_run_perf_benchmark", "assert_no_special_tokens")
