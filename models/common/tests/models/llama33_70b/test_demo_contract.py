# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.demos.utils.trace_region_sizes import resolve_trace_region_size

_DEMO_PATH = "models/common/tests/demos/llama33_70b/demo.py"
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
    assert resolve_trace_region_size("llama3.3-70b", "T3K") == 90_000_000
    assert resolve_trace_region_size("llama3.3-70b", "P150x4") == 96_000_000


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


def test_p150x4_token_accuracy_is_observational_until_a_floor_is_approved():
    source = ast.unparse(_function("_run_token_accuracy"))
    assert "device_name == 'P150x4'" in source
    assert "token accuracy is observational" in source


def test_demo_uses_model_owned_runtime_provider_and_shared_helpers():
    imports = [ast.unparse(node) for node in _DEMO_TREE.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert any("models.common.models.llama33_70b.executor" in statement for statement in imports)
    assert any("models.common.models.llama33_70b.hf_adaptor" in statement for statement in imports)
    assert any("models.common.tests.demos.run_helpers" in statement for statement in imports)
    assert all("models.common.models.executor" not in statement for statement in imports)
    assert all("AutoConfig" not in statement and "AutoTokenizer" not in statement for statement in imports)


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


def test_eval_uses_decode_only_trace_and_registers_representative_prefill_eagerly():
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


def test_prefill_ab_override_does_not_mutate_frozen_model_args():
    assert "model.model_args.disable_batched_prefill = True" not in _DEMO_SOURCE
    assert _DEMO_SOURCE.count("shared prefill runtime reads DISABLE_BATCHED_PREFILL") == 2


def test_shared_special_token_guard_is_used_on_free_running_output():
    assert not any(
        node.name == "assert_no_special_tokens" for node in _DEMO_TREE.body if isinstance(node, ast.FunctionDef)
    )
    assert _calls("_run_perf_benchmark", "assert_no_special_tokens")
