# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.common.tests.demos.llama3_8b.demo_utils import evaluate_seeded_cross_cardinality_consistency
from models.demos.utils.trace_region_sizes import resolve_trace_region_size

_DEMO_PATH = "models/common/tests/demos/llama3_8b/demo.py"
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


def test_demo_exposes_p300_as_ring_two_chip_mesh():
    assert '"P300": (1, 2)' in _DEMO_SOURCE
    assert 'mesh_device_name in {"P300", "P150X4"}' in _DEMO_SOURCE
    assert "ttnn.FabricConfig.FABRIC_1D_RING" in _DEMO_SOURCE


def test_demo_exposes_p150x4_as_ring_four_chip_mesh():
    assert '"P150X4": (1, 4)' in _DEMO_SOURCE
    assert 'mesh_device_name in {"P300", "P150X4"}' in _DEMO_SOURCE


def test_demo_keeps_p300_dp2_case_in_manifest():
    assert '"ci-b1-DP-2": DemoCase(' in _DEMO_SOURCE


def test_p150_batch32_uses_dynamic_trace_allocation():
    assert 'resolve_trace_region_size("llama3.1-8b", mesh_device_name)' in _DEMO_SOURCE
    assert resolve_trace_region_size("llama3.1-8b", "P150") == 0


def test_demo_exposes_seeded_bh_cross_cardinality_qualification_node():
    assert "def test_llama3_8b_bh_seeded_cross_cardinality(ttnn_mesh_device, optimizations):" in _DEMO_SOURCE
    assert '@pytest.mark.parametrize("optimizations", ["performance", "accuracy"])' in _DEMO_SOURCE
    assert "_BH_CROSS_CARDINALITIES = (1, 2, 4, 32)" in _DEMO_SOURCE
    assert 'device_name not in {"P150", "P150x4"}' in _DEMO_SOURCE
    assert "_BH_CROSS_CARDINALITY_SEEDS" in _DEMO_SOURCE
    assert "_install_cross_cardinality_device_seeds" not in _DEMO_SOURCE
    assert "prefill_sampling_params=None" in _DEMO_SOURCE
    assert "DecodeRuntime from SamplingParams.seed" in _DEMO_SOURCE
    assert "allow_batched_prefill_with_device_sampling_for_diagnostics=allow_batched_prefill" in _DEMO_SOURCE
    assert "allow_batched_prefill=True" in _DEMO_SOURCE
    assert '("DISABLE_BATCHED_PREFILL", "DISABLE_BATCHED_EXTRACT")' in _DEMO_SOURCE
    assert "not a serving policy" in _DEMO_SOURCE
    assert "LLAMA3_8B_CROSS_CARDINALITY_VERDICT=" in _DEMO_SOURCE
    assert "llm.runtime_config.disable_batched_prefill is True" in _DEMO_SOURCE


def test_missing_or_incomplete_performance_targets_do_not_block_measurement_on_bh():
    warnings = []
    namespace = {
        "logger": SimpleNamespace(warning=warnings.append),
    }
    exec(
        compile(ast.Module(body=[_function("_expected_for_case")], type_ignores=[]), _DEMO_PATH, "exec"),
        namespace,
    )

    assert namespace["_expected_for_case"]({}, "batch-1", device_name="P150") is None
    assert (
        namespace["_expected_for_case"](
            {"batch-32": {"tok_s_u": 1.0}},
            "batch-32",
            device_name="P150x4",
        )
        is None
    )
    assert len(warnings) == 2
    assert "missing tok_s_u, ttft_ms" in warnings[0]
    assert "Running on P150 without an in-test performance gate" in warnings[0]
    assert "missing ttft_ms" in warnings[1]
    assert "Running on P150x4 without an in-test performance gate" in warnings[1]


def test_performance_target_preflight_preserves_wormhole_missing_target_semantics_and_accepts_valid_targets():
    warnings = []
    namespace = {
        "logger": SimpleNamespace(warning=warnings.append),
    }
    exec(
        compile(ast.Module(body=[_function("_expected_for_case")], type_ignores=[]), _DEMO_PATH, "exec"),
        namespace,
    )

    assert namespace["_expected_for_case"]({}, "batch-1", device_name="N150") is None
    assert warnings and "Running on N150 without an in-test performance gate" in warnings[0]
    assert namespace["_expected_for_case"](
        {"batch-32": {"tok_s_u": 12.5, "ttft_ms": 150.0, "unused": 1}},
        "batch-32",
        device_name="P150",
    ) == {"tok_s_u": 12.5, "ttft_ms": 150.0}


def test_performance_target_preflight_runs_before_model_construction():
    preflight = _calls("test_llama3_8b", "_expected_for_case")
    create = _calls("test_llama3_8b", "create_llama3_for_causal_lm")
    assert len(preflight) == 1
    assert len(create) == 1
    assert preflight[0].lineno < create[0].lineno
    assert "case_performance_expected" in ast.unparse(_function("test_llama3_8b"))


def test_dp_smoke_loads_one_converted_state_dict_for_every_lane():
    function = _function("_run_dp_smoke")
    loads = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_load_dp_converted_state_dict"
    ]
    creates = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "create_llama3_for_causal_lm"
    ]

    assert len(loads) == 1
    assert len(creates) == 1
    assert loads[0].lineno < creates[0].lineno
    converted = next(keyword for keyword in creates[0].keywords if keyword.arg == "converted_state_dict")
    assert ast.unparse(converted.value) == "converted_state_dict"


def test_supplied_performance_targets_fail_on_any_miss_and_accept_all_passes(expect_error):
    namespace = {"PERF_TOLERANCE": 0.05}
    exec(
        compile(ast.Module(body=[_function("_assert_performance_targets")], type_ignores=[]), _DEMO_PATH, "exec"),
        namespace,
    )
    expected = {"tok_s_u": 10.0, "ttft_ms": 100.0}
    passed = SimpleNamespace(
        tok_s_u=10.0,
        ttft_ms=100.0,
        meets_target=lambda targets, tolerance: {"tok_s_u": True, "ttft_ms": True},
    )
    namespace["_assert_performance_targets"](passed, expected, case_name="performance/batch-32")

    failed = SimpleNamespace(
        tok_s_u=9.0,
        ttft_ms=120.0,
        meets_target=lambda targets, tolerance: {"tok_s_u": False, "ttft_ms": False},
    )
    with expect_error(AssertionError, "tok_s_u.*ttft_ms"):
        namespace["_assert_performance_targets"](failed, expected, case_name="performance/batch-32")

    report_source = ast.unparse(_function("_report_performance"))
    assert "_assert_performance_targets(result, expected, case_name=case_name)" in report_source
    assert "logger.warning" not in report_source


def _valid_cross_cardinality_outputs():
    request_ids = tuple(f"request-{index}" for index in range(32))
    controls = {request_id: [index, index + 1] for index, request_id in enumerate(request_ids)}
    outputs = {
        cardinality: {request_id: list(controls[request_id]) for request_id in request_ids[:cardinality]}
        for cardinality in (1, 2, 4, 32)
    }
    return request_ids, controls, outputs


def test_seeded_cross_cardinality_contract_accepts_exact_token_matches():
    request_ids, controls, outputs = _valid_cross_cardinality_outputs()

    verdict, mismatches = evaluate_seeded_cross_cardinality_consistency(
        outputs, controls, request_ids=request_ids, expected_token_count=2
    )
    assert verdict == "INVARIANT"
    assert mismatches == ()


def test_seeded_cross_cardinality_contract_records_complete_token_mismatch_as_rejection():
    request_ids, controls, outputs = _valid_cross_cardinality_outputs()
    outputs[32][request_ids[0]][1] += 1

    verdict, mismatches = evaluate_seeded_cross_cardinality_consistency(
        outputs, controls, request_ids=request_ids, expected_token_count=2
    )

    assert verdict == "BATCHED_PREFILL_REJECTED"
    assert mismatches == (
        {
            "cardinality": 32,
            "request_id": request_ids[0],
            "first_token_difference": 1,
            "control_token_count": 2,
            "batched_token_count": 2,
        },
    )


@pytest.mark.parametrize(
    "failure", ["missing_cardinality", "wrong_request_order", "empty", "truncated", "truncated_control"]
)
def test_seeded_cross_cardinality_contract_fails_closed(failure, expect_error):
    request_ids, controls, outputs = _valid_cross_cardinality_outputs()
    if failure == "missing_cardinality":
        del outputs[4]
    elif failure == "wrong_request_order":
        first, second = tuple(outputs[2])
        outputs[2] = {second: outputs[2][second], first: outputs[2][first]}
    elif failure == "empty":
        outputs[1][request_ids[0]] = []
    elif failure == "truncated":
        outputs[32][request_ids[0]] = outputs[32][request_ids[0]][:-1]
    else:
        controls[request_ids[0]] = controls[request_ids[0]][:-1]

    with expect_error(AssertionError, "seeded cross-cardinality|sequential controls|cardinality|returned"):
        evaluate_seeded_cross_cardinality_consistency(
            outputs, controls, request_ids=request_ids, expected_token_count=2
        )
