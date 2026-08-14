# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from models.common.tests.demos.llama3_8b.demo_utils import assert_seeded_cross_cardinality_consistency
from models.demos.utils.trace_region_sizes import resolve_trace_region_size

_DEMO_SOURCE = Path("models/common/tests/demos/llama3_8b/demo.py").read_text(encoding="utf-8")


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
    assert '_BH_CROSS_CARDINALITIES = (1, 2, 4, 32)' in _DEMO_SOURCE
    assert 'device_name not in {"P150", "P150x4"}' in _DEMO_SOURCE
    assert "_BH_CROSS_CARDINALITY_SEEDS" in _DEMO_SOURCE
    assert "def _install_cross_cardinality_device_seeds(llm, request_indexes)" in _DEMO_SOURCE
    assert "seeds_buffer" in _DEMO_SOURCE and "update(torch.tensor" in _DEMO_SOURCE
    assert "allow_batched_prefill_with_device_sampling_for_diagnostics=allow_batched_prefill" in _DEMO_SOURCE
    assert "allow_batched_prefill=True" in _DEMO_SOURCE
    assert '("DISABLE_BATCHED_PREFILL", "DISABLE_BATCHED_EXTRACT")' in _DEMO_SOURCE
    assert "not a serving policy" in _DEMO_SOURCE


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

    assert_seeded_cross_cardinality_consistency(outputs, controls, request_ids=request_ids)


@pytest.mark.parametrize("failure", ["token", "missing_cardinality", "wrong_request_order", "empty"])
def test_seeded_cross_cardinality_contract_fails_closed(failure):
    request_ids, controls, outputs = _valid_cross_cardinality_outputs()
    if failure == "token":
        outputs[32][request_ids[0]][1] += 1
    elif failure == "missing_cardinality":
        del outputs[4]
    elif failure == "wrong_request_order":
        first, second = tuple(outputs[2])
        outputs[2] = {second: outputs[2][second], first: outputs[2][first]}
    else:
        outputs[1][request_ids[0]] = []

    with pytest.raises(AssertionError):
        assert_seeded_cross_cardinality_consistency(outputs, controls, request_ids=request_ids)
