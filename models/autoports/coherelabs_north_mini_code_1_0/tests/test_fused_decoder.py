# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Stage-02 regression suite for the graph-fused North-Mini decoder."""

from __future__ import annotations

import inspect

import pytest

import models.autoports.coherelabs_north_mini_code_1_0.tests.test_functional_decoder as functional_tests
from models.autoports.coherelabs_north_mini_code_1_0.tt.fused_decoder import FusedDecoder


def _select_fused(monkeypatch):
    # Functional tests intentionally resolve their decoder constructor through
    # this module global.  The temporary substitution reuses the exact accepted
    # semantic/reference checks while every device forward dispatches
    # FusedDecoder.
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", FusedDecoder)


def test_fused_path_source_and_contract_audit():
    assert FusedDecoder._dense_mlp.__module__.endswith(".fused_decoder")
    assert FusedDecoder._attention_decode.__module__.endswith(".fused_decoder")
    assert FusedDecoder._sparse_moe_chunk.__module__.endswith(".fused_decoder")
    assert FusedDecoder._packed_all_expert_moe.__module__.endswith(".fused_decoder")
    runtime_source = "\n".join(
        inspect.getsource(method)
        for method in (
            FusedDecoder._dense_mlp,
            FusedDecoder._attention_decode,
            FusedDecoder._sparse_moe_chunk,
            FusedDecoder._packed_all_expert_moe,
            FusedDecoder._fused_swiglu,
        )
    )
    assert "paged_fused_update_cache" in runtime_source
    assert "deepseek_moe_fast_reduce_nc_fused" in runtime_source
    assert "ttnn.sparse_matmul" in runtime_source
    assert "input_tensor_a_activations=[ttnn.UnaryOpType.SILU]" in runtime_source
    assert 'weights["gate_proj"]' not in runtime_source
    assert 'weights["expert_gate"]' not in runtime_source
    for forbidden in ("import torch", "from_torch", "to_torch", "super()._"):
        assert forbidden not in runtime_source


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [1, 31, 32, 33, 65])
def test_fused_dense_non_aligned_prefill(monkeypatch, mesh_device, seq_len):
    _select_fused(monkeypatch)
    functional_tests.test_dense_paged_prefill_non_aligned_matches_reference(mesh_device, seq_len)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_fused_dense_paged_trace_replay(monkeypatch, mesh_device):
    _select_fused(monkeypatch)
    functional_tests.test_dense_paged_decode_trace_replay_matches_reference(mesh_device)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_fused_dense_serving_batch_32_trace_replay(monkeypatch, mesh_device):
    _select_fused(monkeypatch)
    functional_tests.test_serving_batch_32_paged_decode_trace_replay_matches_reference(mesh_device)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_fused_permuted_cache_and_determinism(monkeypatch, mesh_device):
    _select_fused(monkeypatch)
    functional_tests.test_batch_two_prefill_and_permuted_physical_cache(mesh_device)
    functional_tests.test_random_nonzero_decode_positions_update_expected_physical_slots_and_are_deterministic(
        mesh_device
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", functional_tests.REPRESENTATIVE_LAYERS)
def test_fused_representative_layer_kinds(monkeypatch, mesh_device, layer_idx):
    _select_fused(monkeypatch)
    functional_tests.test_every_meaningful_layer_kind_executes(mesh_device, layer_idx)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,sequence,selected_tokens",
    [(1, 1025, [0, 1023, 1024]), (4, 33, [0, 16, 32])],
)
def test_fused_nonzero_sparse_prefill(
    monkeypatch,
    mesh_device,
    layer_idx,
    sequence,
    selected_tokens,
):
    _select_fused(monkeypatch)
    functional_tests.test_nonzero_sparse_prefill_matches_active_expert_reference(
        mesh_device,
        layer_idx,
        sequence,
        selected_tokens,
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_fused_sliding_window_and_dynamic_history(monkeypatch, mesh_device):
    _select_fused(monkeypatch)
    functional_tests.test_sliding_window_boundary_4097_matches_controlled_reference(mesh_device)
    functional_tests.test_sliding_moe_populated_history_dynamic_trace_replay_matches_reference(mesh_device)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx,batch", [(1, 1), (1, 32), (4, 1)])
def test_fused_nonzero_sparse_traced_decode(monkeypatch, mesh_device, layer_idx, batch):
    _select_fused(monkeypatch)
    functional_tests.test_nonzero_sparse_dynamic_trace_replay_matches_reference(mesh_device, layer_idx, batch)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_fused_real_weight_sliding_moe_decode(monkeypatch, mesh_device):
    _select_fused(monkeypatch)
    functional_tests.test_real_weight_sliding_moe_decode_matches_reference(mesh_device)
