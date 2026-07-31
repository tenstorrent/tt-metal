# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage-02 correctness, trace, topology, and performance coverage."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import pytest
import ttnn

import models.autoports.google_gemma_4_26b_a4b_it.tests.test_functional_decoder as functional_tests
import models.autoports.google_gemma_4_26b_a4b_it.tests.test_trace_mutable_buffers as mutable_tests
import models.autoports.google_gemma_4_26b_a4b_it.tt.fused_decoder as fused_decoder_module
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import FunctionalDecoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import HIDDEN_SIZE, NUM_EXPERTS
from models.autoports.google_gemma_4_26b_a4b_it.tt.fused_decoder import FusedDecoder

ARTIFACT_DIR = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/fused_decoder")


class _NoCacheFusion(FusedDecoder):
    _attention_decode = FunctionalDecoder._attention_decode


class _NoGeGLUFusion(FusedDecoder):
    _dense_mlp = FunctionalDecoder._dense_mlp
    _moe_decode_single_user = FunctionalDecoder._moe_decode_single_user


class _NoRouterFold(FusedDecoder):
    _router_weights = FunctionalDecoder._router_weights


class _ResidualInputNormFusion(FusedDecoder):
    def _finish_parallel_ffn(self, residual, hidden_1, hidden_2):
        hidden_states = ttnn.rms_norm(
            hidden_1,
            residual_input_tensor=hidden_2,
            epsilon=self.eps,
            weight=self.weights.post_ff_ln,
            compute_kernel_config=self.correctness_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden_states = ttnn.add(residual, hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self._apply_layer_scalar(hidden_states)


class _NoDenseGeGLUFusion(FusedDecoder):
    _dense_mlp = FunctionalDecoder._dense_mlp


class _NoSparseGeGLUFusion(FusedDecoder):
    _moe_decode_single_user = FunctionalDecoder._moe_decode_single_user


class _ForceFullCacheFusion(FusedDecoder):
    _attention_decode = FusedDecoder._attention_decode_with_fused_cache


class _BatchRoutingRowMajor(FusedDecoder):
    """Convert all independent routing rows once before per-row sparse MM."""

    def _moe_decode(self, hidden_states, routing_weights):
        batch = hidden_states.shape[2]
        if batch == 1:
            return self._moe_decode_single_user(hidden_states, routing_weights)

        routing_weights = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)
        outputs = []
        for batch_index in range(batch):
            hidden_row = ttnn.slice(
                hidden_states,
                [0, 0, batch_index, 0],
                [1, 1, batch_index + 1, HIDDEN_SIZE],
            )
            routing_row = ttnn.slice(
                routing_weights,
                [0, 0, batch_index, 0],
                [1, 1, batch_index + 1, NUM_EXPERTS],
            )
            outputs.append(self._moe_decode_single_user(hidden_row, routing_row))
        return ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class _BatchedSparseMoE(FusedDecoder):
    """Exercise all independent serving routing rows in three sparse calls."""

    def _moe_decode(self, hidden_states, routing_weights):
        batch = hidden_states.shape[2]
        if batch == 1:
            return FunctionalDecoder._moe_decode_single_user(self, hidden_states, routing_weights)
        return self._moe_decode_batched(hidden_states, routing_weights, batch)

    def _moe_decode_batched(self, hidden_states, routing_weights, batch):
        output_tile = ttnn.Tile([fused_decoder_module.TILE_SIZE, fused_decoder_module.TILE_SIZE])
        hidden_batched = ttnn.reshape(hidden_states, (batch, 1, 1, HIDDEN_SIZE))
        sparsity_down = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)
        sparsity_gate_up = ttnn.reshape(sparsity_down, (batch, 1, 1, NUM_EXPERTS))
        nnz = batch * fused_decoder_module.TOP_K_EXPERTS
        gate_up_config = fused_decoder_module._build_sparse_matmul_config(1, fused_decoder_module.MOE_INTERMEDIATE_SIZE)
        down_config = fused_decoder_module._build_sparse_matmul_config(1, HIDDEN_SIZE)

        gate = ttnn.sparse_matmul(
            hidden_batched,
            self.weights.expert_gate,
            sparsity=sparsity_gate_up,
            nnz=nnz,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=gate_up_config,
            dtype=self.activation_dtype,
            compute_kernel_config=self.correctness_compute_config,
        )
        sparse_intermediate = gate.shape[-1]
        gate = ttnn.reshape(gate, (batch, NUM_EXPERTS, 1, sparse_intermediate))
        up = ttnn.sparse_matmul(
            hidden_batched,
            self.weights.expert_up,
            sparsity=sparsity_gate_up,
            nnz=nnz,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=gate_up_config,
            dtype=self.activation_dtype,
        )
        up = ttnn.reshape(up, (batch, NUM_EXPERTS, 1, sparse_intermediate))
        down_input = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[self.GELU_APPROX],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        down = ttnn.sparse_matmul(
            down_input,
            self.weights.expert_down,
            sparsity=sparsity_down,
            nnz=nnz,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=down_config,
            dtype=self.activation_dtype,
        )
        next_states = ttnn.reshape(down, (batch, NUM_EXPERTS, HIDDEN_SIZE))
        routing_3d = ttnn.reshape(routing_weights, (batch, NUM_EXPERTS, 1))
        next_states = ttnn.mul(next_states, routing_3d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        next_states = ttnn.sum(next_states, dim=1)
        next_states = ttnn.unsqueeze_to_4D(next_states)
        return ttnn.reshape(
            next_states,
            (1, 1, batch, HIDDEN_SIZE),
            (1, 1, max(fused_decoder_module.TILE_SIZE, batch), HIDDEN_SIZE),
        )


class _ChunkedBatchedSparseMoE(_BatchedSparseMoE):
    CHUNK_SIZE = 2

    def _moe_decode(self, hidden_states, routing_weights):
        batch = hidden_states.shape[2]
        if batch == 1:
            return FunctionalDecoder._moe_decode_single_user(self, hidden_states, routing_weights)
        outputs = []
        for start in range(0, batch, self.CHUNK_SIZE):
            end = min(start + self.CHUNK_SIZE, batch)
            hidden_chunk = ttnn.slice(hidden_states, [0, 0, start, 0], [1, 1, end, HIDDEN_SIZE])
            routing_chunk = ttnn.slice(routing_weights, [0, 0, start, 0], [1, 1, end, NUM_EXPERTS])
            outputs.append(self._moe_decode_batched(hidden_chunk, routing_chunk, end - start))
        return ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class _ChunkedBatchedSparseMoE4(_ChunkedBatchedSparseMoE):
    CHUNK_SIZE = 4


class _ChunkedBatchedSparseMoE8(_ChunkedBatchedSparseMoE):
    CHUNK_SIZE = 8


def _route_shared_runner_to_fused(monkeypatch, *, decoder_class=FusedDecoder) -> None:
    """Reuse the accepted functional oracle while forcing the stage-02 class."""

    monkeypatch.setattr(functional_tests, "FunctionalDecoder", decoder_class)
    monkeypatch.setattr(functional_tests, "ARTIFACT_DIR", ARTIFACT_DIR)
    monkeypatch.setenv(
        "GEMMA4_EVIDENCE_WRAPPER_PATH",
        str(Path(__file__).resolve()),
    )
    monkeypatch.setenv(
        "GEMMA4_EVIDENCE_FUSED_DECODER_PATH",
        str(Path(fused_decoder_module.__file__).resolve()),
    )
    if not os.getenv("GEMMA4_EVIDENCE_COMMAND"):
        monkeypatch.setenv(
            "GEMMA4_EVIDENCE_COMMAND",
            "GEMMA4_RANGE_DOWNLOAD=1 pytest -q "
            "models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py",
        )


def test_fused_decoder_is_not_a_functional_fallback():
    assert FusedDecoder is not FunctionalDecoder
    assert FusedDecoder.decode_forward is not FunctionalDecoder.decode_forward
    assert FusedDecoder._prefill_forward_single_user is not FunctionalDecoder._prefill_forward_single_user
    source = inspect.getsource(FusedDecoder)
    for required in (
        "input_tensor_a_activations=[self.GELU_APPROX]",
        "ttnn.experimental.paged_fused_update_cache",
        "self.fused_router_proj",
    ):
        assert required in source


def test_fused_decoder_hot_path_fallback_audit():
    forbidden = ("torch.", "import torch", "ttnn.from_torch", "ttnn.to_torch")
    methods = (
        FusedDecoder._prefill_forward_single_user,
        FusedDecoder.decode_forward,
        FusedDecoder._dense_mlp,
        FusedDecoder._router_weights,
        FusedDecoder._finish_parallel_ffn,
    )
    source = "\n".join(inspect.getsource(method) for method in methods)
    for token in forbidden:
        assert token not in source


def test_fused_decoder_preserves_context_contract():
    contract = json.loads(Path("models/autoports/google_gemma_4_26b_a4b_it/doc/context_contract.json").read_text())
    assert contract["current_supported_context"] == contract["hf_advertised_context"] == 262144
    assert contract["capability_reduction"] is None
    assert contract["functional_decoder"]["decode"]["serving_batch"] == 32


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,shared_physical,decode_pcc,expect_fused_cache_update",
    [
        pytest.param(0, True, 0.995, True, id="sliding_attention_shared_cache"),
        pytest.param(5, False, 0.995, True, id="full_attention_natural_cache"),
        pytest.param(5, True, 0.995, False, id="full_attention_shared_cache_view"),
    ],
)
def test_fused_decoder_real_weights_prefill_decode(
    monkeypatch,
    mesh_device,
    device_params,
    layer_idx,
    shared_physical,
    decode_pcc,
    expect_fused_cache_update,
):
    _route_shared_runner_to_fused(monkeypatch)
    fused_update_calls = 0
    real_fused_update = ttnn.experimental.paged_fused_update_cache

    def counted_fused_update(*args, **kwargs):
        nonlocal fused_update_calls
        fused_update_calls += 1
        return real_fused_update(*args, **kwargs)

    monkeypatch.setattr(ttnn.experimental, "paged_fused_update_cache", counted_fused_update)
    functional_tests.test_functional_decoder_real_weights_prefill_decode(
        mesh_device,
        device_params,
        layer_idx,
        shared_physical,
        decode_pcc,
    )
    assert bool(fused_update_calls) is expect_fused_cache_update


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
def test_fused_decoder_traced_decode_batch_contract(
    monkeypatch,
    mesh_device,
    device_params,
    layer_idx,
    batch,
):
    _route_shared_runner_to_fused(monkeypatch)
    functional_tests.test_traced_decode_batch_contract(mesh_device, device_params, layer_idx, batch)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_fused_decoder_non_aligned_logical_lengths(
    monkeypatch,
    mesh_device,
    device_params,
    layer_idx,
):
    _route_shared_runner_to_fused(monkeypatch)
    functional_tests.test_paged_prefill_logical_boundary_lengths(mesh_device, device_params, layer_idx)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_fused_decoder_bounded_modulo_tail_integrity(monkeypatch, mesh_device, device_params):
    _route_shared_runner_to_fused(monkeypatch)
    functional_tests.test_bounded_modulo_prefill_tail_cache_integrity(mesh_device, device_params)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention_shared_hma"])
def test_fused_decoder_trace_mutable_stable_buffers(
    monkeypatch,
    mesh_device,
    device_params,
    layer_idx,
):
    monkeypatch.setattr(mutable_tests, "FunctionalDecoder", FusedDecoder)
    monkeypatch.setattr(mutable_tests, "ARTIFACT_DIR", ARTIFACT_DIR)
    mutable_tests.test_trace_mutable_stable_buffers(mesh_device, device_params, layer_idx)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_batched_sparse_moe_candidate_trace_batch32(
    monkeypatch,
    mesh_device,
    device_params,
    layer_idx,
):
    if os.getenv("GEMMA4_BATCHED_SPARSE_MOE_CANDIDATE") != "1":
        pytest.skip("set GEMMA4_BATCHED_SPARSE_MOE_CANDIDATE=1 to run the isolated candidate")
    _route_shared_runner_to_fused(monkeypatch, decoder_class=_BatchedSparseMoE)
    monkeypatch.setenv("GEMMA4_EVIDENCE_VARIANT", "batched_sparse_moe")
    functional_tests.test_traced_decode_batch_contract(mesh_device, device_params, layer_idx, 32)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
def test_full_cache_fusion_candidate_correctness(monkeypatch, mesh_device, device_params):
    if os.getenv("GEMMA4_FULL_CACHE_FUSION_CANDIDATE") != "1":
        pytest.skip("set GEMMA4_FULL_CACHE_FUSION_CANDIDATE=1 to run the isolated candidate")
    _route_shared_runner_to_fused(monkeypatch, decoder_class=_ForceFullCacheFusion)
    monkeypatch.setenv("GEMMA4_EVIDENCE_VARIANT", "force_full_cache_fusion")
    candidate_dir = ARTIFACT_DIR / "candidates" / "force_full_cache_fusion"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(functional_tests, "ARTIFACT_DIR", candidate_dir)
    functional_tests.test_functional_decoder_real_weights_prefill_decode(
        mesh_device,
        device_params,
        5,
        False,
        0.995,
    )
    functional_tests.test_traced_decode_batch_contract(mesh_device, device_params, 5, 1)
    functional_tests.test_traced_decode_batch_contract(mesh_device, device_params, 5, 32)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,shared_physical",
    [
        pytest.param(0, True, id="sliding_attention_1024"),
        pytest.param(5, False, id="full_attention_1024"),
    ],
)
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
def test_fused_decoder_perf_profile(
    monkeypatch,
    mesh_device,
    device_params,
    layer_idx,
    shared_physical,
    batch,
):
    if os.getenv("GEMMA4_FUSED_DECODER_PERF") != "1":
        pytest.skip("set GEMMA4_FUSED_DECODER_PERF=1 to run the profiler harness")
    variant = os.getenv("GEMMA4_FUSED_DECODER_VARIANT", "fused")
    decoder_class = {
        "functional": FunctionalDecoder,
        "fused": FusedDecoder,
        "no_cache_fusion": _NoCacheFusion,
        "no_geglu_fusion": _NoGeGLUFusion,
        "no_router_fold": _NoRouterFold,
        "residual_input_norm_fusion": _ResidualInputNormFusion,
        "no_dense_geglu_fusion": _NoDenseGeGLUFusion,
        "no_sparse_geglu_fusion": _NoSparseGeGLUFusion,
        "force_full_cache_fusion": _ForceFullCacheFusion,
        "batch_routing_row_major": _BatchRoutingRowMajor,
        "batched_sparse_moe": _BatchedSparseMoE,
        "batched_sparse_moe_chunk2": _ChunkedBatchedSparseMoE,
        "batched_sparse_moe_chunk4": _ChunkedBatchedSparseMoE4,
        "batched_sparse_moe_chunk8": _ChunkedBatchedSparseMoE8,
    }[variant]
    external_evidence_command = os.getenv("GEMMA4_EVIDENCE_COMMAND")
    _route_shared_runner_to_fused(monkeypatch, decoder_class=decoder_class)
    monkeypatch.setenv("GEMMA4_EVIDENCE_VARIANT", variant)
    if external_evidence_command:
        monkeypatch.setenv("GEMMA4_EVIDENCE_COMMAND", external_evidence_command)
    else:
        monkeypatch.setenv(
            "GEMMA4_EVIDENCE_COMMAND",
            "GEMMA4_FUSED_DECODER_PERF=1 "
            f"GEMMA4_FUSED_DECODER_VARIANT={variant} "
            f"GEMMA4_DECODER_PERF_REPEATS={os.getenv('GEMMA4_DECODER_PERF_REPEATS', '1')} "
            "GEMMA4_RANGE_DOWNLOAD=1 pytest -q "
            "models/autoports/google_gemma_4_26b_a4b_it/tests/"
            "test_fused_decoder.py::test_fused_decoder_perf_profile",
        )
    monkeypatch.setenv("GEMMA4_FUNCTIONAL_DECODER_PERF", "1")
    functional_tests.test_functional_decoder_perf_profile(
        mesh_device,
        device_params,
        layer_idx,
        shared_physical,
        batch,
    )
    source = ARTIFACT_DIR / (
        f"layer{layer_idx}_{functional_tests._load_text_config().layer_types[layer_idx]}"
        f"_seq{os.getenv('GEMMA4_FUNCTIONAL_DECODER_SEQ_LEN', '1024')}_batch{batch}_host_timings.json"
    )
    source.rename(source.with_name(source.stem + f"_{variant}.json"))
