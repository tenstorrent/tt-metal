# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Optimized-decoder contract and hardware evidence harness."""

import ast
import hashlib
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

import pytest

import models.autoports.google_gemma_4_26b_a4b_it.tests.test_functional_decoder as functional_tests
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import FunctionalDecoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import (
    OptimizedDecoder,
    optimization_candidate_matrix,
)

ARTIFACT_DIR = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder")


def test_optimized_path_owns_public_forwards():
    """Guard against an optimized artifact that merely aliases the functional path."""
    assert OptimizedDecoder is not FunctionalDecoder
    assert OptimizedDecoder.__dict__["prefill_forward"] is not FunctionalDecoder.prefill_forward
    assert OptimizedDecoder.__dict__["decode_forward"] is not FunctionalDecoder.decode_forward
    assert OptimizedDecoder.implementation == "optimized"


def test_optimized_hot_path_has_no_host_fallback():
    tt_dir = Path(__file__).parents[1] / "tt"
    forbidden = {"from_torch", "to_torch", "tilize", "untilize", "reshard"}
    hot_methods = {
        "prefill_forward",
        "_prefill_forward_single_user",
        "_prefill_forward_single_user_optimized",
        "decode_forward",
        "forward",
        "_rms_norm",
        "_apply_layer_scalar",
        "_attention_prefill",
        "_fill_prefill_cache",
        "_full_chunked_prefill_attention",
        "_sliding_chunked_prefill_attention",
        "_attention_decode",
        "_cache_view_kwargs",
        "_dense_mlp",
        "_router_weights",
        "_moe_decode",
        "_moe_decode_single_user",
        "_moe_prefill",
        "_moe_prefill_chunk",
    }
    audited = {}
    for filename in ("optimized_decoder.py", "functional_decoder.py"):
        tree = ast.parse((tt_dir / filename).read_text())
        methods = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in hot_methods
        }
        calls = {
            child.func.attr
            for method in methods.values()
            for child in ast.walk(method)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
        }
        audited[filename] = sorted(methods)
        assert not (calls & forbidden), (filename, calls & forbidden)
    assert "decode_forward" in audited["optimized_decoder.py"]
    assert "_attention_decode" in audited["functional_decoder.py"]


def test_optimized_candidate_matrix_covers_required_sweeps():
    matrix = optimization_candidate_matrix()
    assert set(matrix["decode_batches"]) == {"1", "32"}
    assert all(case["per_core_M"] == 1 for case in matrix["decode_batches"].values())
    assert set(matrix["dense_decode"]) == {"packed_gate_up", "dense_down"}
    assert set(matrix["sparse_decode"]) == {"expert_gate_up", "expert_down"}
    pairs = {(item["weight"], item["fidelity"]) for item in matrix["weight_compute_pairs"]}
    assert {item[0] for item in pairs} >= {"bfloat16", "bfloat8_b", "bfloat4_b"}
    assert ("bfloat8_b", "LoFi") in pairs
    assert ("bfloat4_b", "LoFi") in pairs
    assert {item["dtype"] for item in matrix["kv_cache"]} == {"bfloat16", "bfloat8_b"}
    assert 1023 in matrix["prefill_sequence_lengths"]
    assert "large_multicore_reuse" in matrix["prefill_program_families"]
    assert "dram_sharded_weight_l1_width_sharded_output" in matrix["movement_families"]
    for role in matrix["dense_decode"].values():
        assert role["dram_core_counts"]
        assert any(width > 2 for widths in role["in0_block_w"].values() for width in widths)
    for role in matrix["sparse_decode"].values():
        assert any(width > 2 for width in role["in0_block_w"]["1"])


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _optimized_provenance(mesh_device, exact_command):
    tt_dir = Path(__file__).parents[1] / "tt"
    test_path = Path(__file__)
    return {
        "implementation": OptimizedDecoder.implementation,
        "optimized_decoder_sha256": _sha256(tt_dir / "optimized_decoder.py"),
        "functional_decoder_sha256": _sha256(tt_dir / "functional_decoder.py"),
        "optimized_test_sha256": _sha256(test_path),
        "checkout_git_sha": functional_tests._evidence_provenance(mesh_device, exact_command)["checkout_git_sha"],
        "hardware": functional_tests._evidence_provenance(mesh_device, exact_command)["hardware"],
        "exact_command": exact_command,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _rewrite_optimized_timing_provenance(mesh_device, layer_idx, layer_type, batch, destination=ARTIFACT_DIR):
    case_id = f"layer{layer_idx}_{layer_type}_seq1024_batch{batch}"
    path = destination / f"{case_id}_host_timings.json"
    payload = json.loads(path.read_text())
    command = (
        "GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZED_DECODER_PERF=1 "
        "TTNN_CONFIG_OVERRIDES='{\"throw_exception_on_fallback\": true}' "
        "pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/"
        "test_optimized_decoder.py::test_optimized_decoder_perf_profile "
        f"-k '{layer_type} and batch{batch}'"
    )
    payload["implementation"] = OptimizedDecoder.implementation
    payload["provenance"] = _optimized_provenance(mesh_device, command)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _rewrite_optimized_artifact_provenance(mesh_device, path, exact_command):
    payload = json.loads(path.read_text())
    payload["implementation"] = OptimizedDecoder.implementation
    payload["provenance"] = _optimized_provenance(mesh_device, exact_command)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _select_optimized_path(monkeypatch):
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", OptimizedDecoder)
    destination = Path(os.environ.get("GEMMA4_ADVISOR_ORACLE_DIR", str(ARTIFACT_DIR)))
    monkeypatch.setattr(functional_tests, "ARTIFACT_DIR", destination)


def _select_bfp8_cache_candidate(monkeypatch, artifact_dir):
    """Run the normal layer harness with only its exact rank-5 KV cache in BF8."""
    _select_optimized_path(monkeypatch)
    monkeypatch.setattr(functional_tests, "ARTIFACT_DIR", artifact_dir)
    original_as_tt = functional_tests._as_tt

    def cache_aware_as_tt(
        device,
        tensor,
        *,
        dtype=functional_tests.ttnn.bfloat16,
        layout=functional_tests.ttnn.TILE_LAYOUT,
    ):
        # Cache tensors are the only rank-5 tensors in the functional harness.
        # Keep decode K/V updates BF16: paged_update_cache accepts and converts
        # them into the BF8 destination cache.
        if tensor.ndim == 5 and dtype == functional_tests.ttnn.bfloat16:
            dtype = functional_tests.ttnn.bfloat8_b
        return original_as_tt(device, tensor, dtype=dtype, layout=layout)

    monkeypatch.setattr(functional_tests, "_as_tt", cache_aware_as_tt)


DRAM_DENSE_CANDIDATES = {
    # Same physical output padding in every geometry isolates the number of
    # DRAM banks and K-block width.  Gate/down have different K divisors.
    "dram_sharded_dense_bf16_hifi4_g8d6_w11": {
        "weight_dtype": functional_tests.ttnn.bfloat16,
        "gate_cores": 8,
        "down_cores": 6,
        "gate_in0_block_w": 11,
        "down_in0_block_w": 11,
    },
    "dram_sharded_dense_bfp8_hifi4_g8d6_w11": {
        "weight_dtype": functional_tests.ttnn.bfloat8_b,
        "gate_cores": 8,
        "down_cores": 6,
        "gate_in0_block_w": 11,
        "down_in0_block_w": 11,
    },
    "dram_sharded_dense_bfp8_hifi4_g4d3_w22": {
        "weight_dtype": functional_tests.ttnn.bfloat8_b,
        "gate_cores": 4,
        "down_cores": 3,
        "gate_in0_block_w": 22,
        "down_in0_block_w": 22,
    },
    "dram_sharded_dense_bfp8_hifi4_g2d2_w44x33": {
        "weight_dtype": functional_tests.ttnn.bfloat8_b,
        "gate_cores": 2,
        "down_cores": 2,
        "gate_in0_block_w": 44,
        "down_in0_block_w": 33,
    },
}


def _candidate_decoder(
    name,
    *,
    dense_compute_fidelity=None,
    expert_compute_fidelity=OptimizedDecoder.expert_compute_fidelity,
    **dtype_overrides,
):
    dtype_policy = {
        "attention_weight_dtype": functional_tests.ttnn.bfloat16,
        "dense_weight_dtype": functional_tests.ttnn.bfloat16,
        "expert_weight_dtype": functional_tests.ttnn.bfloat8_b,
    }
    dtype_policy.update(dtype_overrides)

    class CandidateDecoder(OptimizedDecoder):
        candidate_name = name
        optimization_candidate = name
        sparse_in0_block_w = int(name.rsplit("_", 1)[1]) if name.startswith("sparse_in0_block_w_") else 11

        @classmethod
        def from_state_dict(cls, state_dict, **kwargs):
            return super().from_state_dict(state_dict, **kwargs, **dtype_policy)

    CandidateDecoder.__name__ = f"OptimizedDecoder_{name}"
    CandidateDecoder.dense_compute_fidelity = dense_compute_fidelity
    CandidateDecoder.expert_compute_fidelity = expert_compute_fidelity
    if name in DRAM_DENSE_CANDIDATES:
        config = DRAM_DENSE_CANDIDATES[name]
        CandidateDecoder.dram_dense_weight_dtype = config["weight_dtype"]
        CandidateDecoder.dram_dense_gate_cores = config["gate_cores"]
        CandidateDecoder.dram_dense_down_cores = config["down_cores"]
        CandidateDecoder.dram_dense_gate_in0_block_w = config["gate_in0_block_w"]
        CandidateDecoder.dram_dense_down_in0_block_w = config["down_in0_block_w"]
    return CandidateDecoder


WHOLE_LAYER_CANDIDATES = (
    *DRAM_DENSE_CANDIDATES,
    "sparse_in0_block_w_2",
    "sparse_in0_block_w_11",
    "large_prefill_multicore",
)


def test_whole_layer_candidates_share_sparse_control():
    """Prevent cross-family timings from silently changing MoE geometry."""
    for name in WHOLE_LAYER_CANDIDATES:
        expected = int(name.rsplit("_", 1)[1]) if name.startswith("sparse_in0_block_w_") else 11
        assert _candidate_decoder(name).sparse_in0_block_w == expected


def _select_whole_layer_candidate(monkeypatch, candidate):
    candidate_decoder = _candidate_decoder(candidate)
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", candidate_decoder)
    destination = ARTIFACT_DIR / "candidates" / "whole_layer" / candidate
    monkeypatch.setattr(functional_tests, "ARTIFACT_DIR", destination)
    return candidate_decoder, destination


def _write_whole_layer_result(
    mesh_device,
    *,
    candidate,
    layer_idx,
    layer_type,
    batch,
    timing,
    destination,
):
    matrix = optimization_candidate_matrix()
    candidate_contract = {
        "sparse_in0_block_w_2": {
            "family": "sparse_expert_gate_up_down",
            "in0_block_w": 2,
            "nnz": 8,
        },
        "sparse_in0_block_w_11": {
            "family": "sparse_expert_gate_up_down",
            "in0_block_w": 11,
            "nnz": 8,
        },
        "large_prefill_multicore": {
            "family": "large_prefill_multicore_reuse",
            "grid": [6, 8],
            "in0_block_w": 11,
            "out_subblock": [1, 2],
            "per_core_M": 4,
            "per_core_N": 22,
        },
    }
    if candidate in DRAM_DENSE_CANDIDATES:
        config = DRAM_DENSE_CANDIDATES[candidate]
        gate_cores, down_cores = config["gate_cores"], config["down_cores"]
        candidate_contract[candidate] = {
            "family": "dram_sharded_weight_l1_width_sharded_activation_output",
            "dtype": str(config["weight_dtype"]),
            "fidelity": "HiFi4",
            "roles": {
                "packed_gate_up": {
                    "core_count": gate_cores,
                    "logical_n": 4224,
                    "padded_n": 4352,
                    "input_shard": [32, 2816 // gate_cores],
                    "weight_shard": [2816, 4352 // gate_cores],
                    "output_shard": [32, 4352 // gate_cores],
                    "in0_block_w": config["gate_in0_block_w"],
                    "per_core_M": 1,
                    "per_core_N": 136 // gate_cores,
                },
                "dense_down": {
                    "core_count": down_cores,
                    "logical_n": 2816,
                    "padded_n": 3072,
                    "input_shard": [32, 2112 // down_cores],
                    "weight_shard": [2112, 3072 // down_cores],
                    "output_shard": [32, 3072 // down_cores],
                    "in0_block_w": config["down_in0_block_w"],
                    "per_core_M": 1,
                    "per_core_N": 96 // down_cores,
                },
            },
        }
    candidate_contract = candidate_contract[candidate]
    candidate_contract["inherited_sparse_in0_block_w"] = (
        int(candidate.rsplit("_", 1)[1]) if candidate.startswith("sparse_in0_block_w_") else 11
    )
    payload = {
        "model_id": functional_tests.MODEL_ID,
        "candidate": candidate,
        "candidate_contract": candidate_contract,
        "candidate_matrix_sha256": hashlib.sha256(json.dumps(matrix, sort_keys=True).encode()).hexdigest(),
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "batch": batch,
        "logical_m": matrix["decode_batches"][str(batch)]["logical_m"],
        "physical_m": matrix["decode_batches"][str(batch)]["physical_m"],
        "per_core_M": matrix["decode_batches"][str(batch)]["per_core_M"],
        "correctness_artifact": str(
            destination / f"pcc_layer{layer_idx}_{layer_type}_shared{int(layer_idx == 0)}.json"
        ),
        "timing": timing,
        "provenance": _optimized_provenance(
            mesh_device,
            "GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZED_WHOLE_LAYER_SWEEP=1 "
            "pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/"
            "test_optimized_decoder.py -k optimized_whole_layer",
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"layer{layer_idx}_{layer_type}_batch{batch}_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,shared_physical",
    [
        pytest.param(0, True, id="sliding_attention"),
        pytest.param(5, False, id="full_attention_natural_cache"),
        pytest.param(5, True, id="full_attention_shared_physical_cache"),
    ],
)
def test_optimized_real_weights_prefill_decode(monkeypatch, mesh_device, device_params, layer_idx, shared_physical):
    _select_optimized_path(monkeypatch)
    functional_tests.test_functional_decoder_real_weights_prefill_decode(
        mesh_device,
        device_params,
        layer_idx,
        shared_physical,
        0.995,
    )
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    artifact_dir = Path(os.environ.get("GEMMA4_ADVISOR_ORACLE_DIR", str(ARTIFACT_DIR)))
    _rewrite_optimized_artifact_provenance(
        mesh_device,
        artifact_dir / f"pcc_layer{layer_idx}_{layer_type}_shared{int(shared_physical)}.json",
        "GEMMA4_RANGE_DOWNLOAD=1 pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/"
        "test_optimized_decoder.py::test_optimized_real_weights_prefill_decode",
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx,shared_physical", [(0, True), (5, False)])
@pytest.mark.parametrize(
    "candidate,dtypes",
    [
        pytest.param(
            "attention_bfp8",
            {"attention_weight_dtype": functional_tests.ttnn.bfloat8_b},
            id="attention_bfp8",
        ),
        pytest.param(
            "dense_bfp8",
            {"dense_weight_dtype": functional_tests.ttnn.bfloat8_b},
            id="dense_bfp8",
        ),
        pytest.param(
            "expert_bfp8",
            {"expert_weight_dtype": functional_tests.ttnn.bfloat8_b},
            id="expert_bfp8",
        ),
        pytest.param(
            "attention_bfp4",
            {"attention_weight_dtype": functional_tests.ttnn.bfloat4_b},
            id="attention_bfp4",
        ),
        pytest.param(
            "dense_bfp4",
            {"dense_weight_dtype": functional_tests.ttnn.bfloat4_b},
            id="dense_bfp4",
        ),
        pytest.param(
            "expert_bfp4",
            {"expert_weight_dtype": functional_tests.ttnn.bfloat4_b},
            id="expert_bfp4",
        ),
        pytest.param(
            "expert_bfp8_hifi2",
            {
                "expert_weight_dtype": functional_tests.ttnn.bfloat8_b,
                "expert_compute_fidelity": functional_tests.ttnn.MathFidelity.HiFi2,
            },
            id="expert_bfp8_hifi2",
        ),
        pytest.param(
            "expert_bfp8_lofi",
            {
                "expert_weight_dtype": functional_tests.ttnn.bfloat8_b,
                "expert_compute_fidelity": functional_tests.ttnn.MathFidelity.LoFi,
            },
            id="expert_bfp8_lofi",
        ),
        pytest.param(
            "expert_bfp4_lofi",
            {
                "expert_weight_dtype": functional_tests.ttnn.bfloat4_b,
                "expert_compute_fidelity": functional_tests.ttnn.MathFidelity.LoFi,
            },
            id="expert_bfp4_lofi",
        ),
    ],
)
def test_optimized_precision_candidate_real_weights(
    monkeypatch, mesh_device, device_params, layer_idx, shared_physical, candidate, dtypes
):
    if os.getenv("GEMMA4_OPTIMIZED_PRECISION_SWEEP") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_PRECISION_SWEEP=1 to reproduce rejected precision candidates")
    candidate_decoder = _candidate_decoder(candidate, **dtypes)
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", candidate_decoder)
    monkeypatch.setattr(functional_tests, "ARTIFACT_DIR", ARTIFACT_DIR / "candidates" / candidate)
    functional_tests.test_functional_decoder_real_weights_prefill_decode(
        mesh_device,
        device_params,
        layer_idx,
        shared_physical,
        0.995,
    )
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    _rewrite_optimized_artifact_provenance(
        mesh_device,
        ARTIFACT_DIR
        / "candidates"
        / candidate
        / f"pcc_layer{layer_idx}_{layer_type}_shared{int(shared_physical)}.json",
        "GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZED_PRECISION_SWEEP=1 "
        "TTNN_CONFIG_OVERRIDES='{\"throw_exception_on_fallback\": true}' "
        "pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/"
        "test_optimized_decoder.py::test_optimized_precision_candidate_real_weights",
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx,shared_physical", [(0, True), (5, False)])
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
@pytest.mark.parametrize(
    "candidate,dtypes",
    [
        (
            "expert_bfp8_hifi2",
            {
                "expert_weight_dtype": functional_tests.ttnn.bfloat8_b,
                "expert_compute_fidelity": functional_tests.ttnn.MathFidelity.HiFi2,
            },
        ),
        (
            "expert_bfp8_lofi",
            {
                "expert_weight_dtype": functional_tests.ttnn.bfloat8_b,
                "expert_compute_fidelity": functional_tests.ttnn.MathFidelity.LoFi,
            },
        ),
        (
            "expert_bfp4_lofi",
            {
                "expert_weight_dtype": functional_tests.ttnn.bfloat4_b,
                "expert_compute_fidelity": functional_tests.ttnn.MathFidelity.LoFi,
            },
        ),
    ],
)
def test_optimized_precision_candidate_repeated_perf(
    monkeypatch,
    mesh_device,
    device_params,
    layer_idx,
    shared_physical,
    batch,
    candidate,
    dtypes,
):
    if os.getenv("GEMMA4_OPTIMIZED_FIDELITY_REPEAT_PERF") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_FIDELITY_REPEAT_PERF=1 to run fidelity timing")
    candidate_dir = ARTIFACT_DIR / "candidates" / candidate
    candidate_decoder = _candidate_decoder(candidate, **dtypes)
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", candidate_decoder)
    monkeypatch.setattr(functional_tests, "ARTIFACT_DIR", candidate_dir)
    monkeypatch.setenv("GEMMA4_FUNCTIONAL_DECODER_PERF", "1")
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    case_id = f"layer{layer_idx}_{layer_type}_seq1024_batch{batch}"
    samples = []
    for _ in range(5):
        functional_tests.test_functional_decoder_perf_profile(
            mesh_device, device_params, layer_idx, shared_physical, batch
        )
        samples.append(_rewrite_optimized_timing_provenance(mesh_device, layer_idx, layer_type, batch, candidate_dir))
    result = {
        "model_id": functional_tests.MODEL_ID,
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "batch": batch,
        "sequence_length": 1024,
        "implementation": OptimizedDecoder.implementation,
        "candidate": candidate,
        "weight_dtype": str(dtypes["expert_weight_dtype"]),
        "expert_compute_fidelity": str(dtypes["expert_compute_fidelity"]),
        "candidate_provenance": {
            "group": "sparse expert gate/up/down",
            "sparse_in0_block_w": candidate_decoder.sparse_in0_block_w,
            "fp32_dest_acc_en": False,
            "math_approx_mode": False,
        },
        "provenance": _optimized_provenance(
            mesh_device,
            "GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZED_FIDELITY_REPEAT_PERF=1 "
            "TTNN_CONFIG_OVERRIDES='{\"throw_exception_on_fallback\": true}' "
            "pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/"
            "test_optimized_decoder.py::test_optimized_precision_candidate_repeated_perf",
        ),
        "decode_trace_host_ms_samples": [item["decode_trace_host_ms"] for item in samples],
        "decode_trace_host_ms_median": statistics.median(item["decode_trace_host_ms"] for item in samples),
    }
    if batch == 1:
        result["prefill_host_ms_samples"] = [item["prefill_host_ms"] for item in samples]
        result["prefill_host_ms_median"] = statistics.median(item["prefill_host_ms"] for item in samples)
    (candidate_dir / f"{case_id}_repeated_timings.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx,shared_physical", [(0, True), (5, False)])
def test_optimized_bfp8_kv_cache_candidate(monkeypatch, mesh_device, device_params, layer_idx, shared_physical):
    """Exact-shape BF8 cache A/B using the normal real-weight layer harness."""
    if os.getenv("GEMMA4_OPTIMIZED_CACHE_SWEEP") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_CACHE_SWEEP=1 to reproduce the BF8 cache candidate")
    _select_bfp8_cache_candidate(monkeypatch, ARTIFACT_DIR / "candidates" / "kv_cache_bfp8")
    functional_tests.test_functional_decoder_real_weights_prefill_decode(
        mesh_device, device_params, layer_idx, shared_physical, 0.995
    )
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    _rewrite_optimized_artifact_provenance(
        mesh_device,
        ARTIFACT_DIR
        / "candidates"
        / "kv_cache_bfp8"
        / f"pcc_layer{layer_idx}_{layer_type}_shared{int(shared_physical)}.json",
        "GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZED_CACHE_SWEEP=1 "
        "TTNN_CONFIG_OVERRIDES='{\"throw_exception_on_fallback\": true}' "
        "pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/"
        "test_optimized_decoder.py::test_optimized_bfp8_kv_cache_candidate",
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
def test_optimized_bfp8_kv_cache_traced_decode(monkeypatch, mesh_device, device_params, layer_idx, batch):
    if os.getenv("GEMMA4_OPTIMIZED_CACHE_SWEEP") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_CACHE_SWEEP=1 to reproduce the BF8 cache candidate")
    candidate_dir = ARTIFACT_DIR / "candidates" / "kv_cache_bfp8"
    _select_bfp8_cache_candidate(monkeypatch, candidate_dir)
    functional_tests.test_traced_decode_batch_contract(mesh_device, device_params, layer_idx, batch)
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    _rewrite_optimized_artifact_provenance(
        mesh_device,
        candidate_dir / f"trace_{layer_type}_batch{batch}.json",
        "GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZED_CACHE_SWEEP=1 "
        "TTNN_CONFIG_OVERRIDES='{\"throw_exception_on_fallback\": true}' "
        "pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/"
        "test_optimized_decoder.py::test_optimized_bfp8_kv_cache_traced_decode",
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,shared_physical",
    [(0, True), (5, False)],
    ids=["sliding_attention", "full_attention"],
)
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
def test_optimized_bfp8_kv_cache_perf(monkeypatch, mesh_device, device_params, layer_idx, shared_physical, batch):
    if os.getenv("GEMMA4_OPTIMIZED_CACHE_SWEEP") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_CACHE_SWEEP=1 to reproduce the BF8 cache candidate")
    candidate_dir = ARTIFACT_DIR / "candidates" / "kv_cache_bfp8"
    _select_bfp8_cache_candidate(monkeypatch, candidate_dir)
    functional_tests.test_functional_decoder_perf_profile(mesh_device, device_params, layer_idx, shared_physical, batch)
    _rewrite_optimized_timing_provenance(
        mesh_device, layer_idx, functional_tests._load_text_config().layer_types[layer_idx], batch, candidate_dir
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx,shared_physical", [(0, True), (5, False)])
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
def test_optimized_bfp8_kv_cache_repeated_perf(
    monkeypatch, mesh_device, device_params, layer_idx, shared_physical, batch
):
    if os.getenv("GEMMA4_OPTIMIZED_CACHE_REPEAT_PERF") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_CACHE_REPEAT_PERF=1 to run repeated BF8 cache timing")
    candidate_dir = ARTIFACT_DIR / "candidates" / "kv_cache_bfp8"
    _select_bfp8_cache_candidate(monkeypatch, candidate_dir)
    monkeypatch.setenv("GEMMA4_FUNCTIONAL_DECODER_PERF", "1")
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    case_id = f"layer{layer_idx}_{layer_type}_seq1024_batch{batch}"
    samples = []
    for _ in range(5):
        functional_tests.test_functional_decoder_perf_profile(
            mesh_device, device_params, layer_idx, shared_physical, batch
        )
        samples.append(_rewrite_optimized_timing_provenance(mesh_device, layer_idx, layer_type, batch, candidate_dir))
    result = {
        "model_id": functional_tests.MODEL_ID,
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "batch": batch,
        "sequence_length": 1024,
        "implementation": OptimizedDecoder.implementation,
        "cache_dtype": "bfloat8_b",
        "provenance": _optimized_provenance(
            mesh_device,
            "GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZED_CACHE_REPEAT_PERF=1 "
            "TTNN_CONFIG_OVERRIDES='{\"throw_exception_on_fallback\": true}' "
            "pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/"
            "test_optimized_decoder.py::test_optimized_bfp8_kv_cache_repeated_perf",
        ),
        "decode_trace_host_ms_samples": [item["decode_trace_host_ms"] for item in samples],
        "decode_trace_host_ms_median": statistics.median(item["decode_trace_host_ms"] for item in samples),
    }
    if batch == 1:
        result["prefill_host_ms_samples"] = [item["prefill_host_ms"] for item in samples]
        result["prefill_host_ms_median"] = statistics.median(item["prefill_host_ms"] for item in samples)
    (candidate_dir / f"{case_id}_repeated_timings.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx,shared_physical", [(0, True), (5, False)])
@pytest.mark.parametrize("in0_block_w", [2, 11])
def test_optimized_sparse_program_candidate(
    monkeypatch, mesh_device, device_params, layer_idx, shared_physical, in0_block_w
):
    if os.getenv("GEMMA4_OPTIMIZED_SPARSE_SWEEP") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_SPARSE_SWEEP=1 to reproduce sparse program candidates")
    candidate_decoder = _candidate_decoder(f"sparse_in0_block_w_{in0_block_w}")
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", candidate_decoder)
    monkeypatch.setattr(
        functional_tests,
        "ARTIFACT_DIR",
        ARTIFACT_DIR / "candidates" / f"sparse_in0_block_w_{in0_block_w}",
    )
    functional_tests.test_functional_decoder_real_weights_prefill_decode(
        mesh_device, device_params, layer_idx, shared_physical, 0.995
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx,shared_physical", [(0, True), (5, False)])
@pytest.mark.parametrize("candidate", WHOLE_LAYER_CANDIDATES)
def test_optimized_whole_layer_candidate_pcc(
    monkeypatch, mesh_device, device_params, layer_idx, shared_physical, candidate
):
    if os.getenv("GEMMA4_OPTIMIZED_WHOLE_LAYER_SWEEP") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_WHOLE_LAYER_SWEEP=1 to run whole-layer candidates")
    _select_whole_layer_candidate(monkeypatch, candidate)
    functional_tests.test_functional_decoder_real_weights_prefill_decode(
        mesh_device, device_params, layer_idx, shared_physical, 0.995
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx,shared_physical", [(0, True), (5, False)])
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
@pytest.mark.parametrize("candidate", WHOLE_LAYER_CANDIDATES)
def test_optimized_whole_layer_candidate_perf(
    monkeypatch, mesh_device, device_params, layer_idx, shared_physical, batch, candidate
):
    if os.getenv("GEMMA4_OPTIMIZED_WHOLE_LAYER_SWEEP") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_WHOLE_LAYER_SWEEP=1 to run whole-layer candidates")
    _, destination = _select_whole_layer_candidate(monkeypatch, candidate)
    monkeypatch.setenv("GEMMA4_FUNCTIONAL_DECODER_PERF", "1")
    functional_tests.test_functional_decoder_perf_profile(mesh_device, device_params, layer_idx, shared_physical, batch)
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    timing = _rewrite_optimized_timing_provenance(mesh_device, layer_idx, layer_type, batch, destination=destination)
    _write_whole_layer_result(
        mesh_device,
        candidate=candidate,
        layer_idx=layer_idx,
        layer_type=layer_type,
        batch=batch,
        timing=timing,
        destination=destination,
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
def test_optimized_traced_decode_contract(monkeypatch, mesh_device, device_params, layer_idx, batch):
    _select_optimized_path(monkeypatch)
    functional_tests.test_traced_decode_batch_contract(mesh_device, device_params, layer_idx, batch)
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    _rewrite_optimized_artifact_provenance(
        mesh_device,
        ARTIFACT_DIR / f"trace_{layer_type}_batch{batch}.json",
        "GEMMA4_RANGE_DOWNLOAD=1 pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/"
        "test_optimized_decoder.py::test_optimized_traced_decode_contract",
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_mutable_trace_aba(monkeypatch, mesh_device, device_params, layer_idx):
    """Prove one captured trace consumes mutable input buffers without stale replay."""
    _select_optimized_path(monkeypatch)
    ft = functional_tests
    cfg = ft._load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    state = ft._load_layer_state(layer_idx)
    batch, current_position = 1, 32
    ft.torch.manual_seed(7100 + layer_idx)
    hidden_a = ft.torch.randn(batch, 1, ft.HIDDEN_SIZE, dtype=ft.torch.bfloat16)
    hidden_b = ft.torch.randn(batch, 1, ft.HIDDEN_SIZE, dtype=ft.torch.bfloat16)
    positions = ft.torch.full((batch, 1), current_position, dtype=ft.torch.long)
    rotary = ft.Gemma4TextRotaryEmbedding(cfg)
    cos, sin = rotary(hidden_a, positions, layer_type=layer_type)

    decoder = OptimizedDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    if layer_type == "full_attention":
        num_kv_heads, block_size, head_dim, blocks_per_user = (
            ft.FULL_NUM_KV_HEADS,
            ft.FULL_BLOCK_SIZE,
            ft.FULL_HEAD_DIM,
            2,
        )
    else:
        num_kv_heads, block_size, head_dim, blocks_per_user = (
            ft.SLIDING_NUM_KV_HEADS,
            ft.SLIDING_BLOCK_SIZE,
            ft.SLIDING_HEAD_DIM,
            4,
        )
    page_table = ft._as_tt(
        mesh_device,
        ft.torch.arange(blocks_per_user, dtype=ft.torch.int32).view(1, blocks_per_user),
        dtype=ft.ttnn.int32,
        layout=ft.ttnn.ROW_MAJOR_LAYOUT,
    )
    cache_shape = (blocks_per_user, num_kv_heads, block_size, head_dim)
    kv_cache = (
        ft._as_tt(mesh_device, ft.torch.zeros(cache_shape, dtype=ft.torch.bfloat16)),
        ft._as_tt(mesh_device, ft.torch.zeros(cache_shape, dtype=ft.torch.bfloat16)),
    )
    if layer_type == "full_attention":
        prefix = ft.torch.randn(1, current_position, ft.HIDDEN_SIZE, dtype=ft.torch.bfloat16)
        prefix_positions = ft.torch.arange(current_position).view(1, -1)
        prefix_cos, prefix_sin = rotary(prefix, prefix_positions, layer_type=layer_type)
        decoder.prefill_forward(
            ft._as_tt(mesh_device, prefix.unsqueeze(1)),
            position_cos=ft._as_tt(mesh_device, prefix_cos.unsqueeze(1)),
            position_sin=ft._as_tt(mesh_device, prefix_sin.unsqueeze(1)),
            page_table=page_table,
            kv_cache=kv_cache,
        )

    tt_hidden = ft._as_tt(mesh_device, hidden_a.transpose(0, 1).unsqueeze(0))
    if layer_type == "sliding_attention":
        tt_cos, tt_sin = cos.unsqueeze(0), sin.unsqueeze(0)
    else:
        tt_cos, tt_sin = cos.transpose(0, 1).unsqueeze(0), sin.transpose(0, 1).unsqueeze(0)
    decode_args = {
        "hidden_states": tt_hidden,
        "position_cos": ft._as_tt(mesh_device, tt_cos),
        "position_sin": ft._as_tt(mesh_device, tt_sin),
        "current_pos": ft._as_tt(
            mesh_device,
            ft.torch.tensor([current_position], dtype=ft.torch.int32),
            dtype=ft.ttnn.int32,
            layout=ft.ttnn.ROW_MAJOR_LAYOUT,
        ),
        "page_table": page_table,
        "kv_cache": kv_cache,
    }
    decoder.decode_forward(**decode_args)
    trace_id = ft.ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(**decode_args)
    ft.ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    def replay(hidden):
        host = ft.ttnn.from_torch(
            hidden.transpose(0, 1).unsqueeze(0),
            dtype=ft.ttnn.bfloat16,
            layout=ft.ttnn.TILE_LAYOUT,
        )
        ft.ttnn.copy_host_to_device_tensor(host, tt_hidden, cq_id=0)
        ft.ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        return ft._to_torch(mesh_device, traced_output).reshape(1, 1, ft.HIDDEN_SIZE).to(ft.torch.bfloat16)

    output_a1 = replay(hidden_a)
    output_b = replay(hidden_b)
    output_a2 = replay(hidden_a)
    ft.ttnn.release_trace(mesh_device, trace_id)
    repeat_ok, repeat_pcc = ft.comp_pcc(output_a1, output_a2, 0.9999)
    _, a_b_pcc = ft.comp_pcc(output_a1, output_b, 0.9999)
    assert repeat_ok, repeat_pcc
    assert float(a_b_pcc) < 0.999, a_b_pcc
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_id": ft.MODEL_ID,
        "implementation": OptimizedDecoder.implementation,
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "batch": batch,
        "current_position": current_position,
        "sequence": ["A", "B", "A"],
        "a1_vs_a2_pcc": float(repeat_pcc),
        "a1_vs_b_pcc": float(a_b_pcc),
        "provenance": _optimized_provenance(
            mesh_device,
            "TT_METAL_WATCHER=10 GEMMA4_RANGE_DOWNLOAD=1 pytest -q "
            "models/autoports/google_gemma_4_26b_a4b_it/tests/"
            "test_optimized_decoder.py::test_optimized_mutable_trace_aba",
        ),
    }
    (ARTIFACT_DIR / f"mutable_trace_aba_{layer_type}.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_non_aligned_prefill_boundaries(monkeypatch, mesh_device, device_params, layer_idx):
    _select_optimized_path(monkeypatch)
    functional_tests.test_paged_prefill_logical_boundary_lengths(mesh_device, device_params, layer_idx)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_batch2_prefill(monkeypatch, mesh_device, device_params, layer_idx):
    _select_optimized_path(monkeypatch)
    functional_tests.test_functional_decoder_real_shape_batch2_prefill(mesh_device, device_params, layer_idx)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_advertised_context_decode(monkeypatch, mesh_device, device_params, layer_idx):
    _select_optimized_path(monkeypatch)
    functional_tests.test_advertised_context_traced_decode(mesh_device, device_params, layer_idx)
    layer_type = "sliding_attention" if layer_idx == 0 else "full_attention"
    _rewrite_optimized_artifact_provenance(
        mesh_device,
        ARTIFACT_DIR / f"advertised_context_decode_{layer_type}.json",
        "GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_FUNCTIONAL_DECODER_CONTEXT=1 "
        "TTNN_CONFIG_OVERRIDES='{\"throw_exception_on_fallback\": true}' "
        "pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/"
        "test_optimized_decoder.py::test_optimized_advertised_context_decode",
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_optimized_bounded_modulo_tail(monkeypatch, mesh_device, device_params):
    _select_optimized_path(monkeypatch)
    functional_tests.test_bounded_modulo_prefill_tail_cache_integrity(mesh_device, device_params)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize(
    "layer_kind",
    [functional_tests.SLIDING_KIND, functional_tests.FULL_KIND],
    ids=["sliding_attention", "full_attention"],
)
def test_optimized_long_prefill_attention(monkeypatch, mesh_device, device_params, layer_kind):
    _select_optimized_path(monkeypatch)
    functional_tests.test_long_prefill_attention_correctness(mesh_device, device_params, layer_kind)
    _rewrite_optimized_artifact_provenance(
        mesh_device,
        ARTIFACT_DIR / f"long_prefill_attention_{layer_kind.name}.json",
        "GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_LONG_ATTN_TEST=1 "
        "TTNN_CONFIG_OVERRIDES='{\"throw_exception_on_fallback\": true}' "
        "pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/"
        "test_optimized_decoder.py::test_optimized_long_prefill_attention",
    )


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
def test_optimized_decoder_perf_profile(monkeypatch, mesh_device, device_params, layer_idx, shared_physical, batch):
    if os.getenv("GEMMA4_OPTIMIZED_DECODER_PERF") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_DECODER_PERF=1 to run the profiler harness")
    _select_optimized_path(monkeypatch)
    monkeypatch.setenv("GEMMA4_FUNCTIONAL_DECODER_PERF", "1")
    functional_tests.test_functional_decoder_perf_profile(
        mesh_device,
        device_params,
        layer_idx,
        shared_physical,
        batch,
    )
    _rewrite_optimized_timing_provenance(
        mesh_device, layer_idx, functional_tests._load_text_config().layer_types[layer_idx], batch
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx,shared_physical", [(0, True), (5, False)])
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
def test_optimized_repeated_perf(monkeypatch, mesh_device, device_params, layer_idx, shared_physical, batch):
    if os.getenv("GEMMA4_OPTIMIZED_DECODER_REPEAT_PERF") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_DECODER_REPEAT_PERF=1 to run repeated timing")
    _select_optimized_path(monkeypatch)
    monkeypatch.setenv("GEMMA4_FUNCTIONAL_DECODER_PERF", "1")
    cfg = functional_tests._load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    case_id = f"layer{layer_idx}_{layer_type}_seq1024_batch{batch}"
    samples = []
    for _ in range(5):
        functional_tests.test_functional_decoder_perf_profile(
            mesh_device,
            device_params,
            layer_idx,
            shared_physical,
            batch,
        )
        payload = _rewrite_optimized_timing_provenance(mesh_device, layer_idx, layer_type, batch)
        samples.append(payload)
    result = {
        "model_id": functional_tests.MODEL_ID,
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "batch": batch,
        "sequence_length": 1024,
        "implementation": OptimizedDecoder.implementation,
        "provenance": _optimized_provenance(
            mesh_device,
            "GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZED_DECODER_REPEAT_PERF=1 "
            "TTNN_CONFIG_OVERRIDES='{\"throw_exception_on_fallback\": true}' "
            "pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/"
            "test_optimized_decoder.py::test_optimized_repeated_perf",
        ),
        "decode_trace_host_ms_samples": [item["decode_trace_host_ms"] for item in samples],
        "decode_trace_host_ms_median": statistics.median(item["decode_trace_host_ms"] for item in samples),
    }
    if batch == 1:
        result["prefill_host_ms_samples"] = [item["prefill_host_ms"] for item in samples]
        result["prefill_host_ms_median"] = statistics.median(item["prefill_host_ms"] for item in samples)
    (ARTIFACT_DIR / f"{case_id}_repeated_timings.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx,shared_physical", [(0, True), (5, False)])
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
def test_functional_baseline_repeated_perf(monkeypatch, mesh_device, device_params, layer_idx, shared_physical, batch):
    if os.getenv("GEMMA4_FUNCTIONAL_BASELINE_REPEAT_PERF") != "1":
        pytest.skip("set GEMMA4_FUNCTIONAL_BASELINE_REPEAT_PERF=1 to run repeated baseline timing")
    baseline_dir = ARTIFACT_DIR / "baseline_repeated"
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", FunctionalDecoder)
    monkeypatch.setattr(functional_tests, "ARTIFACT_DIR", baseline_dir)
    monkeypatch.setenv("GEMMA4_FUNCTIONAL_DECODER_PERF", "1")
    cfg = functional_tests._load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    case_id = f"layer{layer_idx}_{layer_type}_seq1024_batch{batch}"
    samples = []
    for _ in range(5):
        functional_tests.test_functional_decoder_perf_profile(
            mesh_device,
            device_params,
            layer_idx,
            shared_physical,
            batch,
        )
        samples.append(json.loads((baseline_dir / f"{case_id}_host_timings.json").read_text()))
    result = {
        "model_id": functional_tests.MODEL_ID,
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "batch": batch,
        "sequence_length": 1024,
        "decode_trace_host_ms_samples": [item["decode_trace_host_ms"] for item in samples],
        "decode_trace_host_ms_median": statistics.median(item["decode_trace_host_ms"] for item in samples),
    }
    if batch == 1:
        result["prefill_host_ms_samples"] = [item["prefill_host_ms"] for item in samples]
        result["prefill_host_ms_median"] = statistics.median(item["prefill_host_ms"] for item in samples)
    (baseline_dir / f"{case_id}_repeated_timings.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
