# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and path-identity gates for the Gemma-4 optimized decoder.

The functional stage's real-weight/HF oracles are intentionally reused so the
optimized stage cannot weaken thresholds or subtly change cache construction.
Each wrapper replaces the decoder constructor in the oracle module and checks
that optimized material methods were actually entered.
"""

from __future__ import annotations

import inspect
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import pytest
import torch
import ttnn

import models.autoports.google_gemma_4_26b_a4b_it.tests.test_functional_decoder as functional_tests
import models.autoports.google_gemma_4_26b_a4b_it.tests.test_trace_mutable_buffers as mutable_tests
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import FunctionalDecoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder

ARTIFACT_DIR = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder")
OPTIMIZED_SOURCE = Path("models/autoports/google_gemma_4_26b_a4b_it/tt/optimized_decoder.py")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved_policy() -> dict:
    signature = inspect.signature(OptimizedDecoder.from_state_dict)
    defaults = {
        name: str(parameter.default)
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    overrides = {name: value for name, value in sorted(os.environ.items()) if name.startswith("GEMMA4_OPT")}
    return {"constructor_defaults": defaults, "environment_overrides": overrides}


def _stamp_artifact(path: Path, *, exact_command: str | None = None) -> dict:
    contents = json.loads(path.read_text()) if path.exists() else {}
    contents["optimized_stage_provenance"] = {
        "checkout_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "exact_command": exact_command
        or os.getenv("GEMMA4_OPT_EXACT_COMMAND", f"pytest -q {os.getenv('PYTEST_CURRENT_TEST', '').split(' ')[0]}"),
        "hardware": {
            "arch": "Blackhole P300C",
            "platform": platform.platform(),
        },
        "optimized_decoder_sha256": _sha256(OPTIMIZED_SOURCE),
        "optimized_test_sha256": _sha256(Path(__file__)),
        "resolved_policy": _resolved_policy(),
    }
    path.write_text(json.dumps(contents, indent=2, sort_keys=True) + "\n")
    candidate_id = os.getenv("GEMMA4_OPT_CANDIDATE_ID")
    if candidate_id:
        candidate_dir = ARTIFACT_DIR / "candidate_runs"
        candidate_dir.mkdir(exist_ok=True)
        candidate_path = candidate_dir / f"{candidate_id}.json"
        candidate = json.loads(candidate_path.read_text()) if candidate_path.exists() else {}
        candidate[path.name] = contents
        candidate_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    return contents


def _install_optimized_oracle(monkeypatch, module, *, required_methods):
    calls = {name: 0 for name in required_methods}
    monkeypatch.setattr(module, "FunctionalDecoder", OptimizedDecoder)
    monkeypatch.setattr(module, "ARTIFACT_DIR", ARTIFACT_DIR)
    for name in required_methods:
        original = getattr(OptimizedDecoder, name)

        def wrapped(self, *args, __name=name, __original=original, **kwargs):
            calls[__name] += 1
            return __original(self, *args, **kwargs)

        monkeypatch.setattr(OptimizedDecoder, name, wrapped)
    return calls


def test_optimized_material_paths_are_not_functional_fallbacks():
    material_methods = (
        "_attention_prefill",
        "_attention_decode",
        "_dense_mlp",
        "_moe_prefill",
        "_moe_prefill_chunk",
        "_moe_decode_single_user",
    )
    for name in material_methods:
        optimized_method = inspect.getattr_static(OptimizedDecoder, name)
        functional_method = inspect.getattr_static(FunctionalDecoder, name)
        assert optimized_method is not functional_method, name
        assert optimized_method.__module__.endswith(".optimized_decoder"), name


def test_optimized_hot_path_fallback_audit():
    forbidden = ("torch.", "import torch", "ttnn.from_torch", "ttnn.to_torch")
    methods = (
        OptimizedDecoder._attention_prefill,
        OptimizedDecoder._attention_decode,
        OptimizedDecoder._dense_mlp,
        OptimizedDecoder._moe_prefill,
        OptimizedDecoder._moe_prefill_chunk,
        OptimizedDecoder._moe_decode_single_user,
    )
    source = "\n".join(inspect.getsource(method) for method in methods)
    for token in forbidden:
        assert token not in source


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,shared_physical,decode_pcc",
    [
        pytest.param(0, True, 0.995, id="sliding_attention_shared_cache"),
        pytest.param(5, False, 0.995, id="full_attention_natural_cache"),
        pytest.param(5, True, 0.995, id="full_attention_shared_cache_view"),
    ],
)
def test_optimized_real_weights_prefill_decode(
    monkeypatch,
    mesh_device,
    device_params,
    layer_idx,
    shared_physical,
    decode_pcc,
):
    calls = _install_optimized_oracle(
        monkeypatch,
        functional_tests,
        required_methods=(
            "_attention_prefill",
            "_attention_decode",
            "_dense_mlp",
            "_moe_prefill",
            "_moe_prefill_chunk",
            "_moe_decode_single_user",
        ),
    )
    functional_tests.test_functional_decoder_real_weights_prefill_decode(
        mesh_device,
        device_params,
        layer_idx,
        shared_physical,
        decode_pcc,
    )
    assert all(count > 0 for count in calls.values()), calls
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    _stamp_artifact(ARTIFACT_DIR / f"pcc_layer{layer_idx}_{layer_type}_shared{int(shared_physical)}.json")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
def test_optimized_traced_decode_batch_contract(monkeypatch, mesh_device, device_params, layer_idx, batch):
    calls = _install_optimized_oracle(
        monkeypatch,
        functional_tests,
        required_methods=("_attention_decode", "_dense_mlp", "_moe_decode_single_user"),
    )
    per_user_pcc = []
    best_replay_user = []
    if batch == 32:
        original_comp_pcc = functional_tests.comp_pcc

        def comp_pcc_with_per_user_gate(reference, actual, threshold):
            result = original_comp_pcc(reference, actual, threshold)
            if threshold == 0.995 and reference.shape[0] == batch:
                per_user_pcc.extend(
                    float(original_comp_pcc(reference[index : index + 1], actual[index : index + 1], threshold)[1])
                    for index in range(batch)
                )
                reference_rows = torch.nn.functional.normalize(reference.float().reshape(batch, -1), dim=-1)
                actual_rows = torch.nn.functional.normalize(actual.float().reshape(batch, -1), dim=-1)
                best_replay_user.extend((reference_rows @ actual_rows.T).argmax(dim=1).tolist())
            return result

        monkeypatch.setattr(functional_tests, "comp_pcc", comp_pcc_with_per_user_gate)
    functional_tests.test_traced_decode_batch_contract(mesh_device, device_params, layer_idx, batch)
    assert all(count > 0 for count in calls.values()), calls
    if batch == 32:
        assert min(per_user_pcc) >= 0.995, per_user_pcc
        layer_type = functional_tests._load_text_config().layer_types[layer_idx]
        if layer_type == "sliding_attention":
            assert best_replay_user == list(range(batch)), best_replay_user
        artifact = ARTIFACT_DIR / f"trace_{layer_type}_batch{batch}.json"
        contents = json.loads(artifact.read_text())
        contents["hf_vs_trace_replay_per_user_pcc"] = per_user_pcc
        contents["hf_vs_trace_replay_min_user_pcc"] = min(per_user_pcc)
        contents["best_replay_user"] = best_replay_user
        artifact.write_text(json.dumps(contents, indent=2, sort_keys=True) + "\n")
    _stamp_artifact(
        ARTIFACT_DIR / f"trace_{functional_tests._load_text_config().layer_types[layer_idx]}_batch{batch}.json"
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention_shared_hma"])
def test_optimized_trace_mutable_stable_buffers(monkeypatch, mesh_device, device_params, layer_idx):
    calls = _install_optimized_oracle(
        monkeypatch,
        mutable_tests,
        required_methods=("_attention_decode", "_dense_mlp", "_moe_decode_single_user"),
    )
    mutable_tests.test_trace_mutable_stable_buffers(mesh_device, device_params, layer_idx)
    assert all(count > 0 for count in calls.values()), calls
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    _stamp_artifact(ARTIFACT_DIR / f"trace_mutable_buffers_{layer_type}_batch32.json")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_real_shape_batch2_prefill(monkeypatch, mesh_device, device_params, layer_idx):
    calls = _install_optimized_oracle(
        monkeypatch,
        functional_tests,
        required_methods=("_attention_prefill", "_dense_mlp", "_moe_prefill", "_moe_prefill_chunk"),
    )
    functional_tests.test_functional_decoder_real_shape_batch2_prefill(mesh_device, device_params, layer_idx)
    assert all(count > 0 for count in calls.values()), calls
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    _stamp_artifact(ARTIFACT_DIR / f"prefill_batch2_layer{layer_idx}_{layer_type}.json")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_paged_prefill_logical_boundary_lengths(monkeypatch, mesh_device, device_params, layer_idx):
    calls = _install_optimized_oracle(
        monkeypatch,
        functional_tests,
        required_methods=("_attention_prefill", "_dense_mlp", "_moe_prefill", "_moe_prefill_chunk"),
    )
    recorded_pcc = []
    original_comp_pcc = functional_tests.comp_pcc

    def record_all_boundaries(reference, actual, threshold):
        passed, pcc = original_comp_pcc(reference, actual, threshold)
        recorded_pcc.append(float(pcc))
        return True, pcc

    monkeypatch.setattr(functional_tests, "comp_pcc", record_all_boundaries)
    functional_tests.test_paged_prefill_logical_boundary_lengths(mesh_device, device_params, layer_idx)
    assert all(count > 0 for count in calls.values()), calls
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    artifact = ARTIFACT_DIR / f"prefill_boundaries_{layer_type}.json"
    contents = _stamp_artifact(artifact)
    assert len(recorded_pcc) == len(contents["results"])
    assert min(recorded_pcc) >= 0.995, contents["results"]


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_advertised_context_traced_decode(monkeypatch, mesh_device, device_params, layer_idx):
    calls = _install_optimized_oracle(
        monkeypatch,
        functional_tests,
        required_methods=("_attention_decode", "_dense_mlp", "_moe_decode_single_user"),
    )
    functional_tests.test_advertised_context_traced_decode(mesh_device, device_params, layer_idx)
    assert all(count > 0 for count in calls.values()), calls
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    _stamp_artifact(ARTIFACT_DIR / f"advertised_context_decode_{layer_type}.json")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_prefill_capacity_probe(monkeypatch, mesh_device, device_params, layer_idx):
    calls = _install_optimized_oracle(
        monkeypatch,
        functional_tests,
        required_methods=("_attention_prefill", "_dense_mlp", "_moe_prefill", "_moe_prefill_chunk"),
    )
    functional_tests.test_prefill_capacity_probe(mesh_device, device_params, layer_idx)
    assert all(count > 0 for count in calls.values()), calls
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    seq_len = int(os.getenv("GEMMA4_PREFILL_CAPACITY_LENGTH", "262143"))
    _stamp_artifact(ARTIFACT_DIR / f"prefill_capacity_{layer_type}_{seq_len}.json")


def test_optimized_precision_defaults():
    signature = inspect.signature(OptimizedDecoder.from_state_dict)
    assert signature.parameters["weight_dtype"].default == ttnn.bfloat16
    assert signature.parameters["attention_weight_dtype"].default == ttnn.bfloat16
    assert signature.parameters["mlp_weight_dtype"].default == ttnn.bfloat8_b
    assert signature.parameters["prefill_expert_weight_dtype"].default == ttnn.bfloat8_b
    assert signature.parameters["expert_weight_dtype"].default == ttnn.bfloat8_b
    assert signature.parameters["activation_dtype"].default == ttnn.bfloat16
    assert signature.parameters["attention_math_fidelity"].default == ttnn.MathFidelity.HiFi4
    assert signature.parameters["full_attention_math_fidelity"].default == ttnn.MathFidelity.HiFi2
    assert signature.parameters["expert_gate_per_core_n"].default == 2
    assert signature.parameters["expert_down_per_core_n"].default == 2
    assert signature.parameters["expert_decode_input_l1"].default is False
    assert signature.parameters["dense_decode_dram_sharded"].default is False
    assert signature.parameters["packed_dense_gate_up"].default is False
    assert signature.parameters["prefill_expert_chunk_size"].default == 32
    assert signature.parameters["prefill_routed_active"].default is True
    assert signature.parameters["prefill_expert_per_core_n"].default == 2
    assert signature.parameters["prefill_expert_gate_in0_block_w"].default == 44
    assert signature.parameters["prefill_expert_down_in0_block_w"].default == 11
    assert signature.parameters["prefill_expert_tail_per_core_n"].default == 11
    assert signature.parameters["prefill_expert_tail_in0_block_w"].default == 1


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,shared_physical",
    [
        pytest.param(0, True, id="sliding_attention_1024"),
        pytest.param(5, False, id="full_attention_1024"),
    ],
)
def test_optimized_prefill_batch32_perf(monkeypatch, mesh_device, device_params, layer_idx, shared_physical):
    if os.getenv("GEMMA4_OPTIMIZED_PREFILL_BATCH32_PERF") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_PREFILL_BATCH32_PERF=1 to run batch-32 prefill")

    baseline = os.getenv("GEMMA4_OPTIMIZED_PREFILL_BASELINE") == "1"
    calls = (
        {}
        if baseline
        else _install_optimized_oracle(
            monkeypatch,
            functional_tests,
            required_methods=("_attention_prefill", "_dense_mlp", "_moe_prefill", "_moe_prefill_chunk"),
        )
    )
    cfg = functional_tests._load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    state = functional_tests._load_layer_state(layer_idx)
    seq_len = int(os.getenv("GEMMA4_FUNCTIONAL_DECODER_SEQ_LEN", "1024"))
    batch = 32
    torch.manual_seed(1700 + layer_idx)
    one_hidden = torch.randn(1, seq_len, functional_tests.HIDDEN_SIZE, dtype=torch.bfloat16)
    positions = torch.arange(seq_len).unsqueeze(0)
    rotary = functional_tests.Gemma4TextRotaryEmbedding(cfg)
    one_cos, one_sin = rotary(one_hidden, positions, layer_type=layer_type)
    hidden = one_hidden.unsqueeze(1).expand(batch, 1, seq_len, -1)
    cos = one_cos.unsqueeze(1).expand(batch, 1, seq_len, -1)
    sin = one_sin.unsqueeze(1).expand(batch, 1, seq_len, -1)

    decoder_type = FunctionalDecoder if baseline else OptimizedDecoder
    decoder = decoder_type.from_state_dict(
        state,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
    )
    one_user_cache_shape = functional_tests._cache_shape(
        layer_type,
        shared_physical=shared_physical,
        token_capacity=seq_len + 1,
    )
    blocks_per_user = one_user_cache_shape[0]
    page_table = functional_tests._as_tt(
        mesh_device,
        torch.arange(batch * blocks_per_user, dtype=torch.int32).view(batch, blocks_per_user),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cache_shape = (batch * blocks_per_user, *one_user_cache_shape[1:])
    kv_cache = (
        functional_tests._as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
        functional_tests._as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
    )
    kwargs = {
        "hidden_states": functional_tests._as_tt(mesh_device, hidden),
        "position_cos": functional_tests._as_tt(mesh_device, cos),
        "position_sin": functional_tests._as_tt(mesh_device, sin),
        "page_table": page_table,
        "kv_cache": kv_cache,
    }
    decoder.prefill_forward(**kwargs)
    ttnn.synchronize_device(mesh_device)
    start = time.perf_counter()
    output = decoder.prefill_forward(**kwargs)
    ttnn.synchronize_device(mesh_device)
    prefill_ms = (time.perf_counter() - start) * 1000
    assert output.shape[0] == batch
    if not baseline:
        assert all(count > 0 for count in calls.values()), calls

    artifact = ARTIFACT_DIR / f"layer{layer_idx}_{layer_type}_seq{seq_len}_batch32_host_timings.json"
    contents = json.loads(artifact.read_text()) if artifact.exists() else {}
    field = "functional_prefill_batch32_host_ms" if baseline else "prefill_batch32_host_ms"
    contents[field] = prefill_ms
    contents["prefill_batch"] = batch
    artifact.write_text(json.dumps(contents, indent=2, sort_keys=True) + "\n")
    _stamp_artifact(artifact)


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
def test_optimized_decoder_perf_profile(
    monkeypatch,
    mesh_device,
    device_params,
    layer_idx,
    shared_physical,
    batch,
):
    baseline = os.getenv("GEMMA4_OPTIMIZED_PERF_BASELINE") == "1"
    calls = {}
    if baseline:
        monkeypatch.setattr(functional_tests, "ARTIFACT_DIR", ARTIFACT_DIR)
    else:
        calls = _install_optimized_oracle(
            monkeypatch,
            functional_tests,
            required_methods=(
                "_attention_prefill",
                "_attention_decode",
                "_dense_mlp",
                "_moe_prefill",
                "_moe_prefill_chunk",
                "_moe_decode_single_user",
            ),
        )
        if os.getenv("GEMMA4_OPT_DECODE_DEVICE_PROFILE") == "1":
            from tracy import signpost

            original_decode_forward = OptimizedDecoder.decode_forward
            profiled = False

            def profiled_decode_forward(self, *args, **kwargs):
                nonlocal profiled
                if profiled:
                    return original_decode_forward(self, *args, **kwargs)
                profiled = True
                signpost("OPTIMIZED_DECODE_DEVICE")
                output = original_decode_forward(self, *args, **kwargs)
                ttnn.synchronize_device(self.mesh_device)
                signpost("OPTIMIZED_DECODE_DEVICE_END")
                return output

            monkeypatch.setattr(OptimizedDecoder, "decode_forward", profiled_decode_forward)
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    seq_len = int(os.getenv("GEMMA4_FUNCTIONAL_DECODER_SEQ_LEN", "1024"))
    artifact = ARTIFACT_DIR / f"layer{layer_idx}_{layer_type}_seq{seq_len}_batch{batch}_host_timings.json"
    previous = json.loads(artifact.read_text()) if artifact.exists() else {}
    functional_tests.test_functional_decoder_perf_profile(
        mesh_device,
        device_params,
        layer_idx,
        shared_physical,
        batch,
    )
    measured = json.loads(artifact.read_text())
    if baseline:
        if "prefill_host_ms" in measured:
            measured["functional_prefill_host_ms"] = measured.pop("prefill_host_ms")
        measured["functional_decode_trace_host_ms"] = measured.pop("decode_trace_host_ms")
        measured = {**previous, **measured}
        artifact.write_text(json.dumps(measured, indent=2, sort_keys=True) + "\n")
    else:
        assert all(count > 0 for count in calls.values()), calls
        measured = {**previous, **measured}
        artifact.write_text(json.dumps(measured, indent=2, sort_keys=True) + "\n")
    _stamp_artifact(artifact)
