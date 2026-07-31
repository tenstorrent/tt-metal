# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Optimized-decoder gates.

The numerical harness is intentionally shared with the completed functional
stage so before/after PCC and latency use identical inputs.  Each wrapper
rebinds the harness constructor to ``OptimizedDecoder`` and asserts that the
optimized runtime counters were hit.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import pytest

import models.autoports.google_gemma_4_26b_a4b_it.tests.test_functional_decoder as functional_tests
import models.autoports.google_gemma_4_26b_a4b_it.tests.test_trace_mutable_buffers as mutable_tests
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import FunctionalDecoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import (
    ADVISOR_SEED_POLICY,
    ADVISOR_SELECTED_POLICY,
    OptimizedDecoder,
    _advisor_1d_program_config,
)

ARTIFACT_DIR = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder")


def _select_optimized(monkeypatch) -> None:
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", OptimizedDecoder)
    monkeypatch.setattr(functional_tests, "ARTIFACT_DIR", ARTIFACT_DIR)
    if os.getenv("GEMMA4_OPTIMIZATION_POLICY") == "advisor_seed":
        base_policy = (
            ADVISOR_SELECTED_POLICY if os.getenv("GEMMA4_CANDIDATE_CUMULATIVE") == "1" else ADVISOR_SEED_POLICY
        )
        roles = tuple(
            role.strip()
            for role in os.getenv("GEMMA4_ADVISOR_ROLES", ",".join(base_policy.advisor_roles)).split(",")
            if role.strip()
        )
        monkeypatch.setattr(
            OptimizedDecoder,
            "optimization_policy",
            type(base_policy)(**{**base_policy.__dict__, "advisor_roles": roles}),
        )


def _mark_optimized_provenance(path: Path, command: str) -> None:
    data = json.loads(path.read_text())
    provenance = data.setdefault("provenance", {})
    provenance["exact_command"] = command
    provenance["optimized_decoder_sha256"] = hashlib.sha256(
        inspect.getsource(
            __import__(
                "models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder",
                fromlist=["OptimizedDecoder"],
            )
        ).encode()
    ).hexdigest()
    provenance["optimized_test_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance["optimized_policy"] = OptimizedDecoder.optimization_policy.__dict__
    provenance["environment_overrides"] = {
        name: os.environ[name]
        for name in (
            "GEMMA4_OPTIMIZATION_POLICY",
            "GEMMA4_CANDIDATE_CUMULATIVE",
            "GEMMA4_ADVISOR_ROLES",
            "GEMMA4_OPTIMIZED_WEIGHT_DTYPE",
            "GEMMA4_OPTIMIZED_EXPERT_WEIGHT_DTYPE",
            "GEMMA4_OPTIMIZED_ATTENTION_DTYPE",
            "GEMMA4_OPTIMIZED_ATTENTION_FIDELITY",
            "GEMMA4_OPTIMIZED_EXPERT_GATE_FIDELITY",
            "GEMMA4_OPTIMIZED_ARTIFACT_SUFFIX",
        )
        if name in os.environ
    }
    provenance.pop("functional_decoder_sha256", None)
    provenance.pop("test_sha256", None)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _dedicated_profiler_cache_geometry(
    layer_type: str, *, batch: int, token_capacity: int
) -> tuple[bool, tuple[int, ...], tuple[int, int]]:
    """Match the headline profiler's physical-cache policy and geometry."""

    shared_physical = layer_type == "sliding_attention"
    one_user_cache_shape = functional_tests._cache_shape(
        layer_type,
        shared_physical=shared_physical,
        token_capacity=token_capacity,
    )
    blocks_per_user = one_user_cache_shape[0]
    cache_shape = (batch * blocks_per_user, *one_user_cache_shape[1:])
    page_table_shape = (batch, blocks_per_user)
    return shared_physical, cache_shape, page_table_shape


def test_optimized_class_owns_runtime_entry_points():
    assert OptimizedDecoder.prefill_forward.__qualname__.startswith("OptimizedDecoder.")
    assert OptimizedDecoder.decode_forward.__qualname__.startswith("OptimizedDecoder.")
    assert "FunctionalDecoder =" not in inspect.getsource(
        __import__(
            "models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder",
            fromlist=["OptimizedDecoder"],
        )
    )


@pytest.mark.parametrize(
    "layer_type,token_capacity,expected",
    [
        ("sliding_attention", 1025, (True, (17, 8, 64, 256), (1, 17))),
        ("full_attention", 1025, (False, (9, 2, 128, 512), (1, 9))),
        ("sliding_attention", 257, (True, (5, 8, 64, 256), (1, 5))),
        ("full_attention", 257, (False, (3, 2, 128, 512), (1, 3))),
    ],
)
def test_dedicated_profiler_cache_geometry_matches_headline(layer_type, token_capacity, expected):
    assert _dedicated_profiler_cache_geometry(layer_type, batch=1, token_capacity=token_capacity) == expected


def test_optimized_runtime_invocation_counters(monkeypatch):
    decoder = object.__new__(OptimizedDecoder)
    decoder.optimized_prefill_invocations = 0
    decoder.optimized_decode_invocations = 0
    prefill_result = object()
    decode_result = object()

    monkeypatch.setattr(FunctionalDecoder, "prefill_forward", lambda *_args, **_kwargs: prefill_result)

    def fake_decode(parent_decoder, *_args, **_kwargs):
        assert parent_decoder._optimized_decode_active is True
        return decode_result

    monkeypatch.setattr(OptimizedDecoder.__mro__[1], "decode_forward", fake_decode)

    assert decoder.prefill_forward(object()) is prefill_result
    assert decoder.optimized_prefill_invocations == 1
    assert decoder.decode_forward(object()) is decode_result
    assert decoder.optimized_decode_invocations == 1
    assert decoder._optimized_decode_active is False


def test_advisor_program_config_matches_authoritative_ir():
    config = _advisor_1d_program_config(grid_y=8, in0_block_w=2, out_subblock_w=3)
    rendered = str(config)
    assert "num_global_cb_receivers=0" in rendered
    assert "compute_with_storage_grid_size=11-8" in rendered
    assert "in0_block_w=2" in rendered
    assert "out_subblock_w=3" in rendered


def test_advisor_roles_are_independently_selectable(monkeypatch):
    monkeypatch.setenv("GEMMA4_OPTIMIZATION_POLICY", "advisor_seed")
    monkeypatch.setenv("GEMMA4_ADVISOR_ROLES", "qkv,down")
    _select_optimized(monkeypatch)
    assert OptimizedDecoder.optimization_policy.advisor_roles == ("qkv", "down")


def test_default_policy_uses_only_correct_advisor_roles():
    assert OptimizedDecoder.optimization_policy == ADVISOR_SELECTED_POLICY
    assert OptimizedDecoder.optimization_policy.advisor_roles == (
        "sliding_dram_qkv_w1",
        "full_dram_qkv_w2",
        "persistent_o_proj",
        "packed_dense",
        "dense_down_w3",
        "expert_gate_grid_w11",
        "expert_up_w11",
        "expert_up_grid_x11",
        "fused_router_scale",
        "prefill_expert_packed_gate_up_grid_11x4",
        "prefill_expert_packed_gate_up_w11",
        "prefill_expert_packed_gate_up_l1",
        "prefill_expert_down_grid_11x8",
        "prefill_expert_down_w11",
        "prefill_expert_down_l1",
        "b32_dram_packed_dense_w4",
        "b32_sliding_dram_qkv_w2",
    )


def test_moe_batch_orchestration_dispatches_to_optimized_sparse_kernel():
    orchestration = inspect.getsource(FunctionalDecoder._moe_decode)
    optimized_entry = inspect.getsource(OptimizedDecoder._moe_decode)
    assert "self._moe_decode_single_user" in orchestration
    assert "FunctionalDecoder._moe_decode(self" in optimized_entry
    assert OptimizedDecoder._moe_decode_single_user is not FunctionalDecoder._moe_decode_single_user


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,shared_physical",
    [(0, True), (5, False), (5, True)],
    ids=["sliding_attention", "full_attention", "full_attention_shared_hma"],
)
def test_optimized_real_weights_prefill_decode(monkeypatch, mesh_device, device_params, layer_idx, shared_physical):
    _select_optimized(monkeypatch)
    cfg = functional_tests._load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    try:
        functional_tests.test_functional_decoder_real_weights_prefill_decode(
            mesh_device, device_params, layer_idx, shared_physical, 0.995
        )
    finally:
        artifact = ARTIFACT_DIR / f"pcc_layer{layer_idx}_{layer_type}_shared{int(shared_physical)}.json"
        artifact_suffix = os.getenv("GEMMA4_OPTIMIZED_ARTIFACT_SUFFIX")
        if artifact_suffix and artifact.exists():
            assert artifact_suffix.replace("_", "").isalnum(), "artifact suffix must be alphanumeric/underscores"
            suffixed_artifact = artifact.with_name(f"{artifact.stem}_{artifact_suffix}{artifact.suffix}")
            artifact.replace(suffixed_artifact)
            artifact = suffixed_artifact
        if artifact.exists():
            _mark_optimized_provenance(
                artifact,
                "GEMMA4_RANGE_DOWNLOAD=1 python -m pytest -q models/autoports/"
                "google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::"
                "test_optimized_real_weights_prefill_decode",
            )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
def test_optimized_traced_decode_batch_contract(monkeypatch, mesh_device, device_params, layer_idx, batch):
    _select_optimized(monkeypatch)
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    try:
        functional_tests.test_traced_decode_batch_contract(mesh_device, device_params, layer_idx, batch)
    finally:
        artifact = ARTIFACT_DIR / f"trace_{layer_type}_batch{batch}.json"
        artifact_suffix = os.getenv("GEMMA4_OPTIMIZED_ARTIFACT_SUFFIX")
        if artifact_suffix and artifact.exists():
            assert artifact_suffix.replace("_", "").isalnum(), "artifact suffix must be alphanumeric/underscores"
            suffixed_artifact = artifact.with_name(f"{artifact.stem}_{artifact_suffix}{artifact.suffix}")
            artifact.replace(suffixed_artifact)
            artifact = suffixed_artifact
        if artifact.exists():
            _mark_optimized_provenance(
                artifact,
                "GEMMA4_RANGE_DOWNLOAD=1 python -m pytest -q models/autoports/"
                "google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::"
                "test_optimized_traced_decode_batch_contract",
            )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_optimized_bounded_tail_cache_integrity(monkeypatch, mesh_device, device_params):
    _select_optimized(monkeypatch)
    functional_tests.test_bounded_modulo_prefill_tail_cache_integrity(mesh_device, device_params)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_trace_mutable_stable_buffers(monkeypatch, mesh_device, device_params, layer_idx):
    monkeypatch.setattr(mutable_tests, "FunctionalDecoder", OptimizedDecoder)
    monkeypatch.setattr(mutable_tests, "ARTIFACT_DIR", ARTIFACT_DIR)
    mutable_tests.test_trace_mutable_stable_buffers(mesh_device, device_params, layer_idx)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
def test_optimized_repeated_trace_stress(monkeypatch, mesh_device, device_params, layer_idx, batch):
    if os.getenv("GEMMA4_OPTIMIZED_STRESS") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_STRESS=1 to run repeated traced decode")
    _select_optimized(monkeypatch)
    cfg = functional_tests._load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    results = []
    for repetition in range(3):
        functional_tests.test_traced_decode_batch_contract(mesh_device, device_params, layer_idx, batch)
        trace_artifact = ARTIFACT_DIR / f"trace_{layer_type}_batch{batch}.json"
        result = json.loads(trace_artifact.read_text())
        _mark_optimized_provenance(
            trace_artifact,
            "GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZED_STRESS=1 python -m pytest -q "
            "models/autoports/google_gemma_4_26b_a4b_it/tests/"
            "test_optimized_decoder.py::test_optimized_repeated_trace_stress",
        )
        results.append(
            {
                "repetition": repetition,
                "hf_vs_trace_replay_pcc": result["hf_vs_trace_replay_pcc"],
                "eager_vs_trace_replay_pcc": result["eager_vs_trace_replay_pcc"],
                "repeat_replay_pcc": result["repeat_replay_pcc"],
            }
        )
    (ARTIFACT_DIR / f"stress_trace_{layer_type}_batch{batch}.json").write_text(
        json.dumps(
            {
                "model_id": functional_tests.MODEL_ID,
                "optimized_policy": OptimizedDecoder.optimization_policy.__dict__,
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "batch": batch,
                "repetitions": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _mark_optimized_provenance(
        ARTIFACT_DIR / f"stress_trace_{layer_type}_batch{batch}.json",
        "GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZED_STRESS=1 python -m pytest -q "
        "models/autoports/google_gemma_4_26b_a4b_it/tests/"
        "test_optimized_decoder.py::test_optimized_repeated_trace_stress",
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_paged_prefill_logical_boundaries(monkeypatch, mesh_device, device_params, layer_idx):
    _select_optimized(monkeypatch)
    functional_tests.test_paged_prefill_logical_boundary_lengths(mesh_device, device_params, layer_idx)
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    _mark_optimized_provenance(
        ARTIFACT_DIR / f"prefill_boundaries_{layer_type}.json",
        "GEMMA4_RANGE_DOWNLOAD=1 python -m pytest -q models/autoports/"
        "google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::"
        "test_optimized_paged_prefill_logical_boundaries",
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_advertised_context(monkeypatch, mesh_device, device_params, layer_idx):
    _select_optimized(monkeypatch)
    functional_tests.test_advertised_context_traced_decode(mesh_device, device_params, layer_idx)
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    _mark_optimized_provenance(
        ARTIFACT_DIR / f"advertised_context_decode_{layer_type}.json",
        "GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_FUNCTIONAL_DECODER_CONTEXT=1 python -m pytest -q "
        "models/autoports/google_gemma_4_26b_a4b_it/tests/"
        "test_optimized_decoder.py::test_optimized_advertised_context",
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,shared_physical",
    [(0, True), (5, False)],
    ids=["sliding_attention_1024", "full_attention_1024"],
)
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
def test_optimized_perf_profile(monkeypatch, mesh_device, device_params, layer_idx, shared_physical, batch):
    _select_optimized(monkeypatch)
    functional_tests.test_functional_decoder_perf_profile(mesh_device, device_params, layer_idx, shared_physical, batch)
    cfg = functional_tests._load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    seq_len = int(os.getenv("GEMMA4_FUNCTIONAL_DECODER_SEQ_LEN", "1024"))
    artifact = ARTIFACT_DIR / f"layer{layer_idx}_{layer_type}_seq{seq_len}_batch{batch}_host_timings.json"
    artifact_suffix = os.getenv("GEMMA4_OPTIMIZED_ARTIFACT_SUFFIX")
    if artifact_suffix:
        assert artifact_suffix.replace("_", "").isalnum(), "artifact suffix must be alphanumeric/underscores"
        suffixed_artifact = artifact.with_name(f"{artifact.stem}_{artifact_suffix}{artifact.suffix}")
        artifact.replace(suffixed_artifact)
        artifact = suffixed_artifact
    _mark_optimized_provenance(
        artifact,
        "GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_FUNCTIONAL_DECODER_PERF=1 "
        "python -m pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/"
        "test_optimized_decoder.py::test_optimized_perf_profile",
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
@pytest.mark.parametrize("implementation", ["functional", "optimized"])
def test_batch32_prefill_profile(mesh_device, device_params, layer_idx, implementation):
    """Measure real batch-32 prefill in the same seq-1024 regime."""

    if os.getenv("GEMMA4_OPTIMIZED_BATCH32_PREFILL") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_BATCH32_PREFILL=1 to run batch-32 prefill")
    cfg = functional_tests._load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    state = functional_tests._load_layer_state(layer_idx)
    batch = 32
    seq_len = int(os.getenv("GEMMA4_FUNCTIONAL_DECODER_SEQ_LEN", "1024"))
    token_capacity = seq_len + 1
    shared_physical = layer_type == "sliding_attention"
    functional_tests.torch.manual_seed(6000 + layer_idx)
    hidden = functional_tests.torch.randn(
        batch, seq_len, functional_tests.HIDDEN_SIZE, dtype=functional_tests.torch.bfloat16
    )
    positions = functional_tests.torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
    rotary = functional_tests.Gemma4TextRotaryEmbedding(cfg)
    cos, sin = rotary(hidden, positions, layer_type=layer_type)
    decoder_cls = functional_tests.FunctionalDecoder if implementation == "functional" else OptimizedDecoder
    decoder = decoder_cls.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    one_user_shape = functional_tests._cache_shape(
        layer_type, shared_physical=shared_physical, token_capacity=token_capacity
    )
    blocks_per_user = one_user_shape[0]
    cache_shape = (batch * blocks_per_user, *one_user_shape[1:])
    page_table = functional_tests._as_tt(
        mesh_device,
        functional_tests.torch.arange(batch * blocks_per_user, dtype=functional_tests.torch.int32).view(
            batch, blocks_per_user
        ),
        dtype=functional_tests.ttnn.int32,
        layout=functional_tests.ttnn.ROW_MAJOR_LAYOUT,
    )
    kv_cache = (
        functional_tests._as_tt(
            mesh_device, functional_tests.torch.zeros(cache_shape, dtype=functional_tests.torch.bfloat16)
        ),
        functional_tests._as_tt(
            mesh_device, functional_tests.torch.zeros(cache_shape, dtype=functional_tests.torch.bfloat16)
        ),
    )
    args = {
        "hidden_states": functional_tests._as_tt(mesh_device, hidden.unsqueeze(1)),
        "position_cos": functional_tests._as_tt(mesh_device, cos.unsqueeze(1)),
        "position_sin": functional_tests._as_tt(mesh_device, sin.unsqueeze(1)),
        "page_table": page_table,
        "kv_cache": kv_cache,
    }
    warm = decoder.prefill_forward(**args)
    functional_tests.ttnn.synchronize_device(mesh_device)
    functional_tests.ttnn.deallocate(warm)
    start = time.perf_counter()
    output = decoder.prefill_forward(**args)
    functional_tests.ttnn.synchronize_device(mesh_device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert output.shape[0] == batch
    artifact = ARTIFACT_DIR / f"batch32_prefill_{implementation}_layer{layer_idx}_{layer_type}.json"
    artifact.write_text(
        json.dumps(
            {
                "model_id": functional_tests.MODEL_ID,
                "implementation": implementation,
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "batch": batch,
                "sequence_length": seq_len,
                "prefill_host_ms": elapsed_ms,
                "output_shape": list(output.shape),
                "optimized_policy": (
                    OptimizedDecoder.optimization_policy.__dict__ if implementation == "optimized" else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _mark_optimized_provenance(
        artifact,
        "GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZED_BATCH32_PREFILL=1 "
        "python -m pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/"
        "test_optimized_decoder.py::test_batch32_prefill_profile",
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_decode_only_profile(monkeypatch, mesh_device, device_params, layer_idx):
    """Profile one traced decode without overflowing Tracy during cache prefill.

    The cache is intentionally zero initialized.  Its contents affect values,
    but not the decode operation topology or tensor geometry, so this is a
    timing/profile harness rather than a correctness test.
    """

    if os.getenv("GEMMA4_OPTIMIZED_DECODE_PROFILE") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_DECODE_PROFILE=1 to run the decode-only profiler harness")
    try:
        from tracy import signpost
    except ImportError:
        signpost = lambda *_, **__: None

    _select_optimized(monkeypatch)
    cfg = functional_tests._load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    state = functional_tests._load_layer_state(layer_idx)
    seq_len = int(os.getenv("GEMMA4_FUNCTIONAL_DECODER_SEQ_LEN", "1024"))
    batch = 1
    token_capacity = seq_len + 1
    functional_tests.torch.manual_seed(4000 + layer_idx)
    decode_hidden = functional_tests.torch.randn(
        batch, 1, functional_tests.HIDDEN_SIZE, dtype=functional_tests.torch.bfloat16
    )
    rotary = functional_tests.Gemma4TextRotaryEmbedding(cfg)
    positions = functional_tests.torch.full((batch, 1), seq_len, dtype=functional_tests.torch.long)
    decode_cos, decode_sin = rotary(decode_hidden, positions, layer_type=layer_type)

    decoder = OptimizedDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    shared_physical, cache_shape, page_table_shape = _dedicated_profiler_cache_geometry(
        layer_type,
        batch=batch,
        token_capacity=token_capacity,
    )
    page_table = functional_tests._as_tt(
        mesh_device,
        functional_tests.torch.arange(
            page_table_shape[0] * page_table_shape[1], dtype=functional_tests.torch.int32
        ).view(page_table_shape),
        dtype=functional_tests.ttnn.int32,
        layout=functional_tests.ttnn.ROW_MAJOR_LAYOUT,
    )
    kv_cache = (
        functional_tests._as_tt(
            mesh_device, functional_tests.torch.zeros(cache_shape, dtype=functional_tests.torch.bfloat16)
        ),
        functional_tests._as_tt(
            mesh_device, functional_tests.torch.zeros(cache_shape, dtype=functional_tests.torch.bfloat16)
        ),
    )
    if layer_type == "sliding_attention":
        tt_decode_cos = decode_cos.unsqueeze(0)
        tt_decode_sin = decode_sin.unsqueeze(0)
    else:
        tt_decode_cos = decode_cos.transpose(0, 1).unsqueeze(0)
        tt_decode_sin = decode_sin.transpose(0, 1).unsqueeze(0)
    decode_args = {
        "hidden_states": functional_tests._as_tt(mesh_device, decode_hidden.transpose(0, 1).unsqueeze(0)),
        "position_cos": functional_tests._as_tt(mesh_device, tt_decode_cos),
        "position_sin": functional_tests._as_tt(mesh_device, tt_decode_sin),
        "current_pos": functional_tests._as_tt(
            mesh_device,
            functional_tests.torch.full((batch,), seq_len, dtype=functional_tests.torch.int32),
            dtype=functional_tests.ttnn.int32,
            layout=functional_tests.ttnn.ROW_MAJOR_LAYOUT,
        ),
        "page_table": page_table,
        "kv_cache": kv_cache,
    }

    decoder.decode_forward(**decode_args)
    functional_tests.ttnn.synchronize_device(mesh_device)
    trace_id = functional_tests.ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(**decode_args)
    functional_tests.ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    functional_tests.ttnn.synchronize_device(mesh_device)
    functional_tests.ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)

    case_id = f"decode_only_layer{layer_idx}_{layer_type}_seq{seq_len}_batch1"
    artifact_suffix = os.getenv("GEMMA4_OPTIMIZED_ARTIFACT_SUFFIX")
    if artifact_suffix:
        assert artifact_suffix.replace("_", "").isalnum(), "artifact suffix must be alphanumeric/underscores"
        case_id = f"{case_id}_{artifact_suffix}"
    signpost(f"PERF_DECODE_{case_id}", f"cache_shape={cache_shape}")
    start = time.perf_counter()
    functional_tests.ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    functional_tests.ttnn.synchronize_device(mesh_device)
    decode_ms = (time.perf_counter() - start) * 1000
    signpost(f"PERF_DECODE_{case_id}_END", f"cache_shape={cache_shape}")
    functional_tests.ttnn.release_trace(mesh_device, trace_id)

    assert traced_output.shape[-2] >= 1
    assert decoder.optimized_decode_invocations == 2
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / f"{case_id}_host_timings.json").write_text(
        json.dumps(
            {
                "model_id": functional_tests.MODEL_ID,
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "sequence_length": seq_len,
                "decode_current_pos": seq_len,
                "decode_batch": batch,
                "shared_physical_cache": shared_physical,
                "cache_shape": cache_shape,
                "page_table_shape": page_table_shape,
                "zero_initialized_cache_for_profile": True,
                "decode_trace_host_ms": decode_ms,
                "provenance": {
                    "exact_command": "GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZED_DECODE_PROFILE=1 "
                    "python -m pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/"
                    "test_optimized_decoder.py::test_optimized_decode_only_profile"
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _mark_optimized_provenance(
        ARTIFACT_DIR / f"{case_id}_host_timings.json",
        "GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZED_DECODE_PROFILE=1 "
        "python -m pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/"
        "test_optimized_decoder.py::test_optimized_decode_only_profile",
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_prefill_only_profile(monkeypatch, mesh_device, device_params, layer_idx):
    """Profile one warmed prefill independently from decode trace capture."""

    if os.getenv("GEMMA4_OPTIMIZED_PREFILL_PROFILE") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_PREFILL_PROFILE=1 to run the prefill-only profiler harness")
    try:
        from tracy import signpost
    except ImportError:
        signpost = lambda *_, **__: None

    _select_optimized(monkeypatch)
    cfg = functional_tests._load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    state = functional_tests._load_layer_state(layer_idx)
    seq_len = int(os.getenv("GEMMA4_FUNCTIONAL_DECODER_SEQ_LEN", "1024"))
    functional_tests.torch.manual_seed(5000 + layer_idx)
    hidden = functional_tests.torch.randn(
        1, seq_len, functional_tests.HIDDEN_SIZE, dtype=functional_tests.torch.bfloat16
    )
    positions = functional_tests.torch.arange(seq_len).unsqueeze(0)
    rotary = functional_tests.Gemma4TextRotaryEmbedding(cfg)
    cos, sin = rotary(hidden, positions, layer_type=layer_type)
    decoder = OptimizedDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    batch = 1
    shared_physical, cache_shape, page_table_shape = _dedicated_profiler_cache_geometry(
        layer_type,
        batch=batch,
        token_capacity=seq_len + 1,
    )
    page_table = functional_tests._as_tt(
        mesh_device,
        functional_tests.torch.arange(
            page_table_shape[0] * page_table_shape[1], dtype=functional_tests.torch.int32
        ).view(page_table_shape),
        dtype=functional_tests.ttnn.int32,
        layout=functional_tests.ttnn.ROW_MAJOR_LAYOUT,
    )
    kv_cache = (
        functional_tests._as_tt(
            mesh_device, functional_tests.torch.zeros(cache_shape, dtype=functional_tests.torch.bfloat16)
        ),
        functional_tests._as_tt(
            mesh_device, functional_tests.torch.zeros(cache_shape, dtype=functional_tests.torch.bfloat16)
        ),
    )
    args = {
        "hidden_states": functional_tests._as_tt(mesh_device, hidden.unsqueeze(1)),
        "position_cos": functional_tests._as_tt(mesh_device, cos.unsqueeze(1)),
        "position_sin": functional_tests._as_tt(mesh_device, sin.unsqueeze(1)),
        "page_table": page_table,
        "kv_cache": kv_cache,
    }
    decoder.prefill_forward(**args)
    functional_tests.ttnn.synchronize_device(mesh_device)
    case_id = f"prefill_only_layer{layer_idx}_{layer_type}_seq{seq_len}_batch1"
    artifact_suffix = os.getenv("GEMMA4_OPTIMIZED_ARTIFACT_SUFFIX")
    if artifact_suffix:
        assert artifact_suffix.replace("_", "").isalnum(), "artifact suffix must be alphanumeric/underscores"
        case_id = f"{case_id}_{artifact_suffix}"
    signpost(f"PERF_PREFILL_{case_id}", f"cache_shape={cache_shape}")
    start = time.perf_counter()
    output = decoder.prefill_forward(**args)
    functional_tests.ttnn.synchronize_device(mesh_device)
    prefill_ms = (time.perf_counter() - start) * 1000
    signpost(f"PERF_PREFILL_{case_id}_END", f"cache_shape={cache_shape}")
    assert output.shape[0] == batch
    assert decoder.optimized_prefill_invocations == 2
    artifact = ARTIFACT_DIR / f"{case_id}_host_timings.json"
    artifact.write_text(
        json.dumps(
            {
                "model_id": functional_tests.MODEL_ID,
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "sequence_length": seq_len,
                "prefill_batch": batch,
                "shared_physical_cache": shared_physical,
                "cache_shape": cache_shape,
                "page_table_shape": page_table_shape,
                "prefill_host_ms": prefill_ms,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _mark_optimized_provenance(
        artifact,
        "GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZED_PREFILL_PROFILE=1 "
        "python -m pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/"
        "test_optimized_decoder.py::test_optimized_prefill_only_profile",
    )


def test_optimized_hot_path_fallback_audit():
    runtime_methods = (
        OptimizedDecoder.prefill_forward,
        OptimizedDecoder.decode_forward,
        OptimizedDecoder.prefill_forward,
        OptimizedDecoder._attention_decode,
        OptimizedDecoder._dense_mlp,
        OptimizedDecoder._router_weights,
        OptimizedDecoder._moe_decode_single_user,
    )
    source = "\n".join(inspect.getsource(method) for method in runtime_methods)
    assert "torch." not in source
    assert "import torch" not in source
    assert "ttnn.from_torch" not in source
    assert "ttnn.to_torch" not in source
