# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and graph-path coverage for the Gemma-4 fused decoder."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests import test_functional_decoder as functional_tests
from models.autoports.google_gemma_4_26b_a4b_it.tests.test_functional_decoder import (
    HIDDEN_SIZE,
    _as_tt,
    _cache_shape,
    _load_layer_state,
    _load_text_config,
    _page_table,
    _to_torch,
)
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
    MOE_INTERMEDIATE_SIZE,
    FunctionalDecoder,
)
from models.autoports.google_gemma_4_26b_a4b_it.tt.fused_decoder import FusedDecoder
from models.common.utility_functions import comp_pcc

ARTIFACT_DIR = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/fused_decoder")


class _DenseSplitCandidate(FusedDecoder):
    """Best correct rejected dense candidate, retained for same-run selection."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mlp_gate_up = ttnn.concat(
            [self.weights.mlp_gate, self.weights.mlp_up],
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _dense_mlp(self, x, *, fold_activation):
        physical_m = x.padded_shape[-2]
        config = ttnn.MinimalMatmulConfig(
            M_block_size=1 if physical_m == 32 else 4,
            K_block_size=4,
            N_block_size=8,
            subblock_h=1,
            subblock_w=2,
            compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
        )
        gate, up = ttnn.experimental.minimal_matmul_split(
            x,
            self.mlp_gate_up,
            chunks=2,
            dim=-1,
            config=config,
            dtype=self.activation_dtype,
        )
        gate = ttnn.gelu(gate, fast_and_approximate_mode=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.linear(
            hidden,
            self.weights.mlp_down,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


class _AlternativeExpertActivationCandidate(FusedDecoder):
    """Use composite GeGLU instead of the retained explicit lowering."""

    def _packed_expert_activation(self, up_gate, *, use_composite):
        return super()._packed_expert_activation(up_gate, use_composite=True)


class _DenseActivationFoldCandidate(FusedDecoder):
    """Isolate GELU folding into the dense branch's consuming multiply."""

    def _dense_mlp(self, x, *, fold_activation):
        gate = ttnn.linear(
            x,
            self.weights.mlp_gate,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        up = ttnn.linear(
            x,
            self.weights.mlp_up,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden = ttnn.mul(
            gate,
            up,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0)],
        )
        return ttnn.linear(
            hidden,
            self.weights.mlp_down,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


@contextmanager
def _fused_acceptance_oracle():
    """Run the unchanged Stage 01 oracle against the Stage 02 implementation."""
    original_decoder = functional_tests.FunctionalDecoder
    original_artifact_dir = functional_tests.ARTIFACT_DIR
    functional_tests.FunctionalDecoder = FusedDecoder
    functional_tests.ARTIFACT_DIR = ARTIFACT_DIR
    try:
        yield
    finally:
        functional_tests.FunctionalDecoder = original_decoder
        functional_tests.ARTIFACT_DIR = original_artifact_dir


def _rewrite_fused_provenance(path: Path, exact_command: str) -> None:
    artifact = json.loads(path.read_text())
    provenance = artifact.get("provenance", {})
    decoder_path = Path(inspect.getsourcefile(FusedDecoder))
    functional_decoder_path = Path(inspect.getsourcefile(FunctionalDecoder))
    functional_test_path = Path(inspect.getsourcefile(functional_tests))
    provenance.update(
        {
            "exact_command": exact_command,
            "decoder": str(decoder_path),
            "fused_decoder_sha256": hashlib.sha256(decoder_path.read_bytes()).hexdigest(),
            "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "functional_decoder_sha256": hashlib.sha256(functional_decoder_path.read_bytes()).hexdigest(),
            "functional_test_sha256": hashlib.sha256(functional_test_path.read_bytes()).hexdigest(),
        }
    )
    artifact["provenance"] = provenance
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,shared_physical",
    [(0, True), (5, False), (5, True)],
    ids=["sliding_shared", "full_natural", "full_shared_view"],
)
def test_fused_hf_acceptance(mesh_device, device_params, layer_idx, shared_physical):
    with _fused_acceptance_oracle():
        functional_tests.test_functional_decoder_real_weights_prefill_decode(
            mesh_device, device_params, layer_idx, shared_physical, 0.995
        )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
def test_fused_traced_decode_batch_contract(mesh_device, device_params, layer_idx, batch):
    with _fused_acceptance_oracle():
        functional_tests.test_traced_decode_batch_contract(mesh_device, device_params, layer_idx, batch)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_fused_non_aligned_prefill(mesh_device, device_params, layer_idx, monkeypatch):
    monkeypatch.setenv("GEMMA4_BOUNDARY_LENGTHS", "31,33,1025")
    with _fused_acceptance_oracle():
        functional_tests.test_paged_prefill_logical_boundary_lengths(mesh_device, device_params, layer_idx)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_fused_bounded_modulo_cache_integrity(mesh_device, device_params):
    with _fused_acceptance_oracle():
        functional_tests.test_bounded_modulo_prefill_tail_cache_integrity(mesh_device, device_params)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_fused_advertised_context_traced_decode(mesh_device, device_params, layer_idx, monkeypatch):
    monkeypatch.setenv("GEMMA4_FUNCTIONAL_DECODER_CONTEXT", "1")
    with _fused_acceptance_oracle():
        functional_tests.test_advertised_context_traced_decode(mesh_device, device_params, layer_idx)
    layer_type = _load_text_config().layer_types[layer_idx]
    _rewrite_fused_provenance(
        ARTIFACT_DIR / f"advertised_context_decode_{layer_type}.json",
        "GEMMA4_RANGE_DOWNLOAD=1 TTNN_CONFIG_OVERRIDES='{\"throw_exception_on_fallback\": true}' pytest -q "
        "models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::"
        "test_fused_advertised_context_traced_decode",
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_fused_prefill_capacity_probe(mesh_device, device_params, layer_idx):
    """Run the Stage 01 real-weight capacity probe through the fused prefill path."""
    requested = os.getenv("GEMMA4_PREFILL_CAPACITY_LENGTH")
    if requested is None:
        pytest.skip("set GEMMA4_PREFILL_CAPACITY_LENGTH to run fused capacity")
    with _fused_acceptance_oracle():
        functional_tests.test_prefill_capacity_probe(mesh_device, device_params, layer_idx)
    seq_len = int(requested)
    layer_type = _load_text_config().layer_types[layer_idx]
    _rewrite_fused_provenance(
        ARTIFACT_DIR / f"prefill_capacity_{layer_type}_{seq_len}.json",
        f"GEMMA4_PREFILL_CAPACITY_LENGTH={seq_len} GEMMA4_RANGE_DOWNLOAD=1 "
        "TTNN_CONFIG_OVERRIDES='{\"throw_exception_on_fallback\": true}' pytest -q "
        "models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::"
        "test_fused_prefill_capacity_probe",
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx,shared_physical", [(0, True), (5, False)])
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
def test_fused_decoder_perf_profile(mesh_device, device_params, layer_idx, shared_physical, batch):
    """Profiler-signposted fused path; opt-in because Tracy owns the process."""
    if os.getenv("GEMMA4_FUSED_DECODER_PROFILE") != "1":
        pytest.skip("set GEMMA4_FUSED_DECODER_PROFILE=1 to run the profiler harness")
    old_value = os.environ.get("GEMMA4_FUNCTIONAL_DECODER_PERF")
    os.environ["GEMMA4_FUNCTIONAL_DECODER_PERF"] = "1"
    try:
        with _fused_acceptance_oracle():
            functional_tests.test_functional_decoder_perf_profile(
                mesh_device, device_params, layer_idx, shared_physical, batch
            )
        layer_type = _load_text_config().layer_types[layer_idx]
        seq_len = int(os.getenv("GEMMA4_FUNCTIONAL_DECODER_SEQ_LEN", "1024"))
        _rewrite_fused_provenance(
            ARTIFACT_DIR / f"layer{layer_idx}_{layer_type}_seq{seq_len}_batch{batch}_host_timings.json",
            f"GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_FUSED_DECODER_PROFILE=1 "
            f"GEMMA4_FUNCTIONAL_DECODER_SEQ_LEN={seq_len} "
            "TTNN_CONFIG_OVERRIDES='{\"throw_exception_on_fallback\": true}' pytest -q "
            "models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::"
            "test_fused_decoder_perf_profile",
        )
    finally:
        if old_value is None:
            os.environ.pop("GEMMA4_FUNCTIONAL_DECODER_PERF", None)
        else:
            os.environ["GEMMA4_FUNCTIONAL_DECODER_PERF"] = old_value


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 0}],
    indirect=True,
)
def test_moe_compute_gemma4_target_shape_candidate(mesh_device, device_params, monkeypatch):
    """Target-shaped minimal repro for the highest-priority dedicated MoE op."""
    if os.getenv("GEMMA4_MOE_COMPUTE_CANDIDATE") != "1":
        pytest.skip("set GEMMA4_MOE_COMPUTE_CANDIDATE=1 to run the dedicated-op candidate")

    from tests.ttnn.nightly.unit_tests.operations.experimental import test_moe_compute_single_card as candidate
    from ttnn.experimental.moe_compute_utils import (
        auto_output_width_shard_dim,
        effective_matmul_ring_size,
    )
    from ttnn.operations.ccl import MoEActivationFunction

    state = _load_layer_state(0)
    prefix = "model.language_model.layers.0.experts"
    gate_up = state[f"{prefix}.gate_up_proj"]
    down = state[f"{prefix}.down_proj"]
    gate = gate_up[:, :MOE_INTERMEDIATE_SIZE, :].transpose(-2, -1).contiguous().unsqueeze(0)
    up = gate_up[:, MOE_INTERMEDIATE_SIZE:, :].transpose(-2, -1).contiguous().unsqueeze(0)
    down = down.transpose(-2, -1).contiguous().unsqueeze(0)
    monkeypatch.setattr(candidate, "create_torch_w0", lambda *_: up)
    monkeypatch.setattr(candidate, "create_torch_w1", lambda *_: gate)
    monkeypatch.setattr(candidate, "create_torch_w2", lambda *_: down)

    ring_size = effective_matmul_ring_size(mesh_device)
    with pytest.raises(AssertionError, match="Matmul output tensor verification failed"):
        candidate._run_moe_compute_single_card_test(
            mesh_device=mesh_device,
            mesh_shape=(1, 1),
            experts_per_device=128,
            tokens_per_device=32,
            selected_experts_k=8,
            N=704,
            hidden_size=2816,
            output_height_shard_dim=4,
            output_width_shard_dim=auto_output_width_shard_dim(2816, matmul_ring_size=ring_size),
            dtype=ttnn.bfloat16,
            activation_type=MoEActivationFunction.GELU,
            has_bias=False,
        )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = ARTIFACT_DIR / "rejected_moe_compute_candidate.json"
    artifact_path.write_text(
        json.dumps(
            {
                "candidate": "ttnn.experimental.moe_compute(compute_only=True)",
                "weights": "real layer-0 Gemma checkpoint weights packed to the operation's required BF4 format",
                "shape": {"tokens": 32, "experts": 128, "top_k": 8, "hidden": 2816, "intermediate": 704},
                "activation": "GELU",
                "matmul_validation_threshold": 0.984,
                "observed_expert_pcc": {"126": 0.983965, "127": 0.977038},
                "required_active_token_pcc": 0.983,
                "target_shape_and_real_weight_validation": "rejected: expert 127 fails PCC",
                "integration_blocker": (
                    "compute_only returns a 110-core x 2-double-buffer x 32-token x 2816 tensor containing "
                    "only the final two expert buffers; the token-ordered, score-reduced output is produced only "
                    "by selective_reduce_combine in the non-compute-only collective path"
                ),
                "single_device_constraint": (
                    "non-compute-only requires a cluster_axis and fabric combine; that collective path is outside "
                    "the single-device decoder contract"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _rewrite_fused_provenance(
        artifact_path,
        "GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_MOE_COMPUTE_CANDIDATE=1 pytest -q "
        "models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::"
        "test_moe_compute_gemma4_target_shape_candidate",
    )


def _run_prefill_decode(
    decoder, mesh_device, *, layer_type: str, hidden, decode_hidden, cos, sin, decode_cos, decode_sin
):
    page_table = _as_tt(
        mesh_device,
        _page_table(layer_type, shared_physical=False),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cache_shape = _cache_shape(layer_type, shared_physical=False)
    kv_cache = (
        _as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
        _as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
    )
    prefill = decoder.prefill_forward(
        _as_tt(mesh_device, hidden.unsqueeze(1)),
        position_cos=_as_tt(mesh_device, cos.unsqueeze(1)),
        position_sin=_as_tt(mesh_device, sin.unsqueeze(1)),
        page_table=page_table,
        kv_cache=kv_cache,
    )
    if layer_type == "sliding_attention":
        tt_decode_cos = decode_cos.unsqueeze(0)
        tt_decode_sin = decode_sin.unsqueeze(0)
    else:
        tt_decode_cos = decode_cos.transpose(0, 1).unsqueeze(0)
        tt_decode_sin = decode_sin.transpose(0, 1).unsqueeze(0)
    decode = decoder.decode_forward(
        _as_tt(mesh_device, decode_hidden.transpose(0, 1).unsqueeze(0)),
        position_cos=_as_tt(mesh_device, tt_decode_cos),
        position_sin=_as_tt(mesh_device, tt_decode_sin),
        current_pos=_as_tt(
            mesh_device,
            torch.tensor([hidden.shape[1]], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        page_table=page_table,
        kv_cache=kv_cache,
    )
    ttnn.synchronize_device(mesh_device)
    return (
        _to_torch(mesh_device, prefill).reshape(1, hidden.shape[1], HIDDEN_SIZE).to(torch.bfloat16),
        _to_torch(mesh_device, decode).reshape(1, 1, HIDDEN_SIZE).to(torch.bfloat16),
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_fused_rewrites_match_functional_real_weights(mesh_device, device_params, layer_idx):
    """Each retained rewrite must remain equivalent on target shapes/dtypes."""
    cfg = _load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    state = _load_layer_state(layer_idx)
    torch.manual_seed(2200 + layer_idx)
    seq_len = 32
    hidden = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
    decode_hidden = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
    rotary = Gemma4TextRotaryEmbedding(cfg)
    cos, sin = rotary(hidden, torch.arange(seq_len).unsqueeze(0), layer_type=layer_type)
    decode_cos, decode_sin = rotary(
        decode_hidden,
        torch.tensor([[seq_len]], dtype=torch.long),
        layer_type=layer_type,
    )

    functional = FunctionalDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    fused = FusedDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    functional_prefill, functional_decode = _run_prefill_decode(
        functional,
        mesh_device,
        layer_type=layer_type,
        hidden=hidden,
        decode_hidden=decode_hidden,
        cos=cos,
        sin=sin,
        decode_cos=decode_cos,
        decode_sin=decode_sin,
    )
    fused_prefill, fused_decode = _run_prefill_decode(
        fused,
        mesh_device,
        layer_type=layer_type,
        hidden=hidden,
        decode_hidden=decode_hidden,
        cos=cos,
        sin=sin,
        decode_cos=decode_cos,
        decode_sin=decode_sin,
    )

    prefill_ok, prefill_pcc = comp_pcc(functional_prefill, fused_prefill, 0.99)
    decode_ok, decode_pcc = comp_pcc(functional_decode, fused_decode, 0.99)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / f"functional_equivalence_layer{layer_idx}_{layer_type}.json").write_text(
        json.dumps(
            {
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "sequence_length": seq_len,
                "functional_vs_fused_prefill_pcc": float(prefill_pcc),
                "functional_vs_fused_decode_pcc": float(decode_pcc),
                "threshold": 0.99,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    assert prefill_ok, prefill_pcc
    assert decode_ok, decode_pcc


def test_fused_hot_path_is_owned_and_contains_fusions():
    assert FusedDecoder is not FunctionalDecoder
    assert FusedDecoder.decode_forward is not FunctionalDecoder.decode_forward
    assert FusedDecoder._prefill_forward_single_user is not FunctionalDecoder._prefill_forward_single_user
    moe_decode_source = inspect.getsource(FusedDecoder._moe_decode_single_user)
    moe_prefill_source = inspect.getsource(FusedDecoder._moe_prefill_tile)
    expert_activation_source = inspect.getsource(FusedDecoder._packed_expert_activation)
    dense_mlp_source = inspect.getsource(FusedDecoder._dense_mlp)
    attention_source = inspect.getsource(FusedDecoder._attention_decode)
    decode_source = inspect.getsource(FusedDecoder.decode_forward)
    prefill_source = inspect.getsource(FusedDecoder._prefill_forward_single_user)
    assert moe_decode_source.count("ttnn.sparse_matmul(") == 2
    assert moe_prefill_source.count("ttnn.sparse_matmul(") == 2
    assert "self.expert_up_gate" in moe_decode_source
    assert "input_tensor_a_activations" in expert_activation_source
    assert "ttnn.gelu(" in dense_mlp_source
    assert "paged_fused_update_cache" in attention_source
    assert "super().decode_forward" not in decode_source
    assert "super()._prefill_forward_single_user" not in prefill_source


def test_fused_hot_path_fallback_audit():
    methods = (
        FusedDecoder.decode_forward,
        FusedDecoder._prefill_forward_single_user,
        FusedDecoder._attention_decode,
        FusedDecoder._moe_decode_single_user,
        FusedDecoder._moe_prefill_tile,
    )
    source = "\n".join(inspect.getsource(method) for method in methods)
    for token in ("torch.", "import torch", "ttnn.from_torch", "ttnn.to_torch"):
        assert token not in source


def _make_perf_args(decoder, mesh_device, *, layer_type: str, seq_len: int, batch: int):
    torch.manual_seed(3200 + decoder.layer_idx + batch)
    hidden = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
    decode_hidden = torch.randn(batch, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
    rotary = Gemma4TextRotaryEmbedding(decoder.hf_config)
    cos, sin = rotary(hidden, torch.arange(seq_len).unsqueeze(0), layer_type=layer_type)
    decode_positions = torch.full((batch, 1), seq_len, dtype=torch.long)
    decode_cos, decode_sin = rotary(decode_hidden, decode_positions, layer_type=layer_type)
    one_user_shape = _cache_shape(layer_type, shared_physical=False, token_capacity=seq_len + 1)
    blocks_per_user = one_user_shape[0]
    cache_shape = (batch * blocks_per_user, *one_user_shape[1:])
    page_table = _as_tt(
        mesh_device,
        torch.arange(batch * blocks_per_user, dtype=torch.int32).view(batch, blocks_per_user),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    kv_cache = (
        _as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
        _as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
    )
    prefill_args = {
        "hidden_states": _as_tt(mesh_device, hidden.unsqueeze(1)),
        "position_cos": _as_tt(mesh_device, cos.unsqueeze(1)),
        "position_sin": _as_tt(mesh_device, sin.unsqueeze(1)),
        "page_table": page_table,
        "kv_cache": kv_cache,
    }
    if layer_type == "sliding_attention":
        tt_decode_cos = decode_cos.unsqueeze(0)
        tt_decode_sin = decode_sin.unsqueeze(0)
    else:
        tt_decode_cos = decode_cos.transpose(0, 1).unsqueeze(0)
        tt_decode_sin = decode_sin.transpose(0, 1).unsqueeze(0)
    decode_args = {
        "hidden_states": _as_tt(mesh_device, decode_hidden.transpose(0, 1).unsqueeze(0)),
        "position_cos": _as_tt(mesh_device, tt_decode_cos),
        "position_sin": _as_tt(mesh_device, tt_decode_sin),
        "current_pos": _as_tt(
            mesh_device,
            torch.full((batch,), seq_len, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "page_table": page_table,
        "kv_cache": kv_cache,
    }
    return prefill_args, decode_args


def _measure_warmed(decoder, mesh_device, *, prefill_args, decode_args, batch: int, repeats: int = 5):
    result = {}
    if batch == 1:
        decoder.prefill_forward(**prefill_args)
        ttnn.synchronize_device(mesh_device)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            decoder.prefill_forward(**prefill_args)
            ttnn.synchronize_device(mesh_device)
            samples.append((time.perf_counter() - start) * 1000)
        result["prefill_host_ms_median"] = statistics.median(samples)
        result["prefill_host_ms_samples"] = samples

    decoder.decode_forward(**decode_args)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(**decode_args)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh_device)
        samples.append((time.perf_counter() - start) * 1000)
    ttnn.release_trace(mesh_device, trace_id)
    result["decode_trace_host_ms_median"] = statistics.median(samples)
    result["decode_trace_host_ms_samples"] = samples
    result["output_shape"] = list(traced_output.shape)
    return result


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
def test_fused_decoder_functional_ab_latency(mesh_device, device_params, layer_idx, batch):
    if os.getenv("GEMMA4_FUSED_DECODER_PERF") != "1":
        pytest.skip("set GEMMA4_FUSED_DECODER_PERF=1 to run functional/fused A/B")
    cfg = _load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    state = _load_layer_state(layer_idx)
    functional = FunctionalDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    fused = FusedDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    seq_len = int(os.getenv("GEMMA4_FUSED_DECODER_SEQ_LEN", "1024"))
    functional_args = _make_perf_args(functional, mesh_device, layer_type=layer_type, seq_len=seq_len, batch=batch)
    fused_args = _make_perf_args(fused, mesh_device, layer_type=layer_type, seq_len=seq_len, batch=batch)
    functional_result = _measure_warmed(
        functional,
        mesh_device,
        prefill_args=functional_args[0],
        decode_args=functional_args[1],
        batch=batch,
    )
    fused_result = _measure_warmed(
        fused,
        mesh_device,
        prefill_args=fused_args[0],
        decode_args=fused_args[1],
        batch=batch,
    )
    artifact = {
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "sequence_length": seq_len,
        "batch": batch,
        "functional": functional_result,
        "fused": fused_result,
        "decode_speedup": functional_result["decode_trace_host_ms_median"]
        / fused_result["decode_trace_host_ms_median"],
    }
    if batch == 1:
        artifact["prefill_speedup"] = (
            functional_result["prefill_host_ms_median"] / fused_result["prefill_host_ms_median"]
        )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = ARTIFACT_DIR / f"candidate_ab_layer{layer_idx}_{layer_type}_batch{batch}.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    _rewrite_fused_provenance(
        artifact_path,
        "GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_FUSED_DECODER_PERF=1 GEMMA4_FUSED_DECODER_SEQ_LEN=1024 "
        "TTNN_CONFIG_OVERRIDES='{\"throw_exception_on_fallback\": true}' pytest -q "
        "models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::"
        "test_fused_decoder_functional_ab_latency",
    )
    assert fused_result["decode_trace_host_ms_median"] < functional_result["decode_trace_host_ms_median"]


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
@pytest.mark.parametrize(
    "candidate_name,candidate_cls",
    [
        ("dense_split", _DenseSplitCandidate),
        ("alternative_expert_activation", _AlternativeExpertActivationCandidate),
    ],
    ids=["dense_split", "alternative_expert_activation"],
)
def test_final_vs_best_candidates(mesh_device, device_params, layer_idx, batch, candidate_name, candidate_cls):
    """Select the winner with interleaved trace replay in one process."""
    if os.getenv("GEMMA4_FUSED_DECODER_CANDIDATE_AB") != "1":
        pytest.skip("set GEMMA4_FUSED_DECODER_CANDIDATE_AB=1 to rerun candidate selection")

    cfg = _load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    state = _load_layer_state(layer_idx)
    final = FusedDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    candidate = candidate_cls.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    final_args = _make_perf_args(final, mesh_device, layer_type=layer_type, seq_len=1024, batch=batch)
    candidate_args = _make_perf_args(candidate, mesh_device, layer_type=layer_type, seq_len=1024, batch=batch)

    final_prefill_eager = final.prefill_forward(**final_args[0])
    final_eager = final.decode_forward(**final_args[1])
    candidate_prefill_eager = candidate.prefill_forward(**candidate_args[0])
    candidate_eager = candidate.decode_forward(**candidate_args[1])
    ttnn.synchronize_device(mesh_device)
    final_torch = _to_torch(mesh_device, final_eager)
    candidate_torch = _to_torch(mesh_device, candidate_eager)
    equivalent, candidate_pcc = comp_pcc(final_torch, candidate_torch, 0.99)
    assert equivalent, candidate_pcc
    prefill_pcc = None
    if batch == 1:
        prefill_equivalent, prefill_pcc = comp_pcc(
            _to_torch(mesh_device, final_prefill_eager),
            _to_torch(mesh_device, candidate_prefill_eager),
            0.99,
        )
        assert prefill_equivalent, prefill_pcc

    def capture(decoder, decode_args):
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        decoder.decode_forward(**decode_args)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        return trace_id

    final_trace = capture(final, final_args[1])
    candidate_trace = capture(candidate, candidate_args[1])
    final_samples = []
    candidate_samples = []
    final_prefill_samples = []
    candidate_prefill_samples = []

    def sample(trace_id, output):
        start = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh_device)
        output.append((time.perf_counter() - start) * 1000)

    for iteration in range(101):
        ordered = (
            ((final_trace, final_samples), (candidate_trace, candidate_samples))
            if iteration % 2 == 0
            else ((candidate_trace, candidate_samples), (final_trace, final_samples))
        )
        for trace_id, output in ordered:
            sample(trace_id, output)

    ttnn.release_trace(mesh_device, final_trace)
    ttnn.release_trace(mesh_device, candidate_trace)

    if batch == 1:
        for iteration in range(21):
            ordered = (
                (
                    (final, final_args[0], final_prefill_samples),
                    (candidate, candidate_args[0], candidate_prefill_samples),
                )
                if iteration % 2 == 0
                else (
                    (candidate, candidate_args[0], candidate_prefill_samples),
                    (final, final_args[0], final_prefill_samples),
                )
            )
            for decoder, prefill_args, output in ordered:
                start = time.perf_counter()
                decoder.prefill_forward(**prefill_args)
                ttnn.synchronize_device(mesh_device)
                output.append((time.perf_counter() - start) * 1000)

    final_median = statistics.median(final_samples)
    candidate_median = statistics.median(candidate_samples)
    paired_differences = [candidate - final for final, candidate in zip(final_samples, candidate_samples)]
    paired_mean = statistics.mean(paired_differences)
    paired_sem = statistics.stdev(paired_differences) / len(paired_differences) ** 0.5
    artifact = {
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "sequence_length": 1024,
        "batch": batch,
        "functional_equivalence_pcc": float(candidate_pcc),
        "prefill_functional_equivalence_pcc": None if prefill_pcc is None else float(prefill_pcc),
        "equivalence_threshold": 0.99,
        "repeats": 101,
        "ordering": "alternated final/candidate first",
        "final_trace_host_ms_samples": final_samples,
        "candidate_trace_host_ms_samples": candidate_samples,
        "final_trace_host_ms_median": final_median,
        "candidate_trace_host_ms_median": candidate_median,
        "candidate": candidate_name,
        "winner": "final" if final_median < candidate_median else candidate_name,
        "candidate_minus_final_paired_mean_ms": paired_mean,
        "candidate_minus_final_95ci_ms": [paired_mean - 1.96 * paired_sem, paired_mean + 1.96 * paired_sem],
    }
    if batch == 1:
        artifact.update(
            {
                "prefill_repeats": 21,
                "final_prefill_host_ms_samples": final_prefill_samples,
                "candidate_prefill_host_ms_samples": candidate_prefill_samples,
                "final_prefill_host_ms_median": statistics.median(final_prefill_samples),
                "candidate_prefill_host_ms_median": statistics.median(candidate_prefill_samples),
            }
        )
        artifact["prefill_winner"] = (
            "final"
            if artifact["final_prefill_host_ms_median"] < artifact["candidate_prefill_host_ms_median"]
            else candidate_name
        )
    artifact_path = ARTIFACT_DIR / f"final_vs_{candidate_name}_layer{layer_idx}_{layer_type}_batch{batch}.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    _rewrite_fused_provenance(
        artifact_path,
        "GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_FUSED_DECODER_CANDIDATE_AB=1 "
        "TTNN_CONFIG_OVERRIDES='{\"throw_exception_on_fallback\": true}' pytest -q "
        "models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::"
        "test_final_vs_best_candidates",
    )
    if candidate_name == "dense_split":
        assert final_median < candidate_median, artifact
    if candidate_name == "alternative_expert_activation":
        assert artifact["candidate_minus_final_95ci_ms"][1] >= 0.0, artifact
        if batch == 1:
            assert artifact["final_prefill_host_ms_median"] < artifact["candidate_prefill_host_ms_median"], artifact


def test_candidate_matrix_selection():
    """Select expert activation lowering across the complete required matrix."""
    if os.getenv("GEMMA4_FUSED_DECODER_CANDIDATE_AB") != "1":
        pytest.skip("set GEMMA4_FUSED_DECODER_CANDIDATE_AB=1 to verify matrix selection")
    paths = sorted(ARTIFACT_DIR.glob("final_vs_alternative_expert_activation_layer*.json"))
    assert len(paths) == 4
    rows = [json.loads(path.read_text()) for path in paths]
    final_total = sum(row["final_trace_host_ms_median"] for row in rows)
    candidate_total = sum(row["candidate_trace_host_ms_median"] for row in rows)
    final_case_wins = sum(row["final_trace_host_ms_median"] < row["candidate_trace_host_ms_median"] for row in rows)
    significant_candidate_wins = sum(row["candidate_minus_final_95ci_ms"][1] < 0.0 for row in rows)
    summary = {
        "candidate": "composite GeGLU for every decode case",
        "selection": "explicit expert activation lowering",
        "cases": len(rows),
        "final_case_wins": final_case_wins,
        "significant_candidate_wins": significant_candidate_wins,
        "final_matrix_total_ms": final_total,
        "candidate_matrix_total_ms": candidate_total,
        "matrix_advantage_ms": candidate_total - final_total,
        "criterion": "lower sum of four 101-replay host medians, at least three raw case wins, and no paired 95% CI significantly favoring candidate",
    }
    artifact_path = ARTIFACT_DIR / "final_vs_composite_geglu_matrix.json"
    artifact_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _rewrite_fused_provenance(
        artifact_path,
        "GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_FUSED_DECODER_CANDIDATE_AB=1 "
        "TTNN_CONFIG_OVERRIDES='{\"throw_exception_on_fallback\": true}' pytest -q "
        "models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::"
        "test_final_vs_best_candidates "
        "models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::"
        "test_candidate_matrix_selection",
    )
    assert final_case_wins >= 3, summary
    assert significant_candidate_wins == 0, summary
    assert final_total < candidate_total, summary


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
def test_isolated_dense_activation_fold_candidate(mesh_device, device_params):
    """Compare current explicit GELU with a complementary always-folded control."""
    if os.getenv("GEMMA4_DENSE_ACTIVATION_CANDIDATE") != "1":
        pytest.skip("set GEMMA4_DENSE_ACTIVATION_CANDIDATE=1 to run isolated selection")
    cfg = _load_text_config()
    state = _load_layer_state(0)
    final = FusedDecoder.from_state_dict(state, hf_config=cfg, layer_idx=0, mesh_device=mesh_device)
    candidate = _DenseActivationFoldCandidate.from_state_dict(
        state, hf_config=cfg, layer_idx=0, mesh_device=mesh_device
    )
    final_args = _make_perf_args(final, mesh_device, layer_type="sliding_attention", seq_len=1024, batch=1)
    candidate_args = _make_perf_args(candidate, mesh_device, layer_type="sliding_attention", seq_len=1024, batch=1)
    final.prefill_forward(**final_args[0])
    final_output = final.decode_forward(**final_args[1])
    candidate.prefill_forward(**candidate_args[0])
    candidate_output = candidate.decode_forward(**candidate_args[1])
    ttnn.synchronize_device(mesh_device)
    equivalent, pcc = comp_pcc(_to_torch(mesh_device, final_output), _to_torch(mesh_device, candidate_output), 0.99)
    assert equivalent, pcc
    final_result = _measure_warmed(
        final, mesh_device, prefill_args=final_args[0], decode_args=final_args[1], batch=1, repeats=11
    )
    candidate_result = _measure_warmed(
        candidate, mesh_device, prefill_args=candidate_args[0], decode_args=candidate_args[1], batch=1, repeats=11
    )
    prefill_improvement_fraction = (
        final_result["prefill_host_ms_median"] - candidate_result["prefill_host_ms_median"]
    ) / final_result["prefill_host_ms_median"]
    artifact = {
        "candidate": "always_fold_dense_activation",
        "selection": "explicit GELU for prefill and decode",
        "minimum_material_prefill_improvement_fraction": 0.001,
        "observed_prefill_improvement_fraction": prefill_improvement_fraction,
        "layer_idx": 0,
        "layer_type": "sliding_attention",
        "sequence_length": 1024,
        "batch": 1,
        "functional_equivalence_pcc": float(pcc),
        "equivalence_threshold": 0.99,
        "final": final_result,
        "candidate_result": candidate_result,
        "decode_winner": (
            "always_fold_dense_activation"
            if candidate_result["decode_trace_host_ms_median"] < final_result["decode_trace_host_ms_median"]
            else "final"
        ),
        "prefill_winner": (
            "always_fold_dense_activation"
            if candidate_result["prefill_host_ms_median"] < final_result["prefill_host_ms_median"]
            else "final"
        ),
    }
    artifact_path = ARTIFACT_DIR / "isolated_dense_activation_fold_selection.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    _rewrite_fused_provenance(
        artifact_path,
        "GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_DENSE_ACTIVATION_CANDIDATE=1 "
        "TTNN_CONFIG_OVERRIDES='{\"throw_exception_on_fallback\": true}' pytest -q "
        "models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::"
        "test_isolated_dense_activation_fold_candidate",
    )
    assert final_result["decode_trace_host_ms_median"] < candidate_result["decode_trace_host_ms_median"], artifact
    assert prefill_improvement_fraction < artifact["minimum_material_prefill_improvement_fraction"], artifact


def test_fused_evidence_source_binding():
    """Bind the default fused suite and retained evidence to exact source bytes."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    decoder_path = Path(inspect.getsourcefile(FusedDecoder))
    functional_decoder_path = Path(inspect.getsourcefile(FunctionalDecoder))
    functional_test_path = Path(inspect.getsourcefile(functional_tests))
    artifact = {
        "exact_command": (
            "GEMMA4_RANGE_DOWNLOAD=1 TTNN_CONFIG_OVERRIDES='{\"throw_exception_on_fallback\": true}' "
            "pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py"
        ),
        "decoder": str(decoder_path),
        "fused_decoder_sha256": hashlib.sha256(decoder_path.read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "functional_decoder_sha256": hashlib.sha256(functional_decoder_path.read_bytes()).hexdigest(),
        "functional_test_sha256": hashlib.sha256(functional_test_path.read_bytes()).hexdigest(),
    }
    (ARTIFACT_DIR / "source_binding.json").write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
