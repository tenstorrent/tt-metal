# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
import math
import os
import time
from pathlib import Path

import pytest
import torch
from tracy import signpost

import ttnn
from models.autoports.openai_gpt_oss_20b.tests.test_functional_decoder import (
    EMITTED_PREFILL_SEQUENCE,
    LAYER_IDX,
    _assert_pcc,
    _config,
    _position_tensor,
    _real_state_dict,
    _synthetic_state_dict,
    _to_torch,
    _to_tt,
)
from models.autoports.openai_gpt_oss_20b.tt.functional_decoder import _require_tensor
from models.autoports.openai_gpt_oss_20b.tt.multichip_decoder import (
    PAGE_BLOCK_SIZE,
    SUPPORTED_CONTEXT,
    TARGET_MESH_SHAPE,
    TP_DEGREE,
    MultichipConfig,
    MultichipDecoder,
    _validate_qkv_geometry,
)
from models.autoports.openai_gpt_oss_20b.tt.optimized_decoder import OptimizedDecoder
from models.common.lightweightmodule import LightweightModule

FABRIC_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "trace_region_size": 32 * 1024 * 1024,
}
SINGLE_CHIP_DEVICE_PARAMS = {"trace_region_size": 32 * 1024 * 1024}
EVIDENCE_DIR = Path(__file__).parents[1] / "doc" / "multichip_decoder" / "logs"
SYNTHETIC_REFERENCE_PATH = EVIDENCE_DIR / "optimized_reference_synthetic.pt"
CANONICAL_REAL_PCC_SEEDS = {
    # Reuse the optimized decoder's existing boundary/trace seed for sliding
    # attention and the multichip stage seed for full attention.  The crossed
    # seeds deliberately exercise near-tied fourth-place expert routing in the
    # diagnostic below.
    "sliding_attention": 10010,
    "full_attention": 20260717,
}
NEAR_TIED_ROUTER_SEED = 3301


def _real_seed(layer_type):
    override = os.environ.get("MULTICHIP_REAL_SEED")
    return int(override) if override is not None else CANONICAL_REAL_PCC_SEEDS[layer_type]


def _perf_multichip_config_from_env():
    """Build opt-in perf candidates without changing the production default."""

    expert_subblock_candidate = os.environ.get("MULTICHIP_EXPERT_SUBBLOCK_CANDIDATE") == "1"
    expert_prefill_width1_candidate = os.environ.get("MULTICHIP_EXPERT_PREFILL_WIDTH1") == "1"
    if expert_subblock_candidate and expert_prefill_width1_candidate:
        raise ValueError("expert subblock candidates are mutually exclusive")
    kwargs = {}
    if expert_subblock_candidate:
        kwargs.update(
            expert_gate_up_cores=(3, 5),
            expert_down_cores=(5, 6),
            expert_gate_up_subblock_w=3,
            expert_down_subblock_w=3,
            prefill_expert_gate_up_cores=(3, 5),
            prefill_expert_down_cores=(5, 6),
            prefill_expert_gate_up_subblock_w=3,
            prefill_expert_down_subblock_w=3,
        )
    elif expert_prefill_width1_candidate:
        kwargs.update(
            prefill_expert_gate_up_cores=(5, 9),
            prefill_expert_down_cores=(9, 10),
            prefill_expert_gate_up_subblock_w=1,
            prefill_expert_down_subblock_w=1,
        )

    qkv_ab = os.environ.get("MULTICHIP_QKV_AB")
    qkv_candidate = None
    if qkv_ab is not None:
        parts = qkv_ab.split(",")
        if len(parts) != 4:
            raise ValueError("MULTICHIP_QKV_AB must be input_cores,in0_block_w,output_tiles_per_core,out_subblock_w")
        try:
            qkv_candidate = tuple(int(part) for part in parts)
        except ValueError as error:
            raise ValueError("MULTICHIP_QKV_AB fields must be integers") from error
        input_cores, in0_block_w, output_tiles_per_core, out_subblock_w = qkv_candidate
        kwargs.update(
            qkv_input_cores=input_cores,
            qkv_in0_block_w=in0_block_w,
            qkv_output_tiles_per_core=output_tiles_per_core,
            qkv_out_subblock_w=out_subblock_w,
        )

    multichip_config = MultichipConfig(**kwargs)
    if qkv_candidate is not None:
        _validate_qkv_geometry(multichip_config, k_tiles=90, n_tiles=80, grid_x=11, grid_y=10)
    return (multichip_config if kwargs else None), expert_subblock_candidate, qkv_candidate


def _real_reference_path(layer_type, *, seed=None):
    seed = _real_seed(layer_type) if seed is None else seed
    return EVIDENCE_DIR / f"optimized_reference_{layer_type}_seed{seed}.pt"


def _mesh_test(function):
    function = pytest.mark.parametrize("mesh_device", [TARGET_MESH_SHAPE], indirect=True)(function)
    return pytest.mark.parametrize("device_params", [FABRIC_DEVICE_PARAMS], indirect=True)(function)


def _single_chip_test(function):
    function = pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)(function)
    return pytest.mark.parametrize("device_params", [SINGLE_CHIP_DEVICE_PARAMS], indirect=True)(function)


def _decoder(state, config, mesh_device, *, max_cache_len=128, multichip_config=None):
    return MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_cache_len=max_cache_len,
        multichip_config=multichip_config,
    )


def _all_device_torch(tensor):
    return [ttnn.to_torch(local) for local in ttnn.get_device_tensors(tensor)]


def _assert_replicated(name, tensor):
    locals_ = _all_device_torch(tensor)
    assert len(locals_) == TP_DEGREE
    assert torch.equal(locals_[0], locals_[1]), f"{name} differs across TP ranks"


def _host_hidden(hidden_states, mesh_device):
    return ttnn.from_torch(
        hidden_states.unsqueeze(0),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


def _host_position(position, mesh_device):
    return ttnn.from_torch(
        torch.tensor([position], dtype=torch.int32),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def test_multichip_runtime_contract_and_fallback_audit():
    assert issubclass(MultichipDecoder, (OptimizedDecoder, LightweightModule))
    assert TARGET_MESH_SHAPE == (1, 2)
    assert TP_DEGREE == 2
    assert PAGE_BLOCK_SIZE == 64
    assert SUPPORTED_CONTEXT == 131_072

    runtime_methods = (
        MultichipDecoder._prefill_attention,
        MultichipDecoder._decode_attention,
        MultichipDecoder._decode_sliding_attention_mask,
        MultichipDecoder._decode_post_attention_norm,
        MultichipDecoder._active_prefill_expert_chunk,
        MultichipDecoder._active_prefill_sparse_moe,
        MultichipDecoder._moe_forward,
        MultichipDecoder._all_reduce,
        MultichipDecoder.prefill_forward,
        MultichipDecoder.decode_forward,
        MultichipDecoder.forward,
    )
    forbidden = ("torch", "from_torch", "to_torch", "get_device_tensors", "cpu")
    for method in runtime_methods:
        source = inspect.getsource(method)
        assert all(token not in source for token in forbidden), method.__name__

    full_source = inspect.getsource(MultichipDecoder)
    assert "self.experts(" in inspect.getsource(OptimizedDecoder._sparse_moe_forward)
    assert "Experts(" in full_source
    active_prefill_source = inspect.getsource(MultichipDecoder._active_prefill_expert_chunk)
    assert active_prefill_source.count("ttnn.sparse_matmul(") == 3
    assert active_prefill_source.count("nnz=None") == 3
    assert "is_input_a_sparse=True" in active_prefill_source
    assert "is_input_b_sparse=False" in active_prefill_source
    assert "self.experts(" not in active_prefill_source


def test_multichip_recovery_mesh_open_close_smoke():
    """Bounded post-recovery control: open and close the target mesh twice."""

    if os.environ.get("RUN_MULTICHIP_RECOVERY_SMOKE") != "1":
        pytest.skip("set RUN_MULTICHIP_RECOVERY_SMOKE=1 for the explicit recovery smoke")
    for _ in range(2):
        mesh_device = None
        try:
            mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(*TARGET_MESH_SHAPE))
            assert mesh_device.get_num_devices() == TP_DEGREE
            assert tuple(mesh_device.shape) == TARGET_MESH_SHAPE
        finally:
            if mesh_device is not None:
                ttnn.close_mesh_device(mesh_device)


@_mesh_test
def test_active_prefill_uses_four_token_specific_sparse_entries(mesh_device, monkeypatch):
    """Record the three sparse calls for a non-aligned, controlled top-4 mask."""

    config = _config()
    state = _synthetic_state_dict(config)
    decoder = _decoder(state, config, mesh_device, max_cache_len=64)
    seq_len = 17
    generator = torch.Generator().manual_seed(1704)
    hidden = torch.randn(1, seq_len, config.hidden_size, generator=generator).to(torch.bfloat16)
    routing = torch.zeros(seq_len, config.num_local_experts, dtype=torch.bfloat16)
    route_scales = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.bfloat16)
    for token in range(seq_len):
        selected = torch.tensor(
            [token % 32, (token + 7) % 32, (token + 15) % 32, (token + 23) % 32],
            dtype=torch.long,
        )
        routing[token, selected] = route_scales

    tt_hidden = _to_tt(hidden, mesh_device)
    tt_routing = ttnn.from_torch(
        routing,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sparse_calls = []
    original_sparse_matmul = ttnn.sparse_matmul

    def recording_sparse_matmul(input_a, input_b, **kwargs):
        sparsity = kwargs["sparsity"]
        sparse_calls.append(
            {
                "input_a": tuple(input_a.shape),
                "input_b": tuple(input_b.shape),
                "sparsity": tuple(sparsity.shape),
                "active": int(torch.count_nonzero(_to_torch(sparsity))),
                "nnz": kwargs.get("nnz"),
                "input_a_sparse": kwargs.get("is_input_a_sparse", False),
                "input_b_sparse": kwargs.get("is_input_b_sparse", True),
            }
        )
        return original_sparse_matmul(input_a, input_b, **kwargs)

    monkeypatch.setattr(ttnn, "sparse_matmul", recording_sparse_matmul)
    output = decoder._active_prefill_sparse_moe(tt_hidden, tt_hidden, tt_routing, seq_len)
    ttnn.synchronize_device(mesh_device)

    assert tuple(output.shape) == (1, 1, seq_len, config.hidden_size)
    _assert_replicated("active-prefill-controlled-top4", output)
    assert len(sparse_calls) == 3
    assert [call["active"] for call in sparse_calls] == [4 * seq_len] * 3
    assert [call["sparsity"] for call in sparse_calls] == [
        (32, 1, 1, config.num_local_experts),
        (32, 1, 1, config.num_local_experts),
        (1, 1, 32, config.num_local_experts),
    ]
    assert [call["nnz"] for call in sparse_calls] == [None, None, None]
    assert [call["input_a_sparse"] for call in sparse_calls] == [False, False, True]
    assert [call["input_b_sparse"] for call in sparse_calls] == [True, True, False]
    dense_group_entries = math.ceil(seq_len / ttnn.TILE_SIZE) * config.num_local_experts
    assert 4 * seq_len != dense_group_entries


def test_multichip_perf_qkv_ab_env_parsing(monkeypatch):
    monkeypatch.setenv("MULTICHIP_QKV_AB", "30,3,3,3")
    multichip_config, expert_candidate, qkv_candidate = _perf_multichip_config_from_env()
    assert not expert_candidate
    assert qkv_candidate == (30, 3, 3, 3)
    assert multichip_config.qkv_input_cores == 30
    assert multichip_config.qkv_in0_block_w == 3
    assert multichip_config.qkv_output_tiles_per_core == 3
    assert multichip_config.qkv_out_subblock_w == 3
    assert _validate_qkv_geometry(multichip_config, k_tiles=90, n_tiles=80, grid_x=11, grid_y=10) == (3, 27, 3)


def test_multichip_perf_expert_policy_and_candidate_parsing(monkeypatch, expect_error):
    selected = MultichipConfig()
    assert selected.prefill_expert_gate_up_cores == (3, 5)
    assert selected.prefill_expert_down_cores == (5, 6)
    assert selected.prefill_expert_gate_up_subblock_w == selected.prefill_expert_down_subblock_w == 3
    assert selected.expert_gate_up_cores == (5, 9)
    assert selected.expert_down_cores == (9, 10)
    assert selected.expert_gate_up_subblock_w == selected.expert_down_subblock_w == 1

    monkeypatch.setenv("MULTICHIP_EXPERT_PREFILL_WIDTH1", "1")
    width1, expert_candidate, qkv_candidate = _perf_multichip_config_from_env()
    assert not expert_candidate and qkv_candidate is None
    assert width1.prefill_expert_gate_up_cores == width1.expert_gate_up_cores == (5, 9)
    assert width1.prefill_expert_down_cores == width1.expert_down_cores == (9, 10)
    assert width1.prefill_expert_gate_up_subblock_w == width1.prefill_expert_down_subblock_w == 1

    monkeypatch.setenv("MULTICHIP_EXPERT_SUBBLOCK_CANDIDATE", "1")
    with expect_error(ValueError, "mutually exclusive"):
        _perf_multichip_config_from_env()


@pytest.mark.parametrize(
    "candidate,error",
    [
        ("45,3,1,1", "per-core input shard tiles=2"),
        ("30,3,3,2", "must divide qkv_output_tiles_per_core=3"),
        ("18,5,5,5", "FP32-DST limit of 4"),
        ("7,1,1,1", "must divide K tiles=90"),
        ("30,three,3,3", "fields must be integers"),
        ("30,3,3", "must be input_cores"),
    ],
)
def test_multichip_perf_qkv_ab_env_rejects_illegal_tuples(monkeypatch, expect_error, candidate, error):
    monkeypatch.setenv("MULTICHIP_QKV_AB", candidate)
    with expect_error(ValueError, error):
        _perf_multichip_config_from_env()


@_single_chip_test
def test_capture_synthetic_single_chip_optimized_reference(mesh_device):
    """Capture the direct TTNN baseline without overlapping mesh handles."""

    config = _config()
    state = _synthetic_state_dict(config)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_cache_len=128,
    )
    key_cache, value_cache = decoder.create_kv_cache()
    generator = torch.Generator().manual_seed(20260717)
    reference = {}
    for seq_len in (EMITTED_PREFILL_SEQUENCE, 33):
        hidden = torch.randn(1, seq_len, config.hidden_size, generator=generator).to(torch.bfloat16)
        output = decoder.prefill_forward(_to_tt(hidden, mesh_device), key_cache=key_cache, value_cache=value_cache)
        ttnn.synchronize_device(mesh_device)
        reference[f"prefill_{seq_len}"] = _to_torch(output).cpu()
    reference["key_cache"] = ttnn.to_torch(ttnn.get_device_tensors(key_cache)[0]).cpu()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(reference, SYNTHETIC_REFERENCE_PATH)
    print(f"OPTIMIZED_REFERENCE_ARTIFACT {SYNTHETIC_REFERENCE_PATH}")


@_mesh_test
def test_synthetic_non_aligned_prefill_matches_optimized_and_cache_is_head_local(mesh_device):
    assert SYNTHETIC_REFERENCE_PATH.exists(), (
        "run test_capture_synthetic_single_chip_optimized_reference first; "
        "baseline and parent-mesh execution intentionally use separate device sessions"
    )
    reference = torch.load(SYNTHETIC_REFERENCE_PATH, map_location="cpu", weights_only=True)
    config = _config()
    state = _synthetic_state_dict(config)
    max_cache_len = 128
    multichip = _decoder(state, config, mesh_device, max_cache_len=max_cache_len)
    assert multichip.local_num_heads == 32
    assert multichip.local_num_kv_heads == 4
    assert multichip.local_intermediate_size == 1440
    qkv_grid = multichip.tp_qkv_program_config.compute_with_storage_grid_size
    assert (qkv_grid.x, qkv_grid.y) == (11, 4)
    assert multichip.tp_qkv_program_config.in0_block_w == 3
    assert multichip.tp_qkv_program_config.per_core_N == 2
    assert multichip.tp_qkv_program_config.out_subblock_w == 2
    assert multichip.tp_qkv_input_config.shard_spec.grid == ttnn.num_cores_to_corerangeset(
        30, ttnn.CoreCoord(qkv_grid.x, qkv_grid.y), row_wise=True
    )
    assert tuple(multichip.tp_qkv_input_config.shard_spec.shape) == (32, 96)
    assert multichip.tp_qkv_output_config.shard_spec.grid == ttnn.num_cores_to_corerangeset(
        40, ttnn.CoreCoord(qkv_grid.x, qkv_grid.y), row_wise=True
    )
    assert tuple(multichip.tp_qkv_output_config.shard_spec.shape) == (32, 64)
    expert_program = multichip.experts.program_config
    assert expert_program.prefill_gate_up_cores == (3, 5)
    assert expert_program.prefill_down_cores == (5, 6)
    assert expert_program.prefill_gate_up_subblock_w == 3
    assert expert_program.prefill_down_subblock_w == 3
    assert expert_program.decode_gate_up_cores == (5, 9)
    assert expert_program.decode_down_cores == (9, 10)
    assert expert_program.decode_gate_up_subblock_w == 1
    assert expert_program.decode_down_subblock_w == 1
    o_grid = multichip.tp_o_program_config.compute_with_storage_grid_size
    assert multichip.tp_o_output_config.shard_spec.grid == ttnn.num_cores_to_corerangeset(
        90, ttnn.CoreCoord(o_grid.x, o_grid.y), row_wise=True
    )

    reverse_blocks = list(reversed(range(multichip.num_cache_blocks)))
    page_table = multichip.create_page_table(reverse_blocks)
    key_cache, value_cache = multichip.create_kv_cache()
    generator = torch.Generator().manual_seed(20260717)

    for seq_len in (EMITTED_PREFILL_SEQUENCE, 33):
        hidden = torch.randn(1, seq_len, config.hidden_size, generator=generator).to(torch.bfloat16)
        multichip_out = multichip.prefill_forward(
            _to_tt(hidden, mesh_device),
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
        )
        _assert_pcc(
            f"multichip-synthetic-prefill-{seq_len}",
            reference[f"prefill_{seq_len}"],
            _to_torch(multichip_out),
            0.99,
        )
        assert tuple(multichip_out.shape) == (1, 1, seq_len, config.hidden_size)
        _assert_replicated(f"synthetic-prefill-{seq_len}", multichip_out)

    baseline_cache = reference["key_cache"]
    local_caches = _all_device_torch(key_cache)
    physical_page = reverse_blocks[0]
    for rank, local_cache in enumerate(local_caches):
        reference_cache_slice = baseline_cache[0, rank * 4 : (rank + 1) * 4, :PAGE_BLOCK_SIZE]
        _assert_pcc(
            f"rank-{rank}-logical-page-0-local-kv-heads",
            reference_cache_slice,
            local_cache[physical_page],
            0.999,
        )


@_single_chip_test
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"])
def test_capture_real_weight_single_chip_optimized_reference(mesh_device, layer_type):
    """Capture prefill/decode TTNN references in an independent device session."""

    config = _config()
    config.layer_types = list(config.layer_types)
    config.layer_types[LAYER_IDX] = layer_type
    state = _real_state_dict()
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_cache_len=256,
    )
    key_cache, value_cache = decoder.create_kv_cache()
    generator = torch.Generator().manual_seed(_real_seed(layer_type))
    prefill_len = config.sliding_window
    prefill_hidden = torch.randn(1, prefill_len, config.hidden_size, generator=generator).to(torch.bfloat16)
    prefill = decoder.prefill_forward(_to_tt(prefill_hidden, mesh_device), key_cache=key_cache, value_cache=value_cache)
    ttnn.synchronize_device(mesh_device)
    reference = {"prefill": _to_torch(prefill).cpu()}
    for position in range(prefill_len, prefill_len + 3):
        hidden = torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
        attention = decoder._decode_attention(
            _to_tt(hidden, mesh_device),
            key_cache,
            value_cache,
            position,
            _position_tensor(position, mesh_device),
        )
        normalized = ttnn.rms_norm(
            attention,
            epsilon=decoder.eps,
            weight=decoder.weights["post_attention_norm"],
            compute_kernel_config=decoder.compute_kernel_config,
        )
        routing = decoder._route(normalized, 1)
        output = decoder._sparse_moe_forward(attention, normalized, routing, 1)
        ttnn.synchronize_device(mesh_device)
        reference[f"attention_{position}"] = _to_torch(attention).cpu()
        reference[f"routing_{position}"] = _to_torch(routing).cpu()
        reference[f"decode_{position}"] = _to_torch(output).cpu()
    path = _real_reference_path(layer_type)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(reference, path)
    print(f"OPTIMIZED_REFERENCE_ARTIFACT {path}")


@_mesh_test
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"])
def test_real_weight_prefill_decode_matches_single_chip_optimized(mesh_device, layer_type):
    reference_path = _real_reference_path(layer_type)
    assert reference_path.exists(), (
        f"run test_capture_real_weight_single_chip_optimized_reference[{layer_type}] first; "
        "baseline and parent-mesh execution intentionally use separate device sessions"
    )
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    config = _config()
    config.layer_types = list(config.layer_types)
    config.layer_types[LAYER_IDX] = layer_type
    state = _real_state_dict()
    max_cache_len = 256
    multichip = _decoder(
        state,
        config,
        mesh_device,
        max_cache_len=max_cache_len,
        multichip_config=MultichipConfig(
            use_optimized_decode_layouts=os.environ.get("MULTICHIP_DISABLE_OPTIMIZED_DECODE_LAYOUTS") != "1"
        ),
    )
    physical_blocks = (
        range(multichip.num_cache_blocks)
        if os.environ.get("MULTICHIP_IDENTITY_PAGE_TABLE") == "1"
        else reversed(range(multichip.num_cache_blocks))
    )
    page_table = multichip.create_page_table(list(physical_blocks))
    key_cache, value_cache = multichip.create_kv_cache()
    generator = torch.Generator().manual_seed(_real_seed(layer_type))

    prefill_len = config.sliding_window
    prefill_hidden = torch.randn(1, prefill_len, config.hidden_size, generator=generator).to(torch.bfloat16)
    multichip_prefill = multichip.prefill_forward(
        _to_tt(prefill_hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
    )
    _assert_pcc(
        f"multichip-real-{layer_type}-prefill-{prefill_len}",
        reference["prefill"],
        _to_torch(multichip_prefill),
        0.99,
    )
    _assert_replicated(f"real-{layer_type}-prefill", multichip_prefill)

    last_hidden = None
    last_multichip = None
    for position in range(prefill_len, prefill_len + 3):
        last_hidden = torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
        attention = multichip._decode_attention(
            _to_tt(last_hidden, mesh_device),
            key_cache,
            value_cache,
            page_table,
            position,
            _position_tensor(position, mesh_device),
        )
        _assert_pcc(
            f"multichip-real-{layer_type}-attention-{position}",
            reference[f"attention_{position}"],
            _to_torch(attention),
            0.99,
        )
        normalized = multichip._decode_post_attention_norm(attention)
        routing = multichip._route(normalized, 1)
        last_multichip = multichip._sparse_moe_forward(attention, normalized, routing, 1)
        baseline_attention = _to_tt(reference[f"attention_{position}"], mesh_device)
        baseline_normalized = multichip._decode_post_attention_norm(baseline_attention)
        baseline_routing = multichip._route(baseline_normalized, 1)
        _assert_pcc(
            f"multichip-real-{layer_type}-baseline-attention-routing-{position}",
            reference[f"routing_{position}"],
            _to_torch(baseline_routing),
            0.999,
        )
        baseline_attention_output = multichip._sparse_moe_forward(
            baseline_attention, baseline_normalized, baseline_routing, 1
        )
        _assert_pcc(
            f"multichip-real-{layer_type}-baseline-attention-moe-{position}",
            reference[f"decode_{position}"],
            _to_torch(baseline_attention_output),
            0.99,
        )
        _assert_pcc(
            f"multichip-real-{layer_type}-routing-{position}",
            reference[f"routing_{position}"],
            _to_torch(routing),
            0.99,
        )
        _assert_pcc(
            f"multichip-real-{layer_type}-decode-{position}",
            reference[f"decode_{position}"],
            _to_torch(last_multichip),
            0.99,
        )
        _assert_replicated(f"real-{layer_type}-decode-{position}", last_multichip)

    repeated = multichip.decode_forward(
        _to_tt(last_hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        cache_position=prefill_len + 2,
        cache_position_tensor=_position_tensor(prefill_len + 2, mesh_device),
    )
    assert torch.equal(_to_torch(last_multichip), _to_torch(repeated))

    # A decoder output is directly consumable by another layer instance.  No
    # gather, host conversion, or layer-boundary reshaping is permitted here.
    second_key, second_value = multichip.create_kv_cache()
    stacked = multichip.decode_forward(
        repeated,
        key_cache=second_key,
        value_cache=second_value,
        page_table=page_table,
        cache_position=0,
        cache_position_tensor=_position_tensor(0, mesh_device),
    )
    assert tuple(stacked.shape) == tuple(repeated.shape) == (1, 1, 1, config.hidden_size)
    _assert_replicated(f"stacked-{layer_type}", stacked)


@_mesh_test
def test_near_tied_router_isolated_to_tp_attention_rounding(mesh_device):
    """Keep a deterministic top-k discontinuity as a component-level stress gate.

    Seed 3301 has nearly tied fourth/fifth router logits.  The TP attention is
    numerically near-exact, but its legitimate reduction rounding swaps one
    selected expert.  Feeding exact optimized-baseline attention into the same
    TP router and active-expert path proves that the cache, router, and sparse
    expert implementation are not the source of the divergence.
    """

    layer_type = "sliding_attention"
    reference_path = _real_reference_path(layer_type, seed=NEAR_TIED_ROUTER_SEED)
    assert reference_path.exists(), (
        f"run MULTICHIP_REAL_SEED={NEAR_TIED_ROUTER_SEED} "
        "test_capture_real_weight_single_chip_optimized_reference[sliding_attention] first"
    )
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    config = _config()
    config.layer_types = list(config.layer_types)
    config.layer_types[LAYER_IDX] = layer_type
    multichip = _decoder(_real_state_dict(), config, mesh_device, max_cache_len=256)
    page_table = multichip.create_page_table(list(reversed(range(multichip.num_cache_blocks))))
    key_cache, value_cache = multichip.create_kv_cache()
    generator = torch.Generator().manual_seed(NEAR_TIED_ROUTER_SEED)
    prefill_len = config.sliding_window
    prefill_hidden = torch.randn(1, prefill_len, config.hidden_size, generator=generator).to(torch.bfloat16)
    multichip.prefill_forward(
        _to_tt(prefill_hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
    )

    target_position = prefill_len + 1
    for position in range(prefill_len, target_position + 1):
        hidden = torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
        attention = multichip._decode_attention(
            _to_tt(hidden, mesh_device),
            key_cache,
            value_cache,
            page_table,
            position,
            _position_tensor(position, mesh_device),
        )
    _assert_pcc(
        "near-tied-router-tp-attention",
        reference[f"attention_{target_position}"],
        _to_torch(attention),
        0.9998,
    )

    normalized = multichip._decode_post_attention_norm(attention)
    actual_routing = multichip._route(normalized, 1)
    repeated_routing = multichip._route(normalized, 1)
    actual_routing_host = _to_torch(actual_routing)
    assert torch.equal(actual_routing_host, _to_torch(repeated_routing))
    assert not torch.equal(actual_routing_host, reference[f"routing_{target_position}"])
    assert int(torch.count_nonzero(actual_routing_host)) == int(config.num_experts_per_tok) == 4

    baseline_attention = _to_tt(reference[f"attention_{target_position}"], mesh_device)
    baseline_normalized = multichip._decode_post_attention_norm(baseline_attention)
    baseline_routing = multichip._route(baseline_normalized, 1)
    _assert_pcc(
        "near-tied-router-exact-attention-routing",
        reference[f"routing_{target_position}"],
        _to_torch(baseline_routing),
        0.999,
    )
    baseline_attention_output = multichip._sparse_moe_forward(
        baseline_attention, baseline_normalized, baseline_routing, 1
    )
    _assert_pcc(
        "near-tied-router-exact-attention-active-experts",
        reference[f"decode_{target_position}"],
        _to_torch(baseline_attention_output),
        0.99,
    )
    print(
        "NEAR_TIED_ROUTER_DIAGNOSTIC "
        f"seed={NEAR_TIED_ROUTER_SEED} position={target_position} "
        "status=deterministic_fourth_expert_swap component_paths=pass"
    )


@_mesh_test
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"])
def test_warmed_trace_replay_updates_hidden_position_and_paged_cache(mesh_device, layer_type):
    config = _config()
    config.layer_types = list(config.layer_types)
    config.layer_types[LAYER_IDX] = layer_type
    state = _real_state_dict()
    multichip = _decoder(state, config, mesh_device, max_cache_len=256)
    page_table = multichip.create_page_table(list(reversed(range(multichip.num_cache_blocks))))
    eager_key_cache, eager_value_cache = multichip.create_kv_cache()
    trace_key_cache, trace_value_cache = multichip.create_kv_cache()
    generator = torch.Generator().manual_seed(4040)
    prefill_len = config.sliding_window
    prefill_hidden = torch.randn(1, prefill_len, config.hidden_size, generator=generator).to(torch.bfloat16)
    tt_prefill_hidden = _to_tt(prefill_hidden, mesh_device)
    multichip.prefill_forward(
        tt_prefill_hidden,
        key_cache=eager_key_cache,
        value_cache=eager_value_cache,
        page_table=page_table,
    )
    multichip.prefill_forward(
        tt_prefill_hidden,
        key_cache=trace_key_cache,
        value_cache=trace_value_cache,
        page_table=page_table,
    )

    decode_cases = []
    for position in range(prefill_len, prefill_len + 3):
        hidden = torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
        decode_cases.append(
            {
                "position": position,
                "hidden": hidden,
                "eager_hidden": _to_tt(hidden, mesh_device),
                "eager_position": _position_tensor(position, mesh_device),
                "host_hidden": _host_hidden(hidden, mesh_device),
                "host_position": _host_position(position, mesh_device),
            }
        )

    trace_hidden = _to_tt(decode_cases[0]["hidden"], mesh_device)
    trace_position = _position_tensor(prefill_len, mesh_device)
    multichip.decode_forward(
        trace_hidden,
        key_cache=trace_key_cache,
        value_cache=trace_value_cache,
        page_table=page_table,
        cache_position=prefill_len,
        cache_position_tensor=trace_position,
    )
    ttnn.synchronize_device(mesh_device)

    # Compute every eager reference and allocate every replay source before a
    # trace exists.  Allocating or dispatching an eager graph while a trace is
    # active can corrupt trace-owned buffers.
    eager_references = []
    for case in decode_cases:
        eager = multichip.decode_forward(
            case["eager_hidden"],
            key_cache=eager_key_cache,
            value_cache=eager_value_cache,
            page_table=page_table,
            cache_position=case["position"],
            cache_position_tensor=case["eager_position"],
        )
        ttnn.synchronize_device(mesh_device)
        eager_references.append(_to_torch(eager).clone())

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = multichip.decode_forward(
        trace_hidden,
        key_cache=trace_key_cache,
        value_cache=trace_value_cache,
        page_table=page_table,
        cache_position=prefill_len,
        cache_position_tensor=trace_position,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        _assert_pcc(
            f"trace-{layer_type}-capture-position",
            eager_references[0],
            _to_torch(traced_output),
            0.999,
        )
        for case, eager_reference in zip(decode_cases[1:], eager_references[1:]):
            ttnn.copy_host_to_device_tensor(case["host_hidden"], trace_hidden)
            ttnn.copy_host_to_device_tensor(case["host_position"], trace_position)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            _assert_pcc(
                f"trace-{layer_type}-mutable-position-{case['position']}",
                eager_reference,
                _to_torch(traced_output),
                0.999,
            )
            _assert_replicated(f"trace-{layer_type}-{case['position']}", traced_output)

        deterministic_output = _to_torch(traced_output).clone()
        for replay in range(5):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            assert torch.equal(
                deterministic_output, _to_torch(traced_output)
            ), f"{layer_type} trace replay {replay} was not bit deterministic"
    finally:
        ttnn.release_trace(mesh_device, trace_id)

    logical_block = decode_cases[-1]["position"] // PAGE_BLOCK_SIZE
    physical_block = multichip.num_cache_blocks - 1 - logical_block
    for cache_name, eager_cache, trace_cache in (
        ("key", eager_key_cache, trace_key_cache),
        ("value", eager_value_cache, trace_value_cache),
    ):
        for rank, (eager_local, trace_local) in enumerate(
            zip(_all_device_torch(eager_cache), _all_device_torch(trace_cache))
        ):
            assert torch.equal(
                eager_local[physical_block], trace_local[physical_block]
            ), f"{layer_type} rank {rank} {cache_name} trace cache page differs from eager"


@_mesh_test
def test_full_context_cache_allocation_and_last_page_update(mesh_device):
    if os.environ.get("RUN_MULTICHIP_CONTEXT") != "1":
        pytest.skip("set RUN_MULTICHIP_CONTEXT=1 for the 131072-token cache capacity gate")
    config = _config()
    state = _real_state_dict()
    decoder = _decoder(state, config, mesh_device, max_cache_len=SUPPORTED_CONTEXT)
    page_table = decoder.create_page_table()
    key_cache, value_cache = decoder.create_kv_cache()
    assert tuple(key_cache.shape) == (
        SUPPORTED_CONTEXT // PAGE_BLOCK_SIZE,
        config.num_key_value_heads // TP_DEGREE,
        PAGE_BLOCK_SIZE,
        config.head_dim,
    )
    hidden = torch.randn(1, 1, config.hidden_size, generator=torch.Generator().manual_seed(8080)).to(torch.bfloat16)
    output = decoder.decode_forward(
        _to_tt(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        cache_position=SUPPORTED_CONTEXT - 1,
        cache_position_tensor=_position_tensor(SUPPORTED_CONTEXT - 1, mesh_device),
    )
    assert tuple(output.shape) == (1, 1, 1, config.hidden_size)
    _assert_replicated("full-context-last-page", output)


@_mesh_test
def test_sharded_residual_topology_candidate(mesh_device):
    """Measure O projection -> residual -> norm -> router for both contracts.

    This deliberately carries the reduce-scattered residual through distributed
    RMSNorm and a row-sharded router before gathering for sparse gate/up.  An
    immediate gather would only reconstruct the selected all-reduce and would
    not be a useful topology comparison.
    """

    if os.environ.get("RUN_MULTICHIP_TOPOLOGY_PROBE") != "1":
        pytest.skip("set RUN_MULTICHIP_TOPOLOGY_PROBE=1 for the residual-layout comparison")
    config = _config()
    state = _real_state_dict()
    decoder = _decoder(state, config, mesh_device, max_cache_len=128)
    repeats = int(os.environ.get("MULTICHIP_TOPOLOGY_REPEATS", "20"))
    generator = torch.Generator().manual_seed(7070)
    hidden = torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
    local_attention = torch.randn(1, 1, config.num_attention_heads * config.head_dim, generator=generator).to(
        torch.bfloat16
    )

    replicated_hidden = _to_tt(hidden, mesh_device)
    sharded_hidden = ttnn.from_torch(
        hidden.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    sharded_hidden = ttnn.to_memory_config(sharded_hidden, ttnn.L1_MEMORY_CONFIG)
    local_attention = ttnn.from_torch(
        local_attention.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    local_attention = ttnn.to_memory_config(local_attention, ttnn.L1_MEMORY_CONFIG)

    post_attention_weight = _require_tensor(state, LAYER_IDX, "post_attention_layernorm.weight")
    distributed_norm_weight = ttnn.from_torch(
        post_attention_weight.reshape(1, 1, -1, ttnn.TILE_SIZE).to(torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    router_weight = _require_tensor(state, LAYER_IDX, "mlp.router.weight")
    distributed_router_weight = ttnn.from_torch(
        router_weight.transpose(-2, -1).to(torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    router_bias = _require_tensor(state, LAYER_IDX, "mlp.router.bias").float()
    rank_selective_router_bias = torch.stack([router_bias, torch.zeros_like(router_bias)])
    distributed_router_bias = ttnn.from_torch(
        rank_selective_router_bias,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def o_partial():
        partial = ttnn.linear(
            local_attention,
            decoder.weights["o_weight"],
            bias=decoder.weights["o_bias"],
            dtype=ttnn.bfloat16,
            memory_config=decoder.tp_o_output_config,
            program_config=decoder.tp_o_program_config,
            compute_kernel_config=decoder.decode_compute_kernel_config,
        )
        return ttnn.to_memory_config(partial, ttnn.L1_MEMORY_CONFIG)

    def replicated_contract():
        projected = decoder._all_reduce(o_partial(), memory_config=ttnn.L1_MEMORY_CONFIG)
        residual = ttnn.add(replicated_hidden, projected)
        normalized = ttnn.rms_norm(
            residual,
            epsilon=decoder.eps,
            weight=decoder.weights["post_attention_norm"],
            compute_kernel_config=decoder.decode_compute_kernel_config,
        )
        return normalized, decoder._route(normalized, 1)

    def sharded_contract():
        reduced = ttnn.reduce_scatter(
            o_partial(),
            dim=3,
            num_links=decoder.multichip_config.num_links,
            cluster_axis=1,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        residual = ttnn.add(sharded_hidden, reduced)
        stats = ttnn.rms_norm_pre_all_gather(
            residual,
            compute_kernel_config=decoder.decode_compute_kernel_config,
            dtype=ttnn.bfloat16,
        )
        stats = ttnn.all_gather(
            stats,
            dim=3,
            num_links=decoder.multichip_config.num_links,
            cluster_axis=1,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        normalized_local = ttnn.rms_norm_post_all_gather(
            residual,
            stats,
            epsilon=decoder.eps,
            weight=distributed_norm_weight,
            compute_kernel_config=decoder.decode_compute_kernel_config,
        )
        router_input = ttnn.typecast(ttnn.reshape(normalized_local, [1, config.hidden_size // TP_DEGREE]), ttnn.float32)
        router_logits = ttnn.linear(
            router_input,
            distributed_router_weight,
            bias=distributed_router_bias,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=decoder.compute_kernel_config,
        )
        router_logits = decoder._all_reduce(router_logits, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        router_logits = ttnn.typecast(router_logits, ttnn.bfloat16)
        top_values, top_indices = ttnn.topk(router_logits, k=decoder.top_k, dim=-1, sorted=True)
        top_values = ttnn.softmax(top_values, dim=-1, numeric_stable=True)
        routing = ttnn.scatter(
            ttnn.zeros_like(router_logits),
            dim=1,
            index=top_indices,
            src=top_values,
        )
        normalized_for_experts = ttnn.all_gather(
            normalized_local,
            dim=3,
            num_links=decoder.multichip_config.num_links,
            cluster_axis=1,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return normalized_for_experts, routing

    baseline_normalized, baseline_routing = replicated_contract()
    candidate_normalized, candidate_routing = sharded_contract()
    ttnn.synchronize_device(mesh_device)
    _assert_pcc(
        "sharded-residual-distributed-norm",
        _to_torch(baseline_normalized),
        _to_torch(candidate_normalized),
        0.999,
    )
    _assert_pcc(
        "sharded-residual-router",
        _to_torch(baseline_routing),
        _to_torch(candidate_routing),
        0.999,
    )
    _assert_replicated("sharded-residual-gathered-for-router", candidate_normalized)

    start = time.perf_counter()
    for _ in range(repeats):
        replicated_contract()
    ttnn.synchronize_device(mesh_device)
    replicated_ms = (time.perf_counter() - start) * 1000.0 / repeats

    start = time.perf_counter()
    for _ in range(repeats):
        sharded_contract()
    ttnn.synchronize_device(mesh_device)
    sharded_ms = (time.perf_counter() - start) * 1000.0 / repeats

    result = {
        "boundary": "attention O projection -> residual add -> post-attention RMSNorm -> router -> sparse-expert input",
        "mesh": list(TARGET_MESH_SHAPE),
        "repeats": repeats,
        "replicated_all_reduce_ms": replicated_ms,
        "sharded_reduce_scatter_distributed_norm_gather_ms": sharded_ms,
        "replicated_over_sharded_speedup": sharded_ms / replicated_ms,
        "candidate_stack_contract": "width-sharded residual [1,1,1,1440] per device",
        "candidate_next_consumer": "distributed RMSNorm, row-sharded router plus 32-logit all-reduce, then required full-hidden gather for sparse gate/up",
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    result_path = EVIDENCE_DIR / "residual_topology_candidate.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        "RESIDUAL_TOPOLOGY_RESULT "
        f"replicated_ms={replicated_ms:.6f} sharded_ms={sharded_ms:.6f} "
        f"replicated_over_sharded_speedup={sharded_ms / replicated_ms:.6f} artifact={result_path}"
    )


def _perf_reference_path(seq_len):
    return EVIDENCE_DIR / f"single_chip_perf_reference_seq{seq_len}.json"


def _perf_result_path(seq_len):
    override = os.environ.get("MULTICHIP_PERF_RESULT_PATH")
    return Path(override) if override is not None else EVIDENCE_DIR / f"multichip_perf_result_seq{seq_len}.json"


def _time_prefill(decoder, device, hidden, *, repeats, label, page_table=None):
    key_cache, value_cache = decoder.create_kv_cache()
    kwargs = {"key_cache": key_cache, "value_cache": value_cache}
    if page_table is not None:
        kwargs["page_table"] = page_table
    tt_hidden = _to_tt(hidden, device)
    decoder.prefill_forward(tt_hidden, **kwargs)
    ttnn.synchronize_device(device)
    signpost(header=f"PERF_PREFILL_{label}")
    start = time.perf_counter()
    for _ in range(repeats):
        decoder.prefill_forward(tt_hidden, **kwargs)
    ttnn.synchronize_device(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / repeats
    signpost(header=f"PERF_PREFILL_{label}_END")
    return elapsed_ms, key_cache, value_cache


def _time_traced_decode(
    decoder,
    device,
    hidden,
    key_cache,
    value_cache,
    *,
    position,
    trace_replays,
    label,
    page_table=None,
):
    tt_hidden = _to_tt(hidden, device)
    position_tensor = _position_tensor(position, device)
    kwargs = {
        "key_cache": key_cache,
        "value_cache": value_cache,
        "cache_position": position,
        "cache_position_tensor": position_tensor,
    }
    if page_table is not None:
        kwargs["page_table"] = page_table
    decoder.decode_forward(tt_hidden, **kwargs)
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    decoder.decode_forward(tt_hidden, **kwargs)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        signpost(header=f"PERF_DECODE_{label}")
        start = time.perf_counter()
        for _ in range(trace_replays):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / trace_replays
        signpost(header=f"PERF_DECODE_{label}_END")
        return elapsed_ms
    finally:
        ttnn.release_trace(device, trace_id)


@_single_chip_test
def test_capture_single_chip_optimized_perf_reference(mesh_device):
    if os.environ.get("RUN_MULTICHIP_DECODER_PERF") != "1":
        pytest.skip("set RUN_MULTICHIP_DECODER_PERF=1 to capture warmed optimized baseline timing")
    config = _config()
    state = _real_state_dict()
    seq_len = int(os.environ.get("MULTICHIP_DECODER_PERF_SEQ", config.sliding_window))
    repeats = int(os.environ.get("MULTICHIP_DECODER_PREFILL_REPEATS", "10"))
    trace_replays = int(os.environ.get("MULTICHIP_DECODER_TRACE_REPLAYS", "100"))
    baseline = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_cache_len=max(128, seq_len + 1),
    )
    generator = torch.Generator().manual_seed(9191)
    prefill_hidden = torch.randn(1, seq_len, config.hidden_size, generator=generator).to(torch.bfloat16)
    decode_hidden = torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
    prefill_ms, key_cache, value_cache = _time_prefill(
        baseline,
        mesh_device,
        prefill_hidden,
        repeats=repeats,
        label="SINGLE_CHIP",
    )
    decode_ms = _time_traced_decode(
        baseline,
        mesh_device,
        decode_hidden,
        key_cache,
        value_cache,
        position=seq_len,
        trace_replays=trace_replays,
        label="SINGLE_CHIP",
    )
    result = {
        "mesh": [1, 1],
        "seq_len": seq_len,
        "prefill_ms": prefill_ms,
        "traced_decode_ms": decode_ms,
        "prefill_repeats": repeats,
        "trace_replays": trace_replays,
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = _perf_reference_path(seq_len)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        "SINGLE_CHIP_PERF_REFERENCE "
        f"seq={seq_len} prefill_ms={prefill_ms:.6f} traced_decode_ms={decode_ms:.6f} artifact={path}"
    )


@_mesh_test
def test_multichip_decoder_perf(mesh_device):
    if os.environ.get("RUN_MULTICHIP_DECODER_PERF") != "1":
        pytest.skip("set RUN_MULTICHIP_DECODER_PERF=1 to run warmed multichip timing")
    config = _config()
    state = _real_state_dict()
    seq_len = int(os.environ.get("MULTICHIP_DECODER_PERF_SEQ", config.sliding_window))
    repeats = int(os.environ.get("MULTICHIP_DECODER_PREFILL_REPEATS", "10"))
    trace_replays = int(os.environ.get("MULTICHIP_DECODER_TRACE_REPLAYS", "100"))
    reference_path = _perf_reference_path(seq_len)
    assert reference_path.exists(), "capture the single-chip optimized perf reference first"
    baseline = json.loads(reference_path.read_text())
    assert baseline["seq_len"] == seq_len
    multichip_config, expert_subblock_candidate, qkv_candidate = _perf_multichip_config_from_env()
    multichip = _decoder(
        state,
        config,
        mesh_device,
        max_cache_len=max(128, seq_len + 1),
        multichip_config=multichip_config,
    )
    page_table = multichip.create_page_table()
    generator = torch.Generator().manual_seed(9191)
    prefill_hidden = torch.randn(1, seq_len, config.hidden_size, generator=generator).to(torch.bfloat16)
    decode_hidden = torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
    multichip_prefill_ms, key_cache, value_cache = _time_prefill(
        multichip,
        mesh_device,
        prefill_hidden,
        repeats=repeats,
        label="MULTICHIP",
        page_table=page_table,
    )
    multichip_decode_ms = _time_traced_decode(
        multichip,
        mesh_device,
        decode_hidden,
        key_cache,
        value_cache,
        position=seq_len,
        trace_replays=trace_replays,
        label="MULTICHIP",
        page_table=page_table,
    )
    prefill_speedup = baseline["prefill_ms"] / multichip_prefill_ms
    decode_speedup = baseline["traced_decode_ms"] / multichip_decode_ms
    effective_qkv_geometry = [
        multichip.multichip_config.qkv_input_cores,
        multichip.multichip_config.qkv_in0_block_w,
        multichip.multichip_config.qkv_output_tiles_per_core,
        multichip.multichip_config.qkv_out_subblock_w,
    ]
    result = {
        "mesh": list(TARGET_MESH_SHAPE),
        "seq_len": seq_len,
        "single_chip": baseline,
        "multichip_prefill_ms": multichip_prefill_ms,
        "multichip_traced_decode_ms": multichip_decode_ms,
        "prefill_speedup": prefill_speedup,
        "prefill_efficiency": prefill_speedup / TP_DEGREE,
        "decode_speedup": decode_speedup,
        "decode_efficiency": decode_speedup / TP_DEGREE,
        "prefill_repeats": repeats,
        "trace_replays": trace_replays,
        "expert_subblock_candidate": expert_subblock_candidate,
        "expert_prefill_width1_candidate": os.environ.get("MULTICHIP_EXPERT_PREFILL_WIDTH1") == "1",
        "qkv_candidate": qkv_candidate,
        "qkv_geometry": effective_qkv_geometry,
        "expert_geometry": {
            "prefill_gate_up_cores": list(multichip.experts.program_config.prefill_gate_up_cores),
            "prefill_down_cores": list(multichip.experts.program_config.prefill_down_cores),
            "prefill_gate_up_subblock_w": multichip.experts.program_config.prefill_gate_up_subblock_w,
            "prefill_down_subblock_w": multichip.experts.program_config.prefill_down_subblock_w,
            "decode_gate_up_cores": list(multichip.experts.program_config.decode_gate_up_cores),
            "decode_down_cores": list(multichip.experts.program_config.decode_down_cores),
            "decode_gate_up_subblock_w": multichip.experts.program_config.decode_gate_up_subblock_w,
            "decode_down_subblock_w": multichip.experts.program_config.decode_down_subblock_w,
        },
    }
    result_path = _perf_result_path(seq_len)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        "MULTICHIP_PERF_RESULT "
        f"seq={seq_len} baseline_prefill_ms={baseline['prefill_ms']:.6f} "
        f"multichip_prefill_ms={multichip_prefill_ms:.6f} prefill_speedup={prefill_speedup:.6f} "
        f"prefill_efficiency={prefill_speedup / TP_DEGREE:.6f} "
        f"baseline_decode_ms={baseline['traced_decode_ms']:.6f} multichip_decode_ms={multichip_decode_ms:.6f} "
        f"decode_speedup={decode_speedup:.6f} decode_efficiency={decode_speedup / TP_DEGREE:.6f} "
        f"prefill_repeats={repeats} trace_replays={trace_replays} "
        f"qkv_candidate={qkv_candidate} qkv_geometry={effective_qkv_geometry} artifact={result_path}"
    )
