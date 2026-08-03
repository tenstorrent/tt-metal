# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Stage-03 regression suite for the optimized North-Mini decoder."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

import models.autoports.coherelabs_north_mini_code_1_0.tests.test_functional_decoder as functional_tests
import ttnn
from models.autoports.coherelabs_north_mini_code_1_0.tt.functional_decoder import FunctionalDecoder as ReferenceDecoder
from models.autoports.coherelabs_north_mini_code_1_0.tt.optimized_decoder import POLICIES, OptimizedDecoder


def _select_optimized(monkeypatch):
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", OptimizedDecoder)


def test_optimized_path_source_contract():
    sparse_source = inspect.getsource(OptimizedDecoder._sparse_moe_chunk)
    assert OptimizedDecoder._sparse_moe_chunk.__module__.endswith(".optimized_decoder")
    assert "sparsity=sparsity" in sparse_source
    assert "is_input_a_sparse=True" in sparse_source
    assert "nnz=self.top_k" in sparse_source
    assert "src=ttnn.ones_like(top_values)" in sparse_source
    assert "ttnn.reshape(exact_mask" in sparse_source
    assert "memory_config=ttnn.L1_MEMORY_CONFIG" in sparse_source
    assert "self.optimized_batch1_moe_calls += 1" in sparse_source
    runtime_source = "\n".join(
        inspect.getsource(method)
        for method in (
            OptimizedDecoder._qkv_decode,
            OptimizedDecoder._attention_decode,
            OptimizedDecoder._dense_mlp,
            OptimizedDecoder.decode_forward,
            OptimizedDecoder._sparse_moe_chunk,
        )
    )
    for forbidden in ("FunctionalDecoder", "from_torch", "to_torch", "import torch"):
        assert forbidden not in runtime_source


def test_default_policy_selects_measured_optimized_paths():
    policy = POLICIES["default"]
    assert policy.dram_sharded_dense_decode
    assert policy.sharded_dense_residual
    assert policy.sparse_l1_chain
    assert policy.prefill_moe_grid_scale == 8
    assert policy.prefill_gate_up_in0_block_w == 8
    assert policy.prefill_down_in0_block_w == 8
    assert policy.gate_up_cores == 24
    assert policy.down_cores == 32


def _real_layer_state(layer_idx):
    explicit = os.environ.get("NORTH_MINI_REAL_WEIGHT_DIR")
    roots = [Path(explicit)] if explicit else []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.extend((Path(hf_home), Path(hf_home) / "hub"))
    roots.append(Path("/huggingface/hub"))
    snapshot = next(
        (
            path
            for root in roots
            for path in root.glob(f"models--CohereLabs--North-Mini-Code-1.0/snapshots/{functional_tests.REAL_REVISION}")
            if path.is_dir()
        ),
        None,
    )
    if snapshot is None:
        pytest.skip("North-Mini checkpoint not cached; set NORTH_MINI_REAL_WEIGHT_DIR")
    index_path = snapshot / "model.safetensors.index.json"
    if not index_path.is_file():
        pytest.skip("North-Mini safetensors index is not cached")
    prefix = f"model.layers.{layer_idx}."
    weight_map = json.loads(index_path.read_text())["weight_map"]
    shards = sorted({snapshot / shard for key, shard in weight_map.items() if key.startswith(prefix)})
    if not shards or not all(shard.is_file() for shard in shards):
        pytest.skip(f"North-Mini layer-{layer_idx} shards are not cached")
    state = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            state.update({key: handle.get_tensor(key) for key in handle.keys() if key.startswith(prefix)})
    return state


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 4])
def test_selected_policy_real_weight_prefill_and_traced_decode(mesh_device, layer_idx):
    config = functional_tests._config()
    state = _real_layer_state(layer_idx)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=32,
    )
    generator = torch.Generator().manual_seed(20000 + layer_idx)
    prefill_hidden = functional_tests._randn(generator, 1, 1, config.hidden_size, scale=0.02)
    decode_hidden = functional_tests._randn(generator, 1, 1, config.hidden_size, scale=0.02)
    if layer_idx == 0:
        prefill_reference, reference_cache = functional_tests._dense_reference(
            prefill_hidden,
            torch.tensor([[0]]),
            state,
            config,
        )
        decode_reference, _ = functional_tests._dense_reference(
            decode_hidden,
            torch.tensor([[1]]),
            state,
            config,
            cache=reference_cache,
        )
    else:
        reference_decoder = ReferenceDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            batch=1,
            max_cache_len=32,
        )
        reference_key_cache, reference_value_cache = reference_decoder.create_paged_kv_cache()
        reference_page_table = functional_tests._to_tt(
            functional_tests._page_table(1, 1),
            mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        prefill_reference_tt = reference_decoder.prefill_forward(
            functional_tests._to_tt(prefill_hidden.unsqueeze(0), mesh_device),
            key_cache=reference_key_cache,
            value_cache=reference_value_cache,
            page_table=reference_page_table,
        )
        prefill_reference = ttnn.to_torch(prefill_reference_tt).squeeze(0)
        reference_current, _, _ = functional_tests._decode_inputs(
            reference_decoder,
            config,
            mesh_device,
            [1],
        )
        decode_reference_tt = reference_decoder.decode_forward(
            functional_tests._to_tt(decode_hidden.unsqueeze(0), mesh_device),
            key_cache=reference_key_cache,
            value_cache=reference_value_cache,
            page_table=reference_page_table,
            current_positions=reference_current,
        )
        decode_reference = ttnn.to_torch(decode_reference_tt).squeeze(0)

    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = functional_tests._to_tt(
        functional_tests._page_table(1, 1),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    prefill_cos, prefill_sin = decoder.build_rope_rows([0], hf_config=config)
    prefill_actual = decoder.prefill_forward(
        functional_tests._to_tt(prefill_hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        position_cos=functional_tests._to_tt(prefill_cos, mesh_device) if decoder.use_rope else None,
        position_sin=functional_tests._to_tt(prefill_sin, mesh_device) if decoder.use_rope else None,
    )
    prefill_passed, prefill_message = functional_tests.comp_pcc(
        prefill_reference.float(),
        ttnn.to_torch(prefill_actual).squeeze(0).float(),
        pcc=0.995,
    )
    print(f"real-layer{layer_idx}-prefill: {prefill_message}")

    hidden_tt = functional_tests._to_tt(decode_hidden.unsqueeze(0), mesh_device)
    current, decode_cos, decode_sin = functional_tests._decode_inputs(decoder, config, mesh_device, [1])
    kwargs = {
        "key_cache": key_cache,
        "value_cache": value_cache,
        "page_table": page_table,
        "current_positions": current,
        "position_cos": decode_cos if decoder.use_rope else None,
        "position_sin": decode_sin if decoder.use_rope else None,
    }
    decoder.decode_forward(hidden_tt, **kwargs)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    decode_actual = decoder.decode_forward(hidden_tt, **kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        for _ in range(4):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        decode_passed, decode_message = functional_tests.comp_pcc(
            decode_reference.float(),
            ttnn.to_torch(decode_actual).squeeze(0).float(),
            pcc=0.995,
        )
        print(f"real-layer{layer_idx}-traced-decode: {decode_message}")
        assert prefill_passed, f"real-layer{layer_idx}-prefill: {prefill_message}"
        assert decode_passed, f"real-layer{layer_idx}-traced-decode: {decode_message}"
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_exact_sparse_mask_survives_zero_routing_scores_repeated_trace(mesh_device):
    config = functional_tests._config()
    layer_idx = 4
    state = functional_tests._synthetic_state(config, layer_idx, sparse_weights=True)
    prefix = f"model.layers.{layer_idx}."
    state[prefix + "mlp.gate.weight"].fill_(-1000)
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        state[prefix + f"self_attn.{projection}.weight"].zero_()
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=32,
    )
    hidden = torch.ones(1, 1, config.hidden_size, dtype=torch.bfloat16)
    hidden_tt = functional_tests._to_tt(hidden.unsqueeze(0), mesh_device)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = functional_tests._to_tt(
        functional_tests._page_table(1, 1),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    current, _, _ = functional_tests._decode_inputs(decoder, config, mesh_device, [0])
    kwargs = {
        "key_cache": key_cache,
        "value_cache": value_cache,
        "page_table": page_table,
        "current_positions": current,
    }
    decoder.decode_forward(hidden_tt, **kwargs)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    actual = decoder.decode_forward(hidden_tt, **kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        for _ in range(20):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        functional_tests._assert_pcc(
            "zero-routing-score-repeated-trace",
            hidden,
            ttnn.to_torch(actual).squeeze(0),
        )
        assert decoder.optimized_batch1_moe_calls == 2
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [1, 31, 32, 33, 65])
def test_optimized_dense_non_aligned_prefill(monkeypatch, mesh_device, seq_len):
    _select_optimized(monkeypatch)
    functional_tests.test_dense_paged_prefill_non_aligned_matches_reference(mesh_device, seq_len)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_dense_trace_replay(monkeypatch, mesh_device):
    _select_optimized(monkeypatch)
    functional_tests.test_dense_paged_decode_trace_replay_matches_reference(mesh_device)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_cache_determinism(monkeypatch, mesh_device):
    _select_optimized(monkeypatch)
    functional_tests.test_batch_two_prefill_and_permuted_physical_cache(mesh_device)
    functional_tests.test_random_nonzero_decode_positions_update_expected_physical_slots_and_are_deterministic(
        mesh_device
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", functional_tests.REPRESENTATIVE_LAYERS)
def test_optimized_representative_layer_kinds(monkeypatch, mesh_device, layer_idx):
    _select_optimized(monkeypatch)
    functional_tests.test_every_meaningful_layer_kind_executes(mesh_device, layer_idx)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,sequence,selected_tokens",
    [(1, 1025, [0, 1023, 1024]), (4, 33, [0, 16, 32])],
)
def test_optimized_nonzero_sparse_prefill(
    monkeypatch,
    mesh_device,
    layer_idx,
    sequence,
    selected_tokens,
):
    _select_optimized(monkeypatch)
    functional_tests.test_nonzero_sparse_prefill_matches_active_expert_reference(
        mesh_device,
        layer_idx,
        sequence,
        selected_tokens,
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [1, 4])
def test_optimized_nonzero_sparse_traced_batch1(monkeypatch, mesh_device, layer_idx):
    _select_optimized(monkeypatch)
    functional_tests.test_nonzero_sparse_dynamic_trace_replay_matches_reference(mesh_device, layer_idx, 1)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_sliding_history(monkeypatch, mesh_device):
    _select_optimized(monkeypatch)
    functional_tests.test_sliding_window_boundary_4097_matches_controlled_reference(mesh_device)
    functional_tests.test_sliding_moe_populated_history_dynamic_trace_replay_matches_reference(mesh_device)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_real_weight_sliding_moe_decode(monkeypatch, mesh_device):
    _select_optimized(monkeypatch)
    functional_tests.test_real_weight_sliding_moe_decode_matches_reference(mesh_device)
