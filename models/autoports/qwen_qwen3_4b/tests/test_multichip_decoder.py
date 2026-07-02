# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_4b.tt.functional_decoder import HF_MODEL_ID
from models.autoports.qwen_qwen3_4b.tt.multichip_decoder import DEFAULT_MULTICHIP_KV_CONFIG, MultichipDecoder
from models.autoports.qwen_qwen3_4b.tt.optimized_decoder import OptimizedDecoder, PagedKVConfig
from models.common.utility_functions import comp_pcc

try:
    from tracy import signpost
except ImportError:  # pragma: no cover - tracy is optional outside profiling runs.

    def signpost(header):
        return None


SMALL_SEQ_LEN = 16
NON_ALIGNED_SEQ_LEN = 17
LARGE_SEQ_LEN = 64


@pytest.fixture(scope="module")
def hf_config():
    return AutoConfig.from_pretrained(HF_MODEL_ID)


def _synthetic_state_dict(hf_config, layer_idx=0):
    torch.manual_seed(20260702)
    hidden = hf_config.hidden_size
    q_width = hf_config.num_attention_heads * hf_config.head_dim
    kv_width = hf_config.num_key_value_heads * hf_config.head_dim
    inter = hf_config.intermediate_size
    prefix = f"model.layers.{layer_idx}"
    scale = 0.02
    return {
        f"{prefix}.self_attn.q_proj.weight": torch.randn(q_width, hidden, dtype=torch.bfloat16) * scale,
        f"{prefix}.self_attn.k_proj.weight": torch.randn(kv_width, hidden, dtype=torch.bfloat16) * scale,
        f"{prefix}.self_attn.v_proj.weight": torch.randn(kv_width, hidden, dtype=torch.bfloat16) * scale,
        f"{prefix}.self_attn.o_proj.weight": torch.randn(hidden, q_width, dtype=torch.bfloat16) * scale,
        f"{prefix}.self_attn.q_norm.weight": torch.ones(hf_config.head_dim, dtype=torch.bfloat16),
        f"{prefix}.self_attn.k_norm.weight": torch.ones(hf_config.head_dim, dtype=torch.bfloat16),
        f"{prefix}.mlp.gate_proj.weight": torch.randn(inter, hidden, dtype=torch.bfloat16) * scale,
        f"{prefix}.mlp.up_proj.weight": torch.randn(inter, hidden, dtype=torch.bfloat16) * scale,
        f"{prefix}.mlp.down_proj.weight": torch.randn(hidden, inter, dtype=torch.bfloat16) * scale,
        f"{prefix}.input_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
        f"{prefix}.post_attention_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
    }


def _tt_tensor(tensor, mesh_device, *, replicated=False):
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if replicated else None
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def _first_device_to_torch(tt_tensor):
    tensors = ttnn.get_device_tensors(tt_tensor)
    return ttnn.to_torch(tensors[0]).to(torch.float32)


def _all_devices_to_torch(tt_tensor):
    return [ttnn.to_torch(t).to(torch.float32) for t in ttnn.get_device_tensors(tt_tensor)]


def _assert_pcc(reference, actual, threshold):
    passing, pcc = comp_pcc(reference.to(torch.float32), actual.to(torch.float32), threshold)
    print(f"PCC={pcc}")
    assert passing, f"PCC {pcc} below threshold {threshold}"
    return float(pcc)


def _open_single_mesh():
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), trace_region_size=16 << 20)


def _open_tp4_mesh():
    if ttnn.get_num_devices() < 4:
        pytest.skip("MultichipDecoder requires four local TT devices")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=64 << 20)


def _single_chip_prefill(hf_config, state_dict, hidden_states, seq_len):
    mesh = _open_single_mesh()
    try:
        decoder = OptimizedDecoder.from_state_dict(
            state_dict, hf_config=hf_config, layer_idx=0, mesh_device=mesh, max_seq_len=LARGE_SEQ_LEN
        )
        tt_input = _tt_tensor(hidden_states.reshape(1, 1, seq_len, hf_config.hidden_size), mesh)
        output = decoder.prefill_forward(tt_input)
        return ttnn.to_torch(output).reshape_as(hidden_states).to(torch.float32), decoder.timings.prefill_ms
    finally:
        ttnn.close_mesh_device(mesh)


def _multichip_prefill(hf_config, state_dict, hidden_states, seq_len, *, fill_cache=False):
    mesh = _open_tp4_mesh()
    try:
        decoder = MultichipDecoder.from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=0,
            mesh_device=mesh,
            max_seq_len=DEFAULT_MULTICHIP_KV_CONFIG.max_seq_len,
            paged_kv_config=DEFAULT_MULTICHIP_KV_CONFIG,
        )
        kv_cache = decoder.init_paged_kv_cache() if fill_cache else None
        page_table = decoder.make_identity_page_table() if fill_cache else None
        tt_input = _tt_tensor(
            hidden_states.reshape(1, 1, seq_len, hf_config.hidden_size),
            mesh,
            replicated=True,
        )
        output = decoder.prefill_forward(tt_input, kv_cache=kv_cache, page_table=page_table)
        torch_outputs = [out.reshape_as(hidden_states) for out in _all_devices_to_torch(output)]
        return torch_outputs, decoder.timings.prefill_ms, decoder, kv_cache, page_table
    finally:
        ttnn.close_mesh_device(mesh)


@pytest.mark.parametrize("seq_len", [SMALL_SEQ_LEN, NON_ALIGNED_SEQ_LEN, LARGE_SEQ_LEN])
def test_multichip_prefill_matches_single_chip_optimized(hf_config, seq_len):
    state_dict = _synthetic_state_dict(hf_config)
    hidden_states = torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    reference, _ = _single_chip_prefill(hf_config, state_dict, hidden_states, seq_len)
    outputs, _, _, _, _ = _multichip_prefill(hf_config, state_dict, hidden_states, seq_len)

    for device_idx, actual in enumerate(outputs):
        print(f"device={device_idx}")
        _assert_pcc(reference, actual, 0.97)
    for actual in outputs[1:]:
        _assert_pcc(outputs[0], actual, 0.999)


def _single_chip_prefill_then_decode(hf_config, state_dict, hidden_states, prefix_len):
    mesh = _open_single_mesh()
    try:
        decoder = OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=0,
            mesh_device=mesh,
            max_seq_len=LARGE_SEQ_LEN,
            paged_kv_config=PagedKVConfig(max_num_blocks=2, block_size=32),
        )
        kv_cache = decoder.init_paged_kv_cache()
        page_table = decoder.make_identity_page_table()
        tt_prefix = _tt_tensor(hidden_states[:, :prefix_len, :].reshape(1, 1, prefix_len, hf_config.hidden_size), mesh)
        decoder.prefill_forward(tt_prefix, kv_cache=kv_cache, page_table=page_table)
        position_cos, position_sin = decoder.position_tables_for_decode(prefix_len)
        current_pos = decoder.make_current_pos([prefix_len])
        tt_decode_input = _tt_tensor(hidden_states[:, prefix_len:, :].reshape(1, 1, 1, hf_config.hidden_size), mesh)
        output = decoder.decode_forward(
            tt_decode_input,
            current_pos=current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            position_cos=position_cos,
            position_sin=position_sin,
        )
        return ttnn.to_torch(output).reshape(1, 1, hf_config.hidden_size).to(torch.float32), decoder.timings.decode_ms
    finally:
        ttnn.close_mesh_device(mesh)


def _single_chip_warmed_perf(hf_config, state_dict, hidden_states, prefix_len):
    mesh = _open_single_mesh()
    try:
        decoder = OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=0,
            mesh_device=mesh,
            max_seq_len=LARGE_SEQ_LEN,
            paged_kv_config=PagedKVConfig(max_num_blocks=2, block_size=32),
        )
        tt_prefill = _tt_tensor(hidden_states[:, :prefix_len, :].reshape(1, 1, prefix_len, hf_config.hidden_size), mesh)
        decoder.prefill_forward(tt_prefill)
        decoder.prefill_forward(tt_prefill)
        warmed_prefill_ms = decoder.timings.prefill_ms

        kv_cache = decoder.init_paged_kv_cache()
        page_table = decoder.make_identity_page_table()
        decoder.prefill_forward(tt_prefill, kv_cache=kv_cache, page_table=page_table)
        position_cos, position_sin = decoder.position_tables_for_decode(prefix_len)
        current_pos = decoder.make_current_pos([prefix_len])
        tt_decode_input = _tt_tensor(hidden_states[:, prefix_len:, :].reshape(1, 1, 1, hf_config.hidden_size), mesh)
        trace_id, _ = decoder.trace_decode_once(
            tt_decode_input,
            current_pos=current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            position_cos=position_cos,
            position_sin=position_sin,
        )
        assert trace_id is not None
        return warmed_prefill_ms, decoder.timings.traced_decode_ms
    finally:
        ttnn.close_mesh_device(mesh)


def _multichip_prefill_then_decode(hf_config, state_dict, hidden_states, prefix_len, *, trace=False):
    mesh = _open_tp4_mesh()
    try:
        decoder = MultichipDecoder.from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=0,
            mesh_device=mesh,
            max_seq_len=DEFAULT_MULTICHIP_KV_CONFIG.max_seq_len,
            paged_kv_config=DEFAULT_MULTICHIP_KV_CONFIG,
        )
        kv_cache = decoder.init_paged_kv_cache()
        page_table = decoder.make_identity_page_table()
        tt_prefix = _tt_tensor(
            hidden_states[:, :prefix_len, :].reshape(1, 1, prefix_len, hf_config.hidden_size),
            mesh,
            replicated=True,
        )
        decoder.prefill_forward(tt_prefix, kv_cache=kv_cache, page_table=page_table)
        position_cos, position_sin = decoder.position_tables_for_decode(prefix_len)
        current_pos = decoder.make_current_pos([prefix_len])
        tt_decode_input = _tt_tensor(
            hidden_states[:, prefix_len:, :].reshape(1, 1, 1, hf_config.hidden_size),
            mesh,
            replicated=True,
        )
        if trace:
            replay_hidden_states = hidden_states.clone()
            torch.manual_seed(20260703)
            replay_hidden_states[:, prefix_len:, :] = torch.randn_like(replay_hidden_states[:, prefix_len:, :])
            tt_replay_input = _tt_tensor(
                replay_hidden_states[:, prefix_len:, :].reshape(1, 1, 1, hf_config.hidden_size),
                mesh,
                replicated=True,
            )
            eager_output = decoder.decode_forward(
                tt_replay_input,
                current_pos=current_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                position_cos=position_cos,
                position_sin=position_sin,
            )
            trace_kv_cache = decoder.init_paged_kv_cache()
            trace_page_table = decoder.make_identity_page_table()
            decoder.prefill_forward(tt_prefix, kv_cache=trace_kv_cache, page_table=trace_page_table)
            trace_id, traced_output, capture_output = decoder.trace_decode_once(
                tt_decode_input,
                current_pos=current_pos,
                page_table=trace_page_table,
                kv_cache=trace_kv_cache,
                position_cos=position_cos,
                position_sin=position_sin,
                replay_hidden_states=replay_hidden_states[:, prefix_len:, :].reshape(1, 1, 1, hf_config.hidden_size),
                return_capture_output=True,
            )
            assert trace_id is not None
            eager = _first_device_to_torch(eager_output)
            traced = _first_device_to_torch(traced_output)
            capture = capture_output[0].reshape(1, 1, hf_config.hidden_size)
            return eager, traced, capture, decoder.timings.traced_decode_ms
        output = decoder.decode_forward(
            tt_decode_input,
            current_pos=current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            position_cos=position_cos,
            position_sin=position_sin,
        )
        outputs = [_first.reshape(1, 1, hf_config.hidden_size) for _first in _all_devices_to_torch(output)]
        return outputs, decoder.timings.decode_ms
    finally:
        ttnn.close_mesh_device(mesh)


@pytest.mark.parametrize("prefix_len", [SMALL_SEQ_LEN, NON_ALIGNED_SEQ_LEN])
def test_multichip_paged_decode_matches_single_chip_optimized(hf_config, prefix_len):
    state_dict = _synthetic_state_dict(hf_config)
    hidden_states = torch.randn(1, prefix_len + 1, hf_config.hidden_size, dtype=torch.bfloat16)

    reference, _ = _single_chip_prefill_then_decode(hf_config, state_dict, hidden_states, prefix_len)
    outputs, _ = _multichip_prefill_then_decode(hf_config, state_dict, hidden_states, prefix_len)

    for device_idx, actual in enumerate(outputs):
        print(f"device={device_idx}")
        _assert_pcc(reference, actual, 0.97)
    for actual in outputs[1:]:
        _assert_pcc(outputs[0], actual, 0.999)


def test_multichip_kv_cache_layout_contract(hf_config):
    state_dict = _synthetic_state_dict(hf_config)
    mesh = _open_tp4_mesh()
    try:
        decoder = MultichipDecoder.from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=0,
            mesh_device=mesh,
            max_seq_len=DEFAULT_MULTICHIP_KV_CONFIG.max_seq_len,
        )
        kv_cache = decoder.init_paged_kv_cache()
        assert decoder.local_num_attention_heads == 8
        assert decoder.local_num_key_value_heads == 2
        assert tuple(ttnn.get_device_tensors(kv_cache[0])[0].shape) == (
            DEFAULT_MULTICHIP_KV_CONFIG.max_num_blocks,
            2,
            DEFAULT_MULTICHIP_KV_CONFIG.block_size,
            hf_config.head_dim,
        )
        page_table = decoder.make_identity_page_table()
        current_pos = decoder.make_current_pos([NON_ALIGNED_SEQ_LEN])
        assert tuple(ttnn.get_device_tensors(page_table)[0].shape) == (1, DEFAULT_MULTICHIP_KV_CONFIG.max_num_blocks)
        assert tuple(ttnn.get_device_tensors(current_pos)[0].shape) == (1,)
    finally:
        ttnn.close_mesh_device(mesh)


def test_multichip_trace_replay_is_deterministic(hf_config):
    state_dict = _synthetic_state_dict(hf_config)
    hidden_states = torch.randn(1, SMALL_SEQ_LEN + 1, hf_config.hidden_size, dtype=torch.bfloat16)
    eager, traced, capture, traced_ms = _multichip_prefill_then_decode(
        hf_config, state_dict, hidden_states, SMALL_SEQ_LEN, trace=True
    )
    _assert_pcc(eager, traced, 0.999)
    max_delta = torch.max(torch.abs(capture - traced)).item()
    print(f"capture_vs_replay_max_delta={max_delta}")
    assert max_delta > 1e-4
    assert traced_ms is not None


def test_multichip_watcher_single_mesh_stress(hf_config):
    if os.environ.get("QWEN3_4B_MULTICHIP_RUN_WATCHER_STRESS") != "1":
        pytest.skip("set QWEN3_4B_MULTICHIP_RUN_WATCHER_STRESS=1 to run the watcher single-mesh stress")

    state_dict = _synthetic_state_dict(hf_config)
    mesh = _open_tp4_mesh()
    try:
        decoder = MultichipDecoder.from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=0,
            mesh_device=mesh,
            max_seq_len=DEFAULT_MULTICHIP_KV_CONFIG.max_seq_len,
            paged_kv_config=DEFAULT_MULTICHIP_KV_CONFIG,
        )

        kv_cache = decoder.init_paged_kv_cache()
        assert tuple(ttnn.get_device_tensors(kv_cache[0])[0].shape) == (
            DEFAULT_MULTICHIP_KV_CONFIG.max_num_blocks,
            decoder.local_num_key_value_heads,
            DEFAULT_MULTICHIP_KV_CONFIG.block_size,
            hf_config.head_dim,
        )
        page_table = decoder.make_identity_page_table()
        assert tuple(ttnn.get_device_tensors(page_table)[0].shape) == (1, DEFAULT_MULTICHIP_KV_CONFIG.max_num_blocks)

        for seq_len in [SMALL_SEQ_LEN, NON_ALIGNED_SEQ_LEN, LARGE_SEQ_LEN]:
            hidden_states = torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)
            tt_input = _tt_tensor(
                hidden_states.reshape(1, 1, seq_len, hf_config.hidden_size),
                mesh,
                replicated=True,
            )
            output = decoder.prefill_forward(tt_input)
            outputs = [out.reshape_as(hidden_states) for out in _all_devices_to_torch(output)]
            for actual in outputs[1:]:
                _assert_pcc(outputs[0], actual, 0.999)

        for prefix_len in [SMALL_SEQ_LEN, NON_ALIGNED_SEQ_LEN]:
            hidden_states = torch.randn(1, prefix_len + 1, hf_config.hidden_size, dtype=torch.bfloat16)
            kv_cache = decoder.init_paged_kv_cache()
            page_table = decoder.make_identity_page_table()
            tt_prefix = _tt_tensor(
                hidden_states[:, :prefix_len, :].reshape(1, 1, prefix_len, hf_config.hidden_size),
                mesh,
                replicated=True,
            )
            decoder.prefill_forward(tt_prefix, kv_cache=kv_cache, page_table=page_table)
            position_cos, position_sin = decoder.position_tables_for_decode(prefix_len)
            current_pos = decoder.make_current_pos([prefix_len])
            tt_decode_input = _tt_tensor(
                hidden_states[:, prefix_len:, :].reshape(1, 1, 1, hf_config.hidden_size),
                mesh,
                replicated=True,
            )
            output = decoder.decode_forward(
                tt_decode_input,
                current_pos=current_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                position_cos=position_cos,
                position_sin=position_sin,
            )
            outputs = [_first.reshape(1, 1, hf_config.hidden_size) for _first in _all_devices_to_torch(output)]
            for actual in outputs[1:]:
                _assert_pcc(outputs[0], actual, 0.999)

        hidden_states = torch.randn(1, SMALL_SEQ_LEN + 1, hf_config.hidden_size, dtype=torch.bfloat16)
        kv_cache = decoder.init_paged_kv_cache()
        page_table = decoder.make_identity_page_table()
        tt_prefix = _tt_tensor(
            hidden_states[:, :SMALL_SEQ_LEN, :].reshape(1, 1, SMALL_SEQ_LEN, hf_config.hidden_size),
            mesh,
            replicated=True,
        )
        decoder.prefill_forward(tt_prefix, kv_cache=kv_cache, page_table=page_table)
        position_cos, position_sin = decoder.position_tables_for_decode(SMALL_SEQ_LEN)
        current_pos = decoder.make_current_pos([SMALL_SEQ_LEN])
        tt_decode_input = _tt_tensor(
            hidden_states[:, SMALL_SEQ_LEN:, :].reshape(1, 1, 1, hf_config.hidden_size),
            mesh,
            replicated=True,
        )
        eager_output = decoder.decode_forward(
            tt_decode_input,
            current_pos=current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            position_cos=position_cos,
            position_sin=position_sin,
        )
        trace_kv_cache = decoder.init_paged_kv_cache()
        trace_page_table = decoder.make_identity_page_table()
        decoder.prefill_forward(tt_prefix, kv_cache=trace_kv_cache, page_table=trace_page_table)
        trace_id, traced_output = decoder.trace_decode_once(
            tt_decode_input,
            current_pos=current_pos,
            page_table=trace_page_table,
            kv_cache=trace_kv_cache,
            position_cos=position_cos,
            position_sin=position_sin,
        )
        assert trace_id is not None
        _assert_pcc(_first_device_to_torch(eager_output), _first_device_to_torch(traced_output), 0.999)
    finally:
        ttnn.close_mesh_device(mesh)


def test_multichip_reduce_scatter_residual_contract_probe(hf_config, expect_error):
    if os.environ.get("QWEN3_4B_MULTICHIP_RUN_RS_PROBE") != "1":
        pytest.skip("set QWEN3_4B_MULTICHIP_RUN_RS_PROBE=1 to run the reduce-scatter residual probe")

    mesh = _open_tp4_mesh()
    try:
        seq_len = SMALL_SEQ_LEN
        hidden = torch.randn(1, 1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)
        replicated_hidden = _tt_tensor(hidden, mesh, replicated=True)
        reduced_hidden = ttnn.reduce_scatter(
            replicated_hidden,
            dim=3,
            cluster_axis=1,
            num_links=1,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        local_shape = tuple(ttnn.get_device_tensors(reduced_hidden)[0].shape)
        print(f"reduce_scatter_local_shape={local_shape}")
        assert local_shape[-1] == hf_config.hidden_size // 4

        with expect_error(RuntimeError, "Invalid subtile broadcast type") as residual_error:
            ttnn.add(reduced_hidden, replicated_hidden, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        print(f"full_residual_add_error={str(residual_error.value).splitlines()[0]}")

        full_norm_weight = _tt_tensor(torch.ones(hf_config.hidden_size, dtype=torch.bfloat16), mesh, replicated=True)
        with expect_error(RuntimeError, "Input and gamma padded widths must match") as norm_error:
            ttnn.rms_norm(
                reduced_hidden,
                epsilon=1e-6,
                weight=full_norm_weight,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        print(f"full_weight_rms_norm_error={str(norm_error.value).splitlines()[0]}")

        local_norm_weight = ttnn.from_torch(
            torch.ones(hf_config.hidden_size, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        sharded_norm = ttnn.rms_norm(
            reduced_hidden,
            epsilon=1e-6,
            weight=local_norm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        print(f"sharded_norm_local_shape={tuple(ttnn.get_device_tensors(sharded_norm)[0].shape)}")

        gate_weight = ttnn.from_torch(
            torch.randn(hf_config.intermediate_size, hf_config.hidden_size, dtype=torch.bfloat16),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        with expect_error(RuntimeError, "width of the first tensor must be equal to the height") as gate_error:
            ttnn.matmul(
                sharded_norm,
                gate_weight,
                transpose_b=True,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        print(f"gate_up_matmul_error={str(gate_error.value).splitlines()[0]}")
    finally:
        ttnn.close_mesh_device(mesh)


def test_multichip_perf_signposts(hf_config):
    if os.environ.get("QWEN3_4B_MULTICHIP_RUN_PERF") != "1":
        pytest.skip("set QWEN3_4B_MULTICHIP_RUN_PERF=1 to run multichip decoder perf signposts")
    state_dict = _synthetic_state_dict(hf_config)
    hidden_states = torch.randn(1, SMALL_SEQ_LEN + 1, hf_config.hidden_size, dtype=torch.bfloat16)

    single_prefill_ms, single_traced_decode_ms = _single_chip_warmed_perf(
        hf_config, state_dict, hidden_states, SMALL_SEQ_LEN
    )

    mesh = _open_tp4_mesh()
    try:
        decoder = MultichipDecoder.from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=0,
            mesh_device=mesh,
            max_seq_len=DEFAULT_MULTICHIP_KV_CONFIG.max_seq_len,
            paged_kv_config=DEFAULT_MULTICHIP_KV_CONFIG,
        )
        tt_prefill = _tt_tensor(
            hidden_states[:, :SMALL_SEQ_LEN, :].reshape(1, 1, SMALL_SEQ_LEN, hf_config.hidden_size),
            mesh,
            replicated=True,
        )
        decoder.prefill_forward(tt_prefill)
        signpost("PERF_MULTICHIP_PREFILL_WARMED")
        decoder.prefill_forward(tt_prefill)
        warmed_prefill_ms = decoder.timings.prefill_ms
        signpost("PERF_MULTICHIP_PREFILL_WARMED_END")

        kv_cache = decoder.init_paged_kv_cache()
        page_table = decoder.make_identity_page_table()
        decoder.prefill_forward(tt_prefill, kv_cache=kv_cache, page_table=page_table)
        position_cos, position_sin = decoder.position_tables_for_decode(SMALL_SEQ_LEN)
        current_pos = decoder.make_current_pos([SMALL_SEQ_LEN])
        tt_decode_input = _tt_tensor(
            hidden_states[:, SMALL_SEQ_LEN:, :].reshape(1, 1, 1, hf_config.hidden_size),
            mesh,
            replicated=True,
        )
        trace_id, _ = decoder.trace_decode_once(
            tt_decode_input,
            current_pos=current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            position_cos=position_cos,
            position_sin=position_sin,
        )
        assert trace_id is not None

        perf_out = os.environ.get("QWEN3_4B_MULTICHIP_PERF_OUT")
        if perf_out:
            path = Path(perf_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not path.exists()
            with path.open("a", encoding="utf-8") as f:
                if new_file:
                    f.write(
                        "model,decoder,profile,layer,mode,seq_len,decode_pos,batch,traced,"
                        "single_chip_ms,multichip_ms,speedup,efficiency\n"
                    )
                f.write(
                    "Qwen/Qwen3-4B,MultichipDecoder,"
                    f"{MultichipDecoder.mesh_profile['name']},0,prefill,{SMALL_SEQ_LEN},,1,False,"
                    f"{single_prefill_ms:.6f},{warmed_prefill_ms:.6f},"
                    f"{single_prefill_ms / warmed_prefill_ms:.6f},{single_prefill_ms / warmed_prefill_ms / 4:.6f}\n"
                )
                f.write(
                    "Qwen/Qwen3-4B,MultichipDecoder,"
                    f"{MultichipDecoder.mesh_profile['name']},0,decode,1,{SMALL_SEQ_LEN},1,True,"
                    f"{single_traced_decode_ms:.6f},{decoder.timings.traced_decode_ms:.6f},"
                    f"{single_traced_decode_ms / decoder.timings.traced_decode_ms:.6f},"
                    f"{single_traced_decode_ms / decoder.timings.traced_decode_ms / 4:.6f}\n"
                )
    finally:
        ttnn.close_mesh_device(mesh)


def test_multichip_runtime_has_no_host_fallback():
    assert MultichipDecoder.baseline_cls is OptimizedDecoder
    forbidden = ("torch", "from_torch", "to_torch")
    for method in (MultichipDecoder.prefill_forward, MultichipDecoder.decode_forward):
        source = inspect.getsource(method)
        hits = [term for term in forbidden if term in source]
        assert hits == []
