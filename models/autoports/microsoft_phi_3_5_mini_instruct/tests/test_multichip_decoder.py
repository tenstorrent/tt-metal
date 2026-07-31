# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import importlib
import inspect
import os
import time
from contextlib import contextmanager

import pytest
import torch
import ttnn

from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_optimized_decoder import (
    TEST_TABLE_LEN,
    _hf_config,
    _page_table,
    _real_layer0_state_dict,
    _synthetic_state_dict,
)
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.multichip_decoder import (
    HEAD_DIM,
    HIDDEN_SIZE,
    LOCAL_GATE_UP_SIZE,
    LOCAL_HIDDEN_SIZE,
    LOCAL_INTERMEDIATE_SIZE,
    LOCAL_NUM_HEADS,
    LOCAL_NUM_KV_HEADS,
    LOCAL_QKV_SIZE,
    TARGET_MESH_SHAPE,
    TP_FACTOR,
    MultichipDecoder,
    mesh_strategy_summary,
)
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import OptimizedDecoder
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

try:
    from tracy import signpost
except ImportError:

    def signpost(*_args, **_kwargs):
        return None


PCC_THRESHOLD = 0.995


@contextmanager
def _open_single_mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        yield mesh_device
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@contextmanager
def _open_t3k_ring_mesh():
    if ttnn.get_num_devices() < TP_FACTOR:
        pytest.skip(f"Phi multichip decoder requires {TP_FACTOR} devices")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(*TARGET_MESH_SHAPE), trace_region_size=100000000)
    try:
        yield mesh_device
    finally:
        if os.getenv("PHI35_SKIP_MESH_CLOSE") != "1":
            ttnn.close_mesh_device(mesh_device)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _replicate_to_mesh(tensor, mesh_device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _position_tensors(mesh_device, pos):
    current_pos = _replicate_to_mesh(torch.tensor([pos], dtype=torch.int32), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    position_ids = _replicate_to_mesh(
        torch.tensor([pos], dtype=torch.uint32), mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    return current_pos, position_ids


def _to_torch_first_device(tensor):
    try:
        device_tensors = ttnn.get_device_tensors(tensor)
    except RuntimeError:
        return ttnn.to_torch(tensor)
    return ttnn.to_torch(device_tensors[0])


def _assert_replicated_mesh_output(tensor, *, pcc=0.9999):
    device_tensors = ttnn.get_device_tensors(tensor)
    first = ttnn.to_torch(device_tensors[0]).float()
    for idx, device_tensor in enumerate(device_tensors[1:], start=1):
        got = ttnn.to_torch(device_tensor).float()
        ok, msg = comp_pcc(first, got, pcc=pcc)
        assert ok, f"device {idx} output differs from device 0: {msg}"


def _assert_local_cache_layout(kv_cache, *, max_seq_len, block_size):
    expected_blocks = (max_seq_len + block_size - 1) // block_size
    expected_shape = (expected_blocks, LOCAL_NUM_KV_HEADS, block_size, HEAD_DIM)
    for name, cache in zip(("k", "v"), kv_cache):
        device_tensors = ttnn.get_device_tensors(cache)
        assert len(device_tensors) == TP_FACTOR
        for idx, device_tensor in enumerate(device_tensors):
            assert tuple(device_tensor.shape) == expected_shape, f"{name} cache device {idx}: {device_tensor.shape}"


def _run_single_chip_baseline(state_dict, cfg, x_prefill, x_decode, page_table_host, *, seq_len):
    with _open_single_mesh() as mesh_device:
        decoder = OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=cfg,
            layer_idx=0,
            mesh_device=mesh_device,
            max_position_embeddings=max(TEST_TABLE_LEN, seq_len + 32),
        )
        kv_cache = OptimizedDecoder.allocate_paged_kv_cache(
            hf_config=cfg,
            mesh_device=mesh_device,
            max_batch_size=1,
            max_seq_len=seq_len + 32,
            block_size=32,
        )
        page_table = ttnn.Tensor(page_table_host, ttnn.int32).to(mesh_device)
        x_tt = ttnn.Tensor(x_prefill.reshape(1, 1, seq_len, cfg.hidden_size), ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(
            mesh_device
        )

        prefill = decoder.prefill_forward(
            x_tt,
            page_table=page_table,
            kv_cache=kv_cache,
            start_pos=0,
            rope_sequence_length=seq_len,
        )
        ttnn.synchronize_device(mesh_device)
        prefill_torch = ttnn.to_torch(prefill).reshape(1, seq_len, cfg.hidden_size)

        x_decode_tt = ttnn.Tensor(x_decode.reshape(1, 1, 1, cfg.hidden_size), ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(
            mesh_device
        )
        current_pos = ttnn.Tensor(torch.tensor([seq_len], dtype=torch.int32), ttnn.int32).to(mesh_device)
        position_ids = ttnn.Tensor(torch.tensor([seq_len], dtype=torch.uint32), ttnn.uint32).to(mesh_device)
        decode = decoder.decode_forward(
            x_decode_tt,
            current_pos=current_pos,
            position_ids=position_ids,
            page_table=page_table,
            kv_cache=kv_cache,
            rope_sequence_length=seq_len + 1,
        )
        ttnn.synchronize_device(mesh_device)
        decode_torch = ttnn.to_torch(decode).reshape(1, 1, cfg.hidden_size)
        return prefill_torch, decode_torch


def _run_multichip(state_dict, cfg, x_prefill, x_decode, page_table_host, *, seq_len, trace_decode=True):
    with _open_t3k_ring_mesh() as mesh_device:
        decoder = MultichipDecoder.from_state_dict(
            state_dict,
            hf_config=cfg,
            layer_idx=0,
            mesh_device=mesh_device,
            max_position_embeddings=max(TEST_TABLE_LEN, seq_len + 32),
        )
        kv_cache = MultichipDecoder.allocate_paged_kv_cache(
            hf_config=cfg,
            mesh_device=mesh_device,
            max_batch_size=1,
            max_seq_len=seq_len + 32,
            block_size=32,
        )
        _assert_local_cache_layout(kv_cache, max_seq_len=seq_len + 32, block_size=32)
        page_table = _replicate_to_mesh(page_table_host, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        x_tt = _replicate_to_mesh(
            x_prefill.reshape(1, 1, seq_len, cfg.hidden_size).contiguous(),
            mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        warm_prefill = decoder.prefill_forward(
            x_tt,
            page_table=page_table,
            kv_cache=kv_cache,
            start_pos=0,
            rope_sequence_length=seq_len,
        )
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(warm_prefill)

        signpost("PERF_PREFILL")
        prefill = decoder.prefill_forward(
            x_tt,
            page_table=page_table,
            kv_cache=kv_cache,
            start_pos=0,
            rope_sequence_length=seq_len,
        )
        ttnn.synchronize_device(mesh_device)
        signpost("PERF_PREFILL_END")
        _assert_replicated_mesh_output(prefill)
        prefill_torch = _to_torch_first_device(prefill).reshape(1, seq_len, cfg.hidden_size)

        x_decode_tt = _replicate_to_mesh(
            x_decode.reshape(1, 1, 1, cfg.hidden_size).contiguous(),
            mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        current_pos, position_ids = _position_tensors(mesh_device, seq_len)

        decoder.decode_forward(
            x_decode_tt,
            current_pos=current_pos,
            position_ids=position_ids,
            page_table=page_table,
            kv_cache=kv_cache,
            rope_sequence_length=seq_len + 1,
        )
        ttnn.synchronize_device(mesh_device)

        if trace_decode:
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            decode = decoder.decode_forward(
                x_decode_tt,
                current_pos=current_pos,
                position_ids=position_ids,
                page_table=page_table,
                kv_cache=kv_cache,
                rope_sequence_length=seq_len + 1,
            )
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            signpost("PERF_DECODE")
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            signpost("PERF_DECODE_END")
            timing_iters = int(os.getenv("PHI35_HOST_TIMING_ITERS", "0"))
            if timing_iters > 0:
                ttnn.synchronize_device(mesh_device)
                start = time.perf_counter()
                for _ in range(timing_iters):
                    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
                ttnn.synchronize_device(mesh_device)
                elapsed_us = (time.perf_counter() - start) * 1_000_000.0
                print(f"PHI35_MULTICHIP_HOST_TIMED_TRACE_DECODE_E2E_US: {elapsed_us / timing_iters:.3f}")
            ttnn.release_trace(mesh_device, trace_id)
        else:
            decode = decoder.decode_forward(
                x_decode_tt,
                current_pos=current_pos,
                position_ids=position_ids,
                page_table=page_table,
                kv_cache=kv_cache,
                rope_sequence_length=seq_len + 1,
            )

        ttnn.synchronize_device(mesh_device)
        _assert_replicated_mesh_output(decode)
        decode_torch = _to_torch_first_device(decode).reshape(1, 1, cfg.hidden_size)
        if os.getenv("PHI35_READ_DEVICE_PROFILER") == "1":
            ttnn.synchronize_device(mesh_device)
            ttnn.ReadDeviceProfiler(mesh_device)
        return prefill_torch, decode_torch


def _run_multichip_vs_single_chip(state_dict, *, seq_len=32, seed=1, trace_decode=True):
    cfg = _hf_config()
    torch.manual_seed(seed)
    x_prefill = (torch.randn(1, seq_len, cfg.hidden_size, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    x_decode = (torch.randn(1, 1, cfg.hidden_size, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    page_table_host = _page_table((seq_len + 31) // 32 + 1)

    baseline_prefill, baseline_decode = _run_single_chip_baseline(
        state_dict, cfg, x_prefill, x_decode, page_table_host, seq_len=seq_len
    )
    multichip_prefill, multichip_decode = _run_multichip(
        state_dict, cfg, x_prefill, x_decode, page_table_host, seq_len=seq_len, trace_decode=trace_decode
    )

    prefill_ok, prefill_msg = comp_pcc(baseline_prefill.float(), multichip_prefill.float(), pcc=PCC_THRESHOLD)
    decode_ok, decode_msg = comp_pcc(baseline_decode.float(), multichip_decode.float(), pcc=PCC_THRESHOLD)
    print(f"single-chip-vs-multichip prefill PCC: {prefill_msg}")
    print(f"single-chip-vs-multichip decode PCC: {decode_msg}")
    return {
        "prefill_ok": prefill_ok,
        "prefill_msg": prefill_msg,
        "decode_ok": decode_ok,
        "decode_msg": decode_msg,
        "prefill_output": multichip_prefill,
        "decode_output": multichip_decode,
    }


def test_multichip_mesh_plan_static():
    plan = mesh_strategy_summary()
    assert plan["single_chip_baseline"] == "OptimizedDecoder"
    assert plan["mesh_shape"] == TARGET_MESH_SHAPE
    assert plan["tp_factor"] == TP_FACTOR
    assert plan["attention"]["q_heads_per_device"] == LOCAL_NUM_HEADS == 4
    assert plan["attention"]["kv_heads_per_device"] == LOCAL_NUM_KV_HEADS == 4
    assert plan["attention"]["qkv_weight_per_device"] == [HIDDEN_SIZE, LOCAL_QKV_SIZE]
    assert plan["attention"]["o_weight_per_device"] == [LOCAL_HIDDEN_SIZE, HIDDEN_SIZE]
    assert plan["mlp"]["gate_up_weight_per_device"] == [HIDDEN_SIZE, LOCAL_GATE_UP_SIZE]
    assert plan["mlp"]["down_weight_per_device"] == [LOCAL_INTERMEDIATE_SIZE, HIDDEN_SIZE]
    assert plan["moe"] == "not_applicable_dense_phi"


@pytest.mark.timeout(600)
def test_multichip_vs_single_chip_synthetic_prefill_decode_pcc_1x8_ring():
    result = _run_multichip_vs_single_chip(_synthetic_state_dict(), seq_len=32, seed=3, trace_decode=True)
    assert result["prefill_ok"], result["prefill_msg"]
    assert result["decode_ok"], result["decode_msg"]


@pytest.mark.timeout(700)
def test_multichip_dense_layer_real_weights_prefill_decode_pcc_1x8_ring():
    result = _run_multichip_vs_single_chip(_real_layer0_state_dict(), seq_len=32, seed=7, trace_decode=True)
    assert result["prefill_ok"], result["prefill_msg"]
    assert result["decode_ok"], result["decode_msg"]


@pytest.mark.skipif(os.getenv("PHI35_RUN_PERF") != "1", reason="set PHI35_RUN_PERF=1 for profiler-only multichip run")
@pytest.mark.timeout(700)
def test_multichip_dense_layer_real_weights_perf_profile_1x8_ring():
    cfg = _hf_config()
    seq_len = int(os.getenv("PHI35_PERF_SEQ_LEN", "32"))
    torch.manual_seed(23)
    x_prefill = (torch.randn(1, seq_len, cfg.hidden_size, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    x_decode = (torch.randn(1, 1, cfg.hidden_size, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    page_table_host = _page_table((seq_len + 31) // 32 + 1)

    prefill, decode = _run_multichip(
        _real_layer0_state_dict(),
        cfg,
        x_prefill,
        x_decode,
        page_table_host,
        seq_len=seq_len,
        trace_decode=True,
    )

    assert prefill.shape == (1, seq_len, cfg.hidden_size)
    assert decode.shape == (1, 1, cfg.hidden_size)
    assert torch.isfinite(prefill).all()
    assert torch.isfinite(decode).all()


@pytest.mark.timeout(900)
def test_multichip_repeated_input_determinism_1x8_ring():
    state = _synthetic_state_dict(seed=11)
    first = _run_multichip_vs_single_chip(state, seq_len=32, seed=13, trace_decode=False)
    second = _run_multichip_vs_single_chip(state, seq_len=32, seed=13, trace_decode=False)
    prefill_ok, prefill_msg = comp_pcc(first["prefill_output"].float(), second["prefill_output"].float(), pcc=0.9999)
    decode_ok, decode_msg = comp_pcc(first["decode_output"].float(), second["decode_output"].float(), pcc=0.9999)
    assert prefill_ok, prefill_msg
    assert decode_ok, decode_msg


def test_runtime_forward_fallback_audit_static():
    module = importlib.import_module("models.autoports.microsoft_phi_3_5_mini_instruct.tt.multichip_decoder")
    runtime_callables = [
        MultichipDecoder.prefill_forward,
        MultichipDecoder.decode_forward,
        MultichipDecoder._mlp_forward,
        MultichipDecoder._prefill_rope_tables,
        MultichipDecoder._decode_rope_tables,
        MultichipDecoder._all_reduce,
        module._apply_rope,
        module._typecast_if_needed,
    ]
    for callable_obj in runtime_callables:
        source = inspect.getsource(callable_obj)
        forbidden = ("torch.", "ttnn.from_torch", "ttnn.to_torch", "from_torch(", "to_torch(", "from_device(", ".cpu(")
        hits = [token for token in forbidden if token in source]
        assert not hits, f"{callable_obj.__name__} contains forbidden runtime fallback tokens: {hits}"


@pytest.mark.skipif(os.getenv("PHI35_RUN_LONG_CONTEXT") != "1", reason="set PHI35_RUN_LONG_CONTEXT=1 for full-context stress")
@pytest.mark.timeout(900)
def test_multichip_full_context_decode_current_position_and_page_table_1x8_ring():
    cfg = _hf_config()
    seq_len = cfg.max_position_embeddings
    state = _synthetic_state_dict(seed=17)
    with _open_t3k_ring_mesh() as mesh_device:
        decoder = MultichipDecoder.from_state_dict(state, hf_config=cfg, layer_idx=0, mesh_device=mesh_device)
        kv_cache = MultichipDecoder.allocate_paged_kv_cache(
            hf_config=cfg,
            mesh_device=mesh_device,
            max_batch_size=1,
            max_seq_len=seq_len,
            block_size=32,
        )
        _assert_local_cache_layout(kv_cache, max_seq_len=seq_len, block_size=32)
        page_blocks = seq_len // 32
        page_table = _replicate_to_mesh(
            torch.arange(page_blocks, dtype=torch.int32).reshape(1, page_blocks),
            mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        x_decode = _replicate_to_mesh(
            torch.zeros(1, 1, 1, cfg.hidden_size, dtype=torch.bfloat16),
            mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        current_pos, position_ids = _position_tensors(mesh_device, seq_len - 1)
        out = decoder.decode_forward(
            x_decode,
            current_pos=current_pos,
            position_ids=position_ids,
            page_table=page_table,
            kv_cache=kv_cache,
            rope_sequence_length=seq_len,
        )
        got = _to_torch_first_device(out)
        assert got.shape == (1, 1, 1, cfg.hidden_size)
        assert torch.isfinite(got).all()
        _assert_replicated_mesh_output(out)


@pytest.mark.skipif(os.getenv("PHI35_RUN_LONG_PREFILL") != "1", reason="set PHI35_RUN_LONG_PREFILL=1 for long prefill")
@pytest.mark.timeout(1200)
def test_multichip_long_prefill_page_table_1x8_ring():
    cfg = _hf_config()
    seq_len = int(os.getenv("PHI35_LONG_PREFILL_LEN", "4096"))
    assert seq_len % 32 == 0
    state = _synthetic_state_dict(seed=19)
    with _open_t3k_ring_mesh() as mesh_device:
        decoder = MultichipDecoder.from_state_dict(
            state,
            hf_config=cfg,
            layer_idx=0,
            mesh_device=mesh_device,
            max_position_embeddings=seq_len,
        )
        kv_cache = MultichipDecoder.allocate_paged_kv_cache(
            hf_config=cfg,
            mesh_device=mesh_device,
            max_batch_size=1,
            max_seq_len=seq_len,
            block_size=32,
        )
        page_blocks = seq_len // 32
        page_table = _replicate_to_mesh(_page_table(page_blocks), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        x_prefill = _replicate_to_mesh(
            torch.zeros(1, 1, seq_len, cfg.hidden_size, dtype=torch.bfloat16),
            mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        out = decoder.prefill_forward(
            x_prefill,
            page_table=page_table,
            kv_cache=kv_cache,
            start_pos=0,
            rope_sequence_length=seq_len,
        )
        got = _to_torch_first_device(out)
        assert got.shape == (1, 1, seq_len, cfg.hidden_size)
        assert torch.isfinite(got).all()
        _assert_replicated_mesh_output(out)
