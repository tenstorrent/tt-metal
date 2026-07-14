# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
import math
import os
import statistics
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.autoports.google_gemma_4_31b.tests.test_functional_decoder import (
    LAYER_KINDS,
    _assert_pcc,
    _layer_state,
    _rope_host,
)
from models.autoports.google_gemma_4_31b.tests.test_optimized_decoder import _paged_state as _single_paged_state
from models.autoports.google_gemma_4_31b.tt.multichip_decoder import (
    _PERSISTENT_CCL_POOLS,
    DEFAULT_MULTICHIP_OPTIMIZATION_POLICY,
    PAGE_BLOCK_SIZE,
    TARGET_MESH_SHAPE,
    TP_SIZE,
    MultichipDecoder,
    _tp_allreduce,
    _TPOptimizedSharedMLP,
    release_multichip_decoder_resources,
)
from models.autoports.google_gemma_4_31b.tt.optimized_decoder import DEFAULT_OPTIMIZATION_POLICY, OptimizedDecoder
from models.common.utility_functions import comp_pcc

PCC_THRESHOLD = 0.995


def _checkpoint_dir() -> Path:
    root = Path.home() / ".cache/huggingface/hub/models--google--gemma-4-31B/snapshots"
    snapshots = sorted(root.glob("*"))
    if not snapshots:
        pytest.skip("real google/gemma-4-31B checkpoint is not cached")
    return snapshots[-1]


@pytest.fixture(scope="module")
def hf_config():
    return AutoConfig.from_pretrained(_checkpoint_dir(), trust_remote_code=True, local_files_only=True).text_config


def _tt_input(tensor, mesh_device):
    device_shape = (
        tensor.transpose(0, 1).unsqueeze(0)
        if tensor.shape[0] > 1 and tensor.shape[1] == 1
        else tensor.reshape(1, 1, tensor.shape[1], tensor.shape[2])
    )
    return ttnn.from_torch(
        device_shape,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=(ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None),
    )


def _rope_device(hf_config, layer_idx, max_seq_len, mesh_device, *, decode):
    cos, sin = _rope_host(hf_config, layer_idx, torch.arange(max_seq_len))
    if decode:
        cos, sin = cos.squeeze(0), sin.squeeze(0)
        layout = ttnn.ROW_MAJOR_LAYOUT
    else:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
        layout = ttnn.TILE_LAYOUT
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None
    return tuple(
        ttnn.from_torch(
            table,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=layout,
            mesh_mapper=mapper,
        )
        for table in (cos, sin)
    )


def _to_host_first(tensor):
    device = tensor.device()
    if hasattr(device, "get_num_devices") and device.get_num_devices() > 1:
        tensor = ttnn.get_device_tensors(tensor)[0]
    return ttnn.to_torch(tensor).reshape(1, tensor.shape[-2], tensor.shape[-1]).float()


def _replicated_int(host, mesh_device, *, dtype):
    return ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=(ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None),
    )


def _copy_host_to_replicas(host_tensor, mesh_tensor):
    for device_tensor in ttnn.get_device_tensors(mesh_tensor):
        ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)


def _permuted_page_table(page_table, mesh_device):
    """Return a replicated non-identity table over the same physical blocks."""
    rows = torch.arange(page_table.shape[0] * page_table.shape[1], dtype=torch.int32).reshape(
        page_table.shape[0], page_table.shape[1]
    )
    rows = torch.roll(rows, shifts=1, dims=1)
    return _replicated_int(rows, mesh_device, dtype=ttnn.int32)


def _inputs(hf_config, layer_idx):
    torch.manual_seed(20260714 + layer_idx)
    return {
        "prefill33": torch.randn(1, 33, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02,
        "prompt32": torch.randn(1, 32, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02,
        "token": torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02,
    }


@pytest.fixture(scope="module")
def optimized_baseline(hf_config):
    """Collect the frozen single-P150 reference before enabling mesh fabric."""
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    results = {}
    try:
        for layer_idx, layer_kind in LAYER_KINDS:
            values = _inputs(hf_config, layer_idx)
            state = _layer_state(layer_idx)
            decoder = OptimizedDecoder.from_state_dict(
                state, hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh
            )
            cache, page_table = _single_paged_state(hf_config, layer_idx, 128, mesh)
            out = decoder.prefill_forward(
                _tt_input(values["prefill33"], mesh),
                rope_mats=_rope_device(hf_config, layer_idx, 33, mesh, decode=False),
                page_table=page_table,
                kv_cache=cache,
                valid_seq_len=33,
            )
            results[(layer_kind, "prefill33")] = _to_host_first(out).clone()

            cache, page_table = _single_paged_state(hf_config, layer_idx, 128, mesh)
            decoder.prefill_forward(
                _tt_input(values["prompt32"], mesh),
                rope_mats=_rope_device(hf_config, layer_idx, 32, mesh, decode=False),
                page_table=page_table,
                kv_cache=cache,
                valid_seq_len=32,
            )
            pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
            pos_u_host[0, 0] = 32
            out = decoder.decode_forward(
                _tt_input(values["token"], mesh),
                rope_mats=_rope_device(hf_config, layer_idx, 128, mesh, decode=True),
                page_table=page_table,
                kv_cache=cache,
                current_position=_replicated_int(pos_u_host, mesh, dtype=ttnn.uint32),
                current_position_cache=_replicated_int(torch.tensor([32], dtype=torch.int32), mesh, dtype=ttnn.int32),
            )
            results[(layer_kind, "decode32")] = _to_host_first(out).clone()

            batch_size, batch_prompt_len = 32, 33
            torch.manual_seed(20261900 + layer_idx)
            batch_prompt = (
                torch.randn(
                    batch_size,
                    batch_prompt_len,
                    hf_config.hidden_size,
                    dtype=torch.bfloat16,
                )
                * 0.02
            )
            batch_token = torch.randn(batch_size, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
            cache, page_table = _single_paged_state(
                hf_config,
                layer_idx,
                128,
                mesh,
                batch_size=batch_size,
            )
            batch_prefill = decoder.prefill_forward(
                _tt_input(
                    batch_prompt.reshape(1, batch_size * batch_prompt_len, hf_config.hidden_size),
                    mesh,
                ),
                rope_mats=_rope_device(hf_config, layer_idx, batch_prompt_len, mesh, decode=False),
                page_table=page_table,
                kv_cache=cache,
                batch_size=batch_size,
                valid_seq_len=batch_prompt_len,
            )
            assert batch_prefill.shape[-2] == batch_size * batch_prompt_len
            batch_pos_host = torch.full((1, batch_size), batch_prompt_len, dtype=torch.int32)
            batch_decoded = decoder.decode_forward(
                _tt_input(batch_token, mesh),
                rope_mats=_rope_device(hf_config, layer_idx, 128, mesh, decode=True),
                page_table=page_table,
                kv_cache=cache,
                current_position=_replicated_int(batch_pos_host, mesh, dtype=ttnn.uint32),
                current_position_cache=_replicated_int(
                    torch.full((batch_size,), batch_prompt_len, dtype=torch.int32),
                    mesh,
                    dtype=ttnn.int32,
                ),
                batch_size=batch_size,
            )
            results[(layer_kind, "batch32_decode33")] = (
                ttnn.to_torch(batch_decoded).reshape(batch_size, 1, hf_config.hidden_size).float().clone()
            )

            if layer_kind == "sliding_attention":
                for seq_len in (1025, 1057):
                    torch.manual_seed(20262000 + seq_len)
                    prompt = torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
                    token = torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
                    cache, page_table = _single_paged_state(hf_config, layer_idx, seq_len + 1, mesh)
                    prefill = decoder.prefill_forward(
                        _tt_input(prompt, mesh),
                        rope_mats=_rope_device(hf_config, layer_idx, seq_len, mesh, decode=False),
                        page_table=page_table,
                        kv_cache=cache,
                        valid_seq_len=seq_len,
                    )
                    assert prefill.shape[-2] == seq_len
                    pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
                    pos_u_host[0, 0] = seq_len
                    decoded = decoder.decode_forward(
                        _tt_input(token, mesh),
                        rope_mats=_rope_device(hf_config, layer_idx, seq_len + 1, mesh, decode=True),
                        page_table=page_table,
                        kv_cache=cache,
                        current_position=_replicated_int(pos_u_host, mesh, dtype=ttnn.uint32),
                        current_position_cache=_replicated_int(
                            torch.tensor([seq_len], dtype=torch.int32),
                            mesh,
                            dtype=ttnn.int32,
                        ),
                    )
                    results[(layer_kind, f"decode{seq_len}")] = _to_host_first(decoded).clone()

            if "GEMMA4_MULTICHIP_BENCH" in os.environ:
                torch.manual_seed(20262100 + layer_idx)
                prefill_host = torch.randn(1, 128, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
                cache, page_table = _single_paged_state(hf_config, layer_idx, 128, mesh)
                prefill_input = _tt_input(prefill_host, mesh)
                prefill_rope = _rope_device(hf_config, layer_idx, 128, mesh, decode=False)

                def prefill_call():
                    return decoder.prefill_forward(
                        prefill_input,
                        rope_mats=prefill_rope,
                        page_table=page_table,
                        kv_cache=cache,
                        valid_seq_len=128,
                    )

                warm = prefill_call()
                ttnn.synchronize_device(mesh)
                warm.deallocate(True)
                prefill_samples = []
                for _ in range(12):
                    start = time.perf_counter_ns()
                    sample = prefill_call()
                    ttnn.synchronize_device(mesh)
                    prefill_samples.append((time.perf_counter_ns() - start) / 1_000_000)
                    sample.deallocate(True)

                baseline_token = _tt_input(values["token"], mesh)
                baseline_decode_rope = _rope_device(hf_config, layer_idx, 128, mesh, decode=True)
                baseline_pos_host = torch.zeros((1, 32), dtype=torch.int32)
                baseline_pos_host[0, 0] = 32
                baseline_pos_u = _replicated_int(baseline_pos_host, mesh, dtype=ttnn.uint32)
                baseline_pos_i = _replicated_int(
                    torch.tensor([32], dtype=torch.int32),
                    mesh,
                    dtype=ttnn.int32,
                )

                def decode_call():
                    return decoder.decode_forward(
                        baseline_token,
                        rope_mats=baseline_decode_rope,
                        page_table=page_table,
                        kv_cache=cache,
                        current_position=baseline_pos_u,
                        current_position_cache=baseline_pos_i,
                    )

                warm = decode_call()
                ttnn.synchronize_device(mesh)
                warm.deallocate(True)
                trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
                decode_call()
                ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
                try:
                    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                    decode_samples = []
                    for _ in range(12):
                        start = time.perf_counter_ns()
                        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                        decode_samples.append((time.perf_counter_ns() - start) / 1_000_000)
                finally:
                    ttnn.release_trace(mesh, trace_id)
                results[(layer_kind, "baseline_latency")] = {
                    "prefill128_ms": statistics.median(prefill_samples),
                    "prefill128_samples_ms": prefill_samples,
                    "decode_ms": statistics.median(decode_samples),
                    "decode_samples_ms": decode_samples,
                }
    finally:
        ttnn.close_mesh_device(mesh)
    return results


@pytest.fixture(scope="module")
def mesh_device(optimized_baseline):
    if ttnn.get_num_devices() < TP_SIZE:
        pytest.skip("Gemma 4 multichip decoder requires four local devices")
    ring = os.environ.get("GEMMA4_MC_TOPOLOGY") == "ring"
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING if ring else ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*TARGET_MESH_SHAPE), trace_region_size=96 << 20)
    try:
        yield mesh
    finally:
        release_multichip_decoder_resources(mesh)
        assert id(mesh) not in _PERSISTENT_CCL_POOLS
        ttnn.close_mesh_device(mesh)


@pytest.fixture(scope="module")
def profile_mesh_device():
    """Open the target directly so Tracy does not record baseline setup ops."""
    if ttnn.get_num_devices() < TP_SIZE:
        pytest.skip("Gemma 4 multichip decoder requires four local devices")
    ring = os.environ.get("GEMMA4_MC_TOPOLOGY") == "ring"
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING if ring else ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*TARGET_MESH_SHAPE), trace_region_size=96 << 20)
    try:
        yield mesh
    finally:
        release_multichip_decoder_resources(mesh)
        assert id(mesh) not in _PERSISTENT_CCL_POOLS
        ttnn.close_mesh_device(mesh)


@pytest.fixture
def fused_mmrs_ring_mesh_device():
    """Open an isolated TP4 ring mesh for the focused fused MM+RS probe."""
    if ttnn.get_num_devices() < TP_SIZE:
        pytest.skip("Gemma 4 fused MM+RS candidate requires four local devices")

    mesh = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(*TARGET_MESH_SHAPE), trace_region_size=96 << 20)
        yield mesh
    finally:
        if mesh is not None:
            release_multichip_decoder_resources(mesh)
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _decoder(hf_config, mesh_device, layer_idx, *, optimization_policy=None):
    if optimization_policy is None:
        optimization_policy = DEFAULT_MULTICHIP_OPTIMIZATION_POLICY
    attention_dtype = os.environ.get("GEMMA4_MC_ATTENTION_DTYPE")
    if attention_dtype is not None:
        optimization_policy = replace(
            optimization_policy,
            attention_weight_dtype={
                "bfloat8_b": ttnn.bfloat8_b,
                "bfloat4_b": ttnn.bfloat4_b,
            }[attention_dtype],
        )
    if "GEMMA4_MC_QKV_IN0_BLOCK_W" in os.environ:
        optimization_policy = replace(
            optimization_policy,
            qkv_in0_block_w=int(os.environ["GEMMA4_MC_QKV_IN0_BLOCK_W"]),
        )
    mlp_topology = os.environ.get("GEMMA4_MC_MLP_TOPOLOGY")
    if mlp_topology is not None:
        packed_output_dtype = {
            "bfloat16": ttnn.bfloat16,
            "bfloat8_b": ttnn.bfloat8_b,
        }[os.environ.get("GEMMA4_MC_MLP_PACKED_OUTPUT_DTYPE", "bfloat16")]
        optimization_policy = replace(
            optimization_policy,
            mlp_gate_up_topology=mlp_topology,
            mlp_packed_output_dtype=packed_output_dtype,
        )
    communication_dtype = {
        "bfloat16": ttnn.bfloat16,
        "bfloat8_b": ttnn.bfloat8_b,
    }[os.environ.get("GEMMA4_MC_CCL_DTYPE", "bfloat8_b")]
    decoder = MultichipDecoder.from_state_dict(
        _layer_state(layer_idx),
        hf_config=hf_config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        num_links=int(os.environ.get("GEMMA4_MC_NUM_LINKS", "2")),
        qkv_decode_output_cores=int(os.environ.get("GEMMA4_MC_QKV_OUTPUT_CORES", "32")),
        communication_dtype=communication_dtype,
        use_persistent_async_ccl=os.environ.get("GEMMA4_MC_PERSISTENT_ASYNC_CCL", "1") == "1",
        optimization_policy=optimization_policy,
        topology=(ttnn.Topology.Ring if os.environ.get("GEMMA4_MC_TOPOLOGY") == "ring" else ttnn.Topology.Linear),
    )
    if "GEMMA4_MC_MLP_CORES" in os.environ:
        decoder.layer.shared_mlp.policy = _mlp_geometry_policy(
            int(os.environ["GEMMA4_MC_MLP_CORES"]),
            base_policy=decoder.layer.shared_mlp.policy,
        )
    assert type(decoder) is MultichipDecoder
    assert type(decoder.layer.shared_mlp) is _TPOptimizedSharedMLP
    return decoder


def _mlp_geometry_policy(num_cores, *, base_policy):
    block_w = {
        # The packed 2N gate/up row exceeds L1 with block 24; block 12 is the
        # adapted legal divisor used for the packed-policy retry.
        7: (12, 12),
        8: (7, 21),
        12: (7, 7),
        14: (12, 12),
        21: (4, 4),
        24: (7, 7),
        28: (6, 6),
        42: (4, 4),
        56: (3, 3),
        84: (2, 2),
    }
    gate_up_in0_block_w, down_in0_block_w = block_w[num_cores]
    return replace(
        base_policy,
        name=f"tp4_square_mlp_{num_cores}c",
        decode_num_cores=num_cores,
        gate_up_in0_block_w=gate_up_in0_block_w,
        down_in0_block_w=down_in0_block_w,
    )


def test_multichip_persistent_pool_terminal_cleanup(monkeypatch):
    events = []

    class FakeBuffer:
        def __init__(self, name):
            self.name = name

        def deallocate(self, force):
            events.append(("deallocate", self.name, force))

    class FakeMesh:
        pass

    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: events.append(("synchronize", id(mesh))))
    for index in range(2):
        mesh = FakeMesh()
        pool = {
            "mesh_device": mesh,
            "semaphores": [object(), object()],
            "buffers": {"scratch": FakeBuffer(f"scratch-{index}")},
            "slot": 0,
        }
        _PERSISTENT_CCL_POOLS[id(mesh)] = pool
        assert release_multichip_decoder_resources(mesh)
        assert id(mesh) not in _PERSISTENT_CCL_POOLS
        assert pool["mesh_device"] is None
        assert pool["semaphores"] == []
        assert pool["buffers"] == {}
        assert pool["released"] is True
        assert not release_multichip_decoder_resources(mesh)

    assert events == [
        ("synchronize", events[0][1]),
        ("deallocate", "scratch-0", True),
        ("synchronize", events[2][1]),
        ("deallocate", "scratch-1", True),
    ]


@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
def test_multichip_prefill_matches_optimized_baseline(
    hf_config, mesh_device, optimized_baseline, layer_idx, layer_kind
):
    decoder = _decoder(hf_config, mesh_device, layer_idx)
    cache, page_table = decoder.init_paged_kv_cache(max_context=128)
    output = decoder.prefill_forward(
        _tt_input(_inputs(hf_config, layer_idx)["prefill33"], mesh_device),
        rope_mats=_rope_device(hf_config, layer_idx, 33, mesh_device, decode=False),
        page_table=page_table,
        kv_cache=cache,
        valid_seq_len=33,
    )
    reference = optimized_baseline[(layer_kind, "prefill33")]
    device_outputs = [ttnn.to_torch(t).reshape_as(reference).float() for t in ttnn.get_device_tensors(output)]
    for actual in device_outputs:
        _assert_pcc(reference, actual, PCC_THRESHOLD)
    for actual in device_outputs[1:]:
        assert torch.equal(device_outputs[0], actual)


@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
def test_multichip_paged_decode_trace_matches_optimized_baseline(
    hf_config, mesh_device, optimized_baseline, layer_idx, layer_kind
):
    decoder = _decoder(hf_config, mesh_device, layer_idx)
    values = _inputs(hf_config, layer_idx)
    cache, identity_page_table = decoder.init_paged_kv_cache(max_context=128)
    page_table = _permuted_page_table(identity_page_table, mesh_device)
    decoder.prefill_forward(
        _tt_input(values["prompt32"], mesh_device),
        rope_mats=_rope_device(hf_config, layer_idx, 32, mesh_device, decode=False),
        page_table=page_table,
        kv_cache=cache,
        valid_seq_len=32,
    )
    pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
    pos_u_host[0, 0] = 32
    pos_u = _replicated_int(pos_u_host, mesh_device, dtype=ttnn.uint32)
    pos_i = _replicated_int(torch.tensor([32], dtype=torch.int32), mesh_device, dtype=ttnn.int32)
    token = _tt_input(values["token"], mesh_device)
    rope = _rope_device(hf_config, layer_idx, 128, mesh_device, decode=True)

    def call():
        return decoder.decode_forward(
            token,
            rope_mats=rope,
            page_table=page_table,
            kv_cache=cache,
            current_position=pos_u,
            current_position_cache=pos_i,
        )

    warm = call()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    pool = decoder.ccl_manager._persistent_allreduce_pool
    assert _PERSISTENT_CCL_POOLS[id(mesh_device)] is pool
    # Attention-O and MLP-down serialize through one physical-capacity scratch
    # tensor on the common tail grid; their collective epochs remain separate.
    assert len(pool["buffers"]) == 1
    scratch = next(iter(pool["buffers"].values()))
    assert scratch.memory_config().shard_spec.grid.num_cores() == 24
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        first = _to_host_first(output).clone()
        _assert_pcc(optimized_baseline[(layer_kind, "decode32")], first, PCC_THRESHOLD)
        for _ in range(7):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            assert torch.equal(first, _to_host_first(output))

        changed_token_host = ttnn.from_torch(
            (values["token"] + 0.125).reshape(1, 1, 1, hf_config.hidden_size),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        changed_pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
        changed_pos_u_host[0, 0] = 33
        changed_pos_u = ttnn.from_torch(changed_pos_u_host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        changed_pos_i = ttnn.from_torch(
            torch.tensor([33], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        _copy_host_to_replicas(changed_token_host, token)
        _copy_host_to_replicas(changed_pos_u, pos_u)
        _copy_host_to_replicas(changed_pos_i, pos_i)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        changed = _to_host_first(output).clone()
        assert torch.isfinite(changed).all()
        assert not torch.equal(first, changed)

        original_token_host = ttnn.from_torch(
            values["token"].reshape(1, 1, 1, hf_config.hidden_size),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        original_pos_u = ttnn.from_torch(pos_u_host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        original_pos_i = ttnn.from_torch(
            torch.tensor([32], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        _copy_host_to_replicas(original_token_host, token)
        _copy_host_to_replicas(original_pos_u, pos_u)
        _copy_host_to_replicas(original_pos_i, pos_i)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        assert torch.equal(first, _to_host_first(output))
        for replica in ttnn.get_device_tensors(output)[1:]:
            assert torch.equal(first, ttnn.to_torch(replica).reshape_as(first).float())
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("seq_len", [1025, 1057])
def test_multichip_sliding_nonaligned_window_wrap_matches_baseline(hf_config, mesh_device, optimized_baseline, seq_len):
    layer_idx, layer_kind = 0, "sliding_attention"
    decoder = _decoder(hf_config, mesh_device, layer_idx)
    torch.manual_seed(20262000 + seq_len)
    prompt = torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    token_host = torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    cache, page_table = decoder.init_paged_kv_cache(max_context=seq_len + 1)
    output = decoder.prefill_forward(
        _tt_input(prompt, mesh_device),
        rope_mats=_rope_device(hf_config, layer_idx, seq_len, mesh_device, decode=False),
        page_table=page_table,
        kv_cache=cache,
        valid_seq_len=seq_len,
    )
    assert output.shape[-2] == seq_len
    pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
    pos_u_host[0, 0] = seq_len
    decoded = decoder.decode_forward(
        _tt_input(token_host, mesh_device),
        rope_mats=_rope_device(hf_config, layer_idx, seq_len + 1, mesh_device, decode=True),
        page_table=page_table,
        kv_cache=cache,
        current_position=_replicated_int(pos_u_host, mesh_device, dtype=ttnn.uint32),
        current_position_cache=_replicated_int(
            torch.tensor([seq_len], dtype=torch.int32),
            mesh_device,
            dtype=ttnn.int32,
        ),
    )
    reference = optimized_baseline[(layer_kind, f"decode{seq_len}")]
    for replica in ttnn.get_device_tensors(decoded):
        _assert_pcc(
            reference,
            ttnn.to_torch(replica).reshape_as(reference).float(),
            PCC_THRESHOLD,
        )


@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
def test_multichip_batch32_nonaligned_prefill_and_traced_decode(
    hf_config, mesh_device, optimized_baseline, layer_idx, layer_kind
):
    batch_size, prompt_len = 32, 33
    decoder = _decoder(hf_config, mesh_device, layer_idx)
    torch.manual_seed(20261900 + layer_idx)
    prompt = torch.randn(batch_size, prompt_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    token_host = torch.randn(batch_size, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    cache, page_table = decoder.init_paged_kv_cache(max_context=128, batch_size=batch_size)
    prefill = decoder.prefill_forward(
        _tt_input(
            prompt.reshape(1, batch_size * prompt_len, hf_config.hidden_size),
            mesh_device,
        ),
        rope_mats=_rope_device(hf_config, layer_idx, prompt_len, mesh_device, decode=False),
        page_table=page_table,
        kv_cache=cache,
        batch_size=batch_size,
        valid_seq_len=prompt_len,
    )
    assert prefill.shape[-2] == batch_size * prompt_len
    pos_host = torch.full((1, batch_size), prompt_len, dtype=torch.int32)
    token = _tt_input(token_host, mesh_device)
    kwargs = dict(
        rope_mats=_rope_device(hf_config, layer_idx, 128, mesh_device, decode=True),
        page_table=page_table,
        kv_cache=cache,
        current_position=_replicated_int(pos_host, mesh_device, dtype=ttnn.uint32),
        current_position_cache=_replicated_int(
            torch.full((batch_size,), prompt_len, dtype=torch.int32),
            mesh_device,
            dtype=ttnn.int32,
        ),
        batch_size=batch_size,
    )

    def call():
        return decoder.decode_forward(token, **kwargs)

    warm = call()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    pool = decoder.ccl_manager._persistent_allreduce_pool
    assert len(pool["buffers"]) == 1
    scratch = next(iter(pool["buffers"].values()))
    scratch_address = scratch.buffer_address()
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        reference = optimized_baseline[(layer_kind, "batch32_decode33")]
        first = ttnn.to_torch(ttnn.get_device_tensors(output)[0]).reshape_as(reference).float()
        _assert_pcc(reference, first, PCC_THRESHOLD)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        assert torch.equal(
            first,
            ttnn.to_torch(ttnn.get_device_tensors(output)[0]).reshape_as(reference).float(),
        )
        for replica in ttnn.get_device_tensors(output)[1:]:
            assert torch.equal(first, ttnn.to_torch(replica).reshape_as(reference).float())
        assert len(pool["buffers"]) == 1
        assert next(iter(pool["buffers"].values())).buffer_address() == scratch_address
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize(
    "layer_idx,layer_kind,local_heads,head_dim", [(0, "sliding_attention", 4, 256), (5, "full_attention", 1, 512)]
)
def test_multichip_kv_and_layout_contract(
    hf_config, mesh_device, optimized_baseline, layer_idx, layer_kind, local_heads, head_dim
):
    decoder = _decoder(hf_config, mesh_device, layer_idx)
    cache, page_table = decoder.init_paged_kv_cache(max_context=262_144)
    expected_blocks = 16 if layer_kind == "sliding_attention" else 4096
    for tensor in cache:
        assert tuple(tensor.shape) == (expected_blocks, local_heads, PAGE_BLOCK_SIZE, head_dim)
        assert tensor.dtype == ttnn.bfloat8_b
        assert len(ttnn.get_device_tensors(tensor)) == TP_SIZE
    assert tuple(page_table.shape) == (1, expected_blocks)
    assert decoder.layer.self_attn.weights.kv_replicated is False
    assert decoder.mesh_profile["activation_contract"].startswith("replicated BF16")


@pytest.mark.skipif(
    "GEMMA4_MULTICHIP_EXACT_CONTEXT" not in os.environ,
    reason="set GEMMA4_MULTICHIP_EXACT_CONTEXT=1 for advertised-position trace",
)
@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
def test_multichip_advertised_context_traced_decode(hf_config, mesh_device, optimized_baseline, layer_idx, layer_kind):
    max_context = hf_config.max_position_embeddings
    decoder = _decoder(hf_config, mesh_device, layer_idx)
    cache, page_table = decoder.init_paged_kv_cache(max_context=max_context)
    torch.manual_seed(20262300 + layer_idx)
    token = _tt_input(
        torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02,
        mesh_device,
    )
    pos_host = torch.zeros((1, 32), dtype=torch.int32)
    pos_host[0, 0] = max_context - 1
    kwargs = dict(
        rope_mats=_rope_device(hf_config, layer_idx, max_context, mesh_device, decode=True),
        page_table=page_table,
        kv_cache=cache,
        current_position=_replicated_int(pos_host, mesh_device, dtype=ttnn.uint32),
        current_position_cache=_replicated_int(
            torch.tensor([max_context - 1], dtype=torch.int32),
            mesh_device,
            dtype=ttnn.int32,
        ),
    )

    def call():
        return decoder.decode_forward(token, **kwargs)

    warm = call()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        first = _to_host_first(output).clone()
        assert torch.isfinite(first).all()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        assert torch.equal(first, _to_host_first(output))
        for replica in ttnn.get_device_tensors(output)[1:]:
            assert torch.equal(first, ttnn.to_torch(replica).reshape_as(first).float())
    finally:
        ttnn.release_trace(mesh_device, trace_id)


def test_multichip_runtime_source_audit():
    source = "\n".join(
        inspect.getsource(method)
        for method in (
            MultichipDecoder._forward_device,
            MultichipDecoder._prefill_attention_tp,
            MultichipDecoder._decode_attention_tp,
            _TPOptimizedSharedMLP._reduce,
            _TPOptimizedSharedMLP.__call__,
        )
    )
    for required in (
        "paged_fill_cache",
        "paged_scaled_dot_product_attention_decode",
        "_tp_allreduce",
        "local_kv_heads",
        "decode_wqkv",
    ):
        assert required in source
    for forbidden in ("ttnn.from_torch", "ttnn.to_torch", "torch."):
        assert forbidden not in source

    collective_source = inspect.getsource(_tp_allreduce)
    assert "ttnn.experimental.all_reduce_async" in collective_source
    assert "ttnn.to_torch" not in collective_source


@pytest.mark.skipif(
    "GEMMA4_MULTICHIP_MLP_SWEEP" not in os.environ,
    reason="set GEMMA4_MULTICHIP_MLP_SWEEP=1",
)
@pytest.mark.parametrize("num_cores", [7, 8, 12, 14, 21, 24, 28, 42, 56, 84])
def test_multichip_mlp_decode_geometry_sweep(
    hf_config,
    mesh_device,
    optimized_baseline,
    num_cores,
):
    """Measure packed gate/up plus down geometry in MLP-only and full traces."""
    layer_idx, layer_kind = LAYER_KINDS[0]
    decoder = _decoder(hf_config, mesh_device, layer_idx)
    decoder.layer.shared_mlp.policy = _mlp_geometry_policy(
        num_cores,
        base_policy=decoder.layer.shared_mlp.policy,
    )
    values = _inputs(hf_config, layer_idx)
    cache, page_table = decoder.init_paged_kv_cache(max_context=128)
    decoder.prefill_forward(
        _tt_input(values["prompt32"], mesh_device),
        rope_mats=_rope_device(hf_config, layer_idx, 32, mesh_device, decode=False),
        page_table=page_table,
        kv_cache=cache,
        valid_seq_len=32,
    )
    token = _tt_input(values["token"], mesh_device)
    pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
    pos_u_host[0, 0] = 32
    kwargs = dict(
        rope_mats=_rope_device(hf_config, layer_idx, 128, mesh_device, decode=True),
        page_table=page_table,
        kv_cache=cache,
        current_position=_replicated_int(pos_u_host, mesh_device, dtype=ttnn.uint32),
        current_position_cache=_replicated_int(torch.tensor([32], dtype=torch.int32), mesh_device, dtype=ttnn.int32),
    )

    def call():
        return decoder.decode_forward(token, **kwargs)

    warm = call()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        layer_samples = []
        for _ in range(12):
            start = time.perf_counter_ns()
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            layer_samples.append((time.perf_counter_ns() - start) / 1_000_000)
        reference = optimized_baseline[(layer_kind, "decode32")]
        actuals = [ttnn.to_torch(t).reshape_as(reference).float() for t in ttnn.get_device_tensors(output)]
        for actual in actuals:
            _assert_pcc(reference, actual, PCC_THRESHOLD)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    output.deallocate(True)

    mlp = decoder.layer.shared_mlp
    mlp.is_decode = True

    def mlp_call():
        return mlp(token)

    warm = mlp_call()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    mlp_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    mlp_output = mlp_call()
    ttnn.end_trace_capture(mesh_device, mlp_trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, mlp_trace_id, cq_id=0, blocking=True)
        mlp_samples = []
        for _ in range(12):
            start = time.perf_counter_ns()
            ttnn.execute_trace(mesh_device, mlp_trace_id, cq_id=0, blocking=True)
            mlp_samples.append((time.perf_counter_ns() - start) / 1_000_000)
    finally:
        ttnn.release_trace(mesh_device, mlp_trace_id)
    mlp_output.deallocate(True)

    record = {
        "num_cores": num_cores,
        "topology": decoder.layer.shared_mlp.policy.mlp_gate_up_topology,
        "packed_output_dtype": str(decoder.layer.shared_mlp.policy.mlp_packed_output_dtype),
        "gate_up_in0_block_w": decoder.layer.shared_mlp.policy.gate_up_in0_block_w,
        "down_in0_block_w": decoder.layer.shared_mlp.policy.down_in0_block_w,
        "layer_median_ms": statistics.median(layer_samples),
        "layer_samples_ms": layer_samples,
        "mlp_median_ms": statistics.median(mlp_samples),
        "mlp_samples_ms": mlp_samples,
    }
    print("MULTICHIP_MLP_GEOMETRY " + json.dumps(record))


@pytest.mark.skipif(
    "GEMMA4_MULTICHIP_QKV_SWEEP" not in os.environ,
    reason="set GEMMA4_MULTICHIP_QKV_SWEEP=1",
)
@pytest.mark.parametrize("output_cores,in0_block_w", [(8, 3), (16, 3), (32, 1), (32, 3), (32, 7), (32, 21)])
def test_multichip_qkv_decode_geometry_sweep(
    hf_config,
    mesh_device,
    optimized_baseline,
    output_cores,
    in0_block_w,
):
    """Measure legal packed TP-local QKV geometries in the full trace."""
    layer_idx, layer_kind = LAYER_KINDS[0]
    decoder = _decoder(hf_config, mesh_device, layer_idx)
    decoder.qkv_decode_output_cores = output_cores
    decoder.policy = replace(decoder.policy, qkv_in0_block_w=in0_block_w)
    values = _inputs(hf_config, layer_idx)
    cache, page_table = decoder.init_paged_kv_cache(max_context=128)
    decoder.prefill_forward(
        _tt_input(values["prompt32"], mesh_device),
        rope_mats=_rope_device(hf_config, layer_idx, 32, mesh_device, decode=False),
        page_table=page_table,
        kv_cache=cache,
        valid_seq_len=32,
    )
    token = _tt_input(values["token"], mesh_device)
    pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
    pos_u_host[0, 0] = 32
    kwargs = dict(
        rope_mats=_rope_device(hf_config, layer_idx, 128, mesh_device, decode=True),
        page_table=page_table,
        kv_cache=cache,
        current_position=_replicated_int(pos_u_host, mesh_device, dtype=ttnn.uint32),
        current_position_cache=_replicated_int(torch.tensor([32], dtype=torch.int32), mesh_device, dtype=ttnn.int32),
    )

    def call():
        return decoder.decode_forward(token, **kwargs)

    warm = call()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        samples = []
        for _ in range(12):
            start = time.perf_counter_ns()
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
        reference = optimized_baseline[(layer_kind, "decode32")]
        actuals = [ttnn.to_torch(t).reshape_as(reference).float() for t in ttnn.get_device_tensors(output)]
        for actual in actuals:
            _assert_pcc(reference, actual, PCC_THRESHOLD)
        print(
            "MULTICHIP_QKV_GEOMETRY "
            + json.dumps(
                {
                    "output_cores": output_cores,
                    "in0_block_w": in0_block_w,
                    "median_ms": statistics.median(samples),
                    "samples_ms": samples,
                }
            )
        )
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.skipif(
    "GEMMA4_MULTICHIP_PRECISION_SWEEP" not in os.environ,
    reason="set GEMMA4_MULTICHIP_PRECISION_SWEEP=1",
)
@pytest.mark.parametrize(
    "candidate",
    [
        "baseline",
        "attention_bfp4",
        "attention_hifi2",
        "mlp_gate_hifi2",
        "mlp_down_hifi2",
        "kv_bfp4",
        "qkv_split",
        "mlp_packed_bf16",
        "mlp_packed_bfp8",
        "o_block16_full",
        "sdpa_k32",
        "sdpa_k128",
        "sdpa_exp_approx",
        "sdpa_full_grid_full",
    ],
)
def test_multichip_decode_precision_fidelity_sweep(
    hf_config,
    mesh_device,
    optimized_baseline,
    candidate,
):
    """Measure full-trace precision/fidelity candidates without hiding PCC failures."""
    policy = DEFAULT_OPTIMIZATION_POLICY
    if candidate == "attention_bfp4":
        policy = replace(policy, attention_weight_dtype=ttnn.bfloat4_b)
    elif candidate == "attention_hifi2":
        policy = replace(policy, attention_math_fidelity=ttnn.MathFidelity.HiFi2)
    elif candidate == "mlp_gate_hifi2":
        policy = replace(policy, mlp_gate_up_math_fidelity=ttnn.MathFidelity.HiFi2)
    elif candidate == "mlp_down_hifi2":
        policy = replace(policy, mlp_down_math_fidelity=ttnn.MathFidelity.HiFi2)
    elif candidate == "kv_bfp4":
        policy = replace(policy, kv_cache_dtype=ttnn.bfloat4_b)
    elif candidate == "qkv_split":
        policy = replace(policy, attention_projection_topology="split")
    elif candidate == "mlp_packed_bf16":
        policy = replace(policy, mlp_gate_up_topology="packed", mlp_packed_output_dtype=ttnn.bfloat16)
    elif candidate == "mlp_packed_bfp8":
        policy = replace(policy, mlp_gate_up_topology="packed", mlp_packed_output_dtype=ttnn.bfloat8_b)
    elif candidate == "o_block16_full":
        policy = replace(policy, o_proj_in0_block_w=16)

    layer_idx, layer_kind = LAYER_KINDS[1] if candidate in ("o_block16_full", "sdpa_full_grid_full") else LAYER_KINDS[0]
    decoder = _decoder(hf_config, mesh_device, layer_idx, optimization_policy=policy)
    if candidate == "sdpa_k32":
        decoder.sdpa_k_chunk_size = 32
    elif candidate == "sdpa_k128":
        decoder.sdpa_k_chunk_size = 128
    elif candidate == "sdpa_exp_approx":
        decoder.sdpa_exp_approx_mode = True
    elif candidate == "sdpa_full_grid_full":
        decoder.sdpa_force_full_grid = True
    values = _inputs(hf_config, layer_idx)
    cache, page_table = decoder.init_paged_kv_cache(max_context=128)
    decoder.prefill_forward(
        _tt_input(values["prompt32"], mesh_device),
        rope_mats=_rope_device(hf_config, layer_idx, 32, mesh_device, decode=False),
        page_table=page_table,
        kv_cache=cache,
        valid_seq_len=32,
    )
    token = _tt_input(values["token"], mesh_device)
    pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
    pos_u_host[0, 0] = 32
    kwargs = dict(
        rope_mats=_rope_device(hf_config, layer_idx, 128, mesh_device, decode=True),
        page_table=page_table,
        kv_cache=cache,
        current_position=_replicated_int(pos_u_host, mesh_device, dtype=ttnn.uint32),
        current_position_cache=_replicated_int(torch.tensor([32], dtype=torch.int32), mesh_device, dtype=ttnn.int32),
    )

    def call():
        return decoder.decode_forward(token, **kwargs)

    warm = call()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        samples = []
        for _ in range(12):
            start = time.perf_counter_ns()
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
        reference = optimized_baseline[(layer_kind, "decode32")]
        pccs = []
        for tensor in ttnn.get_device_tensors(output):
            _, pcc = comp_pcc(reference.float(), ttnn.to_torch(tensor).reshape_as(reference).float(), PCC_THRESHOLD)
            pccs.append(float(pcc))
        print(
            "MULTICHIP_PRECISION "
            + json.dumps(
                {
                    "candidate": candidate,
                    "pcc": min(pccs),
                    "median_ms": statistics.median(samples),
                    "samples_ms": samples,
                }
            )
        )
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.skipif(
    "GEMMA4_MULTICHIP_FUSED_MMRS" not in os.environ,
    reason="set GEMMA4_MULTICHIP_FUSED_MMRS=1",
)
def test_multichip_fused_mmrs_model_shape(fused_mmrs_ring_mesh_device):
    """Exercise fused MM+RS at Gemma's actual TP4 decode row-projection shape."""
    from tests.ttnn.unit_tests.operations.ccl.test_new_matmul_reduce_scatter import run_reduce_scatter_impl

    # The helper shards global K=5376 over TP4, yielding the model's local K=1344.
    mesh_device = fused_mmrs_ring_mesh_device
    interleaved_dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    run_reduce_scatter_impl(
        mesh_device=mesh_device,
        num_devices=TP_SIZE,
        rs_input_shape=[1, 1, 32, 5376],
        mm_shard_dim=2,
        rs_scatter_dim=3,
        num_links=int(os.environ.get("GEMMA4_MC_NUM_LINKS", "2")),
        mm_weights_shape=[1, 1, 5376, 5376],
        rs_input_dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        matmul_weights_dtype=ttnn.bfloat8_b,
        max_in0_block_w=3,
        use_bias=False,
        mem_config_input=interleaved_dram,
        mem_config_rs=interleaved_dram,
        mem_config_mm=interleaved_dram,
        rs_topology=ttnn.Topology.Ring,
        use_non_fused=False,
        num_iters=1,
        enable_trace=False,
        out_block_w_override=7,
    )


@pytest.mark.skipif(
    "GEMMA4_MULTICHIP_FUSED_AGMM" not in os.environ,
    reason="set GEMMA4_MULTICHIP_FUSED_AGMM=1",
)
@pytest.mark.parametrize(
    "role,gathered_k,local_output_n,weight_dtype,max_in0_block_w",
    [
        ("sliding_attention_o", 8192, 1344, ttnn.bfloat8_b, 8),
        ("full_attention_o", 16384, 1344, ttnn.bfloat8_b, 8),
        ("mlp_down", 21504, 1344, ttnn.bfloat4_b, 7),
        ("mlp_down_bfp8_adapted", 21504, 1344, ttnn.bfloat8_b, 7),
    ],
)
def test_multichip_fused_agmm_model_shape(
    fused_mmrs_ring_mesh_device,
    role,
    gathered_k,
    local_output_n,
    weight_dtype,
    max_in0_block_w,
):
    """Exercise fused AG+MM at both exact Gemma TP4 row-projection shapes.

    Algebraically this is the alternate output-projection decomposition:
    gather the TP-local K input, multiply by a weight sharded on output N,
    and retain a TP-local hidden-width result.  The focused coherent-family
    probe below measures how that local result composes with residual/norm.
    """
    from tests.nightly.t3000.ccl.test_minimal_all_gather_matmul_async import run_all_gather_impl

    mesh_device = fused_mmrs_ring_mesh_device
    interleaved_dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    print(
        "GEMMA4_FUSED_AGMM_SHAPE "
        + json.dumps(
            {
                "role": role,
                "local_input_k": gathered_k // TP_SIZE,
                "gathered_k": gathered_k,
                "local_output_n": local_output_n,
                "global_output_n": local_output_n * TP_SIZE,
                "weight_dtype": str(weight_dtype),
                "persistent_output": True,
            }
        )
    )
    run_all_gather_impl(
        mesh_device=mesh_device,
        num_devices=TP_SIZE,
        ag_output_shape=[1, 1, 32, gathered_k],
        dim=3,
        num_links=1,
        ag_input_dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        matmul_output_dim=local_output_n,
        matmul_weights_dtype=weight_dtype,
        max_in0_block_w=max_in0_block_w,
        use_bias=False,
        mem_config_input=interleaved_dram,
        mem_config_ag=interleaved_dram,
        mem_config_mm=interleaved_dram,
        all_gather_topology=ttnn.Topology.Ring,
        use_non_fused=False,
        use_legacy_allgather=False,
        num_iters=1,
        enable_trace=True,
        use_barrier=True,
        use_persistent_buffers=True,
        out_subblock_w_override=1,
    )


@pytest.mark.skipif(
    "GEMMA4_MULTICHIP_FUSED_AGMM_COHERENT" not in os.environ,
    reason="set GEMMA4_MULTICHIP_FUSED_AGMM_COHERENT=1",
)
@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
def test_multichip_fused_agmm_coherent_boundary(
    hf_config,
    fused_mmrs_ring_mesh_device,
    layer_idx,
    layer_kind,
):
    """Measure the real-weight fused-AGMM family without restoring replication.

    The candidate keeps H/TP local after both O and down, uses distributed
    normalization/residual operations, and lets the next packed projection
    consume the fractured hidden state through another fused AG+matmul.  The
    immediate trailing-all-gather variants are timed independently as
    compatibility candidates; they are not part of the coherent-family time.
    """
    from tracy import signpost

    mesh_device = fused_mmrs_ring_mesh_device
    rows = ttnn.TILE_SIZE
    hidden = int(hf_config.hidden_size)
    intermediate = int(hf_config.intermediate_size)
    local_hidden = hidden // TP_SIZE
    local_intermediate = intermediate // TP_SIZE
    state = _layer_state(layer_idx)
    prefix = f"model.language_model.layers.{layer_idx}."
    attention_k = int(state[prefix + "self_attn.q_proj.weight"].shape[0])
    local_attention_k = attention_k // TP_SIZE

    decoder = MultichipDecoder.from_state_dict(
        state,
        hf_config=hf_config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        communication_dtype=ttnn.bfloat8_b,
        use_persistent_async_ccl=True,
        topology=ttnn.Topology.Ring,
    )
    decoder.layer.shared_mlp.is_decode = True

    interleaved_dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    def mesh_tensor(host, *, dtype, mapper):
        return ttnn.from_torch(
            host.detach().contiguous(),
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=interleaved_dram,
            mesh_mapper=mapper,
        )

    def sharded_weight(host, *, dtype, dim):
        return mesh_tensor(host, dtype=dtype, mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim))

    # O/down are repacked from the production local-K/full-N decomposition to
    # the alternate full-K/local-N decomposition required by OPT-008.
    o_source = state[prefix + "self_attn.o_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
    down_source = state[prefix + "mlp.down_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
    o_local_output_weight = sharded_weight(o_source, dtype=decoder.policy.attention_weight_dtype, dim=3)
    down_local_output_weight = sharded_weight(
        down_source,
        dtype=decoder.policy.mlp_down_weight_dtype,
        dim=3,
    )

    # Pack gate/up in device-major order so each device's local N shard is
    # [gate_i, up_i], matching the selected Stage-05 packed projection.
    gate_source = state[prefix + "mlp.gate_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
    up_source = state[prefix + "mlp.up_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
    gate_chunks = torch.chunk(gate_source, TP_SIZE, dim=3)
    up_chunks = torch.chunk(up_source, TP_SIZE, dim=3)
    packed_gate_up_source = torch.cat(
        [torch.cat((gate_chunks[index], up_chunks[index]), dim=3) for index in range(TP_SIZE)],
        dim=3,
    )
    packed_gate_up_weight = sharded_weight(
        packed_gate_up_source,
        dtype=decoder.policy.mlp_gate_up_weight_dtype,
        dim=3,
    )

    # The next-layer contract is represented by the same real layer kind's
    # packed QKV projection.  Its full-K/local-N weight layout is already the
    # required consumer layout, but an interleaved placement is used because
    # fused AGMM accepts multicast rather than the production DRAM-sharded
    # matmul program.
    q_source = state[prefix + "self_attn.q_proj.weight"]
    k_source = state[prefix + "self_attn.k_proj.weight"]
    v_source = state.get(prefix + "self_attn.v_proj.weight", k_source)
    q_chunks = torch.chunk(q_source, TP_SIZE, dim=0)
    k_chunks = torch.chunk(k_source, TP_SIZE, dim=0)
    v_chunks = torch.chunk(v_source, TP_SIZE, dim=0)
    qkv_per_device = [
        torch.cat(
            (
                q_chunks[index].transpose(-2, -1),
                k_chunks[index].transpose(-2, -1),
                v_chunks[index].transpose(-2, -1),
            ),
            dim=1,
        )
        for index in range(TP_SIZE)
    ]
    packed_qkv_source = torch.cat(qkv_per_device, dim=1).unsqueeze(0).unsqueeze(0)
    packed_qkv_weight = sharded_weight(
        packed_qkv_source,
        dtype=decoder.policy.attention_weight_dtype,
        dim=3,
    )
    local_qkv_width = qkv_per_device[0].shape[-1]

    def local_gamma(name):
        host = state[prefix + name + ".weight"].reshape(1, 1, -1, ttnn.TILE_SIZE)
        return ttnn.from_torch(
            host,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=interleaved_dram,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
        )

    post_attention_gamma = local_gamma("post_attention_layernorm")
    pre_feedforward_gamma = local_gamma("pre_feedforward_layernorm")
    post_feedforward_gamma = local_gamma("post_feedforward_layernorm")
    next_input_gamma = local_gamma("input_layernorm")

    torch.manual_seed(20262800 + layer_idx)
    attention_host = torch.randn(1, 1, rows, attention_k, dtype=torch.bfloat16) * 0.02
    residual_host = torch.randn(1, 1, rows, hidden, dtype=torch.bfloat16) * 0.02
    activated_host = torch.randn(1, 1, rows, intermediate, dtype=torch.bfloat16) * 0.02
    attention_local = mesh_tensor(
        attention_host,
        dtype=ttnn.bfloat16,
        mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    residual_replicated = mesh_tensor(
        residual_host,
        dtype=ttnn.bfloat16,
        mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    residual_fractured = mesh_tensor(
        residual_host,
        dtype=ttnn.bfloat16,
        mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    activated_local = mesh_tensor(
        activated_host,
        dtype=ttnn.bfloat8_b,
        mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )

    # Follow the proven Qwen TP4 setup: one full-worker subdevice, two AG
    # semaphores plus an independent barrier per fused role, and a stable DRAM
    # buffer holding the gathered matmul input.  DRAM avoids making the large
    # full-attention/down gathered tensors an L1-capacity artifact.
    compute_grid = mesh_device.compute_with_storage_grid_size()
    worker_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )
    worker_subdevice = ttnn.SubDevice([worker_cores])
    worker_subdevice_id = ttnn.SubDeviceId(0)
    subdevice_manager = mesh_device.create_sub_device_manager([worker_subdevice], 0)
    mesh_device.load_sub_device_manager(subdevice_manager)
    mesh_device.set_sub_device_stall_group([worker_subdevice_id])

    role_shapes = {
        "attention_o": (attention_k, ttnn.bfloat16),
        "packed_gate_up": (hidden, ttnn.bfloat16),
        "mlp_down": (intermediate, ttnn.bfloat8_b),
        "next_qkv": (hidden, ttnn.bfloat16),
    }
    ag_semaphores = {
        role: [ttnn.create_global_semaphore(mesh_device, worker_cores, 0) for _ in range(2)] for role in role_shapes
    }
    barrier_semaphores = {role: ttnn.create_global_semaphore(mesh_device, worker_cores, 0) for role in role_shapes}
    persistent_ag_buffers = {
        role: mesh_tensor(
            torch.zeros((1, 1, rows, width), dtype=torch.bfloat16),
            dtype=dtype,
            mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        for role, (width, dtype) in role_shapes.items()
    }

    def agmm_program(k, n, in0_block_w):
        # This is the adapted Qwen TP4 multicast geometry.  It owns N padding
        # through ceil(per_core_N), and subblock 1 is legal for every exact
        # Gemma local-output width (including H/TP=1,344).
        grid = (8, 6)
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=max(1, math.ceil((rows // ttnn.TILE_SIZE) / grid[1])),
            per_core_N=max(1, math.ceil((n // ttnn.TILE_SIZE) / grid[0])),
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

    def fused_agmm(role, value, weight, *, k, n, in0_block_w, output_dtype, compute_kernel):
        _, output = ttnn.experimental.all_gather_matmul_async(
            value,
            weight,
            persistent_output_buffer=persistent_ag_buffers[role],
            dim=3,
            multi_device_global_semaphore=ag_semaphores[role],
            all_gather_core_grid_offset=(0, 6),
            num_links=1,
            memory_config_ag=interleaved_dram,
            topology=ttnn.Topology.Ring,
            barrier_semaphore=barrier_semaphores[role],
            subdevice_id=worker_subdevice_id,
            memory_config_mm=interleaved_dram,
            dtype=output_dtype,
            program_config=agmm_program(k, n, in0_block_w),
            compute_kernel_config=compute_kernel,
            num_workers_per_link=1,
        )
        return output

    def distributed_norm(fractured, gamma):
        stats = ttnn.rms_norm_pre_all_gather(fractured, dtype=ttnn.bfloat16)
        gathered_stats = ttnn.all_gather(
            stats,
            dim=3,
            cluster_axis=1,
            num_links=1,
            topology=ttnn.Topology.Ring,
            memory_config=interleaved_dram,
        )
        stats.deallocate(True)
        output = ttnn.rms_norm_post_all_gather(
            fractured,
            gathered_stats,
            epsilon=hf_config.rms_norm_eps,
            weight=gamma,
            dtype=ttnn.bfloat16,
        )
        gathered_stats.deallocate(True)
        return output

    def baseline_o(value):
        input_mem = decoder._decode_memory_config(mesh_device, decoder.policy.decode_num_cores, local_attention_k)
        output_mem = decoder._decode_memory_config(mesh_device, decoder.policy.decode_num_cores, hidden)
        local = ttnn.to_memory_config(value, input_mem)
        partial = ttnn.linear(
            local,
            decoder.decode_o_proj,
            memory_config=output_mem,
            program_config=decoder._decode_matmul_program_config(
                k=local_attention_k,
                n=hidden,
                num_cores=decoder.policy.decode_num_cores,
                in0_block_w=decoder.policy.o_proj_in0_block_w,
            ),
            compute_kernel_config=decoder.attention_compute,
        )
        local.deallocate(True)
        return _tp_allreduce(
            partial,
            decoder.mesh_config,
            decoder.ccl_manager,
            communication_dtype=ttnn.bfloat8_b,
            use_persistent_async=True,
            persistent_role="attention_o",
        )

    def baseline_qkv(value):
        input_mem = decoder._decode_memory_config(mesh_device, decoder.policy.decode_num_cores, hidden)
        output_cores = decoder.qkv_decode_output_cores
        grid = ttnn.num_cores_to_corerangeset(output_cores, compute_grid, row_wise=True)
        output_mem = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, local_qkv_width // output_cores),
            core_grid=grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        local = ttnn.to_memory_config(value, input_mem)
        projected = ttnn.linear(
            local,
            decoder.decode_wqkv,
            memory_config=output_mem,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=decoder.policy.qkv_in0_block_w,
                per_core_M=1,
                per_core_N=local_qkv_width // (ttnn.TILE_SIZE * output_cores),
            ),
            compute_kernel_config=decoder.attention_compute,
        )
        local.deallocate(True)
        interleaved = ttnn.sharded_to_interleaved(projected, interleaved_dram)
        projected.deallocate(True)
        return interleaved

    def baseline_down(value):
        mlp = decoder.layer.shared_mlp
        local_mem = mlp._decode_memory_config(mlp.policy.decode_num_cores, local_intermediate)
        output_mem = mlp._decode_memory_config(mlp.policy.decode_num_cores, hidden)
        local = ttnn.to_memory_config(value, local_mem)
        partial = ttnn.linear(
            local,
            mlp.down_decode,
            memory_config=output_mem,
            program_config=mlp._decode_program_config(
                k=local_intermediate,
                n=hidden,
                num_cores=mlp.policy.decode_num_cores,
                in0_block_w=mlp.policy.down_in0_block_w,
            ),
            compute_kernel_config=mlp.down_compute,
        )
        local.deallocate(True)
        return mlp._reduce(partial)

    def baseline_call():
        projected = baseline_o(attention_local)
        post = decoder.layer.post_attention_layernorm.forward(projected)
        projected.deallocate(True)
        hidden_states = ttnn.add(residual_replicated, post)
        post.deallocate(True)
        normed = decoder.layer.pre_feedforward_layernorm.forward(hidden_states)
        mlp_output = decoder.layer.shared_mlp(normed)
        normed.deallocate(True)
        post_mlp = decoder.layer.post_feedforward_layernorm.forward(mlp_output)
        mlp_output.deallocate(True)
        combined = ttnn.add(hidden_states, post_mlp)
        hidden_states.deallocate(True)
        post_mlp.deallocate(True)
        if decoder.layer.layer_scalar != 1.0:
            scaled = ttnn.mul(combined, decoder.layer.layer_scalar)
            combined.deallocate(True)
            combined = scaled
        next_norm = decoder.layer.input_layernorm.forward(combined)
        combined.deallocate(True)
        output = baseline_qkv(next_norm)
        next_norm.deallocate(True)
        return output

    def coherent_call():
        projected = fused_agmm(
            "attention_o",
            attention_local,
            o_local_output_weight,
            k=attention_k,
            n=local_hidden,
            in0_block_w=8,
            output_dtype=ttnn.bfloat16,
            compute_kernel=decoder.attention_compute,
        )
        post = distributed_norm(projected, post_attention_gamma)
        projected.deallocate(True)
        hidden_states = ttnn.add(residual_fractured, post)
        post.deallocate(True)
        normed = distributed_norm(hidden_states, pre_feedforward_gamma)
        packed = fused_agmm(
            "packed_gate_up",
            normed,
            packed_gate_up_weight,
            k=hidden,
            n=2 * local_intermediate,
            in0_block_w=7,
            output_dtype=ttnn.bfloat8_b,
            compute_kernel=decoder.layer.shared_mlp.gate_up_compute,
        )
        normed.deallocate(True)
        gate = ttnn.slice(packed, [0, 0, 0, 0], [1, 1, rows, local_intermediate])
        up = ttnn.slice(packed, [0, 0, 0, local_intermediate], [1, 1, rows, 2 * local_intermediate])
        packed.deallocate(True)
        gate = ttnn.gelu(gate, fast_and_approximate_mode=True)
        activated = ttnn.mul(gate, up, memory_config=interleaved_dram)
        gate.deallocate(True)
        up.deallocate(True)
        down = fused_agmm(
            "mlp_down",
            activated,
            down_local_output_weight,
            k=intermediate,
            n=local_hidden,
            in0_block_w=7,
            output_dtype=ttnn.bfloat8_b,
            compute_kernel=decoder.layer.shared_mlp.down_compute,
        )
        activated.deallocate(True)
        down_bf16 = ttnn.typecast(down, ttnn.bfloat16)
        down.deallocate(True)
        post_mlp = distributed_norm(down_bf16, post_feedforward_gamma)
        down_bf16.deallocate(True)
        combined = ttnn.add(hidden_states, post_mlp)
        hidden_states.deallocate(True)
        post_mlp.deallocate(True)
        if decoder.layer.layer_scalar != 1.0:
            scaled = ttnn.mul(combined, decoder.layer.layer_scalar)
            combined.deallocate(True)
            combined = scaled
        next_norm = distributed_norm(combined, next_input_gamma)
        combined.deallocate(True)
        output = fused_agmm(
            "next_qkv",
            next_norm,
            packed_qkv_weight,
            k=hidden,
            n=local_qkv_width,
            in0_block_w=7,
            output_dtype=ttnn.bfloat16,
            compute_kernel=decoder.attention_compute,
        )
        next_norm.deallocate(True)
        return output

    def trailing_gather(value):
        communicated = ttnn.typecast(value, ttnn.bfloat8_b) if value.dtype != ttnn.bfloat8_b else value
        gathered = ttnn.all_gather(
            communicated,
            dim=3,
            cluster_axis=1,
            num_links=1,
            topology=ttnn.Topology.Ring,
            memory_config=interleaved_dram,
        )
        if communicated is not value:
            communicated.deallocate(True)
        restored = ttnn.typecast(gathered, ttnn.bfloat16)
        gathered.deallocate(True)
        return restored

    def trailing_o_call():
        local = fused_agmm(
            "attention_o",
            attention_local,
            o_local_output_weight,
            k=attention_k,
            n=local_hidden,
            in0_block_w=8,
            output_dtype=ttnn.bfloat16,
            compute_kernel=decoder.attention_compute,
        )
        output = trailing_gather(local)
        local.deallocate(True)
        return output

    def trailing_down_call():
        local = fused_agmm(
            "mlp_down",
            activated_local,
            down_local_output_weight,
            k=intermediate,
            n=local_hidden,
            in0_block_w=7,
            output_dtype=ttnn.bfloat8_b,
            compute_kernel=decoder.layer.shared_mlp.down_compute,
        )
        output = trailing_gather(local)
        local.deallocate(True)
        return output

    def min_device_pcc(reference, candidate):
        pccs = []
        for expected, actual in zip(ttnn.get_device_tensors(reference), ttnn.get_device_tensors(candidate)):
            expected_host = ttnn.to_torch(expected).float()
            actual_host = ttnn.to_torch(actual).reshape_as(expected_host).float()
            _, pcc = comp_pcc(expected_host, actual_host, PCC_THRESHOLD)
            pccs.append(float(pcc))
        assert min(pccs) >= PCC_THRESHOLD, f"minimum device PCC {min(pccs)} is below {PCC_THRESHOLD}"
        return min(pccs)

    def trace_measure(call, label):
        warm = call()
        ttnn.synchronize_device(mesh_device, sub_device_ids=[worker_subdevice_id])
        warm.deallocate(True)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        output = call()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        try:
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            samples = []
            signpost(label)
            repetitions = 1 if "GEMMA4_MULTICHIP_FUSED_AGMM_PROFILE" in os.environ else 12
            for _ in range(repetitions):
                start = time.perf_counter_ns()
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
                samples.append((time.perf_counter_ns() - start) / 1_000_000)
            signpost(label + "_END")
            return statistics.median(samples), samples
        finally:
            ttnn.release_trace(mesh_device, trace_id)
            output.deallocate(True)

    try:
        baseline = baseline_call()
        coherent = coherent_call()
        baseline_o_output = baseline_o(attention_local)
        trailing_o_output = trailing_o_call()
        baseline_down_output = baseline_down(activated_local)
        trailing_down_output = trailing_down_call()
        ttnn.synchronize_device(mesh_device, sub_device_ids=[worker_subdevice_id])
        coherent_pcc = min_device_pcc(baseline, coherent)
        trailing_o_pcc = min_device_pcc(baseline_o_output, trailing_o_output)
        trailing_down_pcc = min_device_pcc(baseline_down_output, trailing_down_output)
        for tensor in (
            baseline,
            coherent,
            baseline_o_output,
            trailing_o_output,
            baseline_down_output,
            trailing_down_output,
        ):
            tensor.deallocate(True)

        label_prefix = f"AGMM_{layer_kind}"
        baseline_ms, baseline_samples = trace_measure(baseline_call, label_prefix + "_BASELINE_SPINE")
        coherent_ms, coherent_samples = trace_measure(coherent_call, label_prefix + "_COHERENT_SPINE")
        baseline_o_ms, baseline_o_samples = trace_measure(
            lambda: baseline_o(attention_local), label_prefix + "_BASELINE_O"
        )
        trailing_o_ms, trailing_o_samples = trace_measure(trailing_o_call, label_prefix + "_TRAILING_O")
        baseline_down_ms, baseline_down_samples = trace_measure(
            lambda: baseline_down(activated_local), label_prefix + "_BASELINE_DOWN"
        )
        trailing_down_ms, trailing_down_samples = trace_measure(trailing_down_call, label_prefix + "_TRAILING_DOWN")
        print(
            "GEMMA4_FUSED_AGMM_COHERENT "
            + json.dumps(
                {
                    "layer_idx": layer_idx,
                    "layer_kind": layer_kind,
                    "rows": rows,
                    "pcc_threshold": PCC_THRESHOLD,
                    "coherent_pcc": coherent_pcc,
                    "trailing_o_pcc": trailing_o_pcc,
                    "trailing_down_pcc": trailing_down_pcc,
                    "baseline_ms": baseline_ms,
                    "coherent_ms": coherent_ms,
                    "coherent_over_baseline": coherent_ms / baseline_ms,
                    "baseline_o_ms": baseline_o_ms,
                    "trailing_o_ms": trailing_o_ms,
                    "trailing_o_over_baseline": trailing_o_ms / baseline_o_ms,
                    "baseline_down_ms": baseline_down_ms,
                    "trailing_down_ms": trailing_down_ms,
                    "trailing_down_over_baseline": trailing_down_ms / baseline_down_ms,
                    "baseline_samples_ms": baseline_samples,
                    "coherent_samples_ms": coherent_samples,
                    "baseline_o_samples_ms": baseline_o_samples,
                    "trailing_o_samples_ms": trailing_o_samples,
                    "baseline_down_samples_ms": baseline_down_samples,
                    "trailing_down_samples_ms": trailing_down_samples,
                    "o_weight": [attention_k, local_hidden],
                    "down_weight": [intermediate, local_hidden],
                    "packed_gate_up_weight": [hidden, 2 * local_intermediate],
                    "next_qkv_weight": [hidden, local_qkv_width],
                    "residual_contract": "H/TP local through distributed norms/residual and next fused QKV",
                    "trailing_gather_contract": "separate BFP8 immediate replicated-compatibility candidate",
                    "agmm": {
                        "topology": "Ring",
                        "num_links": 1,
                        "dim": 3,
                        "hardcoded_kernel_transfer_count": 4,
                        "num_workers_per_link": 1,
                        "persistent_gathered_input": True,
                        "memory_config_ag": "DRAM_INTERLEAVED",
                        "memory_config_mm": "DRAM_INTERLEAVED",
                        "program_grid": [8, 6],
                        "out_subblock": [1, 1],
                    },
                }
            )
        )
    finally:
        ttnn.synchronize_device(mesh_device, sub_device_ids=[worker_subdevice_id])
        for buffer in persistent_ag_buffers.values():
            buffer.deallocate(True)
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()


@pytest.mark.skipif(
    "GEMMA4_MULTICHIP_MLP_PREFILL_SWEEP" not in os.environ,
    reason="set GEMMA4_MULTICHIP_MLP_PREFILL_SWEEP=1",
)
def test_multichip_mlp_prefill_placement_sweep(hf_config, profile_mesh_device):
    """Compare the interleaved prefill path with legal DRAM-sharded grids."""
    mesh_device = profile_mesh_device
    layer_idx, _ = LAYER_KINDS[0]
    decoder = _decoder(hf_config, mesh_device, layer_idx)
    mlp = decoder.layer.shared_mlp
    hidden = hf_config.hidden_size
    rows = 128
    torch.manual_seed(20262351)
    hidden_input = _tt_input(torch.randn(1, rows, hidden, dtype=torch.bfloat16) * 0.02, mesh_device)

    def interleaved_call():
        mlp.is_decode = False
        return mlp(hidden_input)

    def explicit_1d_call(num_cores):
        policy = _mlp_geometry_policy(num_cores, base_policy=mlp.policy)
        grids = {8: (8, 1), 12: (6, 2), 21: (7, 3), 24: (8, 3), 28: (7, 4), 42: (7, 6), 56: (8, 7)}
        grid = grids[num_cores]
        per_core_n = hidden // (ttnn.TILE_SIZE * num_cores)
        out_subblock_w = max(value for value in range(1, 9) if per_core_n % value == 0)

        def program(in0_block_w, fused_activation=None):
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=grid,
                in0_block_w=in0_block_w,
                per_core_M=rows // ttnn.TILE_SIZE,
                per_core_N=per_core_n,
                out_subblock_h=1,
                out_subblock_w=out_subblock_w,
                fuse_batch=True,
                fused_activation=fused_activation,
                mcast_in0=True,
            )

        up = ttnn.linear(
            hidden_input,
            mlp.up_prefill,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=program(policy.gate_up_in0_block_w),
            compute_kernel_config=mlp.gate_up_compute,
        )
        gate = ttnn.linear(
            hidden_input,
            mlp.gate_prefill,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=program(
                policy.gate_up_in0_block_w,
                ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0),
            ),
            compute_kernel_config=mlp.gate_up_compute,
        )
        activated = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate.deallocate(True)
        up.deallocate(True)
        partial = ttnn.linear(
            activated,
            mlp.down_prefill,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=program(policy.down_in0_block_w),
            compute_kernel_config=mlp.down_compute,
        )
        activated.deallocate(True)
        return mlp._reduce(partial)

    def l1_block_sharded_call():
        grid = (8, 4)
        input_memory_config = ttnn.create_sharded_memory_config(
            shape=(rows // grid[1], hidden // grid[0]),
            core_grid=ttnn.CoreGrid(x=grid[0], y=grid[1]),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        program_args = dict(
            compute_with_storage_grid_size=grid,
            in0_block_w=7,
            out_subblock_h=1,
            out_subblock_w=7,
            per_core_M=1,
            per_core_N=hidden // (ttnn.TILE_SIZE * grid[0]),
            transpose_mcast=False,
        )
        program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            **program_args,
            fused_activation=None,
        )
        gate_program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            **program_args,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0),
        )
        block_input = ttnn.to_memory_config(hidden_input, input_memory_config)
        up = ttnn.linear(
            block_input,
            mlp.up_prefill,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=program,
            compute_kernel_config=mlp.gate_up_compute,
        )
        gate = ttnn.linear(
            block_input,
            mlp.gate_prefill,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=gate_program,
            compute_kernel_config=mlp.gate_up_compute,
        )
        block_input.deallocate(True)
        activated = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate.deallocate(True)
        up.deallocate(True)
        block_activated = ttnn.to_memory_config(activated, input_memory_config)
        activated.deallocate(True)
        partial = ttnn.linear(
            block_activated,
            mlp.down_prefill,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=program,
            compute_kernel_config=mlp.down_compute,
        )
        block_activated.deallocate(True)
        return mlp._reduce(partial)

    def measure(call):
        warm = call()
        ttnn.synchronize_device(mesh_device)
        warm.deallocate(True)
        samples = []
        for _ in range(12):
            start = time.perf_counter_ns()
            output = call()
            ttnn.synchronize_device(mesh_device)
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
            output.deallocate(True)
        return samples

    baseline = interleaved_call()
    baseline_devices = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(baseline)]
    baseline.deallocate(True)
    baseline_samples = measure(interleaved_call)
    candidates = []
    for num_cores in (8, 12, 21, 24, 28, 42, 56):
        candidate = explicit_1d_call(num_cores)
        candidate_devices = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(candidate)]
        for expected, actual in zip(baseline_devices, candidate_devices):
            _assert_pcc(expected, actual, PCC_THRESHOLD)
        candidate.deallocate(True)
        samples = measure(lambda num_cores=num_cores: explicit_1d_call(num_cores))
        policy = _mlp_geometry_policy(num_cores, base_policy=mlp.policy)
        per_core_n = hidden // (ttnn.TILE_SIZE * num_cores)
        candidates.append(
            {
                "placement": "dram_interleaved_explicit_1d",
                "num_cores": num_cores,
                "gate_up_in0_block_w": policy.gate_up_in0_block_w,
                "down_in0_block_w": policy.down_in0_block_w,
                "per_core_M": rows // ttnn.TILE_SIZE,
                "per_core_N": per_core_n,
                "out_subblock_h": 1,
                "out_subblock_w": max(value for value in range(1, 9) if per_core_n % value == 0),
                "median_ms": statistics.median(samples),
                "samples_ms": samples,
            }
        )
    l1_candidate = l1_block_sharded_call()
    l1_candidate_devices = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(l1_candidate)]
    for expected, actual in zip(baseline_devices, l1_candidate_devices):
        _assert_pcc(expected, actual, PCC_THRESHOLD)
    l1_candidate.deallocate(True)
    l1_samples = measure(l1_block_sharded_call)
    candidates.append(
        {
            "placement": "l1_block_sharded_input_explicit_2d",
            "num_cores": 32,
            "gate_up_in0_block_w": 7,
            "down_in0_block_w": 7,
            "per_core_M": 1,
            "per_core_N": hidden // (ttnn.TILE_SIZE * 8),
            "out_subblock_h": 1,
            "out_subblock_w": 7,
            "median_ms": statistics.median(l1_samples),
            "samples_ms": l1_samples,
        }
    )
    record = {
        "rows": rows,
        "baseline": {
            "placement": "dram_interleaved_auto_program",
            "median_ms": statistics.median(baseline_samples),
            "samples_ms": baseline_samples,
        },
        "candidates": candidates,
        "rejected_placement": "DRAM-sharded program hard-requires M=1; see rejected artifact",
    }
    print("MULTICHIP_MLP_PREFILL_PLACEMENT " + json.dumps(record))


@pytest.mark.skipif(
    "GEMMA4_MULTICHIP_FRACTURED_PROBE" not in os.environ,
    reason="set GEMMA4_MULTICHIP_FRACTURED_PROBE=1",
)
@pytest.mark.parametrize("num_rows", [32, 128])
def test_multichip_fractured_residual_boundary_probe(hf_config, profile_mesh_device, num_rows):
    """Compare replicated and fractured O->norm->residual->norm->gate boundaries."""
    mesh_device = profile_mesh_device
    layer_idx, _ = LAYER_KINDS[0]
    decoder = _decoder(hf_config, mesh_device, layer_idx)
    mlp = decoder.layer.shared_mlp
    mlp.policy = _mlp_geometry_policy(24, base_policy=mlp.policy)
    hidden = hf_config.hidden_size
    torch.manual_seed(20262300 + num_rows)
    partial_host = torch.randn(TP_SIZE, 1, num_rows, hidden, dtype=torch.bfloat16) * 0.02
    residual_host = torch.randn(1, 1, num_rows, hidden, dtype=torch.bfloat16) * 0.02
    partial_replicated = ttnn.from_torch(
        partial_host,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    partial_fractured = ttnn.from_torch(
        partial_host,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    residual_replicated = ttnn.from_torch(
        residual_host,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    residual_fractured = ttnn.from_torch(
        residual_host,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    state = _layer_state(layer_idx)
    prefix = f"model.language_model.layers.{layer_idx}."

    def local_gamma(name):
        weight = state[prefix + name + ".weight"].reshape(1, 1, -1, ttnn.TILE_SIZE)
        return ttnn.from_torch(
            weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
        )

    post_gamma = local_gamma("post_attention_layernorm")
    pre_ff_gamma = local_gamma("pre_feedforward_layernorm")

    def projection(full_input):
        if num_rows == 32:
            input_mem = mlp._decode_memory_config(24, hidden)
            output_mem = mlp._decode_memory_config(24, hidden)
            sharded = ttnn.to_memory_config(full_input, input_mem)
            program = mlp._decode_program_config(
                k=hidden,
                n=hidden,
                num_cores=24,
                in0_block_w=7,
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0),
            )
            return ttnn.linear(
                sharded,
                mlp.gate_decode,
                memory_config=output_mem,
                program_config=program,
                compute_kernel_config=mlp.gate_up_compute,
            )
        return ttnn.linear(full_input, mlp.gate_prefill)

    def distributed_norm(fractured, gamma):
        stats = ttnn.rms_norm_pre_all_gather(fractured, dtype=ttnn.bfloat16)
        gathered_stats = ttnn.all_gather(
            stats,
            dim=3,
            cluster_axis=1,
            num_links=2,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.rms_norm_post_all_gather(
            fractured,
            gathered_stats,
            epsilon=hf_config.rms_norm_eps,
            weight=gamma,
            dtype=ttnn.bfloat16,
        )

    def replicated_call():
        # Use the primitive directly because the production helper eagerly
        # deallocates its input after the collective.  Both candidates need a
        # stable source while their warmed timings are sampled repeatedly.
        reduced = ttnn.all_reduce(
            partial_replicated,
            cluster_axis=1,
            num_links=2,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        post = decoder.layer.post_attention_layernorm.forward(reduced)
        residual_sum = ttnn.add(residual_replicated, post)
        normalized = decoder.layer.pre_feedforward_layernorm.forward(residual_sum)
        return projection(normalized)

    def fractured_call():
        reduced = ttnn.reduce_scatter(
            partial_fractured,
            dim=3,
            cluster_axis=1,
            num_links=2,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        post = distributed_norm(reduced, post_gamma)
        residual_sum = ttnn.add(residual_fractured, post)
        normalized = distributed_norm(residual_sum, pre_ff_gamma)
        gathered = ttnn.all_gather(
            normalized,
            dim=3,
            cluster_axis=1,
            num_links=2,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return projection(gathered)

    baseline = replicated_call()
    candidate = fractured_call()
    ttnn.synchronize_device(mesh_device)
    baseline_devices = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(baseline)]
    candidate_devices = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(candidate)]
    for expected, actual in zip(baseline_devices, candidate_devices):
        _assert_pcc(expected, actual, 0.995)

    def measure(call):
        if num_rows == 32:
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            output = call()
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            try:
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
                samples = []
                for _ in range(12):
                    start = time.perf_counter_ns()
                    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
                    samples.append((time.perf_counter_ns() - start) / 1_000_000)
                return samples
            finally:
                ttnn.release_trace(mesh_device, trace_id)
                output.deallocate(True)
        warm = call()
        ttnn.synchronize_device(mesh_device)
        warm.deallocate(True)
        samples = []
        for _ in range(12):
            start = time.perf_counter_ns()
            output = call()
            ttnn.synchronize_device(mesh_device)
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
            output.deallocate(True)
        return samples

    replicated_samples = measure(replicated_call)
    fractured_samples = measure(fractured_call)
    record = {
        "rows": num_rows,
        "replicated_ms": statistics.median(replicated_samples),
        "fractured_ms": statistics.median(fractured_samples),
        "fractured_over_replicated": statistics.median(fractured_samples) / statistics.median(replicated_samples),
        "replicated_samples_ms": replicated_samples,
        "fractured_samples_ms": fractured_samples,
    }
    print("MULTICHIP_FRACTURED_BOUNDARY " + json.dumps(record))


@pytest.mark.skipif("GEMMA4_MULTICHIP_BENCH" not in os.environ, reason="set GEMMA4_MULTICHIP_BENCH=1")
@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
def test_multichip_warmed_latency(hf_config, mesh_device, optimized_baseline, layer_idx, layer_kind):
    decoder = _decoder(hf_config, mesh_device, layer_idx)
    values = _inputs(hf_config, layer_idx)
    cache, page_table = decoder.init_paged_kv_cache(max_context=128)
    torch.manual_seed(20262100 + layer_idx)
    prefill128_host = torch.randn(1, 128, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    prefill128 = _tt_input(prefill128_host, mesh_device)
    prefill128_rope = _rope_device(hf_config, layer_idx, 128, mesh_device, decode=False)

    def prefill_call():
        return decoder.prefill_forward(
            prefill128,
            rope_mats=prefill128_rope,
            page_table=page_table,
            kv_cache=cache,
            valid_seq_len=128,
        )

    warm = prefill_call()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    prefill_samples = []
    for _ in range(12):
        start = time.perf_counter_ns()
        sample = prefill_call()
        ttnn.synchronize_device(mesh_device)
        prefill_samples.append((time.perf_counter_ns() - start) / 1_000_000)
        sample.deallocate(True)

    prompt = _tt_input(values["prompt32"], mesh_device)
    decoder.prefill_forward(
        prompt,
        rope_mats=_rope_device(hf_config, layer_idx, 32, mesh_device, decode=False),
        page_table=page_table,
        kv_cache=cache,
        valid_seq_len=32,
    )
    token = _tt_input(values["token"], mesh_device)
    pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
    pos_u_host[0, 0] = 32
    kwargs = dict(
        rope_mats=_rope_device(hf_config, layer_idx, 128, mesh_device, decode=True),
        page_table=page_table,
        kv_cache=cache,
        current_position=_replicated_int(pos_u_host, mesh_device, dtype=ttnn.uint32),
        current_position_cache=_replicated_int(torch.tensor([32]), mesh_device, dtype=ttnn.int32),
    )
    warm = decoder.decode_forward(token, **kwargs)
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    decoder.decode_forward(token, **kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        samples = []
        for _ in range(12):
            start = time.perf_counter_ns()
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
        baseline = optimized_baseline[(layer_kind, "baseline_latency")]
        multichip_prefill = statistics.median(prefill_samples)
        multichip_decode = statistics.median(samples)
        record = {
            "kind": layer_kind,
            "baseline_prefill128_ms": baseline["prefill128_ms"],
            "multichip_prefill128_ms": multichip_prefill,
            "prefill_speedup": baseline["prefill128_ms"] / multichip_prefill,
            "prefill_tp4_efficiency": baseline["prefill128_ms"] / multichip_prefill / TP_SIZE,
            "baseline_decode_ms": baseline["decode_ms"],
            "multichip_decode_ms": multichip_decode,
            "decode_speedup": baseline["decode_ms"] / multichip_decode,
            "decode_tp4_efficiency": baseline["decode_ms"] / multichip_decode / TP_SIZE,
            "multichip_prefill128_samples_ms": prefill_samples,
            "multichip_decode_samples_ms": samples,
        }
        print("MULTICHIP_WARMED " + json.dumps(record))
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.skipif(
    "GEMMA4_MULTICHIP_PROFILE" not in os.environ,
    reason="set GEMMA4_MULTICHIP_PROFILE=1 and run under Tracy",
)
@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
@pytest.mark.parametrize("perf_mode", ["prefill", "decode"])
def test_multichip_profile(hf_config, profile_mesh_device, layer_idx, layer_kind, perf_mode):
    from tracy import signpost

    mesh_device = profile_mesh_device
    decoder = _decoder(hf_config, mesh_device, layer_idx)
    cache, page_table = decoder.init_paged_kv_cache(max_context=128)
    torch.manual_seed(20262200 + layer_idx)
    if perf_mode == "prefill":
        hidden = _tt_input(
            torch.randn(1, 128, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02,
            mesh_device,
        )
        rope = _rope_device(hf_config, layer_idx, 128, mesh_device, decode=False)

        def call():
            return decoder.prefill_forward(
                hidden,
                rope_mats=rope,
                page_table=page_table,
                kv_cache=cache,
                valid_seq_len=128,
            )

        warm = call()
        ttnn.synchronize_device(mesh_device)
        warm.deallocate(True)
        signpost(f"MC_{layer_kind}_PREFILL")
        output = call()
        ttnn.synchronize_device(mesh_device)
        signpost(f"MC_{layer_kind}_PREFILL_END")
        output.deallocate(True)
        return

    prompt_len = 32
    prompt = _tt_input(
        torch.randn(1, prompt_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02,
        mesh_device,
    )
    decoder.prefill_forward(
        prompt,
        rope_mats=_rope_device(hf_config, layer_idx, prompt_len, mesh_device, decode=False),
        page_table=page_table,
        kv_cache=cache,
        valid_seq_len=prompt_len,
    )
    token = _tt_input(
        torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02,
        mesh_device,
    )
    pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
    pos_u_host[0, 0] = prompt_len
    kwargs = dict(
        rope_mats=_rope_device(hf_config, layer_idx, 128, mesh_device, decode=True),
        page_table=page_table,
        kv_cache=cache,
        current_position=_replicated_int(pos_u_host, mesh_device, dtype=ttnn.uint32),
        current_position_cache=_replicated_int(
            torch.tensor([prompt_len], dtype=torch.int32),
            mesh_device,
            dtype=ttnn.int32,
        ),
    )

    def call():
        return decoder.decode_forward(token, **kwargs)

    warm = call()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        signpost(f"MC_{layer_kind}_DECODE")
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        signpost(f"MC_{layer_kind}_DECODE_END")
    finally:
        ttnn.release_trace(mesh_device, trace_id)
