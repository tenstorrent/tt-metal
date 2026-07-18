# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused fused all-gather/QKV topology gate for the Falcon3 TP4 decoder."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.tiiuae_falcon3_7b_base.tests.test_functional_decoder import _config, _real_layer_state_dict
from models.autoports.tiiuae_falcon3_7b_base.tests.test_multichip_decoder import (
    _release_model,
    _release_tensors,
    _write_result_artifact,
)
from models.autoports.tiiuae_falcon3_7b_base.tt.functional_decoder import IR_REPRESENTATIVE_LAYER
from models.autoports.tiiuae_falcon3_7b_base.tt.multichip_decoder import (
    TARGET_MESH_SHAPE,
    TENSOR_PARALLEL_SIZE,
    MultichipDecoder,
)
from models.common.modules.tt_ccl import TT_CCL
from models.common.utility_functions import comp_pcc

PROBE_PATH = Path(__file__)


def _mesh_concat(tensor, mesh_device) -> torch.Tensor:
    return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))


def _trace_boundary(mesh_device, function, *, samples: int = 5, iterations: int = 100):
    warm = function()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = function()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        for _ in range(10):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        timings = []
        for _ in range(samples):
            start = time.perf_counter()
            for _ in range(iterations):
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            timings.append((time.perf_counter() - start) * 1000.0 / iterations)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        first = _mesh_concat(output, mesh_device)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        second = _mesh_concat(output, mesh_device)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    return output, timings, torch.equal(first, second)


@pytest.mark.skipif(
    os.getenv("FALCON3_RUN_MULTICHIP_FUSED_AGMM") != "1",
    reason="manual fused all-gather plus next-QKV topology gate",
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [TARGET_MESH_SHAPE], indirect=True)
@pytest.mark.timeout(1800)
def test_fractured_residual_fused_all_gather_qkv(mesh_device):
    """Compare replicated, explicit-AG, and fused-AGMM next-layer boundaries."""
    torch.manual_seed(20260718)
    config = _config()
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    model = MultichipDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=IR_REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=128,
    )
    rows = 32
    hidden = config.hidden_size
    partial_host = torch.randn(1, 1, rows, hidden, dtype=torch.bfloat16) * 0.01
    residual_host = torch.randn_like(partial_host)
    gamma_host = state_dict[f"model.layers.{IR_REPRESENTATIVE_LAYER}.input_layernorm.weight"]
    partial = ttnn.from_torch(
        partial_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    residual_replicated = ttnn.from_torch(
        residual_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    residual_fractured = ttnn.from_torch(
        residual_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    gamma_fractured = ttnn.from_torch(
        gamma_host.reshape(1, 1, hidden // 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )

    ag_core_grid = ttnn.num_cores_to_corerangeset(
        TENSOR_PARALLEL_SIZE,
        mesh_device.compute_with_storage_grid_size(),
        row_wise=True,
    )
    ag_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ag_core_grid,
            (rows, hidden // TENSOR_PARALLEL_SIZE),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    qkv_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 1),
        in0_block_w=12,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=5,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    qkv_weight_4d = ttnn.reshape(
        model.qkv_weight,
        (1, 1, hidden, model.local_qkv_size),
    )
    ccl = TT_CCL(mesh_device)

    def next_qkv_dram_sharded(normed):
        qkv_input = model._move_owned(normed, model.qkv_input_memory_config)
        output = ttnn.matmul(
            qkv_input,
            model.qkv_decode_weight,
            dtype=ttnn.bfloat16,
            program_config=model.qkv_decode_program_config,
            compute_kernel_config=model.attention_compute_config,
            memory_config=model.qkv_output_memory_config,
        )
        qkv_input.deallocate(True)
        return output

    def fractured_norm(*, output_memory_config=None):
        scattered = ttnn.reduce_scatter(
            partial,
            dim=3,
            cluster_axis=1,
            num_links=2,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        local_added = ttnn.add(residual_fractured, scattered, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        scattered.deallocate(True)
        stats = ttnn.rms_norm_pre_all_gather(
            local_added,
            compute_kernel_config=model.norm_compute_config,
            dtype=ttnn.bfloat16,
        )
        stats = ttnn.reshape(stats, ttnn.Shape((1, 1, rows, 32)))
        gathered_stats = ttnn.all_gather(
            stats,
            dim=3,
            cluster_axis=1,
            num_links=2,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        stats.deallocate(True)
        local_normed = ttnn.rms_norm_post_all_gather(
            local_added,
            gathered_stats,
            epsilon=model.rms_norm_eps,
            weight=gamma_fractured,
            compute_kernel_config=model.norm_compute_config,
            memory_config=output_memory_config,
        )
        local_added.deallocate(True)
        gathered_stats.deallocate(True)
        return local_normed

    def replicated_boundary():
        reduced = ttnn.all_reduce(
            partial,
            cluster_axis=1,
            num_links=2,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        added = ttnn.add(residual_replicated, reduced, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        reduced.deallocate(True)
        normed = ttnn.rms_norm(
            added,
            epsilon=model.rms_norm_eps,
            weight=model.input_norm_weight,
            compute_kernel_config=model.norm_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        added.deallocate(True)
        return next_qkv_dram_sharded(normed)

    def explicit_ag_boundary():
        local_normed = fractured_norm()
        gathered = ttnn.all_gather(
            local_normed,
            dim=3,
            cluster_axis=1,
            num_links=2,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        local_normed.deallocate(True)
        return next_qkv_dram_sharded(gathered)

    def fused_agmm_boundary():
        local_normed = fractured_norm(output_memory_config=ag_memory_config)
        gathered, qkv = ttnn.experimental.all_gather_matmul_async(
            local_normed,
            qkv_weight_4d,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(),
            all_gather_core_grid_offset=(0, 4),
            barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=1,
            topology=ttnn.Topology.Ring,
            memory_config_ag=ag_memory_config,
            memory_config_mm=model.qkv_output_memory_config,
            dtype=ttnn.bfloat16,
            program_config=qkv_program_config,
            compute_kernel_config=model.attention_compute_config,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        local_normed.deallocate(True)
        gathered.deallocate(True)
        return qkv

    replicated_output = replicated_boundary()
    explicit_output = explicit_ag_boundary()
    fused_output = fused_agmm_boundary()
    replicated_host = _mesh_concat(replicated_output, mesh_device)
    explicit_host = _mesh_concat(explicit_output, mesh_device)
    fused_host = _mesh_concat(fused_output, mesh_device)
    explicit_pcc = comp_pcc(replicated_host.float(), explicit_host.float())[1]
    fused_pcc = comp_pcc(replicated_host.float(), fused_host.float())[1]
    assert explicit_pcc >= 0.999
    assert fused_pcc >= 0.999
    _release_tensors(replicated_output, explicit_output, fused_output)

    replicated_trace, replicated_samples, replicated_deterministic = _trace_boundary(mesh_device, replicated_boundary)
    explicit_trace, explicit_samples, explicit_deterministic = _trace_boundary(mesh_device, explicit_ag_boundary)
    fused_trace, fused_samples, fused_deterministic = _trace_boundary(mesh_device, fused_agmm_boundary)
    assert replicated_deterministic
    assert explicit_deterministic
    assert fused_deterministic
    replicated_ms = sorted(replicated_samples)[len(replicated_samples) // 2]
    explicit_ms = sorted(explicit_samples)[len(explicit_samples) // 2]
    fused_ms = sorted(fused_samples)[len(fused_samples) // 2]
    selected = min(
        (
            (replicated_ms, "replicated"),
            (explicit_ms, "fractured_explicit_all_gather"),
            (fused_ms, "fractured_fused_all_gather_matmul"),
        )
    )[1]
    _write_result_artifact(
        "fused_agmm_boundary.json",
        {
            "boundary": "row-parallel partial -> residual add -> RMSNorm -> real next QKV",
            "mesh": list(TARGET_MESH_SHAPE),
            "shape": [1, 1, rows, hidden],
            "weights": "real layer-14 RMSNorm and interleaved BFP4 QKV",
            "replicated_ms": replicated_ms,
            "replicated_samples": replicated_samples,
            "fractured_explicit_all_gather_ms": explicit_ms,
            "fractured_explicit_all_gather_samples": explicit_samples,
            "fractured_fused_all_gather_matmul_ms": fused_ms,
            "fractured_fused_all_gather_matmul_samples": fused_samples,
            "explicit_pcc_vs_replicated": explicit_pcc,
            "fused_pcc_vs_replicated": fused_pcc,
            "trace_replay_bitwise_deterministic": {
                "replicated": replicated_deterministic,
                "fractured_explicit_all_gather": explicit_deterministic,
                "fractured_fused_all_gather_matmul": fused_deterministic,
            },
            "agmm": {
                "num_links": 1,
                "core_grid_offset": [0, 4],
                "ag_shards": 4,
                "ag_shard_shape": [rows, hidden // TENSOR_PARALLEL_SIZE],
                "qkv_program_grid": [8, 1],
                "in0_block_w": 12,
                "per_core_M": 1,
                "per_core_N": 5,
                "out_subblock_w": 1,
                "chunks_per_sync": 10,
                "num_workers_per_link": 2,
                "num_buffers_per_channel": 2,
            },
            "selected": selected,
            "probe_test_sha256": hashlib.sha256(PROBE_PATH.read_bytes()).hexdigest(),
        },
    )
    _release_tensors(replicated_trace, explicit_trace, fused_trace, partial, residual_replicated, residual_fractured)
    gamma_fractured.deallocate(True)
    _release_model(model)
