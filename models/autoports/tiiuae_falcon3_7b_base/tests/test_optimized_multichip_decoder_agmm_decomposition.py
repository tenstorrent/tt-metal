# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Real-weight gathered-input/local-output O/down decomposition gate."""

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
    _first_rank,
    _release_model,
    _release_tensors,
    _write_result_artifact,
)
from models.autoports.tiiuae_falcon3_7b_base.tt.functional_decoder import IR_REPRESENTATIVE_LAYER
from models.autoports.tiiuae_falcon3_7b_base.tt.multichip_decoder import (
    TARGET_MESH_SHAPE,
    TENSOR_PARALLEL_SIZE,
    MultichipDecoder,
    _mesh_weight,
    _rank_padded_shards,
)
from models.common.modules.tt_ccl import TT_CCL
from models.common.utility_functions import comp_pcc

PROBE_PATH = Path(__file__)
ROWS = 32


def _trace_boundary(mesh_device, function, host_function, *, samples: int = 5, iterations: int = 100):
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
        first = host_function(output)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        second = host_function(output)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    return output, timings, torch.equal(first, second)


def _ag_input_memory_config(mesh_device, global_width: int):
    core_grid = ttnn.num_cores_to_corerangeset(
        TENSOR_PARALLEL_SIZE,
        mesh_device.compute_with_storage_grid_size(),
        row_wise=True,
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            core_grid,
            (ROWS, global_width // TENSOR_PARALLEL_SIZE),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def _local_output_memory_config(local_width: int):
    return ttnn.create_sharded_memory_config(
        shape=(ROWS, local_width // 8),
        core_grid=ttnn.CoreGrid(x=8, y=1),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _agmm_program_config(global_k: int, local_n: int):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 1),
        in0_block_w=global_k // 32 // 8,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=local_n // 32 // 8,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


@pytest.mark.skipif(
    os.getenv("FALCON3_RUN_MULTICHIP_O_DOWN_AGMM") != "1",
    reason="manual gathered-input/local-output O/down decomposition gate",
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [TARGET_MESH_SHAPE], indirect=True)
@pytest.mark.timeout(1800)
def test_gathered_input_local_output_o_down(mesh_device):
    """Cross both row-parallel boundaries without restoring fractured outputs."""
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
        attention_activation_dtype=ttnn.bfloat8_b,
        mlp_activation_dtype=ttnn.bfloat8_b,
    )
    hidden = config.hidden_size
    local_hidden = hidden // TENSOR_PARALLEL_SIZE
    padded_intermediate = model.local_intermediate_decode_size * TENSOR_PARALLEL_SIZE
    layer_prefix = f"model.layers.{IR_REPRESENTATIVE_LAYER}"
    o_host = state_dict[f"{layer_prefix}.self_attn.o_proj.weight"].transpose(-2, -1)
    down_host = state_dict[f"{layer_prefix}.mlp.down_proj.weight"].transpose(-2, -1)
    down_padded_host = _rank_padded_shards(
        down_host,
        tp=TENSOR_PARALLEL_SIZE,
        dim=0,
        padded_local_size=model.local_intermediate_decode_size,
    )
    o_local_output_weight = _mesh_weight(
        o_host.reshape(1, 1, hidden, hidden),
        dtype=ttnn.bfloat4_b,
        mesh_device=mesh_device,
        shard_dim=3,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    down_local_output_weight = _mesh_weight(
        down_padded_host.reshape(1, 1, padded_intermediate, hidden),
        dtype=ttnn.bfloat4_b,
        mesh_device=mesh_device,
        shard_dim=3,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    residual_host = torch.randn(1, 1, ROWS, hidden, dtype=torch.bfloat16) * 0.01
    o_input_host = torch.randn(1, 1, ROWS, hidden, dtype=torch.bfloat16) * 0.01
    down_input_host = torch.randn(1, 1, ROWS, padded_intermediate, dtype=torch.bfloat16) * 0.01
    residual_replicated = ttnn.from_torch(
        residual_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=model.residual_memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    local_output_memory_config = _local_output_memory_config(local_hidden)
    residual_fractured = ttnn.from_torch(
        residual_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=local_output_memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )

    def make_input(host, memory_config):
        return ttnn.from_torch(
            host,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=memory_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        )

    o_selected_input = make_input(o_input_host, ttnn.DRAM_MEMORY_CONFIG)
    down_selected_input = make_input(down_input_host, ttnn.DRAM_MEMORY_CONFIG)
    o_ag_memory_config = _ag_input_memory_config(mesh_device, hidden)
    down_ag_memory_config = _ag_input_memory_config(mesh_device, padded_intermediate)
    o_ag_input = make_input(o_input_host, o_ag_memory_config)
    down_ag_input = make_input(down_input_host, down_ag_memory_config)
    ccl = TT_CCL(mesh_device)

    def selected_boundary(input_tensor, weight, input_memory_config, output_memory_config, program_config, role):
        matmul_input = ttnn.to_memory_config(input_tensor, input_memory_config)
        partial = ttnn.matmul(
            matmul_input,
            weight,
            dtype=ttnn.bfloat8_b,
            program_config=program_config,
            compute_kernel_config=(model.attention_compute_config if role == "attention" else model.mlp_compute_config),
            memory_config=output_memory_config,
        )
        matmul_input.deallocate(True)
        partial = ttnn.to_memory_config(partial, model.residual_memory_config)
        reduced = model._all_reduce_partial(partial, memory_config=model.residual_memory_config, ccl_role=role)
        output = ttnn.add(residual_replicated, reduced, memory_config=model.residual_memory_config)
        reduced.deallocate(True)
        return output

    def explicit_boundary(input_tensor, weight, ag_memory_config, program_config, compute_kernel_config):
        gathered = ttnn.experimental.all_gather_async(
            input_tensor,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=1,
            topology=ttnn.Topology.Ring,
            memory_config=ag_memory_config,
            barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        projected = ttnn.matmul(
            gathered,
            weight,
            dtype=ttnn.bfloat8_b,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=local_output_memory_config,
        )
        gathered.deallocate(True)
        projected_bf16 = ttnn.typecast(projected, ttnn.bfloat16)
        projected.deallocate(True)
        output = ttnn.add(residual_fractured, projected_bf16, memory_config=local_output_memory_config)
        projected_bf16.deallocate(True)
        return output

    def fused_boundary(input_tensor, weight, ag_memory_config, program_config, compute_kernel_config):
        gathered, projected = ttnn.experimental.all_gather_matmul_async(
            input_tensor,
            weight,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(),
            all_gather_core_grid_offset=(0, 4),
            barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=1,
            topology=ttnn.Topology.Ring,
            memory_config_ag=ag_memory_config,
            memory_config_mm=local_output_memory_config,
            dtype=ttnn.bfloat8_b,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        gathered.deallocate(True)
        projected_bf16 = ttnn.typecast(projected, ttnn.bfloat16)
        projected.deallocate(True)
        output = ttnn.add(residual_fractured, projected_bf16, memory_config=local_output_memory_config)
        projected_bf16.deallocate(True)
        return output

    cases = {
        "o": {
            "selected": lambda: selected_boundary(
                o_selected_input,
                model.o_decode_weight,
                model.o_input_memory_config,
                model.o_output_memory_config,
                model.o_decode_program_config,
                "attention",
            ),
            "explicit": lambda: explicit_boundary(
                o_ag_input,
                o_local_output_weight,
                o_ag_memory_config,
                _agmm_program_config(hidden, local_hidden),
                model.attention_compute_config,
            ),
            "fused": lambda: fused_boundary(
                o_ag_input,
                o_local_output_weight,
                o_ag_memory_config,
                _agmm_program_config(hidden, local_hidden),
                model.attention_compute_config,
            ),
            "shape": [ROWS, hidden, hidden],
        },
        "down": {
            "selected": lambda: selected_boundary(
                down_selected_input,
                model.down_decode_weight,
                model.down_input_memory_config,
                model.down_output_memory_config,
                model.down_decode_program_config,
                "mlp",
            ),
            "explicit": lambda: explicit_boundary(
                down_ag_input,
                down_local_output_weight,
                down_ag_memory_config,
                _agmm_program_config(padded_intermediate, local_hidden),
                model.mlp_compute_config,
            ),
            "fused": lambda: fused_boundary(
                down_ag_input,
                down_local_output_weight,
                down_ag_memory_config,
                _agmm_program_config(padded_intermediate, local_hidden),
                model.mlp_compute_config,
            ),
            "shape": [ROWS, padded_intermediate, hidden],
        },
    }
    results = {}
    live_outputs = []
    for name, case in cases.items():
        selected = case["selected"]()
        explicit = case["explicit"]()
        fused = case["fused"]()
        selected_host = _first_rank(selected)
        explicit_host = ttnn.to_torch(explicit, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
        fused_host = ttnn.to_torch(fused, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
        explicit_pcc = comp_pcc(selected_host.float(), explicit_host.float())[1]
        fused_pcc = comp_pcc(selected_host.float(), fused_host.float())[1]
        assert explicit_pcc >= 0.99
        assert fused_pcc >= 0.99
        _release_tensors(selected, explicit, fused)

        selected_trace, selected_samples, selected_deterministic = _trace_boundary(
            mesh_device, case["selected"], _first_rank
        )
        explicit_trace, explicit_samples, explicit_deterministic = _trace_boundary(
            mesh_device,
            case["explicit"],
            lambda tensor: ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3)),
        )
        fused_trace, fused_samples, fused_deterministic = _trace_boundary(
            mesh_device,
            case["fused"],
            lambda tensor: ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3)),
        )
        assert selected_deterministic and explicit_deterministic and fused_deterministic
        live_outputs.extend((selected_trace, explicit_trace, fused_trace))
        results[name] = {
            "logical_matmul_shape": case["shape"],
            "selected_row_parallel_persistent_bfp8_all_reduce_ms": sorted(selected_samples)[2],
            "selected_samples": selected_samples,
            "explicit_bfp8_async_all_gather_local_output_ms": sorted(explicit_samples)[2],
            "explicit_samples": explicit_samples,
            "fused_bfp8_all_gather_matmul_local_output_ms": sorted(fused_samples)[2],
            "fused_samples": fused_samples,
            "explicit_pcc_vs_selected": explicit_pcc,
            "fused_pcc_vs_selected": fused_pcc,
            "trace_replay_bitwise_deterministic": True,
        }

    selected_sum = sum(v["selected_row_parallel_persistent_bfp8_all_reduce_ms"] for v in results.values())
    explicit_sum = sum(v["explicit_bfp8_async_all_gather_local_output_ms"] for v in results.values())
    fused_sum = sum(v["fused_bfp8_all_gather_matmul_local_output_ms"] for v in results.values())
    _write_result_artifact(
        "o_down_agmm_decomposition.json",
        {
            "boundary_contract": "BFP8 projection/CCL, local BF16 residual add; local-output candidates remain hidden-fractured and never restore to replicated layout",
            "mesh": list(TARGET_MESH_SHAPE),
            "weights": "real layer-14 O/down BFP4 LoFi",
            "rows": ROWS,
            "results": results,
            "combined_boundary_ms": {
                "selected_row_parallel_persistent_bfp8_all_reduce": selected_sum,
                "explicit_bfp8_async_all_gather_local_output": explicit_sum,
                "fused_bfp8_all_gather_matmul_local_output": fused_sum,
            },
            "agmm": {
                "num_links": 1,
                "core_grid_offset": [0, 4],
                "matmul_grid": [8, 1],
                "local_output_width": local_hidden,
                "o_in0_block_w": hidden // 32 // 8,
                "down_in0_block_w": padded_intermediate // 32 // 8,
                "per_core_M": 1,
                "per_core_N": local_hidden // 32 // 8,
                "chunks_per_sync": 10,
                "num_workers_per_link": 2,
                "num_buffers_per_channel": 2,
            },
            "selected": min(
                (selected_sum, "row_parallel_persistent_bfp8_all_reduce"),
                (explicit_sum, "explicit_bfp8_async_all_gather_local_output"),
                (fused_sum, "fused_bfp8_all_gather_matmul_local_output"),
            )[1],
            "implementation_sha256": hashlib.sha256(
                Path(model.__class__.__module__.replace(".", "/") + ".py").read_bytes()
            ).hexdigest(),
            "probe_test_sha256": hashlib.sha256(PROBE_PATH.read_bytes()).hexdigest(),
        },
    )
    _release_tensors(
        *live_outputs,
        o_selected_input,
        down_selected_input,
        o_ag_input,
        down_ag_input,
        residual_replicated,
        residual_fractured,
        o_local_output_weight,
        down_local_output_weight,
    )
    _release_model(model)
