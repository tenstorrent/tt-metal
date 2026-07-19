# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Manual exact-shape comparison of shard-advisor and DRAM-sharded matmuls."""

from __future__ import annotations

import json
import os
import statistics
import time

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen2_5_coder_32b_instruct.tt.multichip_decoder import (
    _dram_matmul_program_config,
    _dram_sharded_memory_config,
    _matmul_output_memory_config,
    _width_sharded_memory_config,
)
from models.common.utility_functions import comp_pcc

PROBE_ENV = "QWEN2_5_CODER_32B_MULTICHIP_ADVISOR_PROBE"


def _replicated(host, mesh_device, *, dtype, memory_config):
    return ttnn.from_torch(
        host.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _counted_width_sharded(rows, shard_width, core_count, mesh_device):
    cores = ttnn.num_cores_to_corerangeset(
        core_count,
        mesh_device.compute_with_storage_grid_size(),
        row_wise=True,
    )
    return ttnn.create_sharded_memory_config(
        shape=(rows, shard_width),
        core_grid=cores,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _advisor_program(grid, *, in0_block_w, per_core_n, out_subblock_w):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _trace_median(mesh_device, fn, *, replays=100, trials=5):
    warm = fn()
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = fn()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    samples = []
    try:
        for _ in range(trials):
            start = time.perf_counter()
            for _ in range(replays):
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            samples.append((time.perf_counter() - start) * 1000.0 / replays)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    return warm, traced, samples, statistics.median(samples)


@pytest.mark.skipif(os.getenv(PROBE_ENV) != "1", reason="manual shard-advisor comparison")
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.timeout(900)
def test_advisor_interleaved_1d_vs_dram_sharded_projection_family(mesh_device):
    generator = torch.Generator().manual_seed(190727)
    rows = 32
    roles = {
        "qkv": {
            "k": 5120,
            "n": 2048,
            "dtype": ttnn.bfloat8_b,
            "fidelity": ttnn.MathFidelity.HiFi2,
            "dram_grid": ttnn.CoreGrid(x=8, y=2),
            "advisor_input": ttnn.L1_MEMORY_CONFIG,
            "advisor_output": _counted_width_sharded(rows, 32, 64, mesh_device),
            "advisor_program": _advisor_program((11, 6), in0_block_w=8, per_core_n=1, out_subblock_w=1),
        },
        "o": {
            "k": 1280,
            "n": 5120,
            "dtype": ttnn.bfloat8_b,
            "fidelity": ttnn.MathFidelity.HiFi2,
            "dram_grid": ttnn.CoreGrid(x=8, y=1),
            "advisor_input": _counted_width_sharded(rows, 64, 20, mesh_device),
            "advisor_output": _counted_width_sharded(rows, 64, 80, mesh_device),
            "advisor_program": _advisor_program((11, 8), in0_block_w=2, per_core_n=2, out_subblock_w=2),
        },
        "packed_gate_up": {
            "k": 5120,
            "n": 14336,
            "dtype": ttnn.bfloat4_b,
            "fidelity": ttnn.MathFidelity.LoFi,
            "dram_grid": ttnn.CoreGrid(x=8, y=4),
            "advisor_input": ttnn.L1_MEMORY_CONFIG,
            # The advisor owns two internal output padding tiles: 90*160=14400.
            "advisor_output": _counted_width_sharded(rows, 160, 90, mesh_device),
            "advisor_program": _advisor_program((11, 9), in0_block_w=2, per_core_n=5, out_subblock_w=5),
        },
        "down": {
            "k": 7168,
            "n": 5120,
            "dtype": ttnn.bfloat4_b,
            "fidelity": ttnn.MathFidelity.LoFi,
            "dram_grid": ttnn.CoreGrid(x=8, y=2),
            "advisor_input": _counted_width_sharded(rows, 128, 56, mesh_device),
            "advisor_output": _counted_width_sharded(rows, 64, 80, mesh_device),
            "advisor_program": _advisor_program((11, 8), in0_block_w=2, per_core_n=2, out_subblock_w=2),
        },
    }

    results = {}
    for role, cfg in roles.items():
        host_input = torch.randn((1, 1, rows, cfg["k"]), generator=generator, dtype=torch.bfloat16)
        host_weight = (torch.randn((cfg["k"], cfg["n"]), generator=generator) * 0.02).to(torch.bfloat16)
        dram_input = _replicated(
            host_input,
            mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=_width_sharded_memory_config(rows, cfg["k"], cfg["dram_grid"]),
        )
        advisor_input = _replicated(
            host_input,
            mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=cfg["advisor_input"],
        )
        dram_weight = _replicated(
            host_weight,
            mesh_device,
            dtype=cfg["dtype"],
            memory_config=_dram_sharded_memory_config(mesh_device, cfg["k"], cfg["n"]),
        )
        advisor_weight = _replicated(
            host_weight,
            mesh_device,
            dtype=cfg["dtype"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=cfg["fidelity"],
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        dram_output_memory = _matmul_output_memory_config(
            rows,
            cfg["n"],
            cfg["dram_grid"],
            mesh_device,
        )
        dram_program = _dram_matmul_program_config(
            rows,
            cfg["k"],
            cfg["n"],
            cfg["dram_grid"],
        )

        def dram_call():
            return ttnn.matmul(
                dram_input,
                dram_weight,
                dtype=ttnn.bfloat16,
                program_config=dram_program,
                compute_kernel_config=compute,
                memory_config=dram_output_memory,
            )

        def advisor_call():
            return ttnn.matmul(
                advisor_input,
                advisor_weight,
                dtype=ttnn.bfloat16,
                program_config=cfg["advisor_program"],
                compute_kernel_config=compute,
                memory_config=cfg["advisor_output"],
            )

        dram_warm, _, dram_samples, dram_median = _trace_median(mesh_device, dram_call)
        advisor_warm, _, advisor_samples, advisor_median = _trace_median(mesh_device, advisor_call)
        dram_host = ttnn.to_torch(ttnn.get_device_tensors(dram_warm)[0]).float()[..., : cfg["n"]]
        advisor_host = ttnn.to_torch(ttnn.get_device_tensors(advisor_warm)[0]).float()[..., : cfg["n"]]
        passed, pcc = comp_pcc(dram_host, advisor_host, pcc=0.99)
        assert passed, f"{role} advisor/DRAM result PCC={pcc}"
        results[role] = {
            "pcc": pcc,
            "dram_trace_samples_ms": dram_samples,
            "dram_trace_median_ms": dram_median,
            "advisor_trace_samples_ms": advisor_samples,
            "advisor_trace_median_ms": advisor_median,
            "advisor_over_dram": advisor_median / dram_median,
        }
    print({"advisor_projection_comparison": results})
    output_path = os.getenv("QWEN2_5_CODER_32B_MULTICHIP_ADVISOR_PROBE_RESULT")
    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump({"advisor_projection_comparison": results}, handle, indent=2)
            handle.write("\n")
