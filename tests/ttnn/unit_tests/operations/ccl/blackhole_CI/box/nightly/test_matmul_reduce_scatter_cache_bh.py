# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Program-cache regression test for matmul_reduce_scatter_async on a 4 chip Blackhole box.

Only the first call builds the program. Every later call is a cache hit, and each one passes new tensors
(input, weight, bias, persistent intermediate and output buffers) and a new set of GlobalSemaphores. On a
hit, the op has to re-point every tensor address and re-apply the semaphores, which are not part of the
program hash. After each call its semaphores are set to a large value. A program that still pointed at
them would see every semaphore wait already satisfied and read the intermediate before the other chips
wrote it, so the output check fails instead of passing by accident.

Shapes and program config follow the Qwen3.6-27B GDN prefill out projection
(models/demos/blackhole/qwen36/tt/tp_common.py::matmul_reduce_scatter_prefill), at M=128.
"""
import math

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

TILE = 32
NUM_CALLS = 3
POISON = 1 << 20


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 24576}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "in_dtype, w_dtype, use_bias",
    [
        (ttnn.float32, ttnn.bfloat8_b, False),
        (ttnn.bfloat16, ttnn.bfloat16, True),
    ],
    ids=["fp32_qwen36", "bf16_bias"],
)
def test_matmul_reduce_scatter_cache_hit(mesh_device, in_dtype, w_dtype, use_bias):
    torch.manual_seed(0)
    nd = mesh_device.get_num_devices()
    M, K_local, N = 128, 1536, 5120
    K = K_local * nd
    grid = (8, 8)

    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))})
    # The RS workers sit below the matmul grid, so the semaphores must cover the whole grid.
    grid_size = mesh_device.compute_with_storage_grid_size()
    sem_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))}
    )
    shard_k = lambda dim: ttnn.ShardTensorToMesh(mesh_device, dim=dim)
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)

    def to_device(t, dtype, mapper):
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    def rounded(t, dtype):
        # t as the device stores it in dtype (bfloat16 and bfloat8_b round), computed on the host.
        return ttnn.to_torch(ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()

    # Everything every call needs is created up front and kept alive, so each call's tensors and semaphores
    # live at addresses no other call uses.
    calls = []
    for _ in range(NUM_CALLS):
        x = rounded(torch.randn(1, 1, M, K), in_dtype)
        w = rounded(torch.randn(1, 1, K, N), w_dtype)
        # The bias is added on every chip before the reduce-scatter sums them, so each chip gets 1/nd of it.
        bias = rounded(torch.randn(1, 1, 1, N) / nd, in_dtype) if use_bias else None
        calls.append(
            {
                "x": to_device(x, in_dtype, shard_k(3)),
                "w": to_device(w, w_dtype, shard_k(2)),
                "bias": to_device(bias, in_dtype, replicate) if use_bias else None,
                "interm": to_device(torch.zeros(1, 1, M, N), in_dtype, replicate),
                "out_buf": to_device(torch.zeros(1, 1, M, N // nd), in_dtype, replicate),
                "rs_sems": [ttnn.create_global_semaphore(mesh_device, sem_crs, 0) for _ in range(3)],
                "barrier_sem": ttnn.create_global_semaphore(mesh_device, sem_crs, 0),
                "ref": x @ w + nd * bias if use_bias else x @ w,
            }
        )

    per_core_N = max(1, math.ceil(N / TILE / grid[0]))
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=min(4, max(1, K_local // TILE // grid[0])),
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=max(1, math.ceil(M / TILE / grid[1])),
        per_core_N=per_core_N,
        out_block_w=max(1, per_core_N // 2),
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
        allowed_worker_cores=crs,
    )
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )

    sem_addrs = [
        tuple(ttnn.get_global_semaphore_address(s) for s in call["rs_sems"] + [call["barrier_sem"]]) for call in calls
    ]
    assert len(set(sem_addrs)) == NUM_CALLS, f"the calls must not share semaphore addresses: {sem_addrs}"

    # Only the op runs between the cache-entry readings, so any extra entry is the op missing the cache.
    outputs = []
    cache_entries = []
    for call in calls:
        mm, rs = ttnn.experimental.matmul_reduce_scatter_async(
            call["x"],
            call["w"],
            persistent_intermediate_buffer=call["interm"],
            persistent_output_buffer=call["out_buf"],
            dim=3,
            multi_device_global_semaphore=call["rs_sems"],
            reduce_scatter_core_grid_offset=(0, 8),
            barrier_semaphore=call["barrier_sem"],
            bias=call["bias"],
            num_links=2,
            memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            subdevice_id=None,
            memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
            program_config=pc,
            compute_kernel_config=ckc,
        )
        ttnn.synchronize_device(mesh_device)
        cache_entries.append(mesh_device.num_program_cache_entries())
        for sem in call["rs_sems"] + [call["barrier_sem"]]:
            ttnn.reset_global_semaphore_value(sem, POISON)
        # Keeping mm alive too puts every call's matmul output at its own address. rs is this call's
        # persistent output buffer, which no later call writes.
        outputs.append((mm, rs))

    assert all(
        n == cache_entries[0] for n in cache_entries
    ), f"program cache entries per call: {cache_entries}; the calls after the first must be cache hits"

    for i, (call, (mm, rs)) in enumerate(zip(calls, outputs)):
        ref = call["ref"]
        what = f"call {i} (cache {'miss' if i == 0 else 'hit'})"
        # Each chip's matmul output is its partial sum over its K shard; the partials add up to ref.
        partials = ttnn.to_torch(mm, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3)).float()
        passed, pcc = comp_pcc(ref, sum(torch.chunk(partials, nd, dim=3)), 0.999)
        assert passed, f"{what}, matmul output: {pcc}"
        out = ttnn.to_torch(rs, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3)).float()
        passed, pcc = comp_pcc(ref, out, 0.999)
        assert passed, f"{what}, reduce-scatter output: {pcc}"
