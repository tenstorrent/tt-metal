# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("op", ["ag_1d", "ag_2d", "mm_rs"])
@pytest.mark.parametrize("allowed_cores", [False, True])
@pytest.mark.parametrize(
    "offset", [(0, 0), (3, 0), (4, 0), (0, 6)], ids=["origin", "edge", "unused_grid", "separate_rows"]
)
def test_fused_matmul_core_overlap(expect_error, mesh_device, op, allowed_cores, offset):
    torch.manual_seed(0)
    x = torch.randint(-1, 2, (1, 1, 128, 128)).bfloat16()
    w = torch.randint(-1, 2, (1, 1, 128, 128)).bfloat16()
    reference = x.float() @ w.float()
    dram = ttnn.DRAM_MEMORY_CONFIG
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)

    def put(tensor, mapper=replicate):
        return ttnn.from_torch(
            tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=dram,
            mesh_mapper=mapper,
        )

    a = put(x, replicate if op == "mm_rs" else ttnn.ShardTensorToMesh(mesh_device, dim=3))
    b = put(w)
    intermediate = put(torch.zeros_like(x))
    output = put(torch.zeros(1, 1, 128, 32)) if op == "mm_rs" else None
    grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sems = [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(3)]
    config_args = dict(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        fused_activation=None,
        fuse_batch=False,
    )
    if allowed_cores:
        config_args["allowed_worker_cores"] = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))}
        )
    if op == "ag_1d":
        config_args.update(per_core_M=4, fuse_batch=True)
        config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(**config_args, mcast_in0=True)
    else:
        config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(**config_args, transpose_mcast=False)
    compute = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def invoke():
        common = dict(
            dim=3,
            num_links=1,
            topology=ttnn.Topology.Ring,
            memory_config_mm=dram,
            program_config=config,
            compute_kernel_config=compute,
        )
        if op == "mm_rs":
            return ttnn.experimental.matmul_reduce_scatter_async(
                a,
                b,
                persistent_intermediate_buffer=intermediate,
                persistent_output_buffer=output,
                multi_device_global_semaphore=sems,
                reduce_scatter_core_grid_offset=offset,
                memory_config_rs=dram,
                **common,
            )
        return ttnn.experimental.all_gather_matmul_async(
            a,
            b,
            persistent_output_buffer=intermediate,
            multi_device_global_semaphore=sems[:2],
            all_gather_core_grid_offset=offset,
            memory_config_ag=dram,
            num_workers_per_link=1,
            **common,
        )

    if offset in [(0, 0), (3, 0)]:
        with expect_error(RuntimeError, "overlapping kernel core ranges"):
            invoke()
        return

    # Cores (4,0) onward lie in the configured grid, but outside the actual
    # matmul footprint. A check of the nominal rectangle would reject this.
    for _ in range(2):
        first, second = invoke()
        ttnn.synchronize_device(mesh_device)
        matmul = first if op == "mm_rs" else second
        for tensor in ttnn.get_device_tensors(matmul):
            torch.testing.assert_close(ttnn.to_torch(tensor).float(), reference, rtol=0, atol=0)
        if op == "mm_rs":
            actual = ttnn.to_torch(second, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3)).float()
            torch.testing.assert_close(actual, 4 * reference, rtol=0, atol=0)
        else:
            for tensor in ttnn.get_device_tensors(first):
                torch.testing.assert_close(ttnn.to_torch(tensor), x, rtol=0, atol=0)
