"""
Test matmul_reduce_scatter_async with cluster_axis on 4x8 galaxy.

Validates that the fused MM+RS op with cluster_axis=1 produces correct
results when reduce-scattering along columns (8 devices per ring).
"""

import torch
import pytest
import ttnn
from loguru import logger


def run_mm_rs_cluster_axis(mesh_device, cluster_axis=1):
    num_devices = mesh_device.shape[cluster_axis]
    B, K, N = 128, 256, 256
    scatter_dim = 3

    torch.manual_seed(42)
    input_torch = torch.randn(1, 1, B, K, dtype=torch.bfloat16)
    weight_torch = torch.randn(1, 1, K, N, dtype=torch.bfloat16)

    # Reference: matmul then chunk (simulates reduce-scatter)
    mm_ref = input_torch @ weight_torch
    # All devices have same input (replicated), so reduce = N * single
    rs_ref_chunks = torch.chunk(mm_ref * num_devices, num_devices, dim=scatter_dim)

    tt_input = ttnn.from_torch(
        input_torch,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_weight = ttnn.from_torch(
        weight_torch,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    persistent_intermediate = ttnn.from_torch(
        torch.zeros(1, 1, B, N, dtype=torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    persistent_output = ttnn.from_torch(
        torch.zeros(1, 1, B, N // num_devices, dtype=torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    sems = [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(3)]

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(4, 4),
        in0_block_w=K // 32,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=B // 32 // 4,
        per_core_N=N // 32 // 4,
        transpose_mcast=False,
    )
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
    )

    mm_out, rs_out = ttnn.experimental.matmul_reduce_scatter_async(
        tt_input,
        tt_weight,
        persistent_intermediate,
        persistent_output,
        scatter_dim,
        sems,
        (0, 4),
        topology=ttnn.Topology.Ring,
        cluster_axis=cluster_axis,
        program_config=program_config,
        compute_kernel_config=compute_config,
        memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
        memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.synchronize_device(mesh_device)

    rs_out_torch = ttnn.to_torch(rs_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    all_pass = True
    for dev_idx in range(mesh_device.get_num_devices()):
        col_idx = dev_idx % num_devices
        device_out = rs_out_torch[dev_idx]
        ref = rs_ref_chunks[col_idx]

        ref_flat = ref.float().flatten()
        out_flat = device_out.float().flatten()
        pcc = torch.corrcoef(torch.stack([ref_flat, out_flat]))[0, 1].item() if ref_flat.std() > 0 else 1.0

        status = "PASS" if pcc > 0.98 else "FAIL"
        if pcc <= 0.98:
            all_pass = False
        logger.info(f"Device {dev_idx} (col {col_idx}): PCC={pcc:.6f} {status}")

    return all_pass


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((4, 8), id="4x8")], indirect=True)
def test_mm_rs_cluster_axis(mesh_device, device_params):
    """Test matmul_reduce_scatter_async with cluster_axis=1 on 4x8 mesh."""
    assert run_mm_rs_cluster_axis(mesh_device, cluster_axis=1)
