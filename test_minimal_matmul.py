import pytest
import ttnn
import torch


@pytest.mark.parametrize(
    "device_params",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            }
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_reduce_scatter(mesh_device):
    torch.manual_seed(123)

    memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )

    input_tensors = 32 * [torch.load(f"input/tensor_0.pt")]
    input_tensor = ttnn.from_host_shards(
        [
            ttnn.from_torch(
                tensor,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory_config,
            )
            for tensor in input_tensors
        ],
        mesh_shape=mesh_device.shape,
    ).to(mesh_device)

    input_tensors = 32 * [torch.rand(1, 1, 1024, 2048)]

    weight_tensor = ttnn.from_host_shards(
        [
            ttnn.from_torch(
                tensor,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory_config,
            )
            for tensor in input_tensors
        ],
        mesh_shape=mesh_device.shape,
    ).to(mesh_device)

    minimal_matmul_program_config = ttnn.MinimalMatmulConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(7, 8),
        M_block_size=8,
        K_block_size=8,
        N_block_size=8,
        subblock_h=4,
        subblock_w=2,
    )

    linear_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(7, 10),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=2,
        per_core_M=4,
        per_core_N=10,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    # Test that input tensor is correctly replicated across users
    reshaped_tensor = ttnn.reshape(input_tensor, (1, 32, 128, 1024))
    reshaped_tensor_torch = ttnn.to_torch(ttnn.get_device_tensors(reshaped_tensor)[0]).float()
    for b in range(1, reshaped_tensor_torch.shape[1]):
        assert torch.allclose(
            reshaped_tensor_torch[:, 0, :, :], reshaped_tensor_torch[:, b, :, :], atol=1e-6
        ), f"User {b} does not match user 0"

    output_tensor = ttnn.experimental.minimal_matmul(
        input_tensor=input_tensor,
        weight_tensor=weight_tensor,
        config=minimal_matmul_program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Using ttnn.linear makes the test pass
    # output_tensor = ttnn.linear(
    #     input_tensor,
    #     weight_tensor,
    #     compute_kernel_config=compute_kernel_config,
    #     dtype=ttnn.bfloat8_b,
    #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
    #     program_config=linear_program_config,
    # )

    reshaped_tensor = ttnn.reshape(output_tensor, (1, 32, 128, 2048))
    reshaped_tensor_torch = ttnn.to_torch(ttnn.get_device_tensors(reshaped_tensor)[0]).float()
    for b in range(1, reshaped_tensor_torch.shape[1]):
        assert torch.allclose(
            reshaped_tensor_torch[:, 0, :, :], reshaped_tensor_torch[:, b, :, :], atol=1e-6
        ), f"User {b} does not match user 0"
