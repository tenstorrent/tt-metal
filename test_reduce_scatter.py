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
    input_tensors = []
    for i in range(32):
        input_tensors.append(torch.load(f"input/tensor_{i}.pt"))

    memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )

    host_shards = [
        ttnn.from_torch(
            tensor,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
        )
        for tensor in input_tensors
    ]

    distributed_tensor = ttnn.from_host_shards(
        host_shards,
        mesh_shape=mesh_device.shape,
    ).to(mesh_device)

    reduced_tensor = ttnn.reduce_scatter(
        distributed_tensor,
        dim=3,
        cluster_axis=1,
        memory_config=memory_config,
        num_links=3,
        subdevice_id=ttnn.SubDeviceId(0),
        topology=ttnn.Topology.Ring,  # setting ttnn.Topology.Linear makes the test pass
    )

    reshaped_tensor = ttnn.reshape(reduced_tensor, (1, 32, 128, 32))
    reshaped_tensor_torch = ttnn.to_torch(ttnn.get_device_tensors(reshaped_tensor)[0]).float()
    for b in range(1, reshaped_tensor_torch.shape[1]):
        assert torch.allclose(
            reshaped_tensor_torch[:, 0, :, :], reshaped_tensor_torch[:, b, :, :], atol=1e-6
        ), f"User {b} does not match user 0"
