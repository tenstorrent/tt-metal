import pytest
import torch
import ttnn

from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    SDXL_L1_SMALL_SIZE,
    SDXL_TRACE_REGION_SIZE,
    SDXL_FABRIC_CONFIG,
    MAX_SEQUENCE_LENGTH,
    TEXT_ENCODER_2_PROJECTION_DIM,
    CONCATENATED_TEXT_EMBEDINGS_SIZE,
)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": SDXL_L1_SMALL_SIZE,
            "trace_region_size": SDXL_TRACE_REGION_SIZE,
            "fabric_config": SDXL_FABRIC_CONFIG,
        },
    ],
    indirect=["device_params"],
)
def test_hello_world_mesh(mesh_device):
    """Hello world example for mesh device with tensor sharding"""

    # Print mesh topology
    ttnn.visualize_mesh_device(mesh_device)

    # Create dummy tensor with batch dimension
    batch_size = 32 * mesh_device.get_num_devices()  # Ensure divisible by device count
    torch_tensor = torch.randn(batch_size, 1, 64, 64, dtype=torch.bfloat16)

    # Shard tensor across batch dimension (dim=0)
    sharded_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    # Visualize the sharded tensor distribution
    print("\n=== Tensor Sharding Visualization ===")
    ttnn.visualize_tensor(sharded_tensor)

    # Run all-gather to collect all shards on each device
    # This concatenates along dim=3, making each device have the full tensor
    gathered_tensor = ttnn.all_gather(sharded_tensor, dim=3, num_links=4, topology=ttnn.Topology.Ring)

    print(f"{gathered_tensor.shape=}")

    # Visualize the gathered tensor
    print("\n=== Gathered Tensor Visualization ===")
    ttnn.visualize_tensor(gathered_tensor)
