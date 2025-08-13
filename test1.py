import pytest
import torch
import ttnn
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_1D}),],
    indirect=True,
)
@pytest.mark.parametrize("shape", [(2,1,16384, 4)])
def test_dist(mesh_device, shape):
    t = torch.rand(shape)
    tt = ttnn.from_torch(t, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0))
    mem_config = ttnn.create_sharded_memory_config(
            shape=(2, 1, 16384, 32),
            core_grid=ttnn.CoreGrid(y=8, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
    tt = ttnn.to_memory_config(tt, mem_config)

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    sem = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)

    persistent_output_buffer = ttnn.from_torch(
        torch.zeros((2, 1, 16384, 32)),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=tt.dtype,
        memory_config=tt.memory_config(),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    persistent_output_buffer = ttnn.to_memory_config(persistent_output_buffer, mem_config)

    tt_all_gather_out_tensor = ttnn.experimental.all_gather_async(
        tt,
        persistent_output_buffer=persistent_output_buffer,
        dim=0,
        multi_device_global_semaphore=sem,
        num_links=1,
        memory_config=mem_config,
        topology=ttnn.Topology.Linear,
        subdevice_id=ttnn.SubDeviceId(0),
    )
    ttnn.synchronize_device(mesh_device)
    noise_pred_uncond, noise_pred_text = ttnn.unsqueeze(tt_all_gather_out_tensor[0], 0), ttnn.unsqueeze(tt_all_gather_out_tensor[1], 0)
    noise_pred = noise_pred_uncond + 5.0 * (noise_pred_text - noise_pred_uncond)
    print(f"noise_pred memory config: {noise_pred.memory_config()}")
    print(f"noise_pred layout: {noise_pred.layout}")
    print(f"noise_pred shape: {noise_pred.shape}")

    add_tensor = ttnn.from_torch(
        torch.zeros((1, 1, 16384, 32)),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=tt.dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    print(f"add_tensor shape: {add_tensor.shape}")
    print(f"add_tensor memory config: {add_tensor.memory_config()}")
    print(f"add_tensor layout: {add_tensor.layout}")
    tt = noise_pred - add_tensor
    print(f"tt shape: {tt.shape}")