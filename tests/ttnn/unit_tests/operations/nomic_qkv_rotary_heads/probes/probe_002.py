import torch, ttnn
from ttnn.operations.nomic_residual_layer_norm import input_memory_config as nim, output_memory_config as nom

device = ttnn.open_device(device_id=0)
try:
    print("grid", device.compute_with_storage_grid_size())
    for shape in [(2, 1, 512, 768), (3, 1, 400, 768), (8, 1, 512, 768), (4, 1, 512, 768), (3, 1, 352, 768)]:
        mc = nom(shape, device, nim(shape, device))
        spec = mc.nd_shard_spec
        print(
            shape,
            mc.memory_layout,
            mc.buffer_type,
            spec.shard_shape,
            spec.orientation,
            spec.shard_distribution_strategy,
            spec.grid.num_cores(),
        )
    shape = (3, 1, 400, 768)
    mc = nom(shape, device, nim(shape, device))
    x = ttnn.from_torch(
        torch.randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
    )
    print("addr", x.buffer_address(), x.padded_shape, x.buffer_page_size(), x.is_sharded())
    c = ttnn.CoreCoord(3, 2)
    print("noc", device.worker_core_from_logical_core(c))
    print([a for a in dir(device) if "core" in a.lower()])
    print([a for a in dir(ttnn) if "Mcast" in a or "mcast" in a])
    print(ttnn.DataMovementConfigDescriptor.__init__.__doc__)
    print(ttnn.ComputeConfigDescriptor.__init__.__doc__)
finally:
    ttnn.close_device(device)
