import ttnn, torch
from ttnn.operations.nomic_residual_layer_norm import input_memory_config

dev = ttnn.open_device(device_id=0)
try:
    mc = input_memory_config((8, 1, 512, 768), dev)
    s = mc.nd_shard_spec
    print(mc)
    print([a for a in dir(s) if not a.startswith("_")])
    print(s.shard_shape, s.orientation, s.shard_distribution_strategy, s.grid)
    print([a for a in dir(dev) if "bank" in a.lower() or "alloc" in a.lower()])
    print(dev.compute_with_storage_grid_size())
    t = ttnn.from_torch(
        torch.zeros(1, 1, 2240, 768, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    print(t.buffer_page_size(), t.buffer_aligned_page_size(), t.buffer_num_pages())
    print([a for a in dir(t) if "bank" in a.lower() or "buffer" in a.lower()])
    print(ttnn.CBFormatDescriptor.__init__.__doc__)
finally:
    ttnn.close_device(dev)
