import torch, ttnn

dev = ttnn.open_device(device_id=0)
grid22 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})

for tensor_shape, shard_shape in [
    ([4, 128, 128], [2, 64, 64]),
    ([3, 160, 160], [2, 64, 64]),
    ([5, 4, 160, 160], [2, 3, 64, 96]),
    ([23, 96, 160], [4, 64, 96]),
]:
    nd = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.L1,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=shard_shape, grid=grid22, orientation=ttnn.ShardOrientation.ROW_MAJOR
        ),
    )
    t = ttnn.from_torch(
        torch.rand(tensor_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=nd,
    )
    from math import prod

    padded = list(t.padded_shape)
    folded = prod(padded[:-1])
    W = padded[-1]
    print(f"shape={tensor_shape} shard={shard_shape}")
    print(f"   padded_shape={padded} folded_H={folded} W={W}")
    print(
        f"   page_size={t.buffer_page_size()} aligned={t.buffer_aligned_page_size()} num_pages={t.buffer_num_pages()}"
    )
    print(f"   row_bytes_full={W*2}  ->  row_bytes%page = {(W*2) % t.buffer_page_size()}")
    mc = t.memory_config()
    print(f"   mc.shard_spec={mc.shard_spec} nd={mc.nd_shard_spec}")
    print(
        f"   expected pages if row-major sticks: {folded} * {-(-W*2//t.buffer_page_size())} = {folded * (-(-W*2//t.buffer_page_size()))}"
    )
ttnn.close_device(dev)
