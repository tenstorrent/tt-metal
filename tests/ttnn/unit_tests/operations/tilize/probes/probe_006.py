import torch, ttnn
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)


def sh(scheme, grid, shp):
    return ttnn.MemoryConfig(scheme, ttnn.BufferType.L1, ttnn.ShardSpec(grid, shp, ttnn.ShardOrientation.ROW_MAJOR))


try:
    for shape, th in [([2, 3, 64, 64], 16), ([1, 1, 16384, 64], 8), ([1, 1, 4096, 64], 1), ([1, 1, 2048, 256], 2)]:
        x = torch.randn(shape).to(torch.bfloat16)
        t = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        o = tilize(t, tile=ttnn.Tile([th, 32]))
        y = ttnn.to_torch(o)
        print(shape, th, list(o.tile.tile_shape), torch.equal(y, x))
    g = device.compute_with_storage_grid_size()
    cases = [
        ([1, 1, 32, 1024], ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.num_cores_to_corerangeset(32, g, row_wise=True)),
        (
            [1, 1, 1024, 32],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.num_cores_to_corerangeset(32, g, row_wise=True),
        ),
        (
            [1, 1, 256, 256],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
        ),
    ]
    for shape, scheme, grid in cases:
        mc = sh(scheme, grid, [32, 32])
        for th in (16, 8, 4, 2, 1):
            x = torch.rand(shape).to(torch.bfloat16)
            t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
            o = tilize(t, memory_config=mc, tile=ttnn.Tile([th, 32]))
            y = ttnn.to_torch(o)
            print(shape, scheme, th, list(o.tile.tile_shape), o.memory_config().memory_layout, torch.equal(y, x))
    # crossover DRAM -> height sharded out
    x = torch.randn([1, 1, 1024, 64]).to(torch.bfloat16)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    mc = sh(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.num_cores_to_corerangeset(32, g, row_wise=True), [32, 64])
    o = tilize(t, memory_config=mc, tile=ttnn.Tile([8, 32]))
    print("xover", torch.equal(ttnn.to_torch(o), x))
finally:
    ttnn.close_device(device)
