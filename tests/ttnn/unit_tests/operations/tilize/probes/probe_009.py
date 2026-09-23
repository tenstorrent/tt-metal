import torch, ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as pd

device = ttnn.open_device(device_id=0)
g = device.compute_with_storage_grid_size()


def sh(scheme, grid, shp):
    return ttnn.MemoryConfig(scheme, ttnn.BufferType.L1, ttnn.ShardSpec(grid, shp, ttnn.ShardOrientation.ROW_MAJOR))


def run(tag, shape, ih, oh, in_mc=ttnn.DRAM_MEMORY_CONFIG, out_mc=None):
    x = torch.randn(shape).to(torch.bfloat16)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile([ih, 32]), device=device, memory_config=in_mc
    )
    o = tilize(t, memory_config=out_mc, tile=ttnn.Tile([oh, 32]))
    y = ttnn.to_torch(o)
    print(tag, shape, ih, oh, list(o.tile.tile_shape), o.memory_config().memory_layout, torch.equal(y, x))


try:
    for shape, ih, oh in [
        ([1, 1, 16384, 64], 32, 8),
        ([1, 1, 4096, 64], 1, 32),
        ([1, 1, 2048, 128], 16, 2),
        ([4, 3, 64, 64], 32, 4),
        ([2, 1, 48, 64], 32, 16),
        ([3, 2, 48, 96], 32, 8),
        ([1, 1, 4096, 64], 8, 32),
        ([2, 2, 96, 64], 16, 32),
    ]:
        run("R", shape, ih, oh)
    run("R", [1, 1, 256, 64], 32, 16, in_mc=ttnn.L1_MEMORY_CONFIG, out_mc=ttnn.L1_MEMORY_CONFIG)
    # sharded (translated-test style: 32x32 shards)
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
        for ih, oh in [(32, 16), (16, 32), (8, 1), (1, 8), (4, 2), (32, 1)]:
            run("S", shape, ih, oh, in_mc=mc, out_mc=mc)
    # crossovers: sharded TILE in -> DRAM out; DRAM TILE in -> height-sharded out (out shard 16 rows cuts input 32-tiles)
    mc = sh(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.num_cores_to_corerangeset(32, g, row_wise=True), [32, 64])
    run("S", [1, 1, 1024, 64], 32, 8, in_mc=mc, out_mc=ttnn.DRAM_MEMORY_CONFIG)
    mc16 = sh(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.num_cores_to_corerangeset(32, g, row_wise=True), [16, 64])
    run("S", [1, 1, 512, 64], 32, 16, in_mc=ttnn.DRAM_MEMORY_CONFIG, out_mc=mc16)
    run("S", [1, 1, 512, 64], 32, 8, in_mc=ttnn.DRAM_MEMORY_CONFIG, out_mc=mc16)
finally:
    ttnn.close_device(device)
