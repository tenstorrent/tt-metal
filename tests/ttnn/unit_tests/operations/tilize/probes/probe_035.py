import torch, ttnn
from ttnn.operations.tilize import tilize
from eval.golden_tests.tilize import helpers


def crs(x0, y0, x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})


device = ttnn.open_device(device_id=0)
try:
    # golden retile scenario 3: BLOCK-sharded, same spec, 32 -> 16
    sh = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs(0, 0, 7, 7), (32, 32), ttnn.ShardOrientation.ROW_MAJOR),
    )
    pairs = [
        (ttnn.bfloat16, ttnn.bfloat16),
        (ttnn.float32, ttnn.float32),
        (ttnn.bfloat16, ttnn.float32),
        (ttnn.float32, ttnn.bfloat16),
        (ttnn.bfloat16, ttnn.bfloat8_b),
        (ttnn.uint32, ttnn.uint32),
        (ttnn.uint8, ttnn.uint8),
        (ttnn.uint16, ttnn.uint16),
        (ttnn.int32, ttnn.int32),
    ]
    for dt, odt in pairs:
        for tag, shape, mc, a, b in [
            ("sharded", [1, 1, 256, 256], sh, 32, 16),
            ("dram", [1, 1, 128, 256], ttnn.DRAM_MEMORY_CONFIG, 32, 8),
            ("dram", [1, 1, 128, 256], ttnn.DRAM_MEMORY_CONFIG, 1, 32),
        ]:
            try:
                x = helpers.make_torch_input(dt, shape)
                tt = helpers.create_ttnn_input_tensor(
                    x, device, dtype=dt, memory_config=mc, layout=ttnn.TILE_LAYOUT, tile_height=a
                )
                out = tilize(tt, memory_config=mc, dtype=odt, use_multicore=True, tile=ttnn.Tile([b, 32]))
                got = ttnn.to_torch(out)
                exp = x.to(helpers._COMPARE_TORCH_DTYPE[odt])
                mode, thr = helpers._transition_tolerance(dt, odt)
                helpers.check_identity(got, exp, mode=mode, threshold=thr)
                print(f"OK   retile {tag} {a}->{b} {dt}->{odt}")
            except Exception as e:
                print(f"FAIL retile {tag} {a}->{b} {dt}->{odt}: {type(e).__name__}: {str(e)[:150]}")
finally:
    ttnn.close_device(device)
