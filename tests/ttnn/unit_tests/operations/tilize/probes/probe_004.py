import torch, ttnn
from math import prod

dev = ttnn.open_device(device_id=0)
grid22 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})

cases = [
    ([4, 128, 128], [2, 64, 64], None),
    ([3, 160, 160], [2, 64, 64], None),
    ([5, 4, 160, 160], [2, 3, 64, 96], None),
    ([23, 96, 160], [4, 64, 96], None),
    ([4, 128, 128], [2, 64, 64], [1, 64, 128]),
    ([3, 160, 160], [2, 64, 64], [1, 64, 96]),
    ([5, 4, 160, 160], [2, 3, 64, 96], [3, 2, 96, 64]),
    ([23, 96, 160], [4, 64, 96], [6, 64, 64]),
]
for shape, sh, osh in cases:
    osh = osh or sh
    nd = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.L1,
        nd_shard_spec=ttnn.NdShardSpec(shard_shape=osh, grid=grid22, orientation=ttnn.ShardOrientation.ROW_MAJOR),
    )
    try:
        ot = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, nd)
        p = list(ot.padded_shape)
        print(
            f"OUT shape={shape} oshard={osh}: padded={p} folded={prod(p[:-1])} W={p[-1]} "
            f"page={ot.buffer_page_size()} npages={ot.buffer_num_pages()} "
            f"leading_exact={p[:-1]==list(shape)[:-1]}"
        )
    except Exception as e:
        print(f"OUT shape={shape} oshard={osh}: ALLOC FAIL {type(e).__name__}: {str(e)[:160]}")
ttnn.close_device(dev)
