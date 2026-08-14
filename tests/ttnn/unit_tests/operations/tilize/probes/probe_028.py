import torch, ttnn
from ttnn.operations.tilize import tilize
from eval.golden_tests.tilize import helpers


def crs(x0, y0, x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})


device = ttnn.open_device(device_id=0)


def run(tag, shape, th, dt, odt, in_mc, out_mc, pad=None):
    try:
        x = helpers.make_torch_input(dt, shape)
        tt = ttnn.from_torch(x, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mc)
        kw = {}
        if pad is not None:
            kw["pad_value"] = pad
        out = tilize(tt, memory_config=out_mc, dtype=odt, use_multicore=True, tile=ttnn.Tile([th, 32]), **kw)
        cmp_dtype = helpers._COMPARE_TORCH_DTYPE[odt]
        if pad is None:
            got = ttnn.to_torch(out)
            exp = x.to(cmp_dtype)
        else:
            got = out.cpu().to_torch_with_padded_shape()
            tgt = list(shape)
            tgt[-2] = ((tgt[-2] + th - 1) // th) * th
            tgt[-1] = ((tgt[-1] + 31) // 32) * 32
            exp = helpers.pad_expected(x.to(cmp_dtype), tgt, pad)
        mode, thr = helpers._transition_tolerance(dt, odt)
        helpers.check_identity(got, exp, mode=mode, threshold=thr)
        print(f"OK   {tag} th={th} {dt}->{odt}")
    except Exception as e:
        print(f"FAIL {tag} th={th} {dt}->{odt}: {type(e).__name__}: {str(e)[:200]}")


D = ttnn.DRAM_MEMORY_CONFIG
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
try:
    for th in (16, 8, 1):
        for dt, odt in pairs:
            run("interleaved", [1, 1, 128, 256], th, dt, odt, D, D)
    # golden scenario 3: WIDTH-sharded L1 in and out, shard (32,32) on 32 cores, th=8
    g = crs(0, 0, 7, 3)
    sh = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(g, (32, 32), ttnn.ShardOrientation.ROW_MAJOR),
    )
    for dt, odt in pairs:
        run("width_sharded", [1, 1, 32, 1024], 8, dt, odt, sh, sh)
    # padded tiny tiles (not a golden cell but reachable): widening cast fill
    for th in (16, 4):
        run("padded", [1, 1, 50, 50], th, ttnn.bfloat16, ttnn.float32, D, D, pad=10.2)
        run("padded", [1, 1, 50, 50], th, ttnn.bfloat16, ttnn.bfloat16, D, D, pad=-3.5)
finally:
    ttnn.close_device(device)
