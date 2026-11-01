import ttnn, itertools
from eval.golden_tests.rms_norm import feature_spec as fs
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd
from ttnn.operations.rms_norm.rms_norm import default_compute_kernel_config

device = ttnn.open_device(device_id=0)
HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
cells = set()
for (shape,) in fs.INPUTS:
    for layout, dtype, (gd, gl) in itertools.product(
        [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
        [(None, None)]
        + [
            (a, b)
            for a in (ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b)
            for b in (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT)
        ],
    ):
        if dtype == ttnn.bfloat8_b and (layout == ttnn.ROW_MAJOR_LAYOUT or shape[-1] % 32 or shape[-2] % 32):
            continue
        if gd == ttnn.bfloat8_b and gl == ttnn.ROW_MAJOR_LAYOUT:
            continue
        if layout == ttnn.ROW_MAJOR_LAYOUT and gl == ttnn.TILE_LAYOUT:
            continue
        cells.add((tuple(shape), layout, dtype, gd, gl, "cart"))
for c in fs.LOOSE_CASES:
    if c.get("memory_layout") == HS:
        gd = c["gamma_dtype"] if c["gamma_mode"] == "gamma" else None
        gl = c["gamma_layout"] if gd is not None else None
        cells.add((tuple(c["inputs"][0]), c["layout"], c["dtype"], gd, gl, "loose"))
fails, allocf, n = 0, 0, 0
for shape, layout, dtype, gd, gl, src in sorted(cells, key=str):
    n += 1
    try:
        mc = auto_shard_config(list(shape), HS, layout=layout, dtype=dtype, device=device)
    except Exception as e:
        print("SKIP shardcfg", shape, layout, dtype, str(e)[:80])
        continue
    x = out = g = None
    try:
        x = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, layout, device, mc)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, layout, device, mc)
    except Exception as e:
        allocf += 1
        print("ALLOC", src, shape, layout, dtype, mc.shard_spec.shape)
        for t in (x, out):
            if t is not None:
                ttnn.deallocate(t)
        continue
    try:
        if gd is not None:
            g = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, 1, shape[-1]]), gd, gl, device, ttnn.DRAM_MEMORY_CONFIG
            )
        pd.create_program_descriptor(x, g, out, epsilon=1e-6, compute_kernel_config=default_compute_kernel_config())
    except Exception as e:
        fails += 1
        print("FAIL", src, shape, layout, dtype, gd, gl, mc.shard_spec.shape, str(e).splitlines()[0][:70])
    for t in (x, out, g):
        if t is not None:
            ttnn.deallocate(t)
print("SUM fails", fails, "alloc", allocf, "/", n)
ttnn.close_device(device)
