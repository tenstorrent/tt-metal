import ttnn, itertools
from eval.golden_tests.rms_norm import feature_spec as fs
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd
from ttnn.operations.rms_norm.rms_norm import default_compute_kernel_config

device = ttnn.open_device(device_id=0)
shapes = sorted({tuple(i[0]) for i in fs.INPUTS} | {tuple(c["inputs"][0]) for c in fs.LOOSE_CASES})
HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
fails, n = 0, 0
for shape in shapes:
    for layout, dtype, (gd, gl) in itertools.product(
        [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
        [(None, None), (ttnn.float32, ttnn.TILE_LAYOUT), (ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)],
    ):
        if dtype == ttnn.bfloat8_b and (layout == ttnn.ROW_MAJOR_LAYOUT or shape[-1] % 32 or shape[-2] % 32):
            continue
        if layout == ttnn.ROW_MAJOR_LAYOUT and gl == ttnn.TILE_LAYOUT:
            continue
        try:
            mc = auto_shard_config(list(shape), HS, layout=layout, dtype=dtype, device=device)
        except Exception as e:
            print("SKIP shardcfg", shape, layout, dtype, str(e)[:80])
            continue
        n += 1
        x = out = g = None
        try:
            x = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, layout, device, mc)
            out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, layout, device, mc)
            if gd is not None:
                g = ttnn.allocate_tensor_on_device(
                    ttnn.Shape([1, 1, 1, shape[-1]]), gd, gl, device, ttnn.DRAM_MEMORY_CONFIG
                )
            pd.create_program_descriptor(x, g, out, epsilon=1e-6, compute_kernel_config=default_compute_kernel_config())
        except Exception as e:
            fails += 1
            print("FAIL", shape, layout, dtype, gd, gl, mc.shard_spec.shape, str(e).splitlines()[0][:120])
        for t in (x, out, g):
            if t is not None:
                ttnn.deallocate(t)
print("SUM", fails, "/", n)
ttnn.close_device(device)
