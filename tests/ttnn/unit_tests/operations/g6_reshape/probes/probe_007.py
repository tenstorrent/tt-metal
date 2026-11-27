import importlib, torch, ttnn
from eval.golden_tests.g6_reshape.helpers import make_torch_input, pytorch_g6_reshape, _bits

m = importlib.import_module("ttnn.operations.g6_reshape.g6_reshape")
m.SUPPORTED = {"dtype": [ttnn.bfloat16, ttnn.float32, ttnn.int32, ttnn.uint32], "layout": [ttnn.TILE_LAYOUT]}
device = ttnn.open_device(device_id=0)
cases = [
    ((1, 512), (1, 1, 512, 1), ttnn.float32, None, None),
    ((1, 512), (1, 1, 512, 1), ttnn.float32, "special_values", None),
    ((1, 256), (1, 1, 256, 1), ttnn.float32, None, None),
    ((1, 196), (1, 1, 196, 1), ttnn.float32, None, None),
    ((1, 185), (1, 1, 185, 1), ttnn.float32, None, float("nan")),
    ((1, 207), (1, 1, 207, 1), ttnn.float32, None, None),
    ((1, 16, 6, 256), (1, 1, 16, 1536), ttnn.float32, None, float("nan")),
    ((16, 1, 5120), (1, 1, 16, 5120), ttnn.int32, None, None),
    ((1, 4096), (4096, 1), ttnn.int32, "int32_extremes", None),
    ((1, 1, 1, 32), (32, 1), ttnn.uint32, "uint32_high_bit", None),
]
ok = 0
for src, dst, dt, gen, poison in cases:
    try:
        x = make_torch_input(dt, src, gen)
        t = ttnn.from_torch(
            x,
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            preserve_nan_values=True,
        )
        if poison is not None:
            t = ttnn.fill_implicit_tile_padding(t, poison)
        plan = m.describe_plan(t, list(dst))
        got = ttnn.to_torch(m.g6_reshape(t, list(dst)))
        exp = pytorch_g6_reshape(x, dst)
        bad = int((_bits(got) != _bits(exp)).sum())
        ok += bad == 0
        print(
            "CASE",
            src,
            dst,
            dt,
            gen,
            poison,
            plan["regime"],
            "atom_rows",
            plan["atom_rows"],
            "cores",
            plan["num_cores"],
            "mismatches",
            bad,
        )
    except Exception as e:
        print("ERR", src, dst, dt, repr(e)[:300])
print("SUMMARY", ok, "/", len(cases))
ttnn.close_device(device)
