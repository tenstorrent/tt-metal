import time, torch, ttnn

device = ttnn.open_device(device_id=0)
t = ttnn.from_torch(torch.zeros(1, 128, 2560), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def bench(name, fn, n=2000):
    fn()
    s = time.perf_counter()
    for _ in range(n):
        fn()
    print(f"COST {name}: {(time.perf_counter()-s)/n*1e6:.2f} us")


bench("shape", lambda: t.shape)
bench("shape tuple", lambda: tuple(t.shape))
bench("shape int tuple", lambda: tuple(int(d) for d in t.shape))
bench("dtype", lambda: t.dtype)
bench("layout", lambda: t.layout)
bench("memory_config", lambda: t.memory_config())
for name, f in (
    ("hash memcfg", lambda: hash(t.memory_config())),
    ("spec", lambda: t.spec),
    ("hash spec", lambda: hash(t.spec)),
    ("hash dtype", lambda: hash(t.dtype)),
    ("hash shape", lambda: hash(t.shape)),
):
    try:
        bench(name, f)
    except Exception as e:
        print("COST", name, "ERR", e)
bench("buffer_address", lambda: t.buffer_address())
bench("device", lambda: t.device())
bench("device id", lambda: t.device().id())
cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)
print("CFG attrs", [a for a in dir(cfg) if not a.startswith("_")])
bench(
    "alloc",
    lambda: ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 128, 1536]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    ),
    n=500,
)
ttnn.close_device(device)
