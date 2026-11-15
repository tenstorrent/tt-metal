import time, torch, ttnn

device = ttnn.open_device(device_id=0)
t = ttnn.from_torch(torch.zeros(1, 128, 2560), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def bench(name, fn, n=5000):
    try:
        fn()
        s = time.perf_counter()
        for _ in range(n):
            fn()
        print(f"COST {name}: {(time.perf_counter()-s)/n*1e6:.2f} us -> {fn()!r}"[:200])
    except Exception as e:
        print("COST", name, "ERR", e)


bench("tuple(shape)", lambda: tuple(t.shape))
bench("shape.to_list?", lambda: t.shape.to_list())
bench("str(shape)", lambda: str(t.shape))
bench("repr(shape)", lambda: repr(t.shape))
bench("logical_shape", lambda: tuple(t.logical_shape()))
bench("shape idx", lambda: (t.shape[0], t.shape[1], t.shape[2]))
bench("hash memcfg", lambda: hash(t.memory_config()))
print("SHAPE attrs", [a for a in dir(t.shape) if not a.startswith("__")])
ttnn.close_device(device)
