import time, torch, ttnn, ttml

dev = ttml.autograd.AutoContext.get_instance().get_device()  # chip 0


def timeit(name, fn, reps=20):
    fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(dev)
    print(f"T: {name:55s} {(time.perf_counter()-t0)*1e3/reps:8.3f} ms", flush=True)


for rows, cols, off in (
    (5184, 65536, 5120 // 32),
    (5184, 6144, 160),
    (4160, 5120, 128),
    (5184, 16384, 160),
    (8256, 5120, 256),
):
    w0 = (torch.randn(1, 1, rows, cols) * 0.1).to(torch.bfloat16)
    w = ttnn.from_torch(w0, layout=ttnn.TILE_LAYOUT, device=dev)
    tail0 = (torch.arange(32 * cols, dtype=torch.float32).reshape(1, 1, 32, cols) % 251).to(torch.bfloat16)
    tail = ttnn.from_torch(tail0, layout=ttnn.TILE_LAYOUT, device=dev)
    out = ttml.ops.metal.write_tile_block(w, tail, off)
    back = ttnn.to_torch(w)
    exp = w0.clone()
    exp[0, 0, off * 32 : off * 32 + 32] = tail0[0, 0]
    print(
        f"R: [{rows},{cols}] offset {off}: exact {torch.equal(back, exp)} same buffer {out.buffer_address() == w.buffer_address()}",
        flush=True,
    )
    timeit(f"write_tile_block [{rows},{cols}]", lambda: ttml.ops.metal.write_tile_block(w, tail, off))
    timeit(
        f"fill_cache [{rows},{cols}]", lambda: ttnn.fill_cache(ttnn.reshape(w, [rows // 32, 1, 32, cols]), tail, off)
    )
    # second write (different data) also lands, program cache hit
    tail1 = ttnn.from_torch(tail0 + 1, layout=ttnn.TILE_LAYOUT, device=dev)
    ttml.ops.metal.write_tile_block(w, tail1, off)
    back = ttnn.to_torch(w)
    exp[0, 0, off * 32 : off * 32 + 32] = (tail0 + 1)[0, 0]
    print(f"R: rewrite exact {torch.equal(back, exp)}", flush=True)
