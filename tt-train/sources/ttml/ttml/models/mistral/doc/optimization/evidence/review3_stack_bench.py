import time, torch, ttnn

dev = ttnn.open_device(device_id=0)


def timeit(name, fn, reps=20):
    fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(dev)
    print(f"T: {name:70s} {(time.perf_counter()-t0)*1e3/reps:8.3f} ms", flush=True)


n = 8
dA = [ttnn.from_torch(torch.randn(1, 1, 16, 5120).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev) for _ in range(n)]
dB = [ttnn.from_torch(torch.randn(1, 1, 5120, 16).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev) for _ in range(n)]


# current: concat the 16-row/col tensors, slice back 16 each
def cur_A():
    c = ttnn.concat(dA, dim=2)
    return [ttnn.slice(c, [0, 0, 16 * i, 0], [1, 1, 16 * (i + 1), 5120]) for i in range(n)]


def cur_B():
    c = ttnn.concat(dB, dim=3)
    return [ttnn.slice(c, [0, 0, 0, 16 * i], [1, 1, 5120, 16 * (i + 1)]) for i in range(n)]


# aligned: pad each to 32, concat (aligned), slice aligned 32-blocks, then unpad to 16
def al_A():
    c = ttnn.concat([ttnn.pad(t, padding=[(0, 0), (0, 0), (0, 16), (0, 0)], value=0.0) for t in dA], dim=2)
    return [
        ttnn.slice(ttnn.slice(c, [0, 0, 32 * i, 0], [1, 1, 32 * (i + 1), 5120]), [0, 0, 0, 0], [1, 1, 16, 5120])
        for i in range(n)
    ]


def al_B():
    c = ttnn.concat([ttnn.pad(t, padding=[(0, 0), (0, 0), (0, 0), (0, 16)], value=0.0) for t in dB], dim=3)
    return [
        ttnn.slice(ttnn.slice(c, [0, 0, 0, 32 * i], [1, 1, 5120, 32 * (i + 1)]), [0, 0, 0, 0], [1, 1, 5120, 16])
        for i in range(n)
    ]


# aligned, no unpad: slice the 32-block only (if the optimizer accepted a padded grad)
def al_A32():
    c = ttnn.concat([ttnn.pad(t, padding=[(0, 0), (0, 0), (0, 16), (0, 0)], value=0.0) for t in dA], dim=2)
    return [ttnn.slice(c, [0, 0, 32 * i, 0], [1, 1, 32 * (i + 1), 5120]) for i in range(n)]


for name, fn in [
    ("current A: concat 8x[16,5120] + 8 slices of 16 rows", cur_A),
    ("current B: concat 8x[5120,16] + 8 slices of 16 cols", cur_B),
    ("aligned A: 8 pads to 32 + concat + 8 aligned slices + 8 unpads", al_A),
    ("aligned B: same on columns", al_B),
    ("aligned A without the final unpad", al_A32),
]:
    timeit(name, fn)
r = cur_A()
a = al_A()
print("R: aligned A equals current A:", all(torch.equal(ttnn.to_torch(x), ttnn.to_torch(y)) for x, y in zip(r, a)))
r = cur_B()
a = al_B()
print("R: aligned B equals current B:", all(torch.equal(ttnn.to_torch(x), ttnn.to_torch(y)) for x, y in zip(r, a)))
ttnn.close_device(dev)
