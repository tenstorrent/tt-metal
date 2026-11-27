import torch, ttnn
from ttnn.operations.g6_fill.g6_fill import fill_bits_of
from ttnn.operations.g6_fill import g6_fill_plan

dev = ttnn.open_device(device_id=0)
try:
    for n in (512, 207, 1):
        t = ttnn.from_torch(
            torch.zeros(n),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        print(
            n,
            t.buffer_num_pages(),
            t.buffer_aligned_page_size(),
            t.padded_shape,
            t.tile.tile_shape,
            t.tile.face_shape,
            t.element_size(),
            t.storage_type(),
            t.device().id(),
        )
    print(t.memory_config() == ttnn.DRAM_MEMORY_CONFIG, t.memory_config() == ttnn.L1_MEMORY_CONFIG)
    print(g6_fill_plan(t))
    h = ttnn.from_torch(torch.zeros(4), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    print(h.storage_type())
finally:
    ttnn.close_device(dev)
for v in [
    1 / 3,
    -0.0,
    1e-40,
    float("inf"),
    float("-inf"),
    float("nan"),
    3.4028234663852886e38,
    16777217,
    2**60 + 2**36 + 1,
    0.1,
    1.0,
    -(2**100) - 12345,
    int(3.4028234663852886e38),
]:
    src = torch.float64 if isinstance(v, float) else torch.int64
    try:
        want = torch.tensor(v, dtype=src).to(torch.float32).view(torch.int32).item() & 0xFFFFFFFF
    except RuntimeError as e:
        # int64 overflow; compare against exact via python fractions
        want = None
    got = fill_bits_of(ttnn.float32, v)
    print(repr(v)[:30], hex(got), hex(want) if want is not None else None, got == want)
