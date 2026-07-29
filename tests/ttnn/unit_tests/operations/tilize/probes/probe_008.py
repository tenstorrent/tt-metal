"""Program-cache re-binding on the zero-copy (aliased-CB) sharded path.

The CB base address IS the shard base address. On a program-cache hit the
program is reused, so if the CB address were not re-patched from the tensor the
second call would tilize/write the WRONG L1 region.
"""
import torch, ttnn
from ttnn.operations.tilize import tilize

dev = ttnn.open_device(device_id=0)
try:
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (128, 64), ttnn.ShardOrientation.ROW_MAJOR),
    )
    shape = (1, 1, 512, 64)

    def run(seed):
        torch.manual_seed(seed)
        t = torch.randn(shape).bfloat16()
        x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc)
        y = tilize(x)
        return t, x, y

    t1, x1, y1 = run(1)
    print(
        "call1 exact:",
        torch.equal(ttnn.to_torch(y1), t1),
        "in_addr",
        hex(x1.buffer_address()),
        "out_addr",
        hex(y1.buffer_address()),
    )
    entries1 = dev.num_program_cache_entries()

    # Keep y1 alive so the allocator hands the second call DIFFERENT addresses.
    t2, x2, y2 = run(2)
    entries2 = dev.num_program_cache_entries()
    print(
        "call2 exact:",
        torch.equal(ttnn.to_torch(y2), t2),
        "in_addr",
        hex(x2.buffer_address()),
        "out_addr",
        hex(y2.buffer_address()),
    )
    print("cache entries:", entries1, "->", entries2, "(hit)" if entries1 == entries2 else "(MISS)")
    print("addresses differ:", x1.buffer_address() != x2.buffer_address() or y1.buffer_address() != y2.buffer_address())
    print("call1 still exact after call2:", torch.equal(ttnn.to_torch(y1), t1))
finally:
    ttnn.close_device(dev)
