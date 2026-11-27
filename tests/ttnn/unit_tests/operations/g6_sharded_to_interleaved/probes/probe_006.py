# Knob-turn check: run the deferred TARGET cells through the same program, bypassing validate().
import math, torch, ttnn
from ttnn.operations.g6_sharded_to_interleaved import compute_plan
from ttnn.operations.g6_sharded_to_interleaved.g6_sharded_to_interleaved import _launch

TILE = 32


def make(shape):
    rows, cols = math.prod(shape[:-1]), shape[-1]
    x = torch.randn(rows, cols)
    t = (torch.arange(rows) // TILE)[:, None] * (-(-cols // TILE)) + (torch.arange(cols) // TILE)[None, :]
    return (x * torch.pow(2.0, (t % 16 - 8).float())).reshape(shape).to(torch.bfloat16)


cases = {
    "p1": (
        (1, 128, 256, 32),
        ttnn.bfloat8_b,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        (320, 32),
        [((0, 0), (10, 8)), ((0, 9), (3, 9))],
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.L1_MEMORY_CONFIG,
    ),
    "p1_other": (
        (1, 128, 256, 32),
        ttnn.bfloat8_b,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        (352, 32),
        [((0, 0), (8, 9)), ((9, 0), (9, 3))],
        ttnn.ShardOrientation.COL_MAJOR,
        ttnn.L1_MEMORY_CONFIG,
    ),
    "p2_to_l1": (
        (1, 5120),
        ttnn.bfloat16,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        (32, 160),
        [((0, 0), (7, 3))],
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.L1_MEMORY_CONFIG,
    ),
    "p1_bf16_dram": (
        (1, 128, 256, 32),
        ttnn.bfloat16,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        (320, 32),
        [((0, 0), (10, 8)), ((0, 9), (3, 9))],
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
}
device = ttnn.open_device(device_id=0)
try:
    for name, (shape, dt, ml, shard, grid, orient, out_mc) in cases.items():
        torch.manual_seed(0)
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*a), ttnn.CoreCoord(*b)) for a, b in grid])
        mc = ttnn.MemoryConfig(ml, ttnn.BufferType.L1, ttnn.ShardSpec(crs, shard, orient))
        src = make(shape)
        x = ttnn.from_torch(src, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        ref = src if dt == ttnn.bfloat16 else ttnn.to_torch(x)
        out = ttnn.allocate_tensor_on_device(x.shape, x.dtype, ttnn.TILE_LAYOUT, device, out_mc)
        out = _launch(x, out)
        got = ttnn.to_torch(out)
        diff = int((got.contiguous().view(torch.int16) != ref.contiguous().view(torch.int16)).sum())
        geo, plan = compute_plan(x)
        busy = sum(1 for a in plan if a.num_blocks)
        print(f"case {name}: differing={diff} of {ref.numel()} cores_with_block={busy}/{len(plan)} last={plan[-1]}")
        ttnn.deallocate(out)
        ttnn.deallocate(x)
finally:
    ttnn.close_device(device)
