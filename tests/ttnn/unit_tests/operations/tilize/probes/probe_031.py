import torch, ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as tpd

for shape in [(1, 1, 8192, 32), (1, 1, 4096, 64), (1, 1, 2048, 512), (1, 1, 2048, 32), (1, 1, 2048, 2048)]:
    t = torch.arange(shape[2] * shape[3], dtype=torch.float32).reshape(shape).bfloat16()
    d = ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    probe = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    plan = tpd.build_plan(d, probe, device)
    out = ttnn.to_torch(tilize(d))
    ok = torch.equal(out, t)
    print(
        f"{shape} bt={plan['bank_table']} b13={plan['stateful_read']} blk={plan['blocks_per_core']} "
        f"crb={plan['chunk_row_bytes']} cores={plan['ncores']} EQUAL={ok}"
    )
    assert ok, f"MISMATCH {shape}"

# crossover (alias_out) — the target regime
_L1 = ttnn.BufferType.L1
crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
mc = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.BLOCK_SHARDED, _L1, ttnn.ShardSpec(crs, (256, 64), ttnn.ShardOrientation.ROW_MAJOR)
)
shape = (1, 1, 2048, 512)
t = torch.arange(shape[2] * shape[3], dtype=torch.float32).reshape(shape).bfloat16()
d = ttnn.from_torch(
    t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
probe = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
plan = tpd.build_plan(d, probe, device)
out = ttnn.to_torch(tilize(d, mc))
print(
    f"alias_out bt={plan['bank_table']} b13={plan['stateful_read']} blk={plan['blocks_per_core']} "
    f"crb={plan['chunk_row_bytes']} path={plan['path']} EQUAL={torch.equal(out,t)}"
)
assert torch.equal(out, t)
print("ALL OK")
