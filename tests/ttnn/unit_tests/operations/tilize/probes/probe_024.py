# Does an ND MemoryConfig survive allocation as ND, and is the alias correct for it?
import torch, ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as tpd

device = ttnn.open_device(device_id=0)
try:

    def crs(*rng):
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*s), ttnn.CoreCoord(*e)) for s, e in rng})

    def nd(grid, shape):
        return ttnn.MemoryConfig(
            ttnn.BufferType.L1, ttnn.NdShardSpec(ttnn.Shape(list(shape)), grid, ttnn.ShardOrientation.ROW_MAJOR)
        )

    cases = [
        ("nd_2d_out", (1, 1, 128, 128), None, nd(crs(((0, 0), (1, 1))), (1, 1, 64, 64))),
        ("nd_2d_in", (1, 1, 128, 128), nd(crs(((0, 0), (1, 1))), (1, 1, 64, 64)), None),
        ("nd_rank3_out", (4, 32, 64), None, nd(crs(((0, 0), (1, 0))), (2, 32, 64))),
        ("nd_rank3_in", (4, 32, 64), nd(crs(((0, 0), (1, 0))), (2, 32, 64)), None),
        ("nd_split_batch_out", (2, 64, 64), None, nd(crs(((0, 0), (1, 1))), (1, 32, 64))),
        ("nd_split_batch_in", (2, 64, 64), nd(crs(((0, 0), (1, 1))), (1, 32, 64)), None),
    ]
    for name, shape, in_cfg, out_cfg in cases:
        in_cfg = in_cfg or ttnn.DRAM_MEMORY_CONFIG
        out_cfg = out_cfg or ttnn.DRAM_MEMORY_CONFIG
        n = 1
        for d in shape:
            n *= d
        t = (torch.arange(n, dtype=torch.float32).reshape(shape) % 4096).bfloat16()
        try:
            tt_in = ttnn.from_torch(
                t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_cfg
            )
            probe = ttnn.allocate_tensor_on_device(
                ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, out_cfg
            )
            plan = tpd.build_plan(tt_in, probe, device)
            got = ttnn.to_torch(tilize(tt_in, out_cfg))
            ok = torch.equal(got.float(), t.float())
            src = tt_in if in_cfg is not ttnn.DRAM_MEMORY_CONFIG else probe
            mc = src.memory_config()
            print(
                f"{name:22s} path={plan['path']:10s} layout_after_alloc={mc.memory_layout} "
                f"legacy_spec={'yes' if mc.shard_spec is not None else 'no'} EXACT={ok}"
            )
        except Exception as exc:
            print(f"{name:22s} EXC {type(exc).__name__}: {str(exc)[:120]}")
finally:
    ttnn.close_device(device)
