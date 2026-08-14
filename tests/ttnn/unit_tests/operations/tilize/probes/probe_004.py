import torch, ttnn
from ttnn.operations.tilize import tilize


def crs(*ranges):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*s), ttnn.CoreCoord(*e)) for (s, e) in ranges})


_ROW = ttnn.ShardOrientation.ROW_MAJOR
_COL = ttnn.ShardOrientation.COL_MAJOR
_L1 = ttnn.BufferType.L1
H = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
W = ttnn.TensorMemoryLayout.WIDTH_SHARDED
B = ttnn.TensorMemoryLayout.BLOCK_SHARDED


def legacy(grid, sshape, orient, scheme):
    return ttnn.MemoryConfig(scheme, _L1, ttnn.ShardSpec(grid, sshape, orient))


def nd(grid, sshape, orient=_ROW):
    return ttnn.MemoryConfig(_L1, ttnn.NdShardSpec(ttnn.Shape(sshape), grid, orient))


DRAM = ttnn.DRAM_MEMORY_CONFIG

device = ttnn.open_device(device_id=0)
try:
    cases = [
        # nd same-spec (golden cells)
        (
            "nd 2x2 same",
            [1, 1, 128, 128],
            nd(crs(((0, 0), (1, 1))), (1, 1, 64, 64)),
            nd(crs(((0, 0), (1, 1))), (1, 1, 64, 64)),
        ),
        ("nd rank3 same", [4, 32, 64], nd(crs(((0, 0), (1, 0))), (2, 32, 64)), nd(crs(((0, 0), (1, 0))), (2, 32, 64))),
        # crossovers (golden cells)
        ("DRAM -> HEIGHT", [1, 1, 128, 64], DRAM, legacy(crs(((0, 0), (3, 0))), (32, 64), _ROW, H)),
        ("HEIGHT -> DRAM", [1, 1, 128, 64], legacy(crs(((0, 0), (3, 0))), (32, 64), _ROW, H), DRAM),
        # cross-spec (R2 territory; free via accessor source + local dest)
        (
            "HEIGHTx4 -> HEIGHTx2",
            [1, 1, 128, 64],
            legacy(crs(((0, 0), (3, 0))), (32, 64), _ROW, H),
            legacy(crs(((0, 0), (1, 0))), (64, 64), _ROW, H),
        ),
        # crossovers on WIDTH / BLOCK / COL_MAJOR -> pins the shard->core mapping
        ("DRAM -> WIDTH", [1, 1, 64, 256], DRAM, legacy(crs(((0, 0), (3, 0))), (64, 64), _ROW, W)),
        ("DRAM -> BLOCK ROW", [1, 1, 128, 128], DRAM, legacy(crs(((0, 0), (1, 1))), (64, 64), _ROW, B)),
        ("DRAM -> BLOCK COL", [1, 1, 128, 128], DRAM, legacy(crs(((0, 0), (1, 1))), (64, 64), _COL, B)),
        ("BLOCK COL -> DRAM", [1, 1, 128, 128], legacy(crs(((0, 0), (1, 1))), (64, 64), _COL, B), DRAM),
        ("DRAM -> HEIGHT COL", [1, 1, 256, 64], DRAM, legacy(crs(((0, 0), (0, 3))), (64, 64), _COL, H)),
        ("DRAM -> nd rank3", [4, 32, 64], DRAM, nd(crs(((0, 0), (1, 0))), (2, 32, 64))),
        ("nd rank3 -> DRAM", [4, 32, 64], nd(crs(((0, 0), (1, 0))), (2, 32, 64)), DRAM),
        # L1-interleaved <-> sharded
        (
            "HEIGHT -> L1 interleaved",
            [1, 1, 128, 64],
            legacy(crs(((0, 0), (3, 0))), (32, 64), _ROW, H),
            ttnn.L1_MEMORY_CONFIG,
        ),
    ]
    for name, shape, in_mc, out_mc in cases:
        x = torch.arange(torch.Size(shape).numel()).reshape(shape).float().bfloat16()
        t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mc)
        out = tilize(t, memory_config=out_mc, dtype=ttnn.bfloat16)
        got = ttnn.to_torch(out)
        ok = torch.equal(got.float(), x.float())
        extra = "" if ok else f" mismatch_frac={(got.float()!=x.float()).float().mean():.3f}"
        print(f"{name}: match={ok}{extra}")
        ttnn.deallocate(t)
        ttnn.deallocate(out)
finally:
    ttnn.close_device(device)
