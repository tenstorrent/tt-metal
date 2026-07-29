import os
import torch
import ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize.tilize_program_descriptor import build_plan, create_program_descriptor

_ROW = ttnn.ShardOrientation.ROW_MAJOR
_COL = ttnn.ShardOrientation.COL_MAJOR
_H = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
_B = ttnn.TensorMemoryLayout.BLOCK_SHARDED
_W = ttnn.TensorMemoryLayout.WIDTH_SHARDED


def crs(ex, ey):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ex, ey))})


def shard(scheme, grid, shape, orient=_ROW):
    return ttnn.MemoryConfig(scheme, ttnn.BufferType.L1, ttnn.ShardSpec(grid, shape, orient))


CASES = [
    ("f_small H", (1, 1, 512, 64), shard(_H, crs(3, 0), (128, 64))),
    ("H col", (1, 1, 512, 64), shard(_H, crs(3, 0), (128, 64), _COL)),
    ("B 2x2", (1, 1, 256, 128), shard(_B, crs(1, 1), (128, 64))),
    ("W wide", (1, 1, 64, 256), shard(_W, crs(3, 0), (64, 64))),
    ("1 blk/core", (1, 1, 128, 64), shard(_H, crs(3, 0), (32, 64))),
    ("wide chunk", (1, 1, 128, 256), shard(_B, crs(1, 1), (64, 128))),
]


def run(device, name, shape, cfg, nd, out_dtype=None, dtype=ttnn.bfloat16):
    os.environ["TILIZE_LEVER_ND"] = str(nd)
    n = 1
    for d in shape:
        n *= d
    if dtype == ttnn.float32:
        t = (torch.arange(n, dtype=torch.float32) % 4096.0).reshape(shape)
    else:
        t = ((torch.arange(n, dtype=torch.float32) % 4096.0).reshape(shape)).bfloat16()
    tt_in = ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=cfg)
    probe_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), out_dtype or dtype, ttnn.TILE_LAYOUT, device, cfg
    )
    plan = build_plan(tt_in, probe_out, device)
    desc = create_program_descriptor(tt_in, probe_out, plan)
    srcs = [k.kernel_source.split("/")[-1] for k in desc.kernels]
    out = tilize(tt_in, cfg, dtype=out_dtype)
    got = ttnn.to_torch(out)
    exp = t if out_dtype is None else t.to(torch.bfloat16 if out_dtype != ttnn.float32 else torch.float32)
    ok = torch.equal(got.float(), exp.float())
    print(
        f"  nd={nd} {name:<12} path={plan['path']:<6} dr={plan['drop_reader']} dw={plan['drop_writer']} "
        f"sa={plan['self_arm']} blk={plan['blocks_per_core']} chunk={plan['chunk_wt']} "
        f"kernels={srcs} EXACT={ok}"
    )
    if not ok:
        d = (got.float() - exp.float()).abs()
        print(f"      max_abs={d.max().item()} nmismatch={(d != 0).sum().item()}/{d.numel()}")
    return ok


device = ttnn.open_device(device_id=0)
try:
    allok = True
    for nd in (1, 0, 3):
        print(f"--- ND={nd} ---")
        for name, shape, cfg in CASES:
            allok &= run(device, name, shape, cfg, nd)
    print("ALL EXACT:", allok)
finally:
    os.environ["TILIZE_LEVER_ND"] = "1"
    ttnn.close_device(device)
