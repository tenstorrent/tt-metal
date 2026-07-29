import os
import torch
import ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize.tilize_program_descriptor import build_plan

_ROW = ttnn.ShardOrientation.ROW_MAJOR
_H = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def crs(ex, ey):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ex, ey))})


def shard(scheme, grid, shape):
    return ttnn.MemoryConfig(scheme, ttnn.BufferType.L1, ttnn.ShardSpec(grid, shape, _ROW))


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    if torch.equal(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


CASES = [
    ("dram fp32->bf16", (1, 1, 128, 256), None, ttnn.float32, ttnn.bfloat16),
    ("dram fp32->bf8b", (1, 1, 128, 256), None, ttnn.float32, ttnn.bfloat8_b),
    ("dram fp32->fp32", (1, 1, 128, 256), None, ttnn.float32, ttnn.float32),
    ("shard fp32->bf16", (1, 1, 512, 64), shard(_H, crs(3, 0), (128, 64)), ttnn.float32, ttnn.bfloat16),
    ("shard fp32->bf8b", (1, 1, 512, 64), shard(_H, crs(3, 0), (128, 64)), ttnn.float32, ttnn.bfloat8_b),
    ("shard fp32->fp32", (1, 1, 512, 64), shard(_H, crs(3, 0), (128, 64)), ttnn.float32, ttnn.float32),
    ("dram u32->u32", (1, 1, 128, 256), None, ttnn.uint32, None),
    ("dram bf16->bf8b", (1, 1, 128, 256), None, ttnn.bfloat16, ttnn.bfloat8_b),
]

device = ttnn.open_device(device_id=0)
try:
    for f32 in (1, 0):
        os.environ["TILIZE_LEVER_F32"] = str(f32)
        print(f"--- TILIZE_LEVER_F32={f32} ({'Lossless everywhere' if f32 == 0 else 'gated Fast'}) ---")
        for name, shape, cfg, dt, odt in CASES:
            torch.manual_seed(42)
            if dt == ttnn.float32:
                t = torch.randn(shape, dtype=torch.float32)
            elif dt == ttnn.uint32:
                t = torch.randint(0, 1 << 20, shape, dtype=torch.int32)
            else:
                t = torch.randn(shape).bfloat16()
            mem = cfg if cfg is not None else ttnn.DRAM_MEMORY_CONFIG
            tt_in = ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem)
            po = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), odt or dt, ttnn.TILE_LAYOUT, device, mem)
            plan = build_plan(tt_in, po, device)
            out = tilize(tt_in, mem, dtype=odt)
            got = ttnn.to_torch(out)
            if dt == ttnn.uint32:
                exp = t
                ok = torch.equal(got.to(torch.int32), exp.to(torch.int32))
                print(f"  {name:<18} path={plan['path']:<6} F32={plan['fp32_lossless']} exact={ok}")
                continue
            exp = t.to(torch.float32 if (odt or dt) == ttnn.float32 else torch.bfloat16)
            d = (got.float() - exp.float()).abs()
            print(
                f"  {name:<18} path={plan['path']:<6} F32={plan['fp32_lossless']} "
                f"pcc={pcc(exp, got):.7f} max_abs={d.max().item():.4e} "
                f"nmis={(d != 0).sum().item()}/{d.numel()}"
            )
finally:
    os.environ["TILIZE_LEVER_F32"] = "1"
    ttnn.close_device(device)
