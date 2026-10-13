# Device time of ttnn.typecast, height-sharded in L1 on every core (8 x 8 tiles per core, 4 x 8 for 32-bit), under
# `python -m tracy -r -p --no-web-server -o <dir> prof_tc.py <cases>`: 2 warm-up calls, a signpost, 6 measured calls.
import gc
import sys

import torch
import ttnn
from tracy import signpost

DT = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32, "i32": ttnn.int32, "u32": ttnn.uint32, "u16": ttnn.uint16}
TD = {"bf16": torch.bfloat16, "fp32": torch.float32, "i32": torch.int32, "u32": torch.int64, "u16": torch.int32}


def data(src, shape):
    g = torch.Generator().manual_seed(90)
    if src in ("bf16", "fp32"):
        return (torch.randn(shape, generator=g) * 1000).to(TD[src])
    hi = {"i32": 2**31 - 1, "u32": 2**32 - 1, "u16": 65535}[src]
    lo = {"i32": -(2**31)}.get(src, 0)
    return torch.randint(lo, hi, shape, generator=g, dtype=torch.int64).to(TD[src])


def run_case(d, name):
    _, src, dst = name.split("_")[:3]
    th = 4 if src in ("fp32", "i32", "u32") or dst in ("fp32", "i32", "u32") else 8
    grid = d.compute_with_storage_grid_size(); n = grid.x * grid.y
    crs = ttnn.num_cores_to_corerangeset(n, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                            ttnn.ShardSpec(crs, (32 * th, 32 * 8), ttnn.ShardOrientation.ROW_MAJOR))
    x = ttnn.from_torch(data(src, (1, 1, 32 * th * n, 32 * 8)), dtype=DT[src], layout=ttnn.TILE_LAYOUT, device=d, memory_config=mem)
    for _ in range(2):
        ttnn.typecast(x, DT[dst], memory_config=mem)
    ttnn.synchronize_device(d)
    signpost(header=name)
    for _ in range(6):
        ttnn.typecast(x, DT[dst], memory_config=mem)
    ttnn.synchronize_device(d)


dev = ttnn.open_device(device_id=0)
try:
    for name in sys.argv[1:]:
        gc.collect()
        try:
            run_case(dev, name)
            ttnn.ReadDeviceProfiler(dev)
            print(f"CASE {name} ok", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"CASE {name} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
finally:
    ttnn.close_device(dev)
