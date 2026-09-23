import sys, torch, ttnn
import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize import tilize

tm = sys.modules["ttnn.operations.tilize.tilize"]
device = ttnn.open_device(device_id=0)
orig = pd.create_program_descriptor


def wrap(*a, **k):
    d = orig(*a, **k)
    r = [kd for kd in d.kernels if "reader" in kd.kernel_source][0]
    print(
        "CO",
        list(a[0].shape),
        "co_read",
        r.compile_time_args[28],
        "coalesce",
        r.compile_time_args[26],
        "sems",
        len(d.semaphores),
        "cores",
        r.core_ranges.num_cores(),
    )
    return d


tm.create_program_descriptor = wrap


def hs(w):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (32, w), ttnn.ShardOrientation.ROW_MAJOR),
    )


for shape, mc, omc in [
    ((1, 1, 32, 2048), ttnn.DRAM_MEMORY_CONFIG, None),
    ((1, 1, 128, 64), ttnn.DRAM_MEMORY_CONFIG, None),
    ((1, 1, 2048, 512), ttnn.DRAM_MEMORY_CONFIG, None),
    ((1, 1, 2048, 512), hs(512), ttnn.DRAM_MEMORY_CONFIG),
]:
    x = torch.randn(shape).to(torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    out = tilize(t, memory_config=omc)
    print("CO ok", torch.equal(ttnn.to_torch(out), x))
ttnn.close_device(device)
