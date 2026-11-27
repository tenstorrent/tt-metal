# Idle tail cores (rectangle larger than the work) and repeat-call bit identity of logical values.
import torch, ttnn
from ttnn.operations.g6_unsqueeze import g6_unsqueeze, plan_for
import ttnn.operations.g6_unsqueeze.g6_unsqueeze_program_descriptor as desc

device = ttnn.open_device(device_id=0)
torch.manual_seed(11)


def run(shape, max_cores=None, repeats=3):
    desc.MAX_CORES = max_cores
    x = torch.randn(shape, dtype=torch.float32)
    t = ttnn.from_torch(
        x,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        preserve_nan_values=True,
    )
    t = ttnn.fill_implicit_tile_padding(t, float("nan"))
    plan = plan_for(t, max_cores)
    want = torch.unsqueeze(x, -1).contiguous().view(torch.int32)
    ok = True
    for _ in range(repeats):
        got = ttnn.to_torch(g6_unsqueeze(t, len(shape)))
        ok &= tuple(got.shape) == tuple(want.shape) and torch.equal(got.contiguous().view(torch.int32), want)
    idle = sum(1 for c in plan["cores"] if c["role"] == "idle_tail")
    print(f"CASE {shape} max_cores={max_cores} rect={plan['rectangle']} idle={idle} -> {'OK' if ok else 'FAIL'}")


run((1, 32 * 16 + 5))  # 17 tiles -> 9x2 rectangle, one idle tail core
run((1, 32 * 13))  # 13 tiles -> 7x2 rectangle, one idle tail core
run((1, 32 * 23 + 1), 5)  # 24 tiles on 5 cores -> uneven 5/5/5/5/4
ttnn.close_device(device)
