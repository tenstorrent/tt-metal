import torch, ttnn
from ttnn.operations.g6_fill import g6_fill, g6_fill_plan
from ttnn.operations.g6_fill import g6_fill_program_descriptor as pd

TD = {ttnn.float32: torch.float32, ttnn.int32: torch.int32}


def bits(t):
    return t.contiguous().view(torch.int32) if t.dtype == torch.float32 else t


def expect(dtype, v):
    src = torch.float64 if isinstance(v, float) else torch.int64
    return torch.tensor(v, dtype=src).to(TD[dtype])


def run(device, n, dtype, layout, values, tag=""):
    torch.manual_seed(n)
    xin = torch.randint(1, 100, (n,), dtype=torch.int32) if dtype == ttnn.int32 else torch.randn(n) + 10.0
    x = ttnn.from_torch(xin, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.from_torch(
        xin.clone() + 1, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    # a guard tensor allocated right after out: must stay untouched
    guard_vals = torch.full((64,), 7, dtype=torch.int32) if dtype == ttnn.int32 else torch.full((64,), 7.0)
    guard = ttnn.from_torch(
        guard_vals, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    p = g6_fill_plan(x)
    print(
        "PLAN",
        tag,
        n,
        dtype,
        layout,
        p["regime"],
        "cores",
        p["num_cores"],
        {k: p[k] for k in ("chunks_per_core", "rm_chunk_bytes", "num_rm_chunks") if k in p},
    )
    for v in values:
        g6_fill(x, v, output_tensor=out)
        got = bits(ttnn.to_torch(out)).flatten()
        want = bits(expect(dtype, v)).expand(n)
        ok = (
            torch.equal(got, want)
            and torch.equal(bits(ttnn.to_torch(x)), bits(xin))
            and torch.equal(bits(ttnn.to_torch(guard)), bits(guard_vals))
        )
        print(("OK " if ok else "BAD"), tag, n, dtype, layout, repr(v))


device = ttnn.open_device(device_id=0)
try:
    F = [1 / 3, -0.0, float("nan"), 1e-40, 1.0]
    I = [0, 16777217, -(2**31), 2**31 - 1]
    # off-production pairs
    for n in [1, 196, 513]:
        run(device, n, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, F, "f32RM")
    for n in [1, 185, 512]:
        run(device, n, ttnn.int32, ttnn.TILE_LAYOUT, I, "i32TILE")
    # multi-chunk R2: ragged last chunk, several cores
    for n in [3, 4, 5, 512, 513, 1000, 5000, 70000]:
        run(device, n, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, [7, -1], "i32RM")
    # knob turns on R2
    pd.RM_CHUNK_CAP_BYTES = 16
    run(device, 37, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, [5, -3], "cap16")
    pd.RM_CHUNK_CAP_BYTES = 64
    pd.MAX_FILL_CORES = 3
    run(device, 1000, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, [5, -3], "cap64_cores3")
    pd.FILL_ROW_SOURCE = "common_args"
    run(device, 1000, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, [9], "common_args_ignored_R2")
    run(device, 512, ttnn.float32, ttnn.TILE_LAYOUT, [0.1], "common_args_R1")
finally:
    ttnn.close_device(device)
