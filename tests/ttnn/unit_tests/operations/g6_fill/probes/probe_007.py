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
    p = g6_fill_plan(x)
    print(
        "PLAN",
        tag,
        n,
        p["regime"],
        "cores",
        p["num_cores"],
        p["fill_row_source"],
        {k: p[k] for k in ("rm_chunk_bytes", "num_rm_chunks") if k in p},
    )
    for v in values:
        g6_fill(x, v, output_tensor=out)
        ok = torch.equal(bits(ttnn.to_torch(out)).flatten(), bits(expect(dtype, v)).expand(n)) and torch.equal(
            bits(ttnn.to_torch(x)), bits(xin)
        )
        print(("OK " if ok else "BAD"), tag, n, repr(v))


device = ttnn.open_device(device_id=0)
try:
    pd.RM_CHUNK_CAP_BYTES = 64
    pd.MAX_FILL_CORES = 3
    run(device, 1000, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, [5, -3], "cap64_cores3")
    pd.FILL_ROW_SOURCE = "common_args"
    run(device, 1000, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, [9], "common_args_ignored_R2")
    pd.MAX_FILL_CORES = None
    run(device, 512, ttnn.float32, ttnn.TILE_LAYOUT, [0.1, 1.0], "common_args_R1")
finally:
    ttnn.close_device(device)
