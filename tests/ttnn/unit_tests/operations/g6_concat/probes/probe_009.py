import torch, ttnn
from ttnn.operations.g6_concat import g6_concat_plan

device = ttnn.open_device(device_id=0)
print("dram_alignment", ttnn.get_dram_alignment())
cases = [
    ([(1, 1, 1, 62080)] * 16, 2, "qwen79"),
    ([(1, 4096, 1536)] * 8, 0, "qwen23_p4"),
    ([(1, 1, 1, 128)] * 2, -1, "flux23"),
    ([(1, 2, 31, 64), (1, 2, 33, 64), (1, 2, 7, 64)], 2, "straddle"),
    ([(1, 1, 1, 16384)] * 3 + [(1, 1, 1, 12928)], -1, "qwen160"),
]
for shapes, dim, name in cases:
    ts = [ttnn.from_torch(torch.zeros(s, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device) for s in shapes]
    rep = g6_concat_plan(ts, dim)
    c0 = rep["cores"][0]
    cl = rep["cores"][-1]
    print(
        name,
        rep["regime"],
        "cores",
        rep["num_cores"],
        "block",
        rep["block_pages"],
        "cbout",
        rep["cb_output_pages_bytes"],
        "staging",
        rep["cb_gather_staging_bytes"],
        "first",
        c0["pages"],
        c0["num_blocks"],
        "last",
        cl["pages"],
    )
ttnn.close_device(device)
