import torch, ttnn
from ttnn.operations.g6_concat import g6_concat, g6_concat_plan

device = ttnn.open_device(device_id=0)
try:
    print("dram align", ttnn.get_dram_alignment(), "l1 align", ttnn.get_l1_alignment())
    t = ttnn.from_torch(
        torch.randn(1, 1, 32, 64).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    a = ttnn.TensorAccessorArgs(t)
    print(
        "cta", list(a.get_compile_time_args()), "crta", list(a.get_common_runtime_args()), "storage", t.storage_type()
    )

    def run(shapes, dim):
        xs = [torch.randn(s).bfloat16() for s in shapes]
        exp = torch.cat(xs, dim)
        out = g6_concat(
            [ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for x in xs], dim
        )
        got = ttnn.to_torch(out)
        plan = g6_concat_plan(
            [ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for x in xs], dim
        )
        bad = int((got.view(torch.int16) != exp.view(torch.int16)).sum())
        print(
            shapes[:2],
            dim,
            plan["regime"],
            "cores",
            plan["num_cores"],
            "bp",
            plan["block_pages"],
            "bad",
            bad,
            "/",
            exp.numel(),
        )

    run([(1, 1, 64, 64), (1, 1, 64, 192)], -1)
    run([(1, 6, 31, 64), (1, 6, 31, 192)], -1)
    run([(1, 1, 128, 256)] * 4, 0)
    run([(1, 1, 64, 128), (1, 1, 33, 128)], 2)
    run([(1, 1, 1, 2048)] * 2, 2)
    run([(1, 2, 31, 64), (1, 2, 33, 64), (1, 2, 7, 64)], 2)
    run([(2, 5, 64), (2, 40, 64)], 1)
finally:
    ttnn.close_device(device)
