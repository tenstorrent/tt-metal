import torch, ttnn
from ttnn.operations.g6_expand import g6_expand, g6_expand_plan

device = ttnn.open_device(device_id=0)
try:
    cases = [
        ((1, 1, 1, 512), [1, 1, 32, 512]),  # num_row_groups == 1: channel absent
        ((1, 1, 1, 196), [1, 1, 64, -1]),  # two row groups, w_non_aligned
        ((2, 3, 1, 256), [2, 3, 512, 256]),  # images folded into src_col, blocks cross images, ragged last group
        (
            (1, 700, 1, 320),
            [1, 700, 64, 320],
        ),  # src_block_depth 2: 3 blocks per core, ragged last block, 3 mcast rounds
        ((1, 700, 1, 320), [1, 700, 32, 320]),  # depth 2 with the channel absent
    ]
    for shape, sizes in cases:
        torch.manual_seed(0)
        x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        x_tt = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        x_tt = ttnn.fill_implicit_tile_padding(x_tt, 1000.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        plan = g6_expand_plan(x_tt, sizes)
        expected = x.expand(*[d if s == -1 else s for d, s in zip(shape, sizes)]).contiguous()
        ok = True
        for rep in range(2):
            got = ttnn.to_torch(g6_expand(x_tt, sizes))
            same = torch.equal(got.view(torch.int16), expected.view(torch.int16))
            ok &= same
        print(
            shape,
            sizes,
            "OK" if ok else "MISMATCH",
            {
                k: plan[k]
                for k in (
                    "num_col_groups",
                    "num_row_groups",
                    "core_src_col_tiles",
                    "block_src_col_tiles",
                    "src_block_depth",
                    "l1_bytes_per_core",
                )
            },
        )
finally:
    ttnn.close_device(device)
