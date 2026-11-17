import ttnn
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

device = ttnn.open_device(device_id=0)
L1, DR = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
bf16, bf8, bf4, f32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.float32
cases = []
for m, k, n in [
    (9472, 5120, 3456),
    (9472, 3456, 5120),
    (4096, 4096, 4096),
    (2048, 9216, 6144),
    (2048, 5120, 8704),
    (256, 6144, 9216),
    (256, 9216, 6144),
    (256, 15360, 3072),
    (1, 6144, 6144),
    (384, 1536, 1024),
    (2048, 128, 3072),
    (2048, 3072, 128),
    (32, 1024, 2048),
    (2048, 256, 512),
]:
    cases.append((f"{m}x{k}x{n}", (1, m, k), (k, n), bf16, bf16, bf16, True, DR, "none"))
cases += [
    ("q prefill gate_up", (1, 2048, 5120), (5120, 8704), bf16, bf4, bf16, True, DR, "swiglu"),
    ("q prefill 4160", (1, 2048, 5120), (5120, 4160), bf16, bf4, bf16, True, DR, "none"),
    ("te 512x5120x3072", (1, 512, 5120), (5120, 3072), bf16, bf8, bf16, True, DR, "none"),
    ("ts lin2 fp32", (1, 6144), (6144, 6144), f32, bf16, f32, True, DR, "none"),
    ("ts lin1 fp32", (1, 256), (256, 6144), f32, bf16, f32, True, DR, "none"),
    ("guid lin1", (1, 256), (256, 6144), bf16, bf16, bf16, True, DR, "none"),
    ("time_mod", (1, 6144), (6144, 52224), f32, bf16, bf16, True, DR, "bias"),
    ("pdl 2048x256x512 bf8", (1, 2048, 256), (256, 512), bf8, bf8, bf8, False, DR, "none"),
    ("wpass", (1, 32, 128, 256), (256, 1024), bf8, bf8, bf8, False, L1, "none"),
    ("64x16384x256", (1, 64, 16384), (16384, 256), bf16, bf16, bf16, False, DR, "none"),
    ("ff1", (1, 256, 6144), (6144, 18432), bf16, bf16, bf16, True, DR, "swiglu"),
    ("attn out txt", (1, 256, 6144), (6144, 3072), bf16, bf16, bf16, True, DR, "addcmul"),
    ("attn out 128", (1, 128, 6144), (6144, 3072), bf16, bf16, bf16, True, DR, "addcmul"),
    ("ff1 1024", (1, 1024, 6144), (6144, 18432), bf16, bf16, bf16, True, DR, "swiglu"),
    ("1152x9216x6144", (1, 1152, 9216), (9216, 6144), bf16, bf16, bf16, True, DR, "none"),
    ("ctx 256x15360x3072", (1, 256, 15360), (15360, 3072), bf16, bf16, bf16, True, DR, "none"),
    ("te 512x16384x5120", (1, 512, 16384), (16384, 5120), bf16, bf8, bf16, True, DR, "none"),
    ("q128 gate_up L1", (1, 128, 5120), (5120, 8704), bf16, bf4, bf16, True, L1, "swiglu"),
    ("cart m1", (1, 2048), (2048, 1536), bf16, bf16, bf16, True, DR, "none"),
    ("cart dec32", (1, 1, 32, 1024), (1024, 2048), bf16, bf16, bf16, True, DR, "none"),
    ("cart 384", (1, 384, 1536), (1536, 1024), bf16, bf16, bf16, True, DR, "none"),
    ("cart 128x128x256", (1, 128, 128), (128, 256), bf16, bf16, bf16, True, DR, "none"),
    ("cart 32x64x96", (1, 32, 64), (64, 96), bf16, bf16, bf16, True, DR, "none"),
]


def fmt(g, grid, blk):
    return (
        (grid.grid_m, grid.grid_n, grid.core_m_tiles, grid.core_n_tiles),
        (
            blk.block_m_tiles,
            blk.block_n_tiles,
            blk.block_k_tiles,
            blk.num_k_blocks,
            blk.sb_h,
            blk.sb_w,
            "A" if blk.act_resident else "",
            "W" if blk.weight_resident else "",
            blk.depth_act,
        ),
        "mode",
        pd.operand_mode(g, grid, blk),
    )


for name, ash, bsh, adt, bdt, odt, acc, amc, epi in cases:
    ta = ttnn.allocate_tensor_on_device(ttnn.Shape(list(ash)), adt, ttnn.TILE_LAYOUT, device, amc)
    tb = ttnn.allocate_tensor_on_device(ttnn.Shape(list(bsh)), bdt, ttnn.TILE_LAYOUT, device, DR)
    nout = bsh[-1] // 2 if epi == "swiglu" else bsh[-1]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(ash[:-1]) + [nout]), odt, ttnn.TILE_LAYOUT, device, amc)
    res = []
    for on in (False, True):
        pd.ACT_STREAM_MCAST = on
        pd.Q_N_ROW_AWARE = on
        g, grid, blk, _ = pd.plan(ta, tb, out, acc, device, epilogue=epi)
        res.append(fmt(g, grid, blk))
    print(("SAME " if res[0] == res[1] else "DIFF ") + name)
    if res[0] != res[1]:
        for r in res:
            print("   ", r)
    for t in (ta, tb, out):
        ttnn.deallocate(t)
ttnn.close_device(device)
