import torch, ttnn
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

device = ttnn.open_device(device_id=0)
L1, DR = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
bf16, bf8, bf4 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b
cases = [
    ("PF3", (1, 32, 512, 128), (1, 32, 128, 1024), bf8, bf8, False, DR, L1, DR, "none"),
    ("Wpass", (1, 32, 128, 256), (256, 1024), bf8, bf8, False, L1, DR, L1, "none"),
    ("L1 gate_up128", (1, 128, 5120), (5120, 8704), bf16, bf4, True, L1, DR, L1, "swiglu"),
    ("L1 384", (1, 384, 1536), (1536, 1024), bf16, bf16, True, L1, DR, L1, "none"),
    ("q128 in", (1, 128, 5120), (5120, 4160), bf16, bf4, True, L1, DR, DR, "none"),
    ("q128 out", (1, 128, 1536), (1536, 5120), bf16, bf4, True, L1, DR, DR, "none"),
    ("cart batched", (1, 4, 128, 128), (1, 4, 128, 256), bf16, bf16, True, DR, DR, DR, "none"),
    ("cart batched16", (1, 4, 128, 128), (1, 4, 128, 256), bf16, bf16, False, DR, DR, DR, "none"),
    ("cart pdl", (1, 1, 2048, 256), (256, 512), bf16, bf16, True, DR, DR, DR, "none"),
    ("cart dec32", (1, 1, 32, 1024), (1024, 2048), bf16, bf16, True, DR, DR, DR, "none"),
    ("cart 384", (1, 384, 1536), (1536, 1024), bf16, bf16, True, DR, DR, DR, "none"),
    ("cart m1", (1, 2048), (2048, 1536), bf16, bf16, True, DR, DR, DR, "none"),
]
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
    cases.append((f"guard {m}x{k}x{n}", (1, m, k), (k, n), bf16, bf16, True, DR, DR, DR, "none"))
for name, ash, bsh, adt, bdt, acc, amc, bmc, omc, epi in cases:
    ta = create_ttnn_input_tensor(torch.randn(*ash), device, dtype=adt, layout=ttnn.TILE_LAYOUT, memory_config=amc)
    tb = create_ttnn_input_tensor(torch.randn(*bsh), device, dtype=bdt, layout=ttnn.TILE_LAYOUT, memory_config=bmc)
    nout = bsh[-1] // 2 if epi == "swiglu" else bsh[-1]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(ash[:-1]) + [nout]), adt, ttnn.TILE_LAYOUT, device, omc)
    res = []
    for rsb, l1c in [(1, 1.0), (4, 0.25)]:
        pd.RESIDENT_SPLIT_BLOCKS, pd.L1_READ_COST = rsb, l1c
        g, grid, blk, _ = pd.plan(ta, tb, out, acc, device, epilogue=epi)
        res.append((grid, blk, pd.uses_operand_mcast(g, grid, blk)))
    same = res[0] == res[1]
    print(("SAME " if same else "DIFF ") + name)
    if not same:
        for grid, blk, mc in res:
            print(
                "   ",
                (grid.grid_b, grid.grid_m, grid.grid_n, grid.core_batches, grid.core_m_tiles, grid.core_n_tiles),
                (blk.block_m_tiles, blk.block_n_tiles, blk.block_k_tiles, blk.num_k_blocks, blk.sb_h, blk.sb_w),
                "Wres" if blk.weight_resident else "",
                "Ares" if blk.act_resident else "",
                "mcast" if mc else "",
            )
    ttnn.deallocate(ta)
    ttnn.deallocate(tb)
    ttnn.deallocate(out)
ttnn.close_device(device)
