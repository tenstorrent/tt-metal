import torch, ttnn
from ttnn.operations.genify_matmul.genify_matmul_program_descriptor import plan

dev = ttnn.open_device(device_id=0)
shapes = [
    ((1, 1, 32, 32), (32, 32)),
    ((1, 1, 128, 256), (256, 128)),
    ((1, 384, 1536), (1536, 1024)),
    ((2, 3, 64, 128), (128, 192)),
    ((1, 1, 32, 1024), (1024, 2048)),
    ((64, 256), (256, 160)),
    ((1, 1, 2048, 256), (256, 512)),
    ((1, 1, 128, 8032), (8032, 256)),
    ((1, 1, 2048, 5120), (5120, 8704)),
    ((1, 1, 32, 6144), (6144, 6144)),
    ((1, 1, 4096, 5120), (5120, 17408)),
]
for a, b in shapes:
    A = ttnn.from_torch(torch.zeros(a), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    B = ttnn.from_torch(torch.zeros(b), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    O = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(a[:-1]) + [b[-1]]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
    )
    g, gr, bl, gs = plan(A, B, O, True, dev)
    print(
        "PLAN",
        a,
        b,
        gs,
        (gr.grid_m, gr.grid_n, gr.placement, gr.core_m_tiles, gr.core_n_tiles),
        (
            bl.block_m_tiles,
            bl.block_n_tiles,
            bl.block_k_tiles,
            bl.num_k_blocks,
            bl.last_block_k_tiles,
            bl.sb_h,
            bl.sb_w,
        ),
    )
    A.deallocate()
    B.deallocate()
    O.deallocate()
ttnn.close_device(dev)
