import torch, ttnn
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

device = ttnn.open_device(device_id=0)
bf16 = ttnn.bfloat16
for m, k, n, epi in [
    (256, 6144, 18432, "swiglu"),
    (256, 6144, 9216, "none"),
    (1, 6144, 6144, "none"),
    (256, 9216, 6144, "none"),
    (256, 15360, 3072, "none"),
    (256, 6144, 3072, "none"),
    (32, 1024, 2048, "none"),
    (1, 2048, 1536, "none"),
    (384, 1536, 1024, "none"),
    (2048, 5120, 8704, "swiglu"),
]:
    ta = ttnn.allocate_tensor_on_device(ttnn.Shape([1, m, k]), bf16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    tb = ttnn.allocate_tensor_on_device(ttnn.Shape([k, n]), bf16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    nout = n // 2 if epi == "swiglu" else n
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, m, nout]), bf16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    g, grid, blk, _ = pd.plan(ta, tb, out, True, device, epilogue=epi)
    print(
        (m, k, n, epi),
        (grid.grid_m, grid.grid_n, grid.core_m_tiles, grid.core_n_tiles, grid.q_n),
        (
            blk.block_m_tiles,
            blk.block_n_tiles,
            blk.block_k_tiles,
            blk.num_k_blocks,
            blk.sb_h,
            blk.sb_w,
            blk.act_resident,
            blk.weight_resident,
        ),
        "mode",
        pd.operand_mode(g, grid, blk),
        "MB",
        round(pd.read_traffic(g, grid, blk) / 1e6, 1),
    )
    for t in (ta, tb, out):
        ttnn.deallocate(t)
ttnn.close_device(device)
