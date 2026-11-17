import torch, ttnn
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

device = ttnn.open_device(device_id=0)
bf16 = ttnn.bfloat16


def plan(m, k, n, epi):
    ta = ttnn.allocate_tensor_on_device(ttnn.Shape([1, m, k]), bf16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    tb = ttnn.allocate_tensor_on_device(ttnn.Shape([k, n]), bf16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    nout = n // 2 if epi == "swiglu" else n
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, m, nout]), bf16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    g, grid, blk, _ = pd.plan(ta, tb, out, True, device, epilogue=epi)
    for t in (ta, tb, out):
        ttnn.deallocate(t)
    return (
        (grid.grid_m, grid.grid_n, grid.core_m_tiles, grid.core_n_tiles),
        (
            blk.block_m_tiles,
            blk.block_n_tiles,
            blk.block_k_tiles,
            blk.num_k_blocks,
            blk.act_resident,
            blk.weight_resident,
        ),
        pd.operand_mode(g, grid, blk),
    )


for s in []:
    print(s, plan(*s))
for res in [700 * 1024, 900 * 1024, 1000 * 1024]:
    pd.L1_RESERVE_BYTES = res
    for s in [
        (256, 2048, 8192, "none"),
        (256, 2048, 8192, "swiglu"),
        (256, 4096, 32768, "none"),
        (512, 4096, 32768, "swiglu"),
    ]:
        print(res, s, plan(*s))
ttnn.close_device(device)
