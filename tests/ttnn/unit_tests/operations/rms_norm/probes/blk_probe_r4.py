import pytest, torch, ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd
ML = ttnn.TensorMemoryLayout
CASES = [(640, 6144, (64, 704), (9, 10)), (640, 7168, (64, 1024), (7, 10)), (7168, 1024, (736, 128), (8, 10)),
         (7168, 2048, (736, 192), (11, 10)), (8192, 1024, (1024, 128), (8, 8))]
@pytest.mark.parametrize("case", CASES)
def test_blk(device, case):
    rows, W, sh, (gx, gy) = case
    orig = pd.select_blocking
    def sb(g, R, Wt, C, shard=None, budget=pd.L1_CB_BUDGET_BYTES):
        b = orig(g, R, Wt, C, shard, budget)
        print(f"\nBLK {rows}x{W} budget={budget} {b} per_row={pd._per_row_bytes(g,b.core_w_tiles,b.w_split)} fixed={pd._fixed_bytes(g,b.core_w_tiles)}")
        return b
    pd.select_blocking = sb
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    mc = ttnn.MemoryConfig(ML.BLOCK_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(crs, list(sh), ttnn.ShardOrientation.ROW_MAJOR))
    x = torch.randn((1, 1, rows, W)).to(torch.bfloat16)
    g = torch.randn((1, 1, 1, W)).to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False)
    rms_norm(tx, gamma=tg, compute_kernel_config=cfg, memory_config=mc)
    pd.select_blocking = orig
