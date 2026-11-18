import math, torch, ttnn
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd
from eval.golden_tests.genify_matmul.axes import observed as genify_matmul
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor

orig = pd.choose_split


def traced(geom, gx, gy, budget):
    rows = []
    for g in pd.grid_candidates(geom, gx, gy):
        try:
            blk = pd.choose_blocking(geom, g, budget)
        except RuntimeError:
            continue
        mode = pd.operand_mode(geom, g, blk)
        if g.placement == "bank" and mode != 2:
            continue
        rows.append(
            (
                pd.split_cost(geom, g, blk) / 1350,
                g,
                blk,
                mode,
                [round(t / 1350, 1) for t in pd.split_cost_terms(geom, g, blk)],
            )
        )
    rows.sort(key=lambda r: r[0])
    for c, g, blk, mode, terms in rows[:25]:
        print(
            f"CAND {c:7.1f} mode={mode} {g.grid_m}x{g.grid_n} {g.placement} core={g.core_m_tiles}x{g.core_n_tiles} blk={blk.block_m_tiles}x{blk.block_n_tiles}x{blk.block_k_tiles} sb={blk.sb_h}x{blk.sb_w} terms={terms}"
        )
    r = orig(geom, gx, gy, budget)
    print("CHOSEN", r[0])
    return r


pd.choose_split = traced
device = ttnn.open_device(device_id=0)
A = torch.randn(1, 512, 5120)
B = torch.randn(5120, 16384) / 70
ta = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
tb = create_ttnn_input_tensor(B, device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False)
out = genify_matmul(ta, tb, compute_kernel_config=cfg)
ttnn.close_device(device)
