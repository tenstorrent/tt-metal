import math
import os
from itertools import product

import pytest
import torch

import ttnn

# Reproducible "main optimized baseline vs branch" harness for minimal_matmul (see
# tools/mm_sweep/HANDOFF.md). ONE branch build reproduces both sides:
#   MM_MODE=baseline -> pure unicast == main: explicit block sweep + TT_MM_NO_LARGE_LEVERS=1 (disables
#                       the branch-only DRAM levers; every other branch feature is already gated behind
#                       !config). The downstream parser takes the best block per shape.
#   MM_MODE=branch   -> the production auto path (no config): K-cap + block sizer + subblock maximizer
#                       + gate/slicing, all on.
# Shape from FL_M/FL_K/FL_N. Profiler (tracy) tags each invocation with its MinimalMatmulConfig in the
# ATTRIBUTES column so the parser can find the best baseline block. One PCC check per mode.
# Grid is taken from the device (so it works on Wormhole 8x8 AND Blackhole's larger grid).


def _max_dst(fp32):
    return 4 if fp32 else 8


def _adaptive_sub(mb, nb, max_dst=4):
    # Mirror the branch's subblock chooser: largest pow2 (sbh,sbw) area <= DST depth, each dividing its
    # block dim, balanced tiebreak. Keeps the baseline's subblocks as good as the branch's so the
    # comparison isolates dataflow/blocking, not subblock.
    best_h, best_w = 1, 1
    sh = 1
    while sh <= max_dst:
        if mb % sh == 0:
            sw = 1
            while sh * sw <= max_dst:
                if nb % sw == 0:
                    area, best_area = sh * sw, best_h * best_w
                    if area > best_area or (area == best_area and max(sh, sw) < max(best_h, best_w)):
                        best_h, best_w = sh, sw
                sw <<= 1
        sh <<= 1
    return best_h, best_w


@pytest.mark.timeout(3600)
def test_mm_repro(device):
    if "FL_M" not in os.environ:
        pytest.skip("repro harness: set FL_M/FL_K/FL_N and MM_MODE (baseline|branch)")
    M, K, N = int(os.environ["FL_M"]), int(os.environ["FL_K"]), int(os.environ["FL_N"])
    mode = os.environ.get("MM_MODE", "branch")
    reps = int(os.environ.get("FC_REPS", "10"))
    fp32 = os.environ.get("MM_FP32_ACC", "1") == "1"

    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y

    torch.manual_seed(0)
    ti = torch.randn((M, K), dtype=torch.bfloat16)
    wi = torch.randn((K, N), dtype=torch.bfloat16)
    tt_i = ttnn.from_torch(ti, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_w = ttnn.from_torch(wi, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    cc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=True,
    )

    def check(out):
        got = ttnn.to_torch(out).float()
        ref = ti.float() @ wi.float()
        pcc = torch.corrcoef(torch.stack([ref.flatten(), got.flatten()]))[0, 1].item()
        print(f"REPRO_PCC {mode} M{M}K{K}N{N} grid{gx}x{gy} pcc={pcc:.5f}")
        assert pcc > 0.99, f"PCC too low: {pcc}"

    if mode == "branch":
        out = None
        for _ in range(reps):
            if out is not None:
                out.deallocate()
            out = ttnn.experimental.minimal_matmul(tt_i, tt_w, compute_kernel_config=cc)
            ttnn.synchronize_device(device)
        check(out)
        out.deallocate()
        return

    # baseline: pure unicast (== main). Disable the branch-only DRAM levers, then sweep blocks.
    os.environ["TT_MM_NO_LARGE_LEVERS"] = "1"
    Mt, Kt, Nt = math.ceil(M / 32), math.ceil(K / 32), math.ceil(N / 32)
    # Per-core tile estimate for block seeding (M over rows=gy, N over cols=gx). Seeds only; the
    # blocks-per-core sweep covers a range, so the exact axis assignment isn't critical.
    M_pc = max(1, math.ceil(Mt / gy))
    N_pc = max(1, math.ceil(Nt / gx))
    bpc_m = [int(x) for x in os.environ.get("FC_BPCM", "1,2,4,8").split(",")]
    bpc_n = [int(x) for x in os.environ.get("FC_BPCN", "1,2,4,8").split(",")]
    kbs = [k for k in (int(x) for x in os.environ.get("FC_KBS", "4,8,16,32").split(",")) if k <= Kt]
    cfgs = set()
    for bm, bn, kb in product(bpc_m, bpc_n, kbs):
        mb = max(1, math.ceil(M_pc / bm))
        nb = max(1, math.ceil(N_pc / bn))
        sbh, sbw = _adaptive_sub(mb, nb, _max_dst(fp32))
        cfgs.add((mb, kb, nb, sbh, sbw))

    checked = False
    for mb, kb, nb, sbh, sbw in sorted(cfgs):
        try:
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=mb,
                K_block_size=kb,
                N_block_size=nb,
                subblock_h=sbh,
                subblock_w=sbw,
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            )
            for _ in range(reps):
                out = ttnn.experimental.minimal_matmul(tt_i, tt_w, compute_kernel_config=cc, config=cfg)
                ttnn.synchronize_device(device)
                if not checked:
                    check(out)
                    checked = True
                out.deallocate()
        except Exception as e:  # invalid block for this shape (L1 / divisibility) -> skip
            if isinstance(e, KeyboardInterrupt):
                raise
            continue
