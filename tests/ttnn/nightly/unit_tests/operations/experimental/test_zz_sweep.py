import math
import os
from itertools import product
import pytest
import torch
import ttnn


def _adaptive_sub(mb, nb):
    # Largest valid fp32 subblock (sbh*sbw <= 4), wider dim along the larger block dim, divides block.
    cap = 4

    def largest(v, c):
        for d in (4, 2, 1):
            if d <= c and v % d == 0 and d <= v:
                return d
        return 1

    if nb >= mb:
        sbw = largest(nb, cap)
        sbh = largest(mb, cap // sbw)
    else:
        sbh = largest(mb, cap)
        sbw = largest(nb, cap // sbh)
    return sbh, sbw


@pytest.mark.timeout(3600)
def test_sweep(device):
    # Block-size sweep for ONE shape + ONE dataflow variant (chosen by TT_MM_* env flag at process
    # level). Parametrized by BLOCKS-PER-CORE so block sizes scale with the shape: M/N split 8 ways
    # across the 8x8 grid -> per-core tile count; block = ceil(percore / blocks_per_core). bpc=1 means
    # the whole per-core region in one block. Subblock = adaptive (<=4 tiles, fp32-safe). Profiler
    # captures each invocation tagged with its MinimalMatmulConfig; orchestrator parses best blocking.
    M = int(os.environ["FL_M"])
    K = int(os.environ["FL_K"])
    N = int(os.environ["FL_N"])
    reps = int(os.environ.get("FC_REPS", "5"))
    GX = 8
    M_tiles = math.ceil(M / 32)
    K_tiles = math.ceil(K / 32)
    N_tiles = math.ceil(N / 32)
    M_pc = math.ceil(M_tiles / GX)
    N_pc = math.ceil(N_tiles / GX)  # per-core tiles (M/N parallelized 8-way)
    bpc_m = [int(x) for x in os.environ.get("FC_BPCM", "1,2,4,8").split(",")]
    bpc_n = [int(x) for x in os.environ.get("FC_BPCN", "1,2,4,8").split(",")]
    kbs = [k for k in (int(x) for x in os.environ.get("FC_KBS", "4,8,16,32").split(",")) if k <= K_tiles]

    # Build deduped (mb, kb, nb, sbh, sbw) configs
    cfgs = set()
    for bm, bn, kb in product(bpc_m, bpc_n, kbs):
        mb = max(1, math.ceil(M_pc / bm))
        nb = max(1, math.ceil(N_pc / bn))
        sbh, sbw = _adaptive_sub(mb, nb)
        cfgs.add((mb, kb, nb, sbh, sbw))
    cfgs = sorted(cfgs)

    torch.manual_seed(0)
    ti = torch.randn((M, K), dtype=torch.bfloat16)
    wi = torch.randn((K, N), dtype=torch.bfloat16)
    tt_i = ttnn.from_torch(ti, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_w = ttnn.from_torch(wi, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    cc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    for mb, kb, nb, sbh, sbw in cfgs:
        try:
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=mb,
                K_block_size=kb,
                N_block_size=nb,
                subblock_h=sbh,
                subblock_w=sbw,
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            )
            for _ in range(reps):
                out = ttnn.experimental.minimal_matmul(tt_i, tt_w, compute_kernel_config=cc, config=cfg)
                ttnn.synchronize_device(device)
                out.deallocate()
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            continue
