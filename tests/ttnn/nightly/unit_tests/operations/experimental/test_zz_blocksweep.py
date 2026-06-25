import os
from itertools import product
import pytest
import torch
import ttnn


def _adaptive_sub(mb, nb):
    cap = 4  # fp32 half-sync DST

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
def test_blocksweep(device):
    # Explicit-config block sweep for ONE shape (TT_MM_NUM_SLICES applies via env -> unicast+slicing,
    # since an explicit config disables the gate/prefetch). Profiler tags each config by ATTRIBUTES.
    M = int(os.environ["FL_M"])
    K = int(os.environ["FL_K"])
    N = int(os.environ["FL_N"])
    reps = int(os.environ.get("FC_REPS", "4"))
    mbs = [1, 2, 4, 8]
    kbs = [4, 8, 16]
    nbs = [1, 2, 4, 8]
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
    seen = set()
    for mb, kb, nb in product(mbs, kbs, nbs):
        sbh, sbw = _adaptive_sub(mb, nb)
        key = (mb, kb, nb, sbh, sbw)
        if key in seen:
            continue
        seen.add(key)
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
