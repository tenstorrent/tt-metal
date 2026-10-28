# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""noc_balance bake-off: NoC selection / page order for clamp_square_practice's DM traffic.

Env filters (run_safe_pytest word-splits -k):
  NB_VARIANTS=baseline,rd_alt   NB_SHAPES=1x1x4096x4096   NB_DMONLY=1 (stub compute)
Every run is correctness-gated (bit-exact vs the torch reference unless DM-only).
"""
import os

import pytest
import torch
import ttnn

from ttnn.operations.clamp_square_practice.perf_experiments.noc_balance import variant as V


def _diag(thr):
    """Writer NoC by grid position: NoC0 for the bottom-right part of the grid (x/11 + y/9 > thr),
    NoC1 (home) elsewhere. Logical coords on the 12x10 grid. Reader stays on its own mode."""

    def f(x, y, role):
        return 0 if (x / 11.0 + y / 9.0) > thr else 1

    return f


VARIANTS = {
    "baseline": None,
    "base_dyn": dict(dynamic=True),
    "rd_alt": dict(rd_noc=1),
    "wr_alt": dict(wr_noc=1),
    "both_alt": dict(rd_noc=1, wr_noc=1),
    "both_altblk": dict(rd_noc=2, wr_noc=2),
    "rd_flip": dict(rd_noc=3),
    "wr_flip": dict(wr_noc=3),
    "rot": dict(order="rot"),
    "strided": dict(order="strided"),
    "both_alt_rot": dict(rd_noc=1, wr_noc=1, order="rot"),
    "both_alt_strided": dict(rd_noc=1, wr_noc=1, order="strided"),
    "rd_alt_strided": dict(rd_noc=1, order="strided"),
    "wr_diag": dict(wr_noc=4, core_noc=_diag(1.0)),
    "wr_diag08": dict(wr_noc=4, core_noc=_diag(0.8)),
    "wr_diag12": dict(wr_noc=4, core_noc=_diag(1.2)),
    "wr_diag_rd_alt": dict(rd_noc=1, wr_noc=4, core_noc=_diag(1.0)),
    "wr_diag_strided": dict(wr_noc=4, core_noc=_diag(1.0), order="strided"),
    "wr_diag11": dict(wr_noc=4, core_noc=_diag(1.1)),
    "wr_chk": dict(wr_noc=4, core_noc=lambda x, y, r: (x + y + 1) % 2),
    "wr_halfx": dict(wr_noc=4, core_noc=lambda x, y, r: 0 if x >= 6 else 1),
    "wr_halfy": dict(wr_noc=4, core_noc=lambda x, y, r: 0 if y >= 5 else 1),
    "wr_diag11_rd_hops": dict(rd_noc=5, wr_noc=4, core_noc=_diag(1.1), bank_mask=V.bank_mask_min_hops),
    "wr_diag11_rd_alt": dict(rd_noc=1, wr_noc=4, core_noc=_diag(1.1)),
    "wr_diag11_ded": dict(split_writer=_diag(1.1)),
    "rd_hops": dict(rd_noc=5, bank_mask=V.bank_mask_min_hops),
    "wr_hops": dict(wr_noc=5, bank_mask=V.bank_mask_min_hops),
    "both_hops": dict(rd_noc=5, wr_noc=5, bank_mask=V.bank_mask_min_hops),
    "wr_diag13": dict(wr_noc=4, core_noc=_diag(1.3)),
    "wr_diag14": dict(wr_noc=4, core_noc=_diag(1.4)),
    "wr_diag12_rd_hops": dict(rd_noc=5, wr_noc=4, core_noc=_diag(1.2), bank_mask=V.bank_mask_min_hops),
}
SHAPES = [(1, 1, 4096, 4096), (8, 1, 1024, 1024), (2, 4, 256, 512), (1, 1, 32, 8192), (1, 1, 32, 32), (1, 1, 96, 1312)]


def _ref(x):
    return (x.float().clamp(-1.0, 1.0) ** 2).to(torch.bfloat16)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_noc_balance(device, shape, variant):
    vs, ss = os.environ.get("NB_VARIANTS"), os.environ.get("NB_SHAPES")
    if (vs and variant not in vs.split(",")) or (ss and "x".join(map(str, shape)) not in ss.split(",")):
        pytest.skip("filtered")
    dm_only = os.environ.get("NB_DMONLY") == "1"
    torch.manual_seed(0)
    xt = torch.randn(shape, dtype=torch.bfloat16) * 2
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    kw = VARIANTS[variant]
    abl = os.environ.get("NB_ABLATE", "")  # "r": stub reads (write-only), "w": stub writes (read-only)
    if abl:
        assert dm_only, "ablation only meaningful in the DM-only bench"
        kw = dict(kw or {}, ablate=abl)
    if kw is None:
        desc = V.build_baseline(x, out, -1.0, 1.0, skip_compute=dm_only)
    else:
        desc = V.build(x, out, -1.0, 1.0, skip_compute=dm_only, **kw)
    ttnn.generic_op([x, out], desc)
    ttnn.synchronize_device(device)
    print(f"BENCH_ROW variant={variant} shape={shape} dm_only={dm_only} ablate={os.environ.get('NB_ABLATE', '')}")
    if dm_only:
        return
    y = ttnn.to_torch(out)
    ref = _ref(xt)
    mism = (y != ref).sum().item()
    assert mism == 0, f"{variant} {shape}: {mism} mismatches vs torch"
