# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""posted_stateful_dm bake-off: per-transaction NoC cost of clamp_square_practice's DM stream.

Same descriptor (block 8, 32-block CBs, the op's real compute kernel); only the reader/writer source
and their defines change. See perf_experiments/posted_stateful_dm/kernels/*.cpp for the modes.
Env filters (run_safe_pytest word-splits -k): PS_VARIANTS=base,wr_posted  PS_SHAPES=1x1x4096x4096
PS_DMONLY=1 stubs compute (DM-only); PS_ABLATE=r|w additionally stubs reads|writes. Every full run is gated bit-exact vs the torch reference
(which the op's current kernels match exactly), with the output buffer pre-filled with a sentinel.
"""
import os
from pathlib import Path

import pytest
import torch
import ttnn

from ttnn.operations.clamp_square_practice import clamp_square_practice_program_descriptor as pd

KDIR = Path(pd.__file__).parent / "perf_experiments" / "posted_stateful_dm" / "kernels"

# name: (RD_MODE, WR_MODE, WR_POSTED)
VARIANTS = {
    "base": (0, 0, 0),
    "base_bench": (0, 0, 0),  # same as base but through the bench kernel files (sanity: bench == op)
    "wr_posted": (0, 0, 1),
    "wr_posted_nofence": (0, 0, 2),
    "wr_cache": (0, 1, 0),
    "wr_cache_posted": (0, 1, 1),
    "wr_stateful": (0, 2, 0),
    "wr_stateful_posted": (0, 2, 1),
    "rd_cache": (1, 0, 0),
    "rd_stateful": (2, 0, 0),
    "all_cache_posted": (1, 1, 1),
    "all_stateful_posted": (2, 2, 1),
    "all_stateful": (2, 2, 0),
}
SHAPES = [(1, 1, 4096, 4096), (8, 1, 1024, 1024), (2, 4, 256, 512), (1, 1, 32, 32), (1, 1, 96, 1312)]


def _ref(x):
    return (x.float().clamp(-1.0, 1.0) ** 2).to(torch.bfloat16)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_posted_stateful_dm(device, shape, variant):
    vs, ss = os.environ.get("PS_VARIANTS"), os.environ.get("PS_SHAPES")
    if (vs and variant not in vs.split(",")) or (ss and "x".join(map(str, shape)) not in ss.split(",")):
        pytest.skip("filtered")
    dm_only = os.environ.get("PS_DMONLY") == "1"
    torch.manual_seed(0)
    xt = torch.randn(shape, dtype=torch.bfloat16) * 2
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    # Sentinel-filled output: a variant that drops a write cannot pass on a stale buffer.
    out = ttnn.from_torch(
        torch.full(shape, 7.0, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    rm, wm, wp = VARIANTS[variant]
    desc = pd.create_program_descriptor(x, out, -1.0, 1.0)
    rk, wk, ck = desc.kernels
    abl = os.environ.get("PS_ABLATE", "")  # "r": stub reads (writer-only), "w": stub writes (reader-only)
    assert not abl or dm_only, "ablation only in the DM-only bench"
    if abl and variant == "base":
        pytest.skip("ablation needs the bench kernels; use base_bench")
    if variant != "base":
        rk.kernel_source = str(KDIR / "reader.cpp")
        wk.kernel_source = str(KDIR / "writer.cpp")
        rk.defines = [("RD_MODE", str(rm)), ("RD_STUB", "1" if abl == "r" else "0")]
        wk.defines = [("WR_MODE", str(wm)), ("WR_POSTED", str(wp)), ("WR_STUB", "1" if abl == "w" else "0")]
    if dm_only:
        ck.defines = [("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", "1")]
    desc.kernels = [rk, wk, ck]
    ttnn.generic_op([x, out], desc)
    ttnn.synchronize_device(device)
    print(f"BENCH_ROW variant={variant} shape={shape} dm_only={dm_only}")
    if dm_only:
        return
    y = ttnn.to_torch(out)
    mism = (y != _ref(xt)).sum().item()
    assert mism == 0, f"{variant} {shape}: {mism} mismatches vs torch"
