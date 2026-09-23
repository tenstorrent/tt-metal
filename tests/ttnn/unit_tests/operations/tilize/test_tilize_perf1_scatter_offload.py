# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""scatter_offload bake-off harness (perf_experiments/scatter_offload). Run with --profile.

SO_VARIANTS="name,name,..." picks from VARIANTS below; SO_SHAPES="1x1x16384x64,..." the shapes;
SO_GUARDS=1 adds the low_l1 narrow guard. Every non-ablated cell is checked bit-exact.
The real op is untouched: KERNEL_DIR and BANK_COALESCE_STAGE_DEPTH are monkeypatched, and a
wrapper around create_program_descriptor injects the SO_* defines + two semaphores ONLY when the
host took the bank_coalesced reader (reader CT arg 26 != 0).
"""
import os
import sys
from pathlib import Path

import pytest
import torch
import ttnn

from ttnn.operations.tilize import tilize
import ttnn.operations.tilize.tilize_program_descriptor as pd

tmod = sys.modules["ttnn.operations.tilize.tilize"]
ROOT = Path(__file__).resolve().parents[5]
SO_DIR = ROOT / "ttnn/ttnn/operations/tilize/perf_experiments/scatter_offload/kernels_so"
REAL_DIR = ROOT / "ttnn/ttnn/operations/tilize/kernels"

# name: (kernel dir, SO_MODE, NCRISC share num/den, staging depth, ABL_W); a "_p" suffix before
# "_W" adds SO_PRIO_STORE (BRISC stores before it scatters).
VARIANTS = {
    "real": (REAL_DIR, None, (1, 2), 2, False),
    "base": (SO_DIR, 0, (1, 2), 2, False),
    "base_s3": (SO_DIR, 0, (1, 2), 3, False),
    "half": (SO_DIR, 1, (1, 2), 2, False),
    "half_s3": (SO_DIR, 1, (1, 2), 3, False),
    "all": (SO_DIR, 2, (0, 2), 2, False),
    "all_s3": (SO_DIR, 2, (0, 2), 3, False),
    "split": (SO_DIR, 3, (1, 2), 2, False),
    "split_s3": (SO_DIR, 3, (1, 2), 3, False),
    "split_q3": (SO_DIR, 3, (3, 4), 2, False),  # BRISC reads + scatters a quarter of the banks
    "half_q": (SO_DIR, 1, (3, 4), 2, False),  # BRISC scatters a quarter of the banks
    "half_e": (SO_DIR, 1, (5, 6), 2, False),  # BRISC scatters a sixth (2 of 12 banks)
    "split_e": (SO_DIR, 3, (5, 6), 2, False),  # BRISC reads + scatters a sixth
}
for _n in ("half", "half_q", "all"):
    VARIANTS[_n + "_p"] = VARIANTS[_n]
for _n, (_d, _m, _s, _st, _) in list(VARIANTS.items()):
    VARIANTS[_n + "_W"] = (_d, _m, _s, _st, True)

NAMES = os.environ.get("SO_VARIANTS", "base,half").split(",")
SHAPES = [tuple(int(d) for d in s.split("x")) for s in os.environ.get("SO_SHAPES", "1x1x16384x64").split(",")]
LOW_L1 = [False, True] if os.environ.get("SO_GUARDS") == "1" else [False]


def _wrap(orig, mode, share, abl_w, prio_store=False):
    def create(*args, **kwargs):
        desc = orig(*args, **kwargs)
        ks = desc.kernels
        reader, writer = ks[0], ks[1]
        stage = reader.compile_time_args[26]
        defs = []
        if abl_w:
            defs.append(("ABL_W", "1"))
        if mode is not None and mode != 0 and stage != 0:
            if prio_store:
                defs.append(("SO_PRIO_STORE", "1"))
            defs += [
                ("SO_MODE", str(mode)),
                ("SO_NC_NUM", str(share[0])),
                ("SO_DEN", str(share[1])),
                ("SO_STAGE", str(stage)),
                ("SO_CB_STAGING", str(pd.CB_COALESCE_STAGING)),
                ("SO_SEM_STAGED", "0"),
                ("SO_SEM_DONE", "1"),
            ]
            assert len(desc.semaphores) == 0, "co-read and bank_coalesced are exclusive"
            if mode == 3:
                # BRISC's staging ring sits after NCRISC's (a bank's staged byte range depends on the
                # unit's run structure, so a shared slot would overlap across units).
                cbs = desc.cbs
                for cb in cbs:
                    if any(f.buffer_index == pd.CB_COALESCE_STAGING for f in cb.format_descriptors):
                        cb.total_size = 2 * cb.total_size
                desc.cbs = cbs
            desc.semaphores = [
                ttnn.SemaphoreDescriptor(id=i, core_ranges=reader.core_ranges, initial_value=0) for i in (0, 1)
            ]
        print(f"SO cell: coalesce_depth={stage} mode={mode if stage else 'n/a (not bank_coalesced)'} defs={defs}")
        for k in (reader, writer):
            k.defines = list(k.defines) + defs
        desc.kernels = [reader, writer] + list(ks[2:])
        return desc

    return create


@pytest.mark.parametrize("low_l1", LOW_L1, ids=lambda b: "low_l1" if b else "l1")
@pytest.mark.parametrize("variant", NAMES)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_so(device, monkeypatch, shape, variant, low_l1):
    kdir, mode, share, stage, abl_w = VARIANTS[variant]
    monkeypatch.setattr(pd, "KERNEL_DIR", kdir)
    monkeypatch.setattr(pd, "BANK_COALESCE_STAGE_DEPTH", stage)
    if abl_w and kdir == REAL_DIR:
        monkeypatch.setattr(pd, "KERNEL_DIR", SO_DIR)  # the guard lives in the experiment copy
    monkeypatch.setattr(
        tmod, "create_program_descriptor", _wrap(pd.create_program_descriptor, mode, share, abl_w, "_p" in variant)
    )
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = tilize(t, low_l1=True) if low_l1 else tilize(t)
    ttnn.synchronize_device(device)
    if not abl_w:
        y = ttnn.to_torch(out)
        if not torch.equal(y, x):
            bad = (y != x).reshape(-1, shape[-1]).any(dim=1).nonzero().flatten()
            print(
                f"SO MISMATCH sticks={bad.numel()} first={bad[:40].tolist()} "
                f"mod12={torch.bincount(bad % 12, minlength=12).tolist()} "
                f"by_core={torch.bincount(bad // 256, minlength=64).tolist()}"
            )
            xr, yr = x.reshape(-1, shape[-1]), y.reshape(-1, shape[-1])
            for b in bad[:24].tolist():
                src = (xr == yr[b]).all(dim=1).nonzero().flatten().tolist()
                print(f"SO BAD stick {b} (core {b // 256} local {b % 256}) holds x-stick {src[:3]}")
        assert torch.equal(y, x)
    print(f"SO {variant} {shape} low_l1={low_l1} done")
