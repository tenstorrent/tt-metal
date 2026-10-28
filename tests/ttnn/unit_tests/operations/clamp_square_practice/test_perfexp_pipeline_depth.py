# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""pipeline_depth perf experiment for clamp_square_practice (isolated; does not touch the op).

Variant name grammar: r<R>w<W>i<I>o<O>  = reader DM block R tiles, writer DM block W tiles,
input CB depth I (in R-blocks), output CB depth O (in W-blocks). Compute DEST block stays the op's.
"op" = the real op descriptor + kernels, unmodified (honest baseline).

Env filters (run_safe_pytest word-splits -k):  PD_VARIANTS=op,r8w8i4o4  PD_SHAPES=1x1x4096x4096  PD_REPS=3
Correctness (bit-exact vs torch clamp(x)^2 in bf16) is the only assert; perf is printed (PD_ROW lines).
"""
import os
import re
import statistics

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest
import torch
import ttnn

from ttnn.operations.clamp_square_practice import clamp_square_practice_program_descriptor as pd
from ttnn.operations.clamp_square_practice.perf_experiments.pipeline_depth import variant as pv

SHAPES = [
    (1, 1, 4096, 4096),
    (1, 1, 8192, 8192),
    (1, 1, 1056, 2080),
    (8, 1, 1024, 1024),
    (2, 4, 256, 512),
    (1, 1, 32, 8192),
    (1, 1, 32, 32),
    (1, 1, 64, 128),
    (2, 64, 256),
    (1, 1, 64, 1024),
]
DEFAULT_VARIANTS = "op,r8w8i2o2,r8w8i4o4,r16w16i2o2,r16w16i3o3,r32w32i2o2,r32w32i3o3"
KEY = "DEVICE KERNEL DURATION [ns]"


def _durations(device):
    ttnn.ReadDeviceProfiler(device)
    out = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for p in programs:
            e = (getattr(p, "program_analyses_results", None) or {}).get(KEY)
            if e is not None:
                out.append(float(e.duration))
    return out


def _desc(name, x, out):
    if name.endswith("_nc"):  # ablation: compute payload stubbed (CB sync kept) -> DM-only floor
        desc, info = _desc(name[:-3], x, out)
        rk, wk, ck = desc.kernels
        ck.defines = [("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", "1")]
        desc.kernels = [rk, wk, ck]
        return desc, dict(info, ablation="no_compute")
    if name == "op":
        return pd.create_program_descriptor(x, out, -1.0, 1.0), {}
    m = re.fullmatch(r"r(\d+)w(\d+)i(\d+)o(\d+)", name)
    assert m, name
    r, w, i, o = map(int, m.groups())
    return pv.build(x, out, -1.0, 1.0, rd_tiles=r, wr_tiles=w, in_depth=i, out_depth=o)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_pipeline_depth(device, shape):
    sid = "x".join(map(str, shape))
    ss = os.environ.get("PD_SHAPES")
    if ss and sid not in ss.split(","):
        pytest.skip("filtered")
    variants = os.environ.get("PD_VARIANTS", DEFAULT_VARIANTS).split(",")
    reps = int(os.environ.get("PD_REPS", "3"))
    torch.manual_seed(0)
    xt = (torch.randn(shape, dtype=torch.float32) * 2).to(torch.bfloat16)
    ref = torch.clamp(xt.float(), -1.0, 1.0).pow(2).to(torch.bfloat16)
    x = ttnn.from_torch(
        xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    _durations(device)
    failures = []
    for name in variants:
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        desc, info = _desc(name, x, out)
        ttnn.generic_op([x, out], desc)  # compile + correctness run
        got = ttnn.to_torch(out)
        nbad = int((got.view(torch.int16) != ref.view(torch.int16)).sum())
        pcc = (
            torch.corrcoef(torch.stack([got.float().flatten(), ref.float().flatten()]))[0, 1].item()
            if got.numel() > 1
            else 1.0
        )
        _durations(device)
        for _ in range(reps):
            ttnn.generic_op([x, out], desc)
        ds = _durations(device)
        med = statistics.median(ds) if ds else float("nan")
        print(
            f"PD_ROW shape={sid} variant={name} ns={med:.0f} all={[round(d) for d in ds]} mismatches={nbad} pcc={pcc:.6f} {info}"
        )
        if nbad and not name.endswith("_nc"):
            failures.append((name, nbad))
    assert not failures, failures
