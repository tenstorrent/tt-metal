# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""In-process device-ns probe for mhc_pre (Refinement 4). Reads DEVICE KERNEL DURATION via
ttnn.ReadDeviceProfiler, so it does not depend on the Tracy capture tool. Needs TT_METAL_DEVICE_PROFILER=1
(set on the command line). Perf is printed, never asserted; knob sets come from test_mhc_pre_perf_sweep.
Run:  TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
      scripts/run_safe_pytest.sh --run-all -s <this file>
"""

import os

import pytest
import torch
import ttnn

import ttnn.operations.mhc_pre.mhc_pre_program_descriptor as pd
from ttnn.operations.mhc_pre import mhc_pre
from .test_mhc_pre_perf_sweep import SHAPES, KNOBS

_KEY = "DEVICE KERNEL DURATION [ns]"
_DTYPES = {"xbf16": ttnn.bfloat16, "xf32": ttnn.float32}
_SEL_DT = os.environ.get("MHC_PRE_PERF_DTYPES", "xbf16,xf32").split(",")
_SEL_SH = os.environ.get("MHC_PRE_PERF_SHAPES")
_SHAPES = [s for s in SHAPES if _SEL_SH is None or f"{s[-2]}x{s[-1] // 4}" in _SEL_SH.split(",")]


def _read_ns(device):
    ttnn.ReadDeviceProfiler(device)
    out = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for p in programs:
            e = (getattr(p, "program_analyses_results", None) or {}).get(_KEY)
            if e is not None:
                out.append(float(e.duration))
    return out


@pytest.mark.skipif(os.environ.get("TT_METAL_DEVICE_PROFILER") != "1", reason="needs TT_METAL_DEVICE_PROFILER=1")
def test_mhc_pre_perf_inproc(device, monkeypatch):
    rows = []
    for x_shape in _SHAPES:
        for dt_name in _SEL_DT:
            for knob, patch in KNOBS.items():
                with monkeypatch.context() as m:
                    for name, value in patch.items():
                        m.setattr(pd, name, value)
                    torch.manual_seed(0)
                    nc = x_shape[-1]
                    x = torch.randn(x_shape, dtype=torch.float32)
                    w = torch.randn((nc, 24), dtype=torch.float32) / nc**0.5
                    b = torch.randn((1, 24), dtype=torch.float32)
                    tx = ttnn.from_torch(x, dtype=_DTYPES[dt_name], layout=ttnn.TILE_LAYOUT, device=device)
                    tw = ttnn.from_torch(w, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
                    tb = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
                    _read_ns(device)  # drain anything earlier
                    y, post, comb = mhc_pre(tx, tw, tb, scale=(1.0, 1.0, 1.0))
                    ttnn.synchronize_device(device)
                    ns = _read_ns(device)
                    for t in (y, post, comb):
                        assert torch.isfinite(ttnn.to_torch(t)).all()
                rows.append((f"{x_shape[-2]}x{x_shape[-1] // 4}", dt_name, knob, ns))
    for r in rows:
        print("PERF", *r[:3], " ".join(f"{v / 1000:.1f}us" for v in r[3]))
