# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The exact prefill linear alone, for the device profiler: T2_SHAPE=K,N (default 17408,5120), T2_T tokens (default
2048), T2_CLASS=2 (tprefill/linear2.py, default), 3 (linear3.py) or 6 (linear6.py), T2_GLU=1: gate / up pair records; checks the
output when BONSAI_T2_DEFINES / BONSAI_T6_DEFINES are empty."""

import os

import numpy as np
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.tprefill.linear import reference
from models.demos.blackhole.bonsai2.tt.tprefill.linear2 import TernaryPrefillLinear2


def test_t2_prof(device):
    K, N = (int(v) for v in os.environ.get("T2_SHAPE", "17408,5120").split(","))
    T = int(os.environ.get("T2_T", "2048"))
    cls = TernaryPrefillLinear2
    if os.environ.get("T2_CLASS") == "3":
        from models.demos.blackhole.bonsai2.tt.tprefill.linear3 import TernaryPrefillLinear3 as cls
    if os.environ.get("T2_CLASS") == "6":
        from models.demos.blackhole.bonsai2.tt.tprefill.linear6 import TernaryPrefillLinear6 as cls
    rng = np.random.default_rng(0)
    t = rng.integers(-1, 2, size=(K, N)).astype(np.int8)
    s = (rng.random((N, K // 128)) * 0.02 + 0.005).astype(np.float16)
    lin = cls.from_host(device, t, s)
    x = torch.randn(1, 1, T, K).bfloat16()
    xdt = ttnn.bfloat8_b if os.environ.get("T2_XBF8") == "1" else ttnn.bfloat16
    xt = ttnn.from_torch(x, dtype=xdt, layout=ttnn.TILE_LAYOUT, device=device)
    y = lin(xt)
    if not (os.environ.get("BONSAI_T2_DEFINES") or os.environ.get("BONSAI_T6_DEFINES")):
        s_ref = s
        if os.environ.get("T2_CLASS") == "6":  # folded scales: 5 significant bits
            from models.demos.blackhole.bonsai2.tt.tprefill.linear6 import round_sig

            s_ref = round_sig(s).astype(np.float32)
        ref = reference(x.reshape(T, K), t, s_ref)
        yy = ttnn.to_torch(y).float().reshape(T, N)
        rel = ((yy - ref).norm() / ref.norm()).item()
        print(f"T2REL {rel:.5f}")
        assert rel < 0.02, rel
    for _ in range(4):
        lin(xt)
    ttnn.synchronize_device(device)
    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
        ttnn.ReadDeviceProfiler(device)
