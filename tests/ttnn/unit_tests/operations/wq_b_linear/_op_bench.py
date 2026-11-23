# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-time bench for wq_b_linear (run with scripts/run_safe_pytest.sh --profile).

Dispatches the op 1 warmup + 5 timed times on the single target shape; read the
DEVICE KERNEL DURATION column of the GenericOpDeviceOperation rows (take the median).
WQB_KNOBS="NAME=value,..." overrides program-descriptor knobs (python literals) for an A/B variant;
the PCC vs torch is printed so a variant's correctness is visible in the same run.
"""
import ast
import os

import torch
import ttnn

from ttnn.operations.wq_b_linear import wq_b_linear


def _apply_knobs():
    from ttnn.operations.wq_b_linear import wq_b_linear_program_descriptor as pd

    for kv in filter(None, os.environ.get("WQB_KNOBS", "").split(",")):
        k, v = kv.split("=", 1)
        assert hasattr(pd, k), k
        setattr(pd, k, ast.literal_eval(v))
        print("knob", k, "=", getattr(pd, k))


def test_bench(device):
    _apply_knobs()
    torch.manual_seed(0)
    x = torch.randn(1, 1, 160, 1536).bfloat16()
    w = (torch.randn(1536, 4096) * 0.02).bfloat16()
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    for i in range(6):
        y = wq_b_linear(tx, tw)
        if i == 0:
            ref = (x.float() @ ttnn.to_torch(tw).float()).flatten()
            got = ttnn.to_torch(y).float().flatten()
            print(
                "PCC",
                torch.corrcoef(torch.stack([ref, got]))[0, 1].item(),
                "nan_ref",
                ref.isnan().sum().item(),
                "nan_got",
                got.isnan().sum().item(),
                got.numel(),
                ref.numel(),
            )
        ttnn.deallocate(y)
    ttnn.synchronize_device(device)
