# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""otherlayers: the fused verify-indexer score op (tt/idx_score.py) alone on one device: traced time, profiler zones (DSV41_OL_SCORE_DIAG=1), bit equality of variants against the stock op.
Env: OLS_VARIANTS comma list of env settings sets 'name:K=V+K=V' (default base only)."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_ol_idx_ops import TRACE, traced_us
from models.demos.blackhole.deepseek_v41_flash.tt import idx_score as IS
from models.demos.blackhole.deepseek_v41_flash.tt import pf_fp4


def _inputs(device, U, n, ns, seed=0):
    R = U * n
    g = torch.Generator().manual_seed(seed)
    q = ttnn.from_torch(
        (torch.randn(U, 1, 32 * n, 128, generator=g) * 2).to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    q = pf_fp4.fp4_fused(q)  # fp4-simulated values like the real queries
    kc = ttnn.from_torch(
        torch.randn(U, 1, ns, 128, generator=g).float(), device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
    )
    w = ttnn.from_torch(
        (torch.randn(1, 1, R, 32, generator=g) * 0.1).to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    nv = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
    nv[0, 0, 0, :R] = torch.tensor([ns - 5 - (r % n) for r in range(R)], dtype=torch.int32)
    nvalid = ttnn.from_torch(nv, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    valid = ttnn.from_torch(
        torch.full((1, 1, 1, 1), ns - 5, dtype=torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    return q, kc, w, nvalid, valid


def _variants():
    out = [("base", {})]
    for v in os.environ.get("OLS_VARIANTS", "").split(","):
        if v.strip():
            name, _, envs = v.partition(":")
            out.append((name, dict(kv.split("=", 1) for kv in envs.split("+") if kv)))
    return out


@pytest.mark.parametrize("device_params", TRACE, indirect=True)
@pytest.mark.parametrize("U,n,ns", [(2, 4, 8224), (3, 4, 8224), (2, 4, 4128), (2, 4, 16416)])
def test_score(device, U, n, ns):
    q, kc, w, nvalid, valid = _inputs(device, U, n, ns)
    ref = None
    for name, env in _variants():
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            f = lambda: IS.score_mask_fused(q, kc, w, nvalid, n, ns, valid=valid)
            outs = [ttnn.to_torch(f()) for _ in range(3)]
            eq = all(bool((o == outs[0]).all()) for o in outs)
            if ref is None:
                ref = outs[0]
            same = bool((outs[0].to(torch.float32) == ref.to(torch.float32)).all())
            print(
                f"score U={U} n={n} ns={ns} {name}: deterministic {eq}, equal to base {same}, {traced_us(device, f):.1f} us (traced)",
                flush=True,
            )
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        ttnn.ReadDeviceProfiler(device)
