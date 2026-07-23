# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Refinement 4b regression test — RM-input WIDTH/BLOCK sharded.
# DO NOT DELETE — documents the RM-sharded sub-scheme coverage.
#
# RM WIDTH/BLOCK sharding places a core's W-slice as an arbitrary-width row-major
# shard (a multiple of the RM granule 8/4 el, generally sub-tile). The op reads the
# resident shard via a zero-copy alias + local loopback repack into tile-padded
# sticks, phase-aligns to the global tile grid (g0 = w_offset//32, phase = w_offset%32),
# reduces cross-core with a per-core partial scaler, and untilizes back to the shard.
# These cases exercise: sub-tile widths, per_w_t=2 (wide), boundary cores (W not a
# multiple of sw), H-non-aligned shards, BLOCK (H+W split), gamma/no-gamma, fp32.

import pytest
import torch

import ttnn

from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED
BLOCK = ttnn.TensorMemoryLayout.BLOCK_SHARDED


def _pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _ref(x, g=None, eps=1e-6):
    x = x.float()
    o = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if g is not None:
        o = o * g.float().reshape(-1)
    return o


def _run(device, shape, ml, gamma, dt):
    torch.manual_seed(0)
    tdt = torch.float32 if dt == ttnn.float32 else torch.bfloat16
    ti = torch.randn(shape).to(tdt)
    cfg = auto_shard_config(list(shape), ml, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dt, device=device)
    xt = ttnn.from_torch(ti, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=cfg)
    gt = None
    tg = None
    if gamma:
        W = shape[-1]
        tg = torch.randn(W).to(tdt)
        gt = ttnn.from_torch(tg.reshape(1, 1, 1, W), dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    cc = ttnn.ComputeConfigDescriptor()
    cc.math_fidelity = ttnn.MathFidelity.HiFi4
    cc.fp32_dest_acc_en = dt == ttnn.float32
    cc.math_approx_mode = False
    out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cc, memory_config=xt.memory_config())
    got = ttnn.to_torch(out)
    exp = _ref(ti, tg)
    p = _pcc(got, exp)
    assert p > 0.99, f"{shape} {ml} gamma={gamma} {dt}: PCC={p} maxdiff={(got.float()-exp.float()).abs().max().item()}"


# (shape, memory_layout, gamma, dtype)
_CASES = [
    ((1, 1, 32, 64), WIDTH, True, ttnn.bfloat16),  # sw=8, 1 padded tile
    ((1, 1, 64, 128), WIDTH, True, ttnn.bfloat16),  # HT_LOCAL=2
    ((1, 1, 32, 50), WIDTH, True, ttnn.bfloat16),  # W non-aligned, boundary core
    ((1, 1, 32, 4096), WIDTH, True, ttnn.bfloat16),  # per_w_t=2, K=103, boundary
    ((1, 1, 17, 50), WIDTH, True, ttnn.bfloat16),  # H + W non-aligned
    ((2, 4, 128, 512), WIDTH, True, ttnn.bfloat16),  # HT_LOCAL=32, K=64
    ((1, 32, 128), WIDTH, True, ttnn.bfloat16),  # 3D
    ((32, 64), WIDTH, True, ttnn.bfloat16),  # 2D
    ((1, 1, 32, 64), WIDTH, False, ttnn.bfloat16),  # no gamma
    ((1, 1, 32, 64), WIDTH, False, ttnn.float32),  # fp32
    ((1, 1, 32, 4096), WIDTH, True, ttnn.float32),  # fp32 wide
    ((1, 1, 256, 512), BLOCK, True, ttnn.bfloat16),  # BLOCK sub-tile H (26) + W (48)
    ((2, 4, 128, 512), BLOCK, True, ttnn.bfloat16),  # BLOCK HT_LOCAL=4
    ((1, 1, 64, 128), BLOCK, False, ttnn.bfloat16),  # BLOCK no gamma
]


@pytest.mark.parametrize("shape,ml,gamma,dt", _CASES)
def test_rms_norm_rm_sharded(device, shape, ml, gamma, dt):
    _run(device, shape, ml, gamma, dt)
