# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cross-core (WIDTH/BLOCK) reduction debug tests — DO NOT DELETE.

Captures the Refinement-5 "remaining 2 golden cells" investigation: bf16 +
fp32_dest_acc_en=False + wide-W BLOCK_SHARDED + gamma, shapes (1,1,32,8192) and
(1,32,8192), which reported a DETERMINISTIC uniform ~2.0x scale error
(relRMS ~= 0.9997, PCC ~= 0.9999) on the implementer's build.

The ttnn-expert-debugger investigation established:
  * The error is a uniform 2.0x scale (mean ratio 1.9993 across every core and
    W-tile), NOT a data scramble (PCC ~= 0.9999).
  * It is confined to the pass-2 gamma multiply path (no-gamma passes; the
    cross-core reduce / 1-rms broadcast is correct).
  * It could NOT be reproduced once the compute kernel was recompiled from the
    (byte-identical) current source: the failing configs, fresh wide configs
    (nwb=8, W=12288, WIDTH), and the 3D shape all pass deterministically. The 2x
    was tied to a STALE JIT-cached kernel binary predating the 2c1a6caccd
    "clean cb_partial emit" fix, not to the current kernel logic.

This test asserts the failing configs pass (and their passing siblings stay
green), with several seeds, so a regression / a resurfacing race is caught.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

BLOCK = ttnn.TensorMemoryLayout.BLOCK_SHARDED
WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED


def _cfg(acc):
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = acc
    cfg.math_approx_mode = False
    return cfg


def _ref(x, g):
    xf = x.to(torch.float32)
    inv = 1.0 / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + 1e-6)
    out = xf * inv
    if g is not None:
        out = out * g.to(torch.float32).reshape(-1)
    return out


# (shape, layout, acc, with_gamma) — the exact failing corner + its passing siblings.
CASES = [
    ((1, 1, 32, 8192), BLOCK, False, True),  # briefing FAIL corner (nwb=4, ny=1)
    ((1, 32, 8192), BLOCK, False, True),  # briefing FAIL corner (3D)
    ((1, 1, 32, 8192), BLOCK, True, True),  # sibling: acc=True (reported PASS)
    ((1, 1, 32, 8192), BLOCK, False, False),  # sibling: no gamma (reported PASS)
    ((1, 1, 128, 8192), BLOCK, False, True),  # sibling: ny=4 (reported PASS)
    ((1, 1, 32, 4096), BLOCK, False, True),  # sibling: nwb=2 (reported PASS)
    ((1, 1, 32, 8192), WIDTH, False, True),  # WIDTH wide-W, gamma, acc=False
]


@pytest.mark.parametrize("shape,layout,acc,with_gamma", CASES)
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_cross_core_wide_w_gamma_scale(device, shape, layout, acc, with_gamma, seed):
    """Uniform-scale regression guard: output must match reference within bf16 PCC AND
    the mean elementwise ratio must be ~1.0 (a 2.0x scale bug would trip the ratio
    check even where PCC alone stays ~1)."""
    torch.manual_seed(seed)
    x = torch.randn(shape, dtype=torch.bfloat16)
    g = torch.randn(shape[-1], dtype=torch.bfloat16) if with_gamma else None

    mc = auto_shard_config(list(shape), layout, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    gt = (
        ttnn.from_torch(g.reshape(1, 1, 1, shape[-1]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        if with_gamma
        else None
    )
    out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=_cfg(acc), memory_config=xt.memory_config())
    res = ttnn.to_torch(out).float().reshape(-1, shape[-1])
    exp = _ref(x, g).reshape(-1, shape[-1])

    assert_with_pcc(exp, res, 0.995)
    mask = exp.abs() > 1e-2
    ratio = (res[mask] / exp[mask]).mean().item()
    assert 0.9 < ratio < 1.1, f"uniform-scale bug: mean(out/exp)={ratio:.4f} (expect ~1.0) shape={shape} acc={acc}"
