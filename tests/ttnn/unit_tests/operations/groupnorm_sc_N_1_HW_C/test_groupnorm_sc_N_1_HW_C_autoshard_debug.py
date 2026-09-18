# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 3 debug — the harness's auto-shard RM block-sharded cells (1024x2560 / 1024x1920 on
the 11x10 live grid: shard [103, 240] / [103, 176], K = 8 / 6, staged path with hw_mask) regressed
to garbage (pcc ~0.004, |y| ~1e17) while the pinned SDXL model shards pass. DO NOT DELETE — documents
the debugging process. Runs the exact golden cell two ways (golden helper vs direct call) under the
conftest device so the failing ingredient can be isolated."""
import collections

import pytest
import torch
import ttnn

from eval.sharding import auto_shard_config
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C as op


def _pattern(y, ref, shard_rows, shard_w):
    err = (y - ref).abs()
    bad = err > 0.5
    cnt = collections.Counter()
    for r, c in bad[0, 0].nonzero().tolist():
        cnt[(r // shard_rows, c // shard_w, (r % shard_rows) // 32, (c % shard_w) // 32)] += 1
    return bad.float().mean().item(), err.max().item(), sorted(cnt.items())[:60]


@pytest.mark.parametrize("shape", [(1, 1, 1024, 2560), (1, 1, 1024, 1920)])
@pytest.mark.parametrize("via", ["golden_helper", "direct"])
def test_autoshard_rm_block_sharded_cell(device, shape, via):
    N, _, HW, C = shape
    G = 32
    if via == "golden_helper":
        from eval.golden_tests.groupnorm_sc_N_1_HW_C.helpers import run_groupnorm_sc_N_1_HW_C

        run_groupnorm_sc_N_1_HW_C(
            {"input_tensor": shape, "num_groups": G},
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            affine="gamma_beta",
            affine_dtype=ttnn.bfloat16,
            affine_layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            in_place=True,
            extras={"pcc_threshold": 0.9995},
        )
        return
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    gamma = torch.randn(1, 1, 1, C, dtype=torch.float32).to(torch.bfloat16)
    beta = torch.randn(1, 1, 1, C, dtype=torch.float32).to(torch.bfloat16)
    mc = auto_shard_config(
        list(shape),
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    sh, sw = int(mc.shard_spec.shape[0]), int(mc.shard_spec.shape[1])
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    tg = ttnn.from_torch(gamma, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tb = ttnn.from_torch(beta, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = op(t, G, gamma=tg, beta=tb, in_place=True)
    y = ttnn.to_torch(out).float()
    ref = (
        torch.nn.functional.group_norm(
            x.float().squeeze(1).permute(0, 2, 1), G, gamma.float().reshape(C), beta.float().reshape(C), 1e-5
        )
        .permute(0, 2, 1)
        .unsqueeze(1)
    )
    frac, mx, pattern = _pattern(y, ref, sh, sw)
    print(f"\nshard [{sh},{sw}] bad frac {frac} max {mx} pattern {pattern}")
    assert frac == 0.0, f"bad frac {frac} max {mx} pattern {pattern}"
