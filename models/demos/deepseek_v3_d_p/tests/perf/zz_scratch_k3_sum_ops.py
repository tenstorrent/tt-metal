# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Scratch: the K3 block's final sum (routed y + the shared expert's 3 part outputs, [320, 7168] bf16 tiles per chip),
device time: ttnn.sum + ttnn.add vs sum_ops (one pass) at a few unit sizes / distributions."""
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.perf.zz_scratch_k3_small_ops import run
from ttnn.bringup.combine_rf.sum_ops import sum_parts_prepare


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_sum_ops(mesh_device):
    S, H, P = 320, 7168, 3
    ah, bh = torch.randn(S, H), torch.randn(P * S, H)
    dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # noqa: E731
    a, b = dev(ah), dev(bh)
    ref = ah.bfloat16().float() + bh.bfloat16().float().reshape(P, S, H).sum(0)

    def ttnn_path():
        ys = ttnn.sum(ttnn.reshape(b, [P, S, H]), dim=0)
        y = ttnn.add(a, ys, dtype=ttnn.bfloat16)
        ttnn.deallocate(ys)
        return y

    run(mesh_device, "ttnn.sum + ttnn.add", ttnn_path)
    for bt in (1, 2, 4, 8):
        for dist in ("rr", "contig"):
            f = sum_parts_prepare(mesh_device, P=P, bt=bt, dist=dist)
            y = run(mesh_device, f"sum_ops bt {bt} {dist}", lambda: f(a, b))
            yh = ttnn.to_torch(y).float()
            assert torch.corrcoef(torch.stack([yh.flatten(), ref.flatten()]))[0, 1] > 0.9999
            ttnn.deallocate(y)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_layout_noc(mesh_device):
    """layout_ops with the reads on NOC0 (default) vs NOC1 (LAYOUT_NOC=default): the K3 block's conversions."""
    import os

    from ttnn.bringup.combine_rf.layout_ops import layout_op_prepare

    S = 320
    dev = lambda t, lay: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=lay, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # noqa: E731
    for noc in ("default", "swap"):
        os.environ["LAYOUT_NOC"] = noc
        for W in (3584, 7168):
            x = torch.randn(S, W).bfloat16()
            xt, xr = dev(x, ttnn.TILE_LAYOUT), dev(x, ttnn.ROW_MAJOR_LAYOUT)
            til, until = layout_op_prepare(mesh_device, "tilize"), layout_op_prepare(mesh_device, "untilize")
            a = run(mesh_device, f"{noc} untilize [{S},{W}]", lambda: until(xt))
            b = run(mesh_device, f"{noc} tilize [{S},{W}]", lambda: til(xr))
            assert torch.equal(ttnn.to_torch(a), x) and torch.equal(ttnn.to_torch(b), x)
