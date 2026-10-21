# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch: is a routed expert's device weight layout independent of experts_per_chip (so the weight cache can be
per expert, not per mesh)? Plans each model's shape at two experts-per-chip values (pure DRISC, Galaxy-like grid),
compares the layout fields and the laid-out blocks of the same two experts (flat_weight_regions)."""

import json

import pytest
import torch
from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import flat_weight_regions
from ttnn.bringup.flat_routed_expert_ttnn.flat_layouts import DRISC_ONLY, GLM_FLASH, flat_layout, k3_gu64

import ttnn

SHAPES = {  # name: hidden (routed width), intermediate, (epc a, epc b), global experts, layout
    "Kimi K3": (3584, 3072, (28, 56), 896, {**k3_gu64(True), "drisc": True, "drisc_pure": 1}),
    "Kimi K2.7": (7168, 2048, (12, 24), 384, {**DRISC_ONLY, "drisc": True}),
    "GLM-5.3": (6144, 2048, (8, 16), 256, {**DRISC_ONLY, "drisc": True}),
    "MiniMax M3": (6144, 3072, (4, 8), 128, {**DRISC_ONLY, "drisc": True}),
    "GLM-5.3-Flash": (4096, 2048, (9, 18), 288, {**GLM_FLASH, "drisc": True, "drisc_pure": 1}),
}
LAYOUT_KEYS = (
    "nsg",
    "n_rd",
    "n_rd_sg",
    "np",
    "g",
    "rg",
    "uneven",
    "hg",
    "nu",
    "rgs",
    "ps0",
    "nk_gu",
    "kblk",
    "nd",
    "pcds",
    "col0s",
    "rdown",
    "n_rdn",
    "pcd_r",
    "kd_r",
    "nblk_r",
    "rem_cols",
    "banks",
    "dw_rot",
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_flat_epc_layout(device):
    out = {}
    g = torch.Generator().manual_seed(0)
    for name, (H, I, epcs, E, lay) in SHAPES.items():
        plans = [
            ttnn._ttnn.operations.bringup.flat_routed_expert_plan(
                device,
                H,
                I,
                epc,
                E,
                4096,
                weights_bf8=False,
                pin=1,
                x_bf16=False,
                h_bf16=False,
                layout=flat_layout(lay),
            )
            for epc in epcs
        ]
        diff = {k: (plans[0][k], plans[1][k]) for k in LAYOUT_KEYS if plans[0][k] != plans[1][k]}
        W = [
            (torch.randn(H, I, generator=g), torch.randn(H, I, generator=g), torch.randn(I, H, generator=g))
            for _ in range(2)
        ]
        regs = [flat_weight_regions(W, dict(p)) for p in plans]
        detail = []
        same = True
        for kind, ra, rb in zip(("gate_up", "down", "reader_tail"), regs[0], regs[1]):
            if ra is None and rb is None:
                continue
            if len(ra) != len(rb):
                same = False
                detail.append(f"{kind}: {len(ra)} vs {len(rb)} regions")
                continue
            for i, (x, y) in enumerate(zip(ra, rb)):
                n = min(x.shape[0], y.shape[0])  # (the regions are padded to the plan's size: compare the
                pre = torch.equal(x[:n], y[:n])  # two experts' blocks, then that the rest is zero padding)
                pad = bool((x[n:] == 0).all()) and bool((y[n:] == 0).all())
                if not (pre and pad):
                    same = False
                    detail.append(
                        f"{kind}[{i}]: shapes {tuple(x.shape)} vs {tuple(y.shape)}, prefix equal {pre}, rest zero {pad}"
                    )
                    break
            detail.append(f"{kind}: {len(ra)} regions, shapes {tuple(ra[0].shape)} vs {tuple(rb[0].shape)}")
        out[name] = {
            "epcs": epcs,
            "layout_diff": {k: [str(v[0]), str(v[1])] for k, v in diff.items()},
            "blocks_equal": same,
            "detail": detail,
        }
    print("EPC " + json.dumps(out), flush=True)
