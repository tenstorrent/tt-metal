# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Print flat_routed_expert's plan at GLM's shape (H 4096, I 2048; GLM_PLAN_SHAPES for others; 36 local experts of 288, capacity 8192, bfp4, pin 1)
for the four x / h tile-format regimes (GLM_PLAN_KEYS: comma list of keys to print; default the L1-relevant ones)."""

import os

import pytest

import ttnn

# GLM_PLAN_SHAPES: "H:I,H:I" (default GLM's 4096:2048)
SHAPES = [tuple(int(v) for v in s.split(":")) for s in os.environ.get("GLM_PLAN_SHAPES", "4096:2048").split(",")]
KEYS = os.environ.get("GLM_PLAN_KEYS", "np,g,mt,x_slots,hbuf,gu_rp,arena_tiles,nd,n_rd,nk_gu").split(",")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_flat_plan(device):
    for H, I in SHAPES:
        for xb in (False, True):
            for hb in (False, True):
                p = ttnn._ttnn.operations.bringup.flat_routed_expert_plan(
                    device, H, I, 36, 288, 8192, weights_bf8=False, pin=1, x_bf16=xb, h_bf16=hb
                )
                print(
                    f"[plan] H {H} I {I} x_bf16 {int(xb)} h_bf16 {int(hb)}: " + ", ".join(f"{k} {p[k]}" for k in KEYS),
                    flush=True,
                )
