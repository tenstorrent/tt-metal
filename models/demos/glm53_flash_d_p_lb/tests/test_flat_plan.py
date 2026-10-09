# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Print flat_routed_expert's plan at GLM's shape (H 4096, I 2048, 36 local experts of 288, capacity 8192, bfp4, pin 1)
for the four x / h tile-format regimes (GLM_PLAN_KEYS: comma list of keys to print; default the L1-relevant ones)."""

import os

import pytest

import ttnn

KEYS = os.environ.get("GLM_PLAN_KEYS", "np,g,mt,x_slots,hbuf,gu_rp,arena_tiles,nd,n_rd,nk_gu").split(",")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_flat_plan(device):
    for xb in (False, True):
        for hb in (False, True):
            p = ttnn._ttnn.operations.bringup.flat_routed_expert_plan(
                device, 4096, 2048, 36, 288, 8192, weights_bf8=False, pin=1, x_bf16=xb, h_bf16=hb
            )
            print(f"[plan] x_bf16 {int(xb)} h_bf16 {int(hb)}: " + ", ".join(f"{k} {p[k]}" for k in KEYS), flush=True)
