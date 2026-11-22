# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PB.8 (MHC_POST_V2): the fork's v2 mhc_post against its fast (P.83) program on the model's mesh (resident runner, no
model load): ttnn/ttnn/bringup/mhc_post_ttnn/tests/unit/test_mhc_post_v2.py's cases, smallest first. MHCV2_CASES (a
comma list of case ids) picks some."""

import os

import pytest

from models.demos.common.bringup.testing.harness import mesh_parametrize

BATCH_SHARED = True
BATCH_LAYERS = "executed"
BATCH_ITEM_VARS = ("MHCV2_CASES",)


@mesh_parametrize(timeout_s=900)
def test_mhc_post_v2(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    from ttnn.bringup.mhc_post_ttnn.tests.unit.test_mhc_post_v2 import CASES, IDS, check_v2

    pick = [c for c in os.environ.get("MHCV2_CASES", "").split(",") if c]
    for case, name in zip(CASES, IDS):
        if pick and name not in pick:
            continue
        check_v2(mesh_device, *case)
        print(f"PB.8 v2 == fast: {name} ok", flush=True)
