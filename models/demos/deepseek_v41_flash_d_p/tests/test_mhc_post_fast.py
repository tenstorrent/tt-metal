# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""P.83 (MHC_POST_FAST): the fork's fast mhc_post against its old program on the model's mesh (resident runner, no
model load): ttnn/ttnn/bringup/mhc_post_ttnn/tests/unit/test_mhc_post_fast.py's cases, smallest first."""

import pytest

from models.demos.common.bringup.testing.harness import mesh_parametrize

BATCH_SHARED = True
BATCH_LAYERS = "executed"


@mesh_parametrize(timeout_s=900)
def test_mhc_post_fast(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    from ttnn.bringup.mhc_post_ttnn.tests.unit.test_mhc_post_fast import CASES, IDS, check_fast

    for case, name in zip(CASES, IDS):
        check_fast(mesh_device, *case)
        print(f"P.83 fast == old: {name} ok", flush=True)
