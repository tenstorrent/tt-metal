# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""P.81 (ATTN_ROOFLINE): ttnn.bringup.grouped_matmul against the stock batched matmul and float64 on the model's mesh
(resident runner, no model load): ttnn/ttnn/bringup/grouped_matmul_ttnn/tests/unit/test_grouped_matmul.py's cases."""

import pytest

from models.demos.common.bringup.testing.harness import mesh_parametrize

BATCH_SHARED = True
BATCH_LAYERS = "executed"


@mesh_parametrize(timeout_s=900)
def test_grouped_matmul_wo_a(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    from ttnn.bringup.grouped_matmul_ttnn.tests.unit.test_grouped_matmul import CASES, IDS, check, check_refuses

    for case, name in zip(CASES, IDS):
        check(mesh_device, *case)
        print(f"P.81 grouped_matmul: {name} ok", flush=True)
    check_refuses(mesh_device)
