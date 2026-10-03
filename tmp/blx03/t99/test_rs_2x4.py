# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""t99: fused RMSNorm(a + b) + sum on a 2x4 submesh of the BH galaxy (the unit-test shapes)."""

import importlib.util
import os

import pytest

import ttnn

_UT = "tests/ttnn/nightly/unit_tests/operations/experimental/test_dit_rms_norm_unary_fused.py"
_spec = importlib.util.spec_from_file_location("ut_rs", os.path.join(os.environ.get("T99_SRC", "."), _UT))
ut = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ut)

SHAPES = [
    (1, 19, 17, 15, 1024),
    (1, 5, 34, 30, 512),
    (1, 5, 36, 32, 512),
    (1, 3, 68, 60, 256),
    (1, 3, 72, 64, 256),
    (1, 2, 136, 120, 128),
    (1, 2, 144, 128, 128),
]


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_rs_2x4(mesh_device, device_params):
    # A bare 2x4 open on the BH galaxy fails fabric init; open the system mesh and carve the 2x4.
    sub = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
    for shape in SHAPES:
        print(f"RS shape={shape}", flush=True)
        ut.run_residual_sum_test(sub, shape)
        print(f"RS_OK shape={shape}", flush=True)
