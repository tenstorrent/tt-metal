# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The intake smoke (e.g. "capital of France" -> "Paris") on the device model, shipped defaults (testing/smoke.py)."""

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.common.bringup.testing.smoke import run_smoke

S = spec()
pytestmark = device_timeout(S)


@mesh_parametrize
def test_smoke(mesh_device):
    out = run_smoke(S, mesh_device)
    assert out["ok"], out
