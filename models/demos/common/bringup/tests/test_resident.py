# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F281: the resident runner's device process (models/demos/common/bringup/resident.py starts it under
scripts/run_safe_pytest.sh): the mesh opened once, requests served until stop / yield / a restart it asks for
(testing/resident.py). No timeout of its own: each run keeps its own pytest-timeout; hangs are the safe runner's
dispatch timeout."""

from models.demos.common.bringup.testing import resident
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec

S = spec()


@mesh_parametrize(timeout_s=0)
def test_resident(mesh_device):
    assert resident.serve(mesh_device, S.mesh) in ("stop", "yield", "restart")
