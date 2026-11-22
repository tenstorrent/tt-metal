# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Entry point of serve/server.py under pytest: the mesh opens through the bring-up fixtures (FABRIC_2D, the spec's
device params) and scripts/run_safe_pytest.sh gives the device lock, hang triage and the reset afterwards. Blocks until
SIGTERM / SIGINT (serve/stop.sh). Not a CI test."""

import pytest

from models.demos.common.bringup.testing.harness import mesh_parametrize


@pytest.mark.timeout(0)
@mesh_parametrize
def test_serve(mesh_device):
    from models.demos.glm53_flash_d_p_lb.serve.server import serve_forever

    serve_forever(mesh_device)
