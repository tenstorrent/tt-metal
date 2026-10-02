# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Entry point of the OpenAI-compatible server (serve/server.py) under pytest, so the mesh opens through the same
fixtures as the model tests (fabric config, device params) and scripts/run_safe_pytest.sh gives the device lock, hang
triage and the reset afterwards. Blocks until SIGTERM / SIGINT (serve/stop.sh). Not a CI test."""

import pytest

from models.demos.mimo_v2_d_p.serve.server import serve_forever
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS


@pytest.mark.timeout(0)
@MESH_PARAMS
def test_serve(mesh_device, device_params):
    serve_forever(mesh_device, device_params)
