# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Entry point of the serve stack (serve/server.py) under pytest, so scripts/run_safe_pytest.sh holds the device
lock for the server's whole life, gives hang triage and resets the device afterwards. The runner subprocess opens the
mesh; this process never does. Blocks until SIGTERM / SIGINT (serve/stop.sh). Not a CI test."""

import os

import pytest


@pytest.mark.timeout(0)
def test_serve():
    if os.environ.get("UP_FRONT_COLLECT") == "1":
        pytest.skip("precompile pass: the runner subprocess needs the chips; runs in the real pass only")
    from models.demos.xing40_a4b_d_p.serve.server import serve_forever

    serve_forever()
