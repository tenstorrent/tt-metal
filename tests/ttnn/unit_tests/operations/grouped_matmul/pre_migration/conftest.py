# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures of the grouped_matmul pre-migration suite.

``gmm_mesh``: the whole system mesh, opened once a session with FABRIC_2D (the bring-up owner rule: every mesh opens
with 2D fabric; the op itself moves no data between chips). Inputs are replicated unless a test says otherwise, so every chip must produce the same bytes.
Cache-count tests clear the program cache themselves before they measure.
"""

import pytest
import ttnn


@pytest.fixture(scope="session")
def gmm_mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    mesh = ttnn.open_mesh_device(l1_small_size=24576)
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
