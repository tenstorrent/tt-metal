# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Folder-local mesh_device fixture for the deepseek pipeline smoke test.

Why this exists: the parent ``exabox/conftest.py:75`` provides its own
``mesh_device`` fixture that pops only ``fabric_config`` from
``device_params`` before calling ``ttnn.open_mesh_device(**device_params)``.
The deepseek pipeline tests pass ``fabric_router_config`` (and on other
configs, ``worker_l1_size``) through ``device_params``, which leaks into
``open_mesh_device`` as an unknown kwarg and fails with::

  TypeError: open_mesh_device() got an unexpected keyword argument
  'fabric_router_config'

The top-level ``conftest.bh_2d_mesh_device_context`` knows how to thread
all the BH fabric-config kwargs through ``set_fabric`` before opening the
mesh, so we override the fixture here to use it. This mirrors the pattern
the (now-removed) single_pod folder used.
"""

from __future__ import annotations

import pytest

from conftest import bh_2d_mesh_device_context


@pytest.fixture(scope="function")
def mesh_device(request, device_params):
    try:
        param = request.param
    except (ValueError, AttributeError):
        param = (4, 2)
    assert param == (4, 2), f"deepseek pipeline tests expect mesh_device=(4, 2) per rank, got {param}"

    with bh_2d_mesh_device_context(device_params) as md:
        yield md
