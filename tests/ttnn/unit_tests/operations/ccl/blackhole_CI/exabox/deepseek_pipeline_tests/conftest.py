# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Folder-local mesh device fixture for the deepseek pipeline smoke test.

Named ``deepseek_pipeline_mesh_device`` (not ``mesh_device``) so it does
not shadow the parent ``exabox/conftest.py``'s ``mesh_device`` or the
root ``mesh_device`` — shadowed fixtures are awkward to work around when
running adjacent tests locally.

Why this fixture exists at all: the parent ``exabox/conftest.py``'s
``mesh_device`` pops only ``fabric_config`` from ``device_params`` before
calling ``ttnn.open_mesh_device(**device_params)``. The deepseek
pipeline tests pass ``fabric_router_config`` (and on other configs,
``worker_l1_size``) through ``device_params``, which would leak into
``open_mesh_device`` as an unknown kwarg and fail with::

  TypeError: open_mesh_device() got an unexpected keyword argument
  'fabric_router_config'

The top-level ``conftest.bh_2d_mesh_device_context`` knows how to thread
all the BH fabric-config kwargs through ``set_fabric`` before opening
the mesh, so we wrap it here.
"""

from __future__ import annotations

import pytest

from conftest import bh_2d_mesh_device_context


@pytest.fixture(scope="function")
def deepseek_pipeline_mesh_device(request, device_params):
    try:
        param = request.param
    except (ValueError, AttributeError):
        param = (4, 2)
    assert param == (4, 2), f"deepseek pipeline tests expect deepseek_pipeline_mesh_device=(4, 2) per rank, got {param}"

    with bh_2d_mesh_device_context(device_params) as md:
        yield md
