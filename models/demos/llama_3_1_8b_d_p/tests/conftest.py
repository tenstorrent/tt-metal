# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the device test suites.

They live HERE rather than in ``test_factory.py`` because pytest only auto-discovers fixtures from a
``conftest.py`` — a fixture defined in an imported helper module is invisible unless every test also
imports it by name. ``test_factory.py`` keeps the plain helpers and the mesh parametrization
decorator, which import normally.
"""

import os

import pytest

from models.demos.llama_3_1_8b_d_p.config import MeshConfig
from models.demos.llama_3_1_8b_d_p.reference.config import LlamaConfigConstants
from models.demos.llama_3_1_8b_d_p.tt.ccl import CCLManager, default_topology
from models.demos.llama_3_1_8b_d_p.utils.general_utils import get_default_num_links

from .test_factory import SP, TARGET_MESH, TP


@pytest.fixture(scope="function")
def config() -> LlamaConfigConstants:
    """The real model dims. Module tests run at real dims — only depth is ever reduced."""
    return LlamaConfigConstants.from_json()


@pytest.fixture(scope="function")
def hf_config(config):
    return config.to_hf_config()


@pytest.fixture(scope="function")
def mesh_config(mesh_device):
    """TP on the cols, SP on the rows — the spec's sp8 x tp4 on an (8, 4) mesh."""
    rows, cols = tuple(mesh_device.shape)
    cfg = MeshConfig(mesh_device.shape, tp=cols)
    if (rows, cols) == TARGET_MESH:
        assert (cfg.sp, cfg.tp) == (SP, TP), f"mesh gives sp{cfg.sp}xtp{cfg.tp}, spec asks sp{SP}xtp{TP}"
    return cfg


@pytest.fixture(scope="function")
def ccl_manager(mesh_device):
    return CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=default_topology())


@pytest.fixture(scope="function")
def topology_name():
    """Which fabric topology this measurement was taken on. Recorded with every PCC number, because
    linear and torus collective costs are not comparable."""
    return "torus" if os.getenv("LLAMA31_8B_FABRIC_TORUS") == "1" else "linear"


