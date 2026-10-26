# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import gc
import os

import pytest

from models.demos.qwen25_vl.tt.model_config import default_data_parallel
from models.tt_transformers.tt.generator import create_submeshes


@pytest.fixture(autouse=True)
def ensure_gc():
    gc.collect()


@pytest.fixture
def qwen25_vl_mesh_device(mesh_device):
    data_parallel = default_data_parallel(os.environ.get("HF_MODEL", ""), mesh_device.get_num_devices())
    if data_parallel == 1:
        yield mesh_device
        return

    # Submeshes are owned by the parent mesh and released when it closes.
    yield create_submeshes(mesh_device, data_parallel)[0]
