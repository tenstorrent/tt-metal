# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import gc
import os

import pytest

import ttnn
from models.tt_transformers.tt.generator import create_submeshes


@pytest.fixture(autouse=True)
def ensure_gc():
    gc.collect()


def _qwen25_vl_test_data_parallel():
    data_parallel = os.environ.get("TT_DATA_PARALLEL") or os.environ.get("DATA_PARALLEL")
    if data_parallel:
        return int(data_parallel)

    hf_model = os.environ.get("HF_MODEL", "")
    if os.environ.get("MESH_DEVICE") == "TG" and ("olmOCR-2-7B" in hf_model or "Qwen2.5-VL-7B" in hf_model):
        return 4

    return 1


@pytest.fixture
def qwen25_vl_mesh_device(mesh_device):
    data_parallel = _qwen25_vl_test_data_parallel()
    if data_parallel == 1:
        yield mesh_device
        return

    submeshes = create_submeshes(mesh_device, data_parallel)
    try:
        yield submeshes[0]
    finally:
        for submesh in submeshes:
            ttnn.close_mesh_device(submesh)
