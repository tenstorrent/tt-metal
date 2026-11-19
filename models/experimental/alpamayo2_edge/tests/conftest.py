# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.experimental.alpamayo2_edge.reference.config import load_config, resolve_model_dir
from models.experimental.alpamayo2_edge.reference.inputs import prepare_sample_inputs, synthetic_sample

P100_MESH = pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)


@pytest.fixture(scope="session")
def model_dir():
    return resolve_model_dir()


@pytest.fixture(scope="session")
def cfg(model_dir):
    return load_config(model_dir)


@pytest.fixture(scope="session")
def sample_inputs(cfg):
    """`synthetic_sample(0)` prepared for the model: the input of every full-pipeline golden."""
    return prepare_sample_inputs(cfg, synthetic_sample(0))
