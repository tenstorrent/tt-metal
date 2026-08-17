# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the Qwen3.6-35B-A3B functional-decoder tests.

Building a layer materialises ~2.2 GiB of expert weights, so pairs are cached per
(kind, batch, context, weights, extra config) for the whole session and the device is
opened once.
"""

import gc

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests import harness


@pytest.fixture(scope="session")
def qwen_device():
    torch.set_num_threads(16)
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    yield device
    ttnn.close_mesh_device(device)


@pytest.fixture(scope="session")
def layer_pairs(qwen_device):
    cache: dict = {}

    def get(kind, *, max_batch_size=1, supported_context=4096, real_weights=False, **cfg_kwargs):
        key = (kind, max_batch_size, supported_context, real_weights, tuple(sorted(cfg_kwargs.items())))
        if key not in cache:
            cache[key] = harness.build_layer_pair(
                qwen_device,
                kind=kind,
                max_batch_size=max_batch_size,
                supported_context=supported_context,
                real_weights=real_weights,
                **cfg_kwargs,
            )
        pair = cache[key]
        pair.tt.reset_state()
        return pair

    yield get
    cache.clear()
    gc.collect()
