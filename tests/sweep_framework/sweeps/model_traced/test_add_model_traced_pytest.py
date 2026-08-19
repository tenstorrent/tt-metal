# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Pytest-based sweep test for ttnn.add (model-traced configs).

This is a migration of add_model_traced.py from the custom sweep runner
to native pytest. The run() function is reused as-is.
"""

import os
import sys

import pytest

# Add sweep framework to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from conftest_model_traced import load_vectors_for_module, vector_id
from framework.serialize import deserialize_vector_structured
from sweeps.model_traced.add_model_traced import run


TIMEOUT = 30


def _get_vectors():
    """Load vectors, returning empty list if no vector files found."""
    try:
        return load_vectors_for_module("model_traced.add_model_traced")
    except Exception:
        return []


vectors = _get_vectors()


@pytest.mark.timeout(TIMEOUT)
@pytest.mark.parametrize("vector", vectors, ids=[vector_id(v) for v in vectors])
def test_add(vector, mesh_device):
    """Run a single add config from model trace."""
    if isinstance(mesh_device, tuple):
        device, device_name = mesh_device
    else:
        device = mesh_device

    # Deserialize vector values (strings → ttnn objects, tuples, etc.)
    vector = deserialize_vector_structured(dict(vector))

    # Extract named params that run() expects as positional args
    kwargs = dict(vector)

    # Remove infra keys
    for key in ["input_hash", "validity", "invalid_reason", "status", "sweep_name", "suite_name", "timestamp", "tag"]:
        kwargs.pop(key, None)

    input_a_shape = kwargs.pop("input_a_shape")
    input_a_dtype = kwargs.pop("input_a_dtype")
    input_a_layout = kwargs.pop("input_a_layout")
    input_a_memory_config = kwargs.pop("input_a_memory_config")
    input_b_shape = kwargs.pop("input_b_shape")
    input_b_dtype = kwargs.pop("input_b_dtype")
    input_b_layout = kwargs.pop("input_b_layout")
    input_b_memory_config = kwargs.pop("input_b_memory_config")
    output_memory_config = kwargs.pop("output_memory_config", None)
    storage_type = kwargs.pop("storage_type", "StorageType::DEVICE")

    result = run(
        input_a_shape=input_a_shape,
        input_a_dtype=input_a_dtype,
        input_a_layout=input_a_layout,
        input_a_memory_config=input_a_memory_config,
        input_b_shape=input_b_shape,
        input_b_dtype=input_b_dtype,
        input_b_layout=input_b_layout,
        input_b_memory_config=input_b_memory_config,
        output_memory_config=output_memory_config,
        storage_type=storage_type,
        device=device,
        **kwargs,
    )

    # result is [pcc_result, e2e_perf]
    pcc = result[0]
    if isinstance(pcc, tuple):
        passed, pcc_value = pcc
    else:
        passed = float(pcc) >= 0.999 if isinstance(pcc, (int, float)) else bool(pcc)
        pcc_value = pcc

    assert passed, f"PCC check failed: {pcc_value}"
