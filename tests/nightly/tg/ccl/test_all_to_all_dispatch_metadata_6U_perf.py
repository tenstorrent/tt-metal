# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Performance wrapper tests for all_to_all_dispatch_metadata operation.

These tests run test_decode_perf via subprocess with tracy profiling and
measure device kernel duration for the AllToAllDispatchMetadataDeviceOperation.

Prerequisites:
    - RUN_ALL_TO_ALL_PERF=1
    - TT_MESH_GRAPH_DESC_PATH=tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_galaxy_16x1_torus_graph_descriptor.textproto

Example:
    RUN_ALL_TO_ALL_PERF=1 TT_MESH_GRAPH_DESC_PATH=tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_galaxy_16x1_torus_graph_descriptor.textproto pytest tests/nightly/tg/ccl/test_all_to_all_dispatch_metadata_6U_perf.py -v
"""

import os
import pytest
from loguru import logger

from models.perf.device_perf_utils import run_device_perf_detailed


# Mesh graph descriptor path for 16x1 mesh configuration
MESH_GRAPH_DESC_16x1 = (
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_galaxy_16x1_torus_graph_descriptor.textproto"
)


def is_perf_env_configured():
    """Check if both RUN_ALL_TO_ALL_PERF and TT_MESH_GRAPH_DESC_PATH are set correctly."""
    return (
        os.environ.get("RUN_ALL_TO_ALL_PERF") == "1"
        and os.environ.get("TT_MESH_GRAPH_DESC_PATH") == MESH_GRAPH_DESC_16x1
    )


@pytest.mark.skipif(
    not is_perf_env_configured(),
    reason=f"Requires RUN_ALL_TO_ALL_PERF=1 and TT_MESH_GRAPH_DESC_PATH={MESH_GRAPH_DESC_16x1}",
)
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "select_experts_k, packet_size, expected_lower_ns, expected_upper_ns",
    [
        pytest.param(1, "double", 47_000, 55_000, id="k1_double"),
        pytest.param(2, "double", 68_000, 76_000, id="k2_double"),
        pytest.param(1, "single", 52_000, 60_000, id="k1_single"),
        pytest.param(2, "single", 75_000, 83_000, id="k2_single"),
    ],
)
def test_perf_16x1(select_experts_k, packet_size, expected_lower_ns, expected_upper_ns):
    """Performance validation: persistent, random_sequential_experts, direct, 16x1 mesh"""
    k_filter = f"k{select_experts_k}"
    command = (
        f"pytest tests/nightly/tg/ccl/test_all_to_all_dispatch_metadata_6U.py::test_decode_perf "
        f"-k 'persistent and random_sequential_experts and direct and {k_filter} and {packet_size} and 16x1' -v"
    )
    subdir = f"all_to_all_dispatch_metadata_{k_filter}_{packet_size}_16x1"
    cols = ["DEVICE KERNEL"]
    op_name = "AllToAllDispatchMetadataDeviceOperation"

    results = run_device_perf_detailed(
        command=command,
        subdir=subdir,
        cols=cols,
        op_name=op_name,
        has_signposts=True,
        warmup_iters=10,
    )

    device_kernel_duration = results["DEVICE KERNEL"]["AVG"]
    logger.info(f"{k_filter}_{packet_size}_16x1 Device Kernel Duration: {device_kernel_duration:.2f} ns")

    assert (
        device_kernel_duration >= expected_lower_ns
    ), f"Device kernel duration {device_kernel_duration:.2f} ns is below expected minimum {expected_lower_ns} ns"
    assert (
        device_kernel_duration <= expected_upper_ns
    ), f"Device kernel duration {device_kernel_duration:.2f} ns exceeds expected maximum {expected_upper_ns} ns"
