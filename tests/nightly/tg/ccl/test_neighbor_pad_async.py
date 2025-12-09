# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
import os
from loguru import logger
import ttnn
from models.common.utility_functions import skip_for_blackhole
from tests.nightly.t3000.ccl.test_neighbor_pad_async import run_neighbor_pad_impl


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.timeout(900)  # test is slow so bump up the default timeout
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 10),
        (False, 1),
    ],
    ids=["perf", "check"],
)
@pytest.mark.parametrize(
    "input_shape, halo_shard_dim, other_shard_dim, layout, input_dtype, padding_left, padding_right, padding_mode, cluster_axis, num_links",
    [
        # ([28, 60, 106, 768], 0, 2, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 0, "replicate", 1, True, 10),  # perf
        # ([82, 120, 212, 512], 0, 2, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 0, "replicate", 1, False, 1),  # check
        # ([28, 60, 106, 768], 2, 0, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 2, "replicate", 1, True, 10),  # perf
        # ([28, 60, 106, 768], 2, 0, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 2, "zeros", 1, False, 1),  # check
        ([1, 3, 23 * 4, 20 * 8, 32], 2, 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 0, 3),
        ([3, 25 * 4, 20 * 8, 32], 2, 1, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 1, 4),
        ([1, 3, 23 * 4, 20 * 8, 384], 2, 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 0, 3),
        ([3, 25 * 4, 20 * 8, 384], 2, 1, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 1, 4),
        ([1, 2, 46 * 4, 40 * 8, 384], 2, 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 0, 2),
        ([2, 48 * 4, 40 * 8, 384], 2, 1, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 1, 4),
        ([1, 4, 46 * 4, 40 * 8, 192], 2, 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 0, 4),
        ([4, 48 * 4, 40 * 8, 192], 2, 1, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 1, 4),
        ([1, 4, 46 * 4, 40 * 8, 384], 2, 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 0, 4),
        ([4, 48 * 4, 40 * 8, 384], 2, 1, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 1, 4),
        ([1, 4, 92 * 4, 80 * 8, 384], 2, 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 0, 4),
        ([4, 94 * 4, 80 * 8, 384], 2, 1, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 1, 4),
        ([1, 6, 92 * 4, 80 * 8, 192], 2, 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 0, 4),
        ([6, 94 * 4, 80 * 8, 192], 2, 1, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 1, 4),
        ([1, 4, 184 * 4, 160 * 8, 192], 2, 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 0, 4),
        ([4, 186 * 4, 160 * 8, 192], 2, 1, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 1, 4),
        ([1, 6, 184 * 4, 160 * 8, 96], 2, 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 0, 4),
        ([6, 186 * 4, 160 * 8, 96], 2, 1, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 1, 4),
    ],
    ids=[
        # "mochi_vae_1-perf",
        # "mochi_vae_2-check",
        # "replicate_width_dim-perf",
        # "zeros_width_dim-check",
        "Wan_1",
        "Wan_2",
        "Wan_3",
        "Wan_4",
        "Wan_5",
        "Wan_6",
        "Wan_7",
        "Wan_8",
        "Wan_9",
        "Wan_10",
        "Wan_11",
        "Wan_12",
        "Wan_13",
        "Wan_14",
        "Wan_15",
        "Wan_16",
        "Wan_17",
        "Wan_18",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_output",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "device_params, neighbor_pad_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
def test_neighbor_pad_async(
    mesh_device,
    input_shape,
    halo_shard_dim,
    other_shard_dim,
    padding_left,
    padding_right,
    padding_mode,
    cluster_axis,
    num_links,
    input_dtype,
    layout,
    mem_config_input,
    mem_config_output,
    enable_trace,
    neighbor_pad_topology,
    num_iters,
):
    run_neighbor_pad_impl(
        mesh_device,
        input_shape=input_shape,
        halo_shard_dim=halo_shard_dim,
        other_shard_dim=other_shard_dim,
        padding_left=padding_left,
        padding_right=padding_right,
        padding_mode=padding_mode,
        cluster_axis=cluster_axis,
        num_links=num_links,
        input_dtype=input_dtype,
        layout=layout,
        mem_config_input=mem_config_input,
        mem_config_output=mem_config_output,
        enable_trace=enable_trace,
        neighbor_pad_topology=neighbor_pad_topology,
        num_iters=num_iters,
    )
