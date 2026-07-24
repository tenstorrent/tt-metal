# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from models.common.utility_functions import skip_for_blackhole
from tests.nightly.t3000.ccl.test_all_gather import run_all_gather_impl


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "ag_output_shape, dim, layout, ag_input_dtype",
    [
        # Gather on dim 0
        ([16, 1, 8, 8], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ([16, 16, 8, 8], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ([8, 16, 8, 8], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        # Gather on dim 1
        ([1, 16, 8, 8], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ([16, 16, 8, 8], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ([16, 8, 8, 8], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        # Gather on dim 2
        ([1, 16, 512, 8], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ([16, 1, 512, 8], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ([16, 16, 512, 8], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        # # Gather on dim 3
        ([1, 16, 8, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ([16, 1, 8, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ([16, 16, 8, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
    ids=[
        "tt_training_test_one",
        "tt_training_test_two",
        "tt_training_test_three",
        "tt_training_test_four",
        "tt_training_test_five",
        "tt_training_test_six",
        "tt_training_test_seven",
        "tt_training_test_eight",
        "tt_training_test_nine",
        "tt_training_test_ten",
        "tt_training_test_eleven",
        "tt_training_test_twelve",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 3),
    ],
    ids=["perf"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 90112},
    ],
    indirect=True,
    ids=["fabric_2d_linear"],
)
def test_all_gather_training_shapes(
    mesh_device,
    ag_output_shape,
    dim,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    num_iters,
):
    run_all_gather_impl(
        mesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=enable_trace,
        num_iters=num_iters,
        use_persistent_buffers=False,
    )
