# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from tests.nightly.t3000.ccl.test_minimal_reduce_scatter_async import run_reduce_scatter_impl
from models.utility_functions import skip_for_blackhole


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "rs_input_shape, dim, layout, rs_input_dtype",
    [
        # Scatter on dim 0
        ([16, 1, 8, 8], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ([16, 16, 128, 128], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ([8, 16, 8, 8], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        # Scatter on dim 1
        ([1, 16, 8, 8], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ([16, 16, 128, 128], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ([16, 8, 8, 8], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        # Scatter on dim 2
        ([1, 16, 512, 8], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ([16, 1, 512, 128], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ([16, 16, 512, 8], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        # # Scatter on dim 3
        ([1, 16, 8, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ([16, 1, 128, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
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
    "mem_config_input, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace,num_iters",
    [
        (True, 10),
        (False, 1),
    ],
    ids=["perf", "check"],
)
@pytest.mark.parametrize(
    "ones_tensor",
    [
        True,
        False,
    ],
    ids=["ones", "random"],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_2D_DYNAMIC, "trace_region_size": 1171456}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_2d_dynamic_linear"],
)
def test_reduce_scatter_async_training_shapes(
    mesh_device,
    rs_input_shape,
    dim,
    num_links,
    rs_input_dtype,
    layout,
    mem_config_input,
    mem_config_rs,
    enable_trace,
    rs_topology,
    num_iters,
    ones_tensor,
):
    run_reduce_scatter_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        rs_input_shape,
        dim,
        num_links,
        rs_input_dtype,
        layout,
        mem_config_input,
        mem_config_rs,
        rs_topology=rs_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        ones_tensor=ones_tensor,
        use_barrier=True,
        use_persistent_buffers=False,
    )
