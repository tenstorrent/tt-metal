# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull, get_devices_for_t3000
from tests.ttnn.unit_tests.operations.test_all_gather import (
    is_unsupported_case,
    run_line_all_gather,
    run_all_gather_deprecated,
    run_all_gather_sharded,
)
from ttnn import ShardTensorToMesh


nightly_all_gather_shape_dim_layout_configs = [
    ([4, 1, 33, 256], 0, ttnn.ROW_MAJOR_LAYOUT),
    ([4, 1, 256, 32], 0, ttnn.TILE_LAYOUT),
    ([8, 5, 13, 512], 3, ttnn.ROW_MAJOR_LAYOUT),
    ([8, 5, 32, 512], 3, ttnn.TILE_LAYOUT),
    ([8, 5, 13, 384], 3, ttnn.ROW_MAJOR_LAYOUT),
    ([8, 5, 32, 384], 3, ttnn.TILE_LAYOUT),
    ([8, 8, 256, 384], 0, ttnn.ROW_MAJOR_LAYOUT),
    ([8, 8, 256, 384], 0, ttnn.TILE_LAYOUT),
    ([8, 8, 256, 384], 1, ttnn.ROW_MAJOR_LAYOUT),
    ([8, 8, 256, 384], 1, ttnn.TILE_LAYOUT),
    ([8, 8, 256, 384], 2, ttnn.ROW_MAJOR_LAYOUT),
    ([8, 8, 256, 384], 2, ttnn.TILE_LAYOUT),
    ([8, 8, 256, 384], 3, ttnn.ROW_MAJOR_LAYOUT),
    ([8, 8, 256, 384], 3, ttnn.TILE_LAYOUT),
    ([8, 8, 256, 768], 3, ttnn.ROW_MAJOR_LAYOUT),
    ([8, 8, 256, 768], 3, ttnn.TILE_LAYOUT),
    ([8, 8, 1024, 4096], 1, ttnn.TILE_LAYOUT),
    ([8, 8, 2048, 4096], 1, ttnn.TILE_LAYOUT),
    ([8, 8, 128, 4096], 1, ttnn.ROW_MAJOR_LAYOUT),
    ([8, 8, 1024, 4096], 1, ttnn.ROW_MAJOR_LAYOUT),
    ([8, 8, 2048, 4096], 1, ttnn.ROW_MAJOR_LAYOUT),
    # Only for BFP8B
    # ([1, 1, 640, 32768], 3, ttnn.TILE_LAYOUT),
    # MLP AllGather. Llama 2 decode attn, mlp. Llama2, Falcon 40B decode mlp attn
    # Mixtral 8x7B, functional bringup with expanded tensor getting allgathered
    # Full shape for 8 chips
    ([1, 1, 32, 32768], 3, ttnn.TILE_LAYOUT),
    ([1, 1, 32, 32768], 3, ttnn.ROW_MAJOR_LAYOUT),
    # Input, Selfout, Final AllGather
    ([1, 1, 32, 8192], 3, ttnn.TILE_LAYOUT),
    ([1, 1, 32, 8192], 3, ttnn.ROW_MAJOR_LAYOUT),
    # MLP AllGather. Llama 2 decode attn, mlp. Llama2, Falcon 40B decode mlp attn
    # Half shape for 4 chips, same per chip shape as 8 chips
    ([1, 1, 32, 16384], 3, ttnn.TILE_LAYOUT),
    ([1, 1, 32, 16384], 3, ttnn.ROW_MAJOR_LAYOUT),
    # Input, Selfout, Final AllGather. Llama2, Falcon 40B decode mlp attn
    # Full shape for 8 chips
    ([1, 1, 32, 8192], 3, ttnn.TILE_LAYOUT),
    ([1, 1, 32, 8192], 3, ttnn.ROW_MAJOR_LAYOUT),
    # Input, Selfout, Final AllGather. Llama2, Falcon 40B decode mlp attn
    # Half shape for running on 4 chips, same per chip shape as for 8 chips
    ([1, 1, 32, 4096], 3, ttnn.TILE_LAYOUT),
    ([1, 1, 32, 4096], 3, ttnn.ROW_MAJOR_LAYOUT),
    # Falcon 40B prefill
    # 8 chips
    ([1, 1, 2048, 8192], 3, ttnn.TILE_LAYOUT),
    ([1, 1, 2048, 8192], 3, ttnn.ROW_MAJOR_LAYOUT),
    # 4 chips, same per chip shape as 8 chips
    ([1, 1, 2048, 4096], 3, ttnn.TILE_LAYOUT),
    ([1, 1, 2048, 4096], 3, ttnn.ROW_MAJOR_LAYOUT),
    # Falcon 40B prefill
    # 8 chips
    ([1, 1, 2048, 32768], 3, ttnn.TILE_LAYOUT),
    ([1, 1, 2048, 32768], 3, ttnn.ROW_MAJOR_LAYOUT),
    # 4 chips, same per chip shape as 8 chips
    ([1, 1, 2048, 16384], 3, ttnn.TILE_LAYOUT),
    ([1, 1, 2048, 16384], 3, ttnn.ROW_MAJOR_LAYOUT),
    # Mixtral 8x7B, Min sequence length
    # 8 chips
    # ([1, 1, 32768, 32768], 3, ttnn.ROW_MAJOR_LAYOUT),
    ([1, 1, 32768, 32768], 3, ttnn.TILE_LAYOUT),  # ultra slow?
    # 4 chips, per chip shape same as 8 chips
    # ([1, 1, 32768, 16384], 3, ttnn.ROW_MAJOR_LAYOUT),
    # ([1, 1, 32768, 16384], 3, ttnn.TILE_LAYOUT),
    # Llama galaxy mlp weights stationary -> emulation of row/col reduce
    ([1, 1, 128, 1024], 2, ttnn.ROW_MAJOR_LAYOUT),
    ([1, 1, 128, 1024], 2, ttnn.TILE_LAYOUT),
    # ([1, 1, 32, 8192], 3, ttnn.ROW_MAJOR_LAYOUT), # ALREADY LISTED PREVIOUSLY
    # ([1, 1, 32, 8192], 3, ttnn.TILE_LAYOUT),      # ALREADY LISTED PREVIOUSLY
    ([1, 1, 128, 4096], 2, ttnn.ROW_MAJOR_LAYOUT),  #
    ([1, 1, 128, 4096], 2, ttnn.TILE_LAYOUT),
    # ([1, 1, 32, 16384], 3, ttnn.ROW_MAJOR_LAYOUT), # ALREADY LISTED PREVIOUSLY. Update for 8 chip, actuall 32k for 8 chip but we are halving it for our 4 chip test
    # ([1, 1, 32, 16384], 3, ttnn.TILE_LAYOUT),      # ALREADY LISTED PREVIOUSLY. Update for 8 chip, actuall 32k for 8 chip but we are halving it for our 4 chip test
    ([1, 1, 8192, 32], 2, ttnn.ROW_MAJOR_LAYOUT),
    ([1, 1, 8192, 32], 2, ttnn.TILE_LAYOUT),
    ([1, 1, 1024, 128], 3, ttnn.ROW_MAJOR_LAYOUT),  # double on reduction dim for 8 chip
    ([1, 1, 1024, 128], 3, ttnn.TILE_LAYOUT),  # double on reduction dim for 8 chip
    ([1, 1, 16384, 32], 2, ttnn.ROW_MAJOR_LAYOUT),  # double on reduction dim for 8 chip
    ([1, 1, 16384, 32], 2, ttnn.TILE_LAYOUT),  # double on reduction dim for 8 chip
    ([1, 1, 32768, 32], 2, ttnn.ROW_MAJOR_LAYOUT),  # double on reduction dim for 8 chip
    ([1, 1, 32768, 32], 2, ttnn.TILE_LAYOUT),  # double on reduction dim for 8 chip
    ([1, 1, 4096, 128], 3, ttnn.ROW_MAJOR_LAYOUT),  # only for 4 chip
    ([1, 1, 4096, 128], 3, ttnn.TILE_LAYOUT),  # only for 4 chip
    ([1, 1, 128, 2048], 2, ttnn.ROW_MAJOR_LAYOUT),  # double on reduction dim for 8 chip
    ([1, 1, 128, 2048], 2, ttnn.TILE_LAYOUT),  # double on reduction dim for 8 chip
    # ([1, 1, 32, 8192], 3, ttnn.ROW_MAJOR_LAYOUT), # only for 4 chip - ALREADY LISTED PREVIOUSLY
    # ([1, 1, 32, 8192], 3, ttnn.TILE_LAYOUT),      # only for 4 chip - ALREADY LISTED PREVIOUSLY
    ([1, 1, 128, 8192], 2, ttnn.ROW_MAJOR_LAYOUT),  # double on reduction dim for 8 chip
    ([1, 1, 128, 8192], 2, ttnn.TILE_LAYOUT),  # double on reduction dim for 8 chip
    ([4, 1, 256, 32], 0, ttnn.TILE_LAYOUT),
    ([8, 8, 256, 384], 1, ttnn.ROW_MAJOR_LAYOUT),
    ([1, 1, 256, 1024], 2, ttnn.ROW_MAJOR_LAYOUT),
    ([1, 1, 1024, 256], 3, ttnn.ROW_MAJOR_LAYOUT),
    ([1, 1, 256, 2048], 2, ttnn.ROW_MAJOR_LAYOUT),
    ([1, 1, 256, 8192], 2, ttnn.ROW_MAJOR_LAYOUT),  # double on reduction dim for 8 chip
]


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (2, 1),
        (4, 1),
        (8, 1),
    ],
)
@pytest.mark.parametrize("input_shape, dim, layout", nightly_all_gather_shape_dim_layout_configs)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
def test_all_gather_on_t3000_nightly(
    t3k_mesh_device,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
):
    if (
        input_shape == [8, 8, 256, 384]
        and dim == 1
        and layout == ttnn.TILE_LAYOUT
        and num_devices == 4
        and num_links == 1
        and input_dtype == ttnn.bfloat16
        and mem_config.buffer_type == ttnn.BufferType.DRAM
    ):
        pytest.xfail(reason="Known failure")

    if (
        input_shape == [8, 8, 256, 384]
        and dim == 2
        and layout == ttnn.TILE_LAYOUT
        and num_devices == 4
        and num_links == 1
        and input_dtype == ttnn.bfloat16
        and mem_config.buffer_type == ttnn.BufferType.DRAM
    ):
        pytest.xfail(reason="Known failure")

    if (
        input_shape == [8, 8, 256, 384]
        and dim == 2
        and layout == ttnn.TILE_LAYOUT
        and num_devices == 4
        and num_links == 1
        and input_dtype == ttnn.bfloat8_b
        and mem_config.buffer_type == ttnn.BufferType.DRAM
    ):
        pytest.xfail(reason="Known failure")

    run_all_gather_on_t3000_impl(
        t3k_mesh_device,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        all_gather_operation=ttnn.all_gather,
    )


## We currently don't have the clean way with the test infra to describe the inner ring of a
## t3k device using the `t3k_mesh_device` fixture, so we use this test entry for all
## 4-chip, 2-link tests.
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (4, 2),
    ],
)
@pytest.mark.parametrize("input_shape, dim, layout", nightly_all_gather_shape_dim_layout_configs)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
def test_all_gather_on_t3000_2link_tests_nightly(
    all_devices,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
):
    if (
        input_shape == [8, 8, 256, 384]
        and dim == 2
        and layout == ttnn.TILE_LAYOUT
        and num_devices == 4
        and num_links == 2
        and input_dtype == ttnn.bfloat8_b
        and mem_config.buffer_type == ttnn.BufferType.DRAM
    ):
        pytest.xfail(reason="Known failure")

    run_all_gather_deprecated(
        all_devices,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        True,
        all_gather_operation=ttnn.all_gather,
        num_iters=1,
    )


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, input_shape, dim, layout",
    [
        (4, 2, [4, 1, 256, 32], 0, ttnn.TILE_LAYOUT),
        (8, 1, [8, 1, 256, 32], 0, ttnn.TILE_LAYOUT),
        (8, 1, [1, 1, 32, 16384], 3, ttnn.TILE_LAYOUT),
        (4, 2, [1, 1, 32, 32768], 3, ttnn.TILE_LAYOUT),
        (4, 2, [4, 1, 256, 32], 0, ttnn.ROW_MAJOR_LAYOUT),
        (8, 1, [8, 1, 256, 32], 0, ttnn.ROW_MAJOR_LAYOUT),
        (8, 1, [1, 1, 32, 16384], 3, ttnn.ROW_MAJOR_LAYOUT),
        (4, 2, [1, 1, 32, 32768], 3, ttnn.ROW_MAJOR_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,        # https://github.com/tenstorrent/tt-metal/issues/9686
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        # ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),        # https://github.com/tenstorrent/tt-metal/issues/9686
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("num_iters", [1000])  # TODO: restore to 500
@pytest.mark.parametrize("enable_async", [True, False])
def test_all_gather_on_t3000_nightly_commit_looping(
    t3k_mesh_device,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_all_gather_on_t3000_impl_tight_loop(
        t3k_mesh_device,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        all_gather_operation=ttnn.all_gather,
        num_iters=num_iters,
        enable_async=enable_async,
    )


# @pytest.mark.parametrize("num_devices", [4, 8])
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("num_devices", [8])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT])
# @pytest.mark.parametrize("num_cores", [1])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,  # https://github.com/tenstorrent/tt-metal/issues/9686
        # ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "tensor_mem_layout",
    [
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        # ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        # ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize(
    "input_shape, input_shard_shape,shard_grid",
    (
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 128),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 256),
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 64, 128),
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 64, 256),
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 64),
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 128),
            (32, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 64),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 128),
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        (
            (1, 1, 32, 128),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
        ),
        (
            (1, 1, 64, 128),
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
        ),
        (
            (1, 1, 32, 256),
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
        ),
        (
            (1, 1, 64, 256),
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 256),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 512),
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
        ),
        # LLama
        (
            (1, 1, 32, 1024),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 4096),
            (32, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 4096),
            (32, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 2048),
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 1792),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))}),
        ),
    ),
)
@pytest.mark.parametrize("all_gather_operation", [ttnn.all_gather, ttnn.line_all_gather])
def test_sharded_all_gather_nightly(
    t3k_mesh_device,
    num_devices,
    input_shape,
    input_shard_shape,
    shard_grid,
    dim,
    num_links,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    # num_cores,
    use_program_cache,
    function_level_defaults,
    all_gather_operation,
):
    run_all_gather_sharded(
        t3k_mesh_device,
        num_devices,
        input_shape,
        input_shard_shape,
        shard_grid,
        dim,
        num_links,
        orientation,
        input_dtype,
        tensor_layout,
        tensor_mem_layout,
        # num_cores,
        use_program_cache,
        function_level_defaults,
        all_gather_operation=all_gather_operation,
    )


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, input_shape, dim, layout",
    [
        (4, 1, [4, 1, 33, 256], 0, ttnn.ROW_MAJOR_LAYOUT),
        (8, 1, [8, 1, 33, 256], 0, ttnn.ROW_MAJOR_LAYOUT),
        (8, 1, [8, 1, 256, 32], 0, ttnn.TILE_LAYOUT),
        (8, 1, [8, 8, 256, 384], 1, ttnn.ROW_MAJOR_LAYOUT),
        # (4, 2, [8, 8, 256, 384], 1, ttnn.TILE_LAYOUT),
        (8, 1, [8, 8, 256, 384], 1, ttnn.TILE_LAYOUT),
        (4, 1, [8, 5, 13, 384], 3, ttnn.ROW_MAJOR_LAYOUT),
        (8, 1, [8, 5, 13, 512], 3, ttnn.ROW_MAJOR_LAYOUT),
        (4, 1, [8, 5, 32, 384], 3, ttnn.TILE_LAYOUT),
        (8, 1, [8, 5, 32, 512], 3, ttnn.TILE_LAYOUT),
        (4, 1, [1, 1, 32, 16384], 3, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("enable_async", [True, False])
def test_line_all_gather_on_t3000_nightly(
    t3k_mesh_device,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    enable_async,
    num_iters=1,
):
    run_line_all_gather(
        t3k_mesh_device,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        enable_async,
        num_iters,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, input_shape, dim, layout",
    [
        (
            4,
            2,
            [8, 8, 256, 384],
            1,
            ttnn.TILE_LAYOUT,
        ),  # test cases with num_links = 2 is currently not supported by new mesh fixture
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("enable_async", [True, False])
def test_line_all_gather_on_t3000_nightly_two_link(
    all_devices,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    enable_async,
    num_iters=1,
):
    run_all_gather_deprecated(
        all_devices,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        enable_async,
        ttnn.line_all_gather,
        num_iters,
    )


def run_line_all_gather_instances(
    t3k_mesh_device,
    num_devices,
    num_instances,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    enable_async,
    num_iters=1,
):
    if t3k_mesh_device.get_num_devices() != 8:
        pytest.skip("Not T3000!")

    for device in t3k_mesh_device.get_devices():
        device.enable_async(enable_async)

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")

    (is_known_failure, message) = is_unsupported_case(
        input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

    t3k_device = []

    for device in t3k_mesh_device.get_devices():
        t3k_device.append(device)

    t3000_device_rows = [
        [t3k_device[4], t3k_device[0], t3k_device[3], t3k_device[7]],
        [t3k_device[5], t3k_device[1], t3k_device[2], t3k_device[6]],
    ]
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")

    input_tensor = torch.rand(input_shape).bfloat16()

    ttnn_tensor = ttnn.from_torch(input_tensor, mesh_mapper=ShardTensorToMesh(t3k_mesh_device, dim=dim))
    input_tensor_mesh = ttnn.to_device(ttnn_tensor, t3k_mesh_device)

    result_mesh_tensors = []
    for loop in range(num_iters):
        for i, devices in enumerate(t3000_device_rows):
            tt_out_tensor = ttnn.line_all_gather(input_tensor_mesh, dim, num_links=num_links, memory_config=mem_config)
            result_mesh_tensors.append(tt_out_tensor)

    for loop in range(num_iters):
        ## Wait for completion
        for i, devices in enumerate(t3000_device_rows):
            for d in devices:
                ttnn.synchronize_device(d)

        for tt_out_tensor in result_mesh_tensors:
            for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
                tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
                if input_dtype == ttnn.bfloat16:
                    eq, output = comp_equal(tt_output_tensor, input_tensor)
                else:
                    eq, output = comp_pcc(tt_output_tensor, input_tensor)
                if not eq:
                    logger.error(f"output mismatch for tensor {i}")
                assert eq, f"{i} FAILED: {output}"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_instances, num_links, input_shape, dim, layout",
    [
        (4, 1, 1, [4, 1, 33, 256], 0, ttnn.ROW_MAJOR_LAYOUT),
        # (4, 1, 2, [8, 8, 256, 384], 1, ttnn.TILE_LAYOUT),
        (4, 1, 1, [8, 5, 13, 384], 3, ttnn.ROW_MAJOR_LAYOUT),
        (4, 1, 1, [8, 5, 32, 384], 3, ttnn.TILE_LAYOUT),
        (4, 1, 1, [1, 1, 32, 16384], 3, ttnn.TILE_LAYOUT),
        (4, 2, 1, [1, 1, 32, 16384], 3, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("enable_async", [True, False])
def test_line_all_gather_on_t3000_nightly_instances(
    t3k_mesh_device,
    num_devices,
    num_instances,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    enable_async,
    num_iters=1,
):
    run_line_all_gather_instances(
        t3k_mesh_device,
        num_devices,
        num_instances,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        enable_async,
        num_iters,
    )
