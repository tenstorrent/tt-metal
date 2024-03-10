# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull, get_devices_for_t3000
import itertools


def is_unsupported_case(input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout):
    if layout == ttl.tensor.Layout.ROW_MAJOR and input_dtype == ttl.tensor.DataType.BFLOAT8_B:
        return True, "Invalid combination"

    if num_devices < 2:
        return True, "Requires multiple devices to run"
    elif num_devices == 2 and num_links <= 2:
        return True, "Not enough links to run"

    if layout == ttl.tensor.Layout.ROW_MAJOR and input_dtype == ttl.tensor.DataType.BFLOAT8_B:
        return True, "Invalid combination"

    if num_devices < 2:
        return True, "Requires multiple devices to run"
    elif num_devices == 2 and num_links <= 2:
        return True, "Not enough links to run"

    if input_shape[dim] % num_devices != 0 or (dim == 3 and input_shape[dim] // num_devices % 32 != 0):
        return True, "Unsupported test case"

    ## Check that we can readback results
    fast_dispatch_page_size_limit = 55 * 1024
    elem_size = 2 if input_dtype == ttl.tensor.DataType.BFLOAT16 else 1
    if layout == ttl.tensor.Layout.ROW_MAJOR and (input_shape[dim] * elem_size) > fast_dispatch_page_size_limit:
        # Fast dispatch currently can't breakup readback of large pages into multiple smaller pages and is
        # limited to ~55K pages.
        return True, "Fast dispatch can't support reading back this page size in one shot"

    # Check that we can fit in L1 (if L1 config)
    tensor_size_bytes = elem_size
    for i in input_shape:
        tensor_size_bytes *= i
    num_l1_banks = 64
    if mem_config.buffer_type == ttl.tensor.BufferType.L1 and tensor_size_bytes > num_l1_banks * 50 * 1024:
        return True, "L1 buffer can't support large tensor sizes"

    # Check that each chip has a non-zero amount of data available
    min_sized_chunks_on_dim = input_shape[dim]
    if dim == 3:
        min_sized_chunks_on_dim //= 32
    if dim == 2:
        if layout == ttl.tensor.Layout.TILE:
            min_sized_chunks_on_dim //= 32
    if min_sized_chunks_on_dim < num_devices:
        return (
            True,
            f"Input shape {input_shape} incompatible with {num_devices} on dim {dim} because some chips will have no tensor",
        )

    return False, ""


def run_all_gather_on_t3000_impl(
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
    if len(all_devices) != 8:
        pytest.skip("Not T3000!")

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")

    (is_known_failure, message) = is_unsupported_case(
        input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

    devices = get_devices_for_t3000(all_devices, num_devices)

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")

    input_tensor = torch.rand(input_shape).bfloat16()

    input_tensors = torch.chunk(input_tensor, num_devices, dim)
    tt_input_tensors = []
    for i, t in enumerate(input_tensors):
        tt_input_tensors.append(ttl.tensor.Tensor(t, input_dtype).to(layout).to(devices[i], mem_config))

    tt_out_tensors = ttl.tensor.all_gather(tt_input_tensors, dim, num_links, output_mem_config=mem_config)

    torch.set_printoptions(sci_mode=False)
    # import time
    # time.sleep(5)
    for i, t in enumerate(tt_out_tensors):
        tt_output_tensor = t.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
        print((tt_output_tensor != input_tensor).nonzero())
        if input_dtype == ttl.tensor.DataType.BFLOAT16:
            eq, output = comp_equal(tt_output_tensor, input_tensor)
        else:
            eq, output = comp_pcc(tt_output_tensor, input_tensor)
        assert eq, f"{i} FAILED: {output}"


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, input_shape, dim, layout",
    [
        (4, 2, [4, 1, 33, 256], 0, ttl.tensor.Layout.ROW_MAJOR),
        (8, 1, [8, 1, 33, 256], 0, ttl.tensor.Layout.ROW_MAJOR),
        (4, 2, [4, 1, 256, 32], 0, ttl.tensor.Layout.TILE),
        (8, 1, [8, 1, 256, 32], 0, ttl.tensor.Layout.TILE),
        (4, 2, [8, 8, 256, 384], 1, ttl.tensor.Layout.ROW_MAJOR),
        (8, 1, [8, 8, 256, 384], 1, ttl.tensor.Layout.ROW_MAJOR),
        (4, 2, [8, 8, 256, 384], 1, ttl.tensor.Layout.TILE),
        (8, 1, [8, 8, 256, 384], 1, ttl.tensor.Layout.TILE),
        (4, 2, [8, 5, 13, 384], 3, ttl.tensor.Layout.ROW_MAJOR),
        (8, 1, [8, 5, 13, 512], 3, ttl.tensor.Layout.ROW_MAJOR),
        (4, 2, [8, 5, 32, 384], 3, ttl.tensor.Layout.TILE),
        (8, 1, [8, 5, 32, 512], 3, ttl.tensor.Layout.TILE),
        # Only for BFP8B
        # ([1, 1, 640, 32768], 3, ttl.tensor.Layout.TILE),
        # MLP AllGather,  Llama 2 decode attn, mlp. Llama2, Falcon 40B decode mlp attn
        (8, 1, [1, 1, 32, 32768], 3, ttl.tensor.Layout.TILE),
        # (8, 1, [1, 1, 32, 32768], 3, ttl.tensor.Layout.ROW_MAJOR),
        # Input, Selfout, Final AllGather,  Llama2, Falcon 40B decode mlp attn
        (8, 1, [1, 1, 32, 8192], 3, ttl.tensor.Layout.TILE),
        (8, 1, [1, 1, 32, 8192], 3, ttl.tensor.Layout.ROW_MAJOR),
        # Falcon 40B prefill
        # 8 chips
        (8, 1, [1, 1, 2048, 8192], 3, ttl.tensor.Layout.TILE),
        (8, 1, [1, 1, 2048, 8192], 3, ttl.tensor.Layout.ROW_MAJOR),
        # Falcon 40B prefill, also mixtral expert reduction (w/ zero filled tensor)
        # 8 chips
        (8, 1, [1, 1, 2048, 32768], 3, ttl.tensor.Layout.TILE),
        # Llama/falcon40B galaxy mlp weights stationary -> emulation of row/col reduce
        (8, 1, [1, 1, 256, 1024], 2, ttl.tensor.Layout.ROW_MAJOR),
        (8, 1, [1, 1, 256, 1024], 2, ttl.tensor.Layout.TILE),
        (8, 1, [1, 1, 246, 4096], 2, ttl.tensor.Layout.ROW_MAJOR),
        (8, 1, [1, 1, 246, 4096], 2, ttl.tensor.Layout.TILE),
        (8, 1, [1, 1, 8192, 32], 2, ttl.tensor.Layout.ROW_MAJOR),
        (8, 1, [1, 1, 8192, 32], 2, ttl.tensor.Layout.TILE),
        (8, 1, [1, 1, 1024, 256], 3, ttl.tensor.Layout.ROW_MAJOR),
        (8, 1, [1, 1, 1024, 256], 3, ttl.tensor.Layout.TILE),
        (8, 1, [1, 1, 32768, 32], 2, ttl.tensor.Layout.ROW_MAJOR),
        (8, 1, [1, 1, 32768, 32], 2, ttl.tensor.Layout.TILE),
        (8, 1, [1, 1, 256, 2048], 2, ttl.tensor.Layout.ROW_MAJOR),
        (8, 1, [1, 1, 256, 2048], 2, ttl.tensor.Layout.TILE),
        (8, 1, [1, 1, 256, 8192], 2, ttl.tensor.Layout.ROW_MAJOR),  # double on reduction dim for 8 chip
        (8, 1, [1, 1, 256, 8192], 2, ttl.tensor.Layout.TILE),  # double on reduction dim for 8 chip
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttl.tensor.MemoryConfig(buffer_type=ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(buffer_type=ttl.tensor.BufferType.L1),
    ],
)
def test_all_gather_on_t3000_post_commit(
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
    run_all_gather_on_t3000_impl(
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
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (2, 1),
        (4, 1),
        (4, 2),
        (8, 1),
    ],
)
@pytest.mark.parametrize(
    "input_shape, dim, layout",
    [
        ([4, 1, 33, 256], 0, ttl.tensor.Layout.ROW_MAJOR),
        ([4, 1, 256, 32], 0, ttl.tensor.Layout.TILE),
        ([8, 5, 13, 512], 3, ttl.tensor.Layout.ROW_MAJOR),
        ([8, 5, 32, 512], 3, ttl.tensor.Layout.TILE),
        ([8, 5, 13, 384], 3, ttl.tensor.Layout.ROW_MAJOR),
        ([8, 5, 32, 384], 3, ttl.tensor.Layout.TILE),
        ([8, 8, 256, 384], 0, ttl.tensor.Layout.ROW_MAJOR),
        ([8, 8, 256, 384], 0, ttl.tensor.Layout.TILE),
        ([8, 8, 256, 384], 1, ttl.tensor.Layout.ROW_MAJOR),
        ([8, 8, 256, 384], 1, ttl.tensor.Layout.TILE),
        ([8, 8, 256, 384], 2, ttl.tensor.Layout.ROW_MAJOR),
        ([8, 8, 256, 384], 2, ttl.tensor.Layout.TILE),
        ([8, 8, 256, 384], 3, ttl.tensor.Layout.ROW_MAJOR),
        ([8, 8, 256, 384], 3, ttl.tensor.Layout.TILE),
        ([8, 8, 256, 768], 3, ttl.tensor.Layout.ROW_MAJOR),
        ([8, 8, 256, 768], 3, ttl.tensor.Layout.TILE),
        # Only for BFP8B
        # ([1, 1, 640, 32768], 3, ttl.tensor.Layout.TILE),
        # MLP AllGather
        ([1, 1, 32, 32768], 3, ttl.tensor.Layout.TILE),
        ([1, 1, 32, 32768], 3, ttl.tensor.Layout.ROW_MAJOR),
        # Input, Selfout, Final AllGather
        ([1, 1, 32, 8192], 3, ttl.tensor.Layout.TILE),
        ([1, 1, 32, 8192], 3, ttl.tensor.Layout.ROW_MAJOR),
        # MLP AllGather. Llama 2 decode attn, mlp. Llama2, Falcon 40B decode mlp attn
        # Half shape for 4 chips, same per chip shape as 8 chips
        ([1, 1, 32, 16384], 3, ttl.tensor.Layout.TILE),
        ([1, 1, 32, 16384], 3, ttl.tensor.Layout.ROW_MAJOR),
        # Input, Selfout, Final AllGather. Llama2, Falcon 40B decode mlp attn
        # Full shape for 8 chips
        # ([1, 1, 32, 8192], 3, ttl.tensor.Layout.TILE),
        # ([1, 1, 32, 8192], 3, ttl.tensor.Layout.ROW_MAJOR),
        # Input, Selfout, Final AllGather. Llama2, Falcon 40B decode mlp attn
        # Half shape for running on 4 chips, same per chip shape as for 8 chips
        ([1, 1, 32, 4096], 3, ttl.tensor.Layout.TILE),
        ([1, 1, 32, 4096], 3, ttl.tensor.Layout.ROW_MAJOR),
        # Falcon 40B prefill
        # 8 chips
        ([1, 1, 2048, 8192], 3, ttl.tensor.Layout.TILE),
        ([1, 1, 2048, 8192], 3, ttl.tensor.Layout.ROW_MAJOR),
        # 4 chips, same per chip shape as 8 chips
        ([1, 1, 2048, 4096], 3, ttl.tensor.Layout.TILE),
        ([1, 1, 2048, 4096], 3, ttl.tensor.Layout.ROW_MAJOR),
        # Falcon 40B prefill
        # 8 chips
        ([1, 1, 2048, 32768], 3, ttl.tensor.Layout.TILE),
        ([1, 1, 2048, 32768], 3, ttl.tensor.Layout.ROW_MAJOR),
        # 4 chips, same per chip shape as 8 chips
        ([1, 1, 2048, 16384], 3, ttl.tensor.Layout.TILE),
        ([1, 1, 2048, 16384], 3, ttl.tensor.Layout.ROW_MAJOR),
        # Mixtral 8x7B, Min sequence length
        # 8 chips
        # ([1, 1, 32768, 32768], 3, ttl.tensor.Layout.ROW_MAJOR),
        ([1, 1, 32768, 32768], 3, ttl.tensor.Layout.TILE),  # ultra slow?
        # 4 chips, per chip shape same as 8 chips
        # ([1, 1, 32768, 16384], 3, ttl.tensor.Layout.ROW_MAJOR),
        # ([1, 1, 32768, 16384], 3, ttl.tensor.Layout.TILE),
        # Llama galaxy mlp weights stationary -> emulation of row/col reduce
        ([1, 1, 128, 1024], 2, ttl.tensor.Layout.ROW_MAJOR),
        ([1, 1, 128, 1024], 2, ttl.tensor.Layout.TILE),
        # ([1, 1, 32, 8192], 3, ttl.tensor.Layout.ROW_MAJOR), # ALREADY LISTED PREVIOUSLY
        # ([1, 1, 32, 8192], 3, ttl.tensor.Layout.TILE),     # ALREADY LISTED PREVIOUSLY
        ([1, 1, 128, 4096], 2, ttl.tensor.Layout.ROW_MAJOR),  #
        ([1, 1, 128, 4096], 2, ttl.tensor.Layout.TILE),
        # ([1, 1, 32, 16384], 3, ttl.tensor.Layout.ROW_MAJOR), # duplicate of above. Update for 8 chip, actuall 32k for 8 chip but we are halving it for our 4 chip test
        # ([1, 1, 32, 16384], 3, ttl.tensor.Layout.TILE),      # duplicate of above. Update for 8 chip, actuall 32k for 8 chip but we are halving it for our 4 chip test
        ([1, 1, 8192, 32], 2, ttl.tensor.Layout.ROW_MAJOR),
        ([1, 1, 8192, 32], 2, ttl.tensor.Layout.TILE),
        ([1, 1, 1024, 128], 3, ttl.tensor.Layout.ROW_MAJOR),  # double on reduction dim for 8 chip
        ([1, 1, 1024, 128], 3, ttl.tensor.Layout.TILE),  # double on reduction dim for 8 chip
        ([1, 1, 16384, 32], 2, ttl.tensor.Layout.ROW_MAJOR),  # double on reduction dim for 8 chip
        ([1, 1, 16384, 32], 2, ttl.tensor.Layout.TILE),  # double on reduction dim for 8 chip
        ([1, 1, 4096, 128], 3, ttl.tensor.Layout.ROW_MAJOR),  # only for 4 chip
        ([1, 1, 4096, 128], 3, ttl.tensor.Layout.TILE),  # only for 4 chip
        ([1, 1, 128, 2048], 2, ttl.tensor.Layout.ROW_MAJOR),  # double on reduction dim for 8 chip
        ([1, 1, 128, 2048], 2, ttl.tensor.Layout.TILE),  # double on reduction dim for 8 chip
        # ([1, 1, 32, 8192], 3, ttl.tensor.Layout.ROW_MAJOR), # only for 4 chip
        # ([1, 1, 32, 8192], 3, ttl.tensor.Layout.TILE),      # only for 4 chip
        ([1, 1, 128, 8192], 2, ttl.tensor.Layout.ROW_MAJOR),  # double on reduction dim for 8 chip
        ([1, 1, 128, 8192], 2, ttl.tensor.Layout.TILE),  # double on reduction dim for 8 chip
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttl.tensor.MemoryConfig(buffer_type=ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(buffer_type=ttl.tensor.BufferType.L1),
    ],
)
def test_all_gather_on_t3000_nightly(
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
    run_all_gather_on_t3000_impl(
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
    )
