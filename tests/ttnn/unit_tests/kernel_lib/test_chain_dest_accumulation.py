# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Functional coverage for the eltwise-chain sticky DEST accumulator."""

import pytest
import torch
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib


KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/dest_accumulation/dest_accumulation.cpp"
DEEPSEEK_FAST_REDUCE_KERNEL = (
    "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc/device/kernels/"
    "deepseek_moe_fast_reduce_nc_reduce.cpp"
)
DEEPSEEK_FUSED_FAST_REDUCE_KERNEL = (
    "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/device/kernels/"
    "deepseek_moe_fast_reduce_nc_fused_compute.cpp"
)


def _run(device, block_size, caller_managed):
    n = 8
    num_outputs = 3
    total_input_tiles = n * num_outputs
    dt = ttnn.bfloat16
    input_shape = [1, 1, 32, 32 * total_input_tiles]
    output_shape = [1, 1, 32, 32 * num_outputs]
    core_grid = lib.single_core_grid()

    torch_local, tt_local = lib.make_input(input_shape, dt, device, seed=1701)
    torch_remote, tt_remote = lib.make_input(input_shape, dt, device, seed=1702)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    cbs = [
        lib.cb_descriptor(0, dt, total_input_tiles, core_grid),
        lib.cb_descriptor(1, dt, total_input_tiles, core_grid),
        lib.cb_descriptor(16, dt, num_outputs, core_grid),
    ]
    reader = lib.build_reader_kernel([tt_local, tt_remote], total_input_tiles, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, num_outputs, core_grid)
    compute = lib.build_compute_kernel(
        KERNEL,
        [n, block_size, int(caller_managed), num_outputs],
        core_grid,
    )
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    out = ttnn.to_torch(ttnn.generic_op([tt_local, tt_remote, tt_out], program)).to(torch.float32)

    # Each outer row is an independent reduction into a freshly acquired sticky D0.
    local_tiles = torch.stack(torch_local.to(torch.float32).split(32, dim=-1)).reshape(num_outputs, n, 1, 1, 32, 32)
    remote_tiles = torch.stack(torch_remote.to(torch.float32).split(32, dim=-1)).reshape(num_outputs, n, 1, 1, 32, 32)
    reduced = (local_tiles + remote_tiles).sum(dim=1)
    golden = torch.cat([reduced[i] for i in range(num_outputs)], dim=-1)
    return golden, out


@pytest.mark.parametrize("block_size", [1, 2, 8])
@pytest.mark.parametrize("caller_managed", [False, True])
def test_dest_accumulation(device, block_size, caller_managed):
    golden, out = _run(device, block_size, caller_managed)
    pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([ttnn.bfloat16]))
    logger.info(f"DEST accumulation block={block_size}, caller_managed={caller_managed} | {msg}")
    assert pcc_ok, msg


def test_migrated_deepseek_fast_reduce_uses_chunked_dest_accumulation(device):
    """Run the production compute kernel with a block size greater than one."""
    n = 8
    num_outputs = 3
    total_input_tiles = n * num_outputs
    input_granularity = 2
    dt = ttnn.bfloat16
    core_grid = lib.single_core_grid()

    torch_partial, tt_partial = lib.make_input([1, 1, 32, 32 * total_input_tiles], dt, device, seed=1901)
    torch_bias, tt_bias = lib.make_input([1, 1, 32, 32], dt, device, seed=1902)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32 * num_outputs]), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    program = ttnn.ProgramDescriptor(
        kernels=[
            lib.build_reader_asym_kernel([tt_partial, tt_bias], [total_input_tiles, 1], core_grid),
            lib.build_writer_1out_kernel(tt_out, num_outputs, core_grid),
            lib.build_compute_kernel(
                DEEPSEEK_FAST_REDUCE_KERNEL,
                [num_outputs, n, input_granularity, 0, 1, 16],
                core_grid,
            ),
        ],
        semaphores=[],
        cbs=[
            lib.cb_descriptor(0, dt, total_input_tiles, core_grid),
            lib.cb_descriptor(1, dt, 1, core_grid),
            lib.cb_descriptor(16, dt, num_outputs, core_grid),
        ],
    )
    out = ttnn.to_torch(ttnn.generic_op([tt_partial, tt_bias, tt_out], program)).to(torch.float32)

    partial_tiles = torch.stack(torch_partial.to(torch.float32).split(32, dim=-1)).reshape(num_outputs, n, 1, 1, 32, 32)
    reduced = (partial_tiles + torch_bias.to(torch.float32)).sum(dim=1)
    golden = torch.cat([reduced[i] for i in range(num_outputs)], dim=-1)
    pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([dt]))
    logger.info(f"DeepSeek DEST accumulation block={input_granularity} | {msg}")
    assert pcc_ok, msg


def test_migrated_deepseek_fused_fast_reduce_uses_chunked_dest_accumulation(device):
    """Run the production broadcast-MAC kernel with a block size greater than one."""
    n = 8
    num_outputs = 3
    total_activation_tiles = n * num_outputs
    input_granularity = 2
    dt = ttnn.bfloat16
    core_grid = lib.single_core_grid()

    torch_activation, tt_activation = lib.make_input([1, 1, 32, 32 * total_activation_tiles], dt, device, seed=1911)
    torch_score, tt_score = lib.make_input([1, 1, 32, 32 * n], dt, device, seed=1912)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32 * num_outputs]), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    program = ttnn.ProgramDescriptor(
        kernels=[
            lib.build_reader_asym_kernel([tt_activation, tt_score], [total_activation_tiles, n], core_grid),
            lib.build_writer_1out_kernel(tt_out, num_outputs, core_grid),
            lib.build_compute_kernel(
                DEEPSEEK_FUSED_FAST_REDUCE_KERNEL,
                [num_outputs, n, input_granularity, 0, 1, 16],
                core_grid,
            ),
        ],
        semaphores=[],
        cbs=[
            lib.cb_descriptor(0, dt, total_activation_tiles, core_grid),
            lib.cb_descriptor(1, dt, n, core_grid),
            lib.cb_descriptor(16, dt, num_outputs, core_grid),
        ],
    )
    out = ttnn.to_torch(ttnn.generic_op([tt_activation, tt_score, tt_out], program)).to(torch.float32)

    activation_tiles = torch.stack(torch_activation.to(torch.float32).split(32, dim=-1)).reshape(
        num_outputs, n, 1, 1, 32, 32
    )
    score_tiles = torch.stack(torch_score.to(torch.float32).split(32, dim=-1))
    reduced = (activation_tiles * score_tiles[..., :1]).sum(dim=1)
    golden = torch.cat([reduced[i] for i in range(num_outputs)], dim=-1)
    pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([dt]))
    logger.info(f"DeepSeek fused DEST accumulation block={input_granularity} | {msg}")
    assert pcc_ok, msg
