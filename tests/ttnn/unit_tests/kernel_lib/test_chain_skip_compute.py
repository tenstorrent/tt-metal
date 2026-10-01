# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the CKL_ELTWISE_CHAIN_SKIP_COMPUTE profiling knob.

Each test compiles an existing functional chain with the build knob enabled. Completion proves
that the helper retained its CB and tile-register synchronization lifecycle; disagreement with the
real golden proves that helper-owned init, reconfiguration, compute, and pack execution were
elided. The ordinary and unified accumulation fixtures are intentionally shared with the normal
correctness tests so skip-on and skip-off exercise identical call sites.
"""

import pytest
import torch
import ttnn
from loguru import logger

import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib
from tests.ttnn.utils_for_testing import comp_pcc

HOIST_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/axes/hoist.cpp"
ACCUMULATION_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/accumulation.cpp"
SKIP_DEFINE = [("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", "1")]


@pytest.mark.parametrize("n", [8, 32])
def test_skip_compute_ordinary_walk_preserves_handshake(device, n):
    dtype = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()
    torch_in, tt_in = lib.make_input(shape, dtype, device, seed=90011)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    program = ttnn.ProgramDescriptor(
        kernels=[
            lib.build_reader_kernel([tt_in], n, core_grid),
            lib.build_writer_1out_kernel(tt_out, n, core_grid),
            lib.build_compute_kernel(HOIST_KERNEL, [n, 0], core_grid, defines=SKIP_DEFINE),
        ],
        semaphores=[],
        cbs=[
            lib.cb_descriptor(0, dtype, 2, core_grid),
            lib.cb_descriptor(16, dtype, 2, core_grid),
        ],
    )

    output = ttnn.to_torch(ttnn.generic_op([tt_in, tt_out], program)).to(torch.float32)
    golden = torch.exp(torch_in.to(torch.float32))
    matches, message = comp_pcc(golden, output, 0.99)
    logger.debug(f"skip ordinary n={n} | completed without deadlock | {message}")
    assert not matches, f"Skip output matched exp(x) ({message}); compute was not elided"


@pytest.mark.parametrize("whole_shape", [False, True], ids=["per-row", "whole-shape"])
@pytest.mark.parametrize("block_size", [1, 8])
@pytest.mark.parametrize("caller_managed", [False, True], ids=["managed", "caller-managed"])
def test_skip_compute_dest_accumulation_preserves_handshake(device, whole_shape, block_size, caller_managed):
    tiles_per_output = 8
    num_outputs = 3
    total_input_tiles = tiles_per_output * num_outputs
    output_tiles = 1 if whole_shape else num_outputs
    dtype = ttnn.bfloat16
    core_grid = lib.single_core_grid()
    input_shape = [1, 1, 32, 32 * total_input_tiles]
    torch_a, tt_a = lib.make_input(input_shape, dtype, device, seed=90021)
    torch_b, tt_b = lib.make_input(input_shape, dtype, device, seed=90022)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32 * output_tiles]),
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    program = ttnn.ProgramDescriptor(
        kernels=[
            lib.build_reader_kernel([tt_a, tt_b], total_input_tiles, core_grid),
            lib.build_writer_1out_kernel(tt_out, output_tiles, core_grid),
            lib.build_compute_kernel(
                ACCUMULATION_KERNEL,
                [0, tiles_per_output, block_size, int(caller_managed), num_outputs, int(whole_shape)],
                core_grid,
                defines=SKIP_DEFINE,
            ),
        ],
        semaphores=[],
        cbs=[
            lib.cb_descriptor(0, dtype, total_input_tiles, core_grid),
            lib.cb_descriptor(1, dtype, total_input_tiles, core_grid),
            lib.cb_descriptor(16, dtype, output_tiles, core_grid),
        ],
    )

    output = ttnn.to_torch(ttnn.generic_op([tt_a, tt_b, tt_out], program)).to(torch.float32)
    a_tiles = torch.stack(torch_a.to(torch.float32).split(32, dim=-1)).reshape(
        num_outputs, tiles_per_output, 1, 1, 32, 32
    )
    b_tiles = torch.stack(torch_b.to(torch.float32).split(32, dim=-1)).reshape(
        num_outputs, tiles_per_output, 1, 1, 32, 32
    )
    reduced = (a_tiles + b_tiles).sum(dim=1)
    golden = reduced.sum(dim=0) if whole_shape else torch.cat([reduced[i] for i in range(num_outputs)], dim=-1)
    matches, message = comp_pcc(golden, output, 0.99)
    logger.debug(
        f"skip DEST whole_shape={whole_shape}, block={block_size}, caller_managed={caller_managed} | "
        f"completed without deadlock | {message}"
    )
    assert not matches, f"Skip output matched the DEST reduction ({message}); compute was not elided"


@pytest.mark.parametrize("caller_managed", [False, True], ids=["managed", "caller-managed"])
def test_skip_compute_l1_accumulation_preserves_handshake(device, caller_managed):
    n = 8
    dtype = ttnn.bfloat16
    core_grid = lib.single_core_grid()
    shape = [1, 1, 32, 32 * n]
    torch_in, tt_in = lib.make_input(shape, dtype, device, seed=90031, scale=0.125, bias=0.0)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32]), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    program = ttnn.ProgramDescriptor(
        kernels=[
            lib.build_reader_kernel([tt_in], n, core_grid),
            lib.build_writer_1out_kernel(tt_out, 1, core_grid),
            lib.build_compute_kernel(
                ACCUMULATION_KERNEL,
                [1, n, 1, int(caller_managed), 1, 0],
                core_grid,
                defines=SKIP_DEFINE,
            ),
        ],
        semaphores=[],
        cbs=[
            lib.cb_descriptor(0, dtype, 2, core_grid),
            lib.cb_descriptor(15, dtype, 1, core_grid),
            lib.cb_descriptor(16, dtype, 2, core_grid),
        ],
    )

    output = ttnn.to_torch(ttnn.generic_op([tt_in, tt_out], program)).to(torch.float32)
    golden = torch_in.to(torch.float32).reshape(1, 1, 32, n, 32).sum(dim=3)
    matches, message = comp_pcc(golden, output, 0.99)
    logger.debug(f"skip L1 caller_managed={caller_managed} | completed without deadlock | {message}")
    assert not matches, f"Skip output matched the L1 accumulation ({message}); compute was not elided"
