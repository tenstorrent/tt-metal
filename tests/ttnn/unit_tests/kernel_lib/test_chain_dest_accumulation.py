# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib
from tests.ttnn.utils_for_testing import comp_pcc

KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/accumulation.cpp"


def _run_configuration(
    device,
    tt_a,
    tt_b,
    tiles_per_output,
    total_input_tiles,
    output_tiles,
    num_outputs,
    block_size,
    caller_managed,
    whole_shape,
):
    dtype = ttnn.bfloat16
    core_grid = lib.single_core_grid()
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
                KERNEL,
                [0, tiles_per_output, block_size, int(caller_managed), num_outputs, int(whole_shape)],
                core_grid,
            ),
        ],
        semaphores=[],
        cbs=[
            lib.cb_descriptor(0, dtype, total_input_tiles, core_grid),
            lib.cb_descriptor(1, dtype, total_input_tiles, core_grid),
            lib.cb_descriptor(16, dtype, output_tiles, core_grid),
        ],
    )
    return ttnn.to_torch(ttnn.generic_op([tt_a, tt_b, tt_out], program)).to(torch.float32)


@pytest.mark.parametrize("whole_shape", [False, True], ids=["per-row", "whole-shape"])
def test_dest_accumulation_modes_and_lifecycle_equivalence(device, whole_shape):
    """Each scope checks its golden plus block-size and managed/caller-managed equivalence."""
    n = 8
    num_outputs = 3
    total_input_tiles = n * num_outputs
    dtype = ttnn.bfloat16

    torch_a, tt_a = lib.make_input([1, 1, 32, 32 * total_input_tiles], dtype, device, seed=1701)
    torch_b, tt_b = lib.make_input([1, 1, 32, 32 * total_input_tiles], dtype, device, seed=1702)
    output_tiles = 1 if whole_shape else num_outputs

    a_tiles = torch.stack(torch_a.to(torch.float32).split(32, dim=-1)).reshape(num_outputs, n, 1, 1, 32, 32)
    b_tiles = torch.stack(torch_b.to(torch.float32).split(32, dim=-1)).reshape(num_outputs, n, 1, 1, 32, 32)
    reduced = (a_tiles + b_tiles).sum(dim=1)
    golden = reduced.sum(dim=0) if whole_shape else torch.cat([reduced[i] for i in range(num_outputs)], dim=-1)
    # Whole-shape reduction preserves one hardware DEST accumulation across all rows, so its
    # addition order intentionally differs from torch's tree reduction over the reshaped tensor.
    threshold = 0.999 if whole_shape else lib.pcc_threshold([dtype])
    results = {}
    for block_size in (1, 2, 8):
        for caller_managed in (False, True):
            out = _run_configuration(
                device,
                tt_a,
                tt_b,
                n,
                total_input_tiles,
                output_tiles,
                num_outputs,
                block_size,
                caller_managed,
                whole_shape,
            )
            pcc_ok, message = comp_pcc(golden, out, threshold)
            logger.debug(
                f"DEST accumulation block={block_size}, caller_managed={caller_managed}, "
                f"whole_shape={whole_shape} | {message}"
            )
            assert pcc_ok, message
            results[(block_size, caller_managed)] = out

    reference = results[(1, False)]
    for config, out in results.items():
        assert torch.equal(out, reference), f"DEST accumulation changed across lifecycle/block config {config}"


def _run_l1_configuration(device, tt_in, caller_managed):
    n = 8
    dtype = ttnn.bfloat16
    core_grid = lib.single_core_grid()
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32]), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    program = ttnn.ProgramDescriptor(
        kernels=[
            lib.build_reader_kernel([tt_in], n, core_grid),
            lib.build_writer_1out_kernel(tt_out, 1, core_grid),
            lib.build_compute_kernel(KERNEL, [1, n, 1, int(caller_managed), 1, 0], core_grid),
        ],
        semaphores=[],
        cbs=[
            lib.cb_descriptor(0, dtype, 2, core_grid),
            lib.cb_descriptor(15, dtype, 1, core_grid),
            lib.cb_descriptor(16, dtype, 2, core_grid),
        ],
    )
    return ttnn.to_torch(ttnn.generic_op([tt_in, tt_out], program)).to(torch.float32)


def test_l1_accumulation_managed_and_caller_managed_are_equivalent(device):
    n = 8
    dtype = ttnn.bfloat16
    torch_in, tt_in = lib.make_input([1, 1, 32, 32 * n], dtype, device, seed=1701, scale=0.125)

    golden = torch_in.to(torch.float32).reshape(1, 1, 32, n, 32).sum(dim=3)
    outputs = {}
    for caller_managed in (False, True):
        out = _run_l1_configuration(device, tt_in, caller_managed)
        pcc_ok, message = comp_pcc(golden, out, 0.999)
        logger.debug(f"L1 accumulation caller_managed={caller_managed} | {message}")
        assert pcc_ok, message
        outputs[caller_managed] = out

    assert torch.equal(outputs[False], outputs[True]), "L1 accumulation changed with lifecycle ownership"
