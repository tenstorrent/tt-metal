# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

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
    output_base=None,
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
                [0, tiles_per_output, block_size, int(caller_managed), num_outputs, int(whole_shape), output_tiles],
                core_grid,
                defines=[("OUTPUT_OFFSET", str(output_base))] if output_base is not None else None,
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
            lib.assert_close(
                golden,
                out,
                f"DEST accumulation block={block_size}, caller_managed={caller_managed}, whole_shape={whole_shape}",
                rtol=0.1,
                atol=0.1,
            )
            results[(block_size, caller_managed)] = out

    reference = results[(1, False)]
    for config, out in results.items():
        assert torch.equal(out, reference), f"DEST accumulation changed across lifecycle/block config {config}"


@pytest.mark.parametrize("output_base", [0, 3])
@pytest.mark.parametrize("block_size", [1, 2])
def test_per_row_offset_packs_contiguous_results(device, output_base, block_size):
    n = 3
    num_outputs = 3
    total_input_tiles = n * num_outputs
    a = torch.arange(1, total_input_tiles + 1, dtype=torch.float32).to(torch.bfloat16)
    a = a.repeat_interleave(32).reshape(1, 1, 1, -1).expand(1, 1, 32, -1).contiguous()
    tt_a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(torch.zeros_like(a), layout=ttnn.TILE_LAYOUT, device=device)

    # Include guard tiles and room for the old, incorrectly spaced writes so failure is numerical, not OOB.
    output_tiles = output_base + (num_outputs - 1) * n + 2
    out = _run_configuration(
        device,
        tt_a,
        tt_b,
        n,
        total_input_tiles,
        output_tiles,
        num_outputs,
        block_size,
        caller_managed=True,
        whole_shape=False,
        output_base=output_base,
    )

    # The kernel fills the output window from input tile 0 before writing the reduced results.
    golden = torch.ones_like(out)
    for row in range(num_outputs):
        value = sum(range(row * n + 1, (row + 1) * n + 1))
        tile = output_base + row
        golden[..., tile * 32 : (tile + 1) * 32] = value
    torch.testing.assert_close(out, golden, rtol=0, atol=0)


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
        lib.assert_close(golden, out, f"L1 accumulation caller_managed={caller_managed}", rtol=0.1, atol=0.1)
        outputs[caller_managed] = out

    assert torch.equal(outputs[False], outputs[True]), "L1 accumulation changed with lifecycle ownership"
