# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal harness for driving ``ttnn.generic_op`` from the Python API.

``ttnn.generic_op`` takes a hand-assembled ``ttnn.ProgramDescriptor`` -- kernels,
circular buffers, compile-time args and per-core runtime args -- and runs it on
device.  This file wires up the smallest interesting program:

    reader kernel (DRAM -> CB0)  ->  compute kernel (SFPU)  ->  writer kernel (CB16 -> DRAM)

and checks the result against a torch golden.  It is meant as a starting point:
swap ``SFPU_OPS`` / the kernel paths / the runtime args to run your own kernels.

Run as a pytest (uses the shared ``device`` fixture)::

    pytest tests/ttnn/unit_tests/operations/debug/test_generic_op_harness.py -v

Run standalone (opens and closes its own device)::

    python tests/ttnn/unit_tests/operations/debug/test_generic_op_harness.py --num-tiles 2 --op exp

Under the ttsim simulator, set these first (see ttsim's README and
.github/actions/setup-ttsim/action.yml)::

    export TT_METAL_SIMULATOR_HOME=$HOME/sim
    export TT_METAL_SIMULATOR=$HOME/sim/libttsim.so
    export TT_METAL_SLOW_DISPATCH_MODE=1
    export TT_METAL_DISABLE_SFPLOADMACRO=1

The simulator is much slower than silicon, so prefer a small core grid and a
small tile count there.
"""

import argparse
import sys

import pytest
import torch
from loguru import logger

import ttnn

# Kernels used by the harness.  All three are already in the repo; the reader and
# writer are the stock interleaved unary dataflow kernels and the compute kernel is
# the generic SFPU driver whose op is selected by preprocessor defines.
READER_KERNEL = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp"
WRITER_KERNEL = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp"
COMPUTE_KERNEL = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_sfpu.cpp"

# CB indices are fixed by the kernels above: the reader hardcodes CB0 as its
# destination and eltwise_sfpu.cpp hardcodes CB0 in / CB16 out.
IN_CB = 0
OUT_CB = 16

TILE_HW = 32 * 32
BFLOAT16_TILE_BYTES = TILE_HW * 2

# Each entry is (compute-kernel defines, torch golden).  ``SFPU_OP_*_INCLUDE``
# pulls the op's LLK header into the kernel; ``SFPU_OP_CHAIN_0`` is the actual
# call, run on whatever is sitting in DST register 0.
SFPU_OPS = {
    "exp": (
        [("SFPU_OP_EXP_INCLUDE", "1"), ("SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);")],
        torch.exp,
    ),
    "sqrt": (
        [("SFPU_OP_SQRT_INCLUDE", "1"), ("SFPU_OP_CHAIN_0", "sqrt_tile_init(); sqrt_tile(0);")],
        torch.sqrt,
    ),
    "gelu": (
        [("SFPU_OP_GELU_INCLUDE", "1"), ("SFPU_OP_CHAIN_0", "gelu_tile_init(); gelu_tile(0);")],
        torch.nn.functional.gelu,
    ),
    "recip": (
        [("SFPU_OP_RECIP_INCLUDE", "1"), ("SFPU_OP_CHAIN_0", "recip_tile_init(); recip_tile(0);")],
        torch.reciprocal,
    ),
    "neg": (
        [("SFPU_OP_NEG_INCLUDE", "1"), ("SFPU_OP_CHAIN_0", "negative_tile_init(); negative_tile(0);")],
        torch.neg,
    ),
}


def build_unary_sfpu_program(input_tensor, output_tensor, num_tiles, defines, core_grid):
    """Assemble the ProgramDescriptor for a per-tile SFPU op over ``num_tiles`` tiles.

    Work is split by whole tiles across ``core_grid`` in row-major order.  Cores
    are only included in the program if they get at least one tile, so the
    descriptor stays valid for tile counts smaller than the grid.
    """
    cores = [ttnn.CoreCoord(x, y) for y in range(core_grid.y) for x in range(core_grid.x)]
    num_cores = min(len(cores), num_tiles)
    cores = cores[:num_cores]

    # eltwise_sfpu.cpp takes the tile count as a compile-time arg, so every core in
    # a single program must process the same number of tiles.  Rather than emit two
    # core groups with two compute kernels, require an even split.
    if num_tiles % num_cores != 0:
        raise ValueError(
            f"num_tiles={num_tiles} does not divide evenly over {num_cores} cores; "
            f"the compute kernel takes tiles-per-core as a compile-time arg, so this "
            f"harness needs an even split (try a power-of-two tile count or --core-grid 1x1)"
        )
    tiles_per_core = num_tiles // num_cores

    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

    cb_format = lambda idx: ttnn.CBFormatDescriptor(
        buffer_index=idx,
        data_format=ttnn.bfloat16,
        page_size=BFLOAT16_TILE_BYTES,
    )
    # Double-buffered: two pages so the reader can fill one tile while compute
    # consumes the other.
    cb_descriptor = lambda idx: ttnn.CBDescriptor(
        total_size=2 * BFLOAT16_TILE_BYTES,
        core_ranges=core_ranges,
        format_descriptors=[cb_format(idx)],
    )

    # The reader takes only its TensorAccessor args (its CB is hardcoded to CB0);
    # the writer takes the output CB index first, then its TensorAccessor args.
    reader_ct_args = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    writer_ct_args = [OUT_CB] + list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    # (per_core_block_cnt, per_core_block_dim) -- one tile per inner iteration.
    compute_ct_args = [tiles_per_core, 1]

    # Both dataflow kernels take (buffer_address, num_pages, start_page_id).
    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    for i, core in enumerate(cores):
        start_tile = i * tiles_per_core
        reader_rt_args[core.x][core.y] = [input_tensor.buffer_address(), tiles_per_core, start_tile]
        writer_rt_args[core.x][core.y] = [output_tensor.buffer_address(), tiles_per_core, start_tile]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=READER_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_ranges,
            compile_time_args=reader_ct_args,
            runtime_args=reader_rt_args,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=WRITER_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_ranges,
            compile_time_args=writer_ct_args,
            runtime_args=writer_rt_args,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=COMPUTE_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_ranges,
            compile_time_args=compute_ct_args,
            defines=defines,
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(),
        ),
    ]

    return ttnn.ProgramDescriptor(
        kernels=kernels,
        semaphores=[],
        cbs=[cb_descriptor(IN_CB), cb_descriptor(OUT_CB)],
    )


def run_unary_sfpu_generic_op(device, op="exp", num_tiles=1, core_grid=None, pcc=0.99, seed=0):
    """Run ``op`` over ``num_tiles`` bfloat16 tiles via ttnn.generic_op.

    Returns ``(torch_output, torch_golden)``.
    """
    if op not in SFPU_OPS:
        raise ValueError(f"unknown op {op!r}; known ops: {sorted(SFPU_OPS)}")
    defines, golden_fn = SFPU_OPS[op]

    if core_grid is None:
        core_grid = ttnn.CoreCoord(1, 1)

    shape = [1, num_tiles, 32, 32]
    torch.manual_seed(seed)
    # Keep inputs in (0, 1] so sqrt/recip/log-ish ops stay in a sane range.
    torch_input = torch.rand(shape).to(torch.bfloat16) + 0.01

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # generic_op does not allocate: the output must exist before the program runs
    # and be passed as the last element of io_tensors.
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    program_descriptor = build_unary_sfpu_program(input_tensor, output_tensor, num_tiles, defines, core_grid)

    logger.info(f"running generic_op: op={op} num_tiles={num_tiles} core_grid={core_grid}")
    output = ttnn.generic_op([input_tensor, output_tensor], program_descriptor)

    torch_output = ttnn.to_torch(output).to(torch.float32)
    torch_golden = golden_fn(torch_input.to(torch.float32))

    return torch_output, torch_golden


def assert_pcc(actual, golden, pcc=0.99):
    """Pearson correlation check, so bfloat16/SFPU approximation error is tolerated."""
    a = actual.flatten().to(torch.float32)
    g = golden.flatten().to(torch.float32)
    if torch.equal(a, g):
        measured = 1.0
    else:
        measured = torch.corrcoef(torch.stack([a, g]))[0, 1].item()
    logger.info(f"PCC = {measured:.6f} (threshold {pcc})")
    assert measured >= pcc, f"PCC {measured} below threshold {pcc}"
    return measured


@pytest.mark.parametrize("op", ["exp", "sqrt", "gelu", "neg"])
@pytest.mark.parametrize("num_tiles", [1, 4])
def test_unary_sfpu_generic_op(device, op, num_tiles):
    output, golden = run_unary_sfpu_generic_op(device, op=op, num_tiles=num_tiles)
    assert_pcc(output, golden)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--op", default="exp", choices=sorted(SFPU_OPS), help="SFPU op to run")
    parser.add_argument("--num-tiles", type=int, default=1, help="number of 32x32 tiles")
    parser.add_argument(
        "--core-grid",
        default="1x1",
        help="cores to spread the tiles over, as XxY (default 1x1; keep small on ttsim)",
    )
    parser.add_argument("--pcc", type=float, default=0.99, help="PCC threshold")
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args(argv)

    grid_x, grid_y = (int(v) for v in args.core_grid.lower().split("x"))

    device = ttnn.open_device(device_id=args.device_id)
    try:
        output, golden = run_unary_sfpu_generic_op(
            device,
            op=args.op,
            num_tiles=args.num_tiles,
            core_grid=ttnn.CoreCoord(grid_x, grid_y),
        )
        assert_pcc(output, golden, pcc=args.pcc)
    finally:
        ttnn.close_device(device)

    logger.info(f"PASS: generic_op {args.op} over {args.num_tiles} tile(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
