# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""rand_keyed_tile and dropout_keyed_tile through the public compute API, driven by ttnn.generic_op.

Both draw from a counter-based generator: each element's value is a pure function of (seed, tile index,
position in the tile), so it can be reproduced on the host, and does not depend on which core computes
the tile or on what ran before.
"""

import struct

import numpy as np
import pytest
import torch
import ttnn
from tt_py_test_utils_common.utility_functions import is_quasar

from tests.ttnn.unit_tests.operations.eltwise.keyed_rand_reference import reference_draw31, reference_uniform

pytestmark = pytest.mark.use_module_device

# On Quasar the keyed APIs reseed the existing stream kernels per tile instead of computing the counter hash, so
# only the placement-independence tests apply there.
counter_hash_only = pytest.mark.skipif(is_quasar(), reason="Quasar falls back to reseeded stream kernels")

KERNEL = "tests/tt_metal/tt_metal/test_kernels/compute/keyed_rand_sfpu.cpp"
READER = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp"
WRITER = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp"
TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.float32: 4096}
INT_MAX = 2**31 - 1


def f2u(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def worker_cores(device, count):
    grid = device.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    if count > len(cores):
        pytest.skip(f"needs {count} worker cores, device has {len(cores)}")
    return cores[:count]


def run_keyed(device, num_tiles, num_cores, seed, param0, param1, dtype=ttnn.float32, input_tensor=None, tile_offset=0):
    """Runs the test kernel with tiles split contiguously over num_cores cores; output tile i is tile
    tile_offset + i of the stream. Returns [tiles, 32, 32]."""
    cores = worker_cores(device, num_cores)
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in cores])
    output = ttnn.from_torch(
        torch.zeros([1, 1, 32 * num_tiles, 32]), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    compute_args, reader_args, writer_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    base, extra = divmod(num_tiles, num_cores)
    start = 0
    for k, core in enumerate(cores):
        count = base + (1 if k < extra else 0)
        compute_args[core.x][core.y] = [count, start, seed, param0, param1, tile_offset]
        writer_args[core.x][core.y] = [output.buffer_address(), count, start]
        if input_tensor is not None:
            reader_args[core.x][core.y] = [input_tensor.buffer_address(), count, start]
        start += count

    def cb(index, cb_dtype):
        return ttnn.CBDescriptor(
            total_size=2 * TILE_BYTES[cb_dtype],
            core_ranges=core_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=cb_dtype, page_size=TILE_BYTES[cb_dtype])
            ],
        )

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=[16] + ttnn.TensorAccessorArgs(output).get_compile_time_args(),
            runtime_args=writer_args,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=[],
            defines=[("KEYED_DROPOUT", "1")] if input_tensor is not None else [],
            runtime_args=compute_args,
            config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=dtype == ttnn.float32),
        ),
    ]
    cbs = [cb(16, dtype)]
    if input_tensor is not None:
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=READER,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_set,
                compile_time_args=ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args(),
                runtime_args=reader_args,
                config=ttnn.ReaderConfigDescriptor(),
            )
        )
        cbs.append(cb(0, input_tensor.dtype))
        io_tensors = [input_tensor, output]
    else:
        # generic_op takes at least one tensor besides the output; the rand kernel reads none.
        dummy = ttnn.from_torch(torch.zeros([32, 32]), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        io_tensors = [dummy, output]
    ttnn.generic_op(io_tensors, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return ttnn.to_torch(output).float().reshape(num_tiles, 32, 32)


@pytest.mark.parametrize("seed", [0, 1, 12345, 0xFFFFFFFF])
@counter_hash_only
def test_rand_keyed_matches_host_reference(device, seed):
    num_tiles = 16
    result = run_keyed(device, num_tiles, 1, seed, f2u(0.0), f2u(1.0))
    expected = torch.from_numpy(reference_uniform(seed, range(num_tiles)))
    assert torch.equal(result, expected)


@pytest.mark.parametrize(
    "low, scale",
    [(-1.5, 3.0), (0.0, 1.5 * 2.0**-97)],
    ids=["normalization_folded_into_scale", "normalization_per_row"],
)
@counter_hash_only
def test_rand_keyed_from_and_scale_match_host_reference(device, low, scale):
    num_tiles = 16
    seed = 77
    result = run_keyed(device, num_tiles, 1, seed, f2u(low), f2u(scale))
    # A fused multiply-add: the product and sum are exact in float64, so one rounding to FP32 matches.
    unit = reference_uniform(seed, range(num_tiles)).astype(np.float64)
    expected = torch.from_numpy((unit * np.float64(np.float32(scale)) + np.float64(np.float32(low))).astype(np.float32))
    assert torch.equal(result, expected)


@counter_hash_only
def test_rand_keyed_range(device):
    # [from, from + scale], inclusive, with a mean near the midpoint.
    result = run_keyed(device, 64, 4, 7, f2u(-3.0), f2u(5.0))
    assert result.min() >= -3.0 and result.max() <= 2.0
    assert abs(result.mean().item() + 0.5) < 0.05
    expected = torch.from_numpy(reference_uniform(7, range(64))) * 5.0 - 3.0
    torch.testing.assert_close(result, expected, rtol=0, atol=1e-6)


@counter_hash_only
def test_rand_keyed_bfloat16(device):
    result = run_keyed(device, 64, 4, 11, f2u(0.0), f2u(1.0), dtype=ttnn.bfloat16)
    expected = torch.from_numpy(reference_uniform(11, range(64)))
    assert result.min() >= 0.0 and result.max() <= 1.0
    torch.testing.assert_close(result, expected, rtol=1e-2, atol=0)


@pytest.mark.parametrize("num_cores", [2, 7, 64])
def test_rand_keyed_does_not_depend_on_core_split(device, num_cores):
    num_tiles = 128
    single_core = run_keyed(device, num_tiles, 1, 99, f2u(0.0), f2u(1.0))
    # A run with another seed in between: the generator carries no state from one program to the next.
    run_keyed(device, num_tiles, num_cores, 5, f2u(0.0), f2u(1.0))
    split = run_keyed(device, num_tiles, num_cores, 99, f2u(0.0), f2u(1.0))
    assert torch.equal(single_core, split)


@pytest.mark.parametrize("tile_offset", [2**22 - 2, 2**32 - 2], ids=["past_2_22", "last_indices"])
@counter_hash_only
def test_rand_keyed_tile_indices_past_counter_range(device, tile_offset):
    # The counter repeats every 2^22 tiles; past that the tile index's high bits go into the key instead.
    seed = 7
    num_tiles = 2 if tile_offset == 2**32 - 2 else 4
    result = run_keyed(device, num_tiles, 1, seed, f2u(0.0), f2u(1.0), tile_offset=tile_offset)
    tiles = [(tile_offset + i) % 2**32 for i in range(num_tiles)]
    assert torch.equal(result, torch.from_numpy(reference_uniform(seed, tiles)))
    # The last tile is past 2^22, so it must differ from the tile 2^22 below it, whose counters it repeats.
    wrapped = run_keyed(device, 1, 1, seed, f2u(0.0), f2u(1.0), tile_offset=tiles[-1] % 2**22)
    assert not torch.equal(result[-1], wrapped[0])


@counter_hash_only
def test_rand_keyed_seeds_and_tiles_are_independent(device):
    num_tiles = 64
    a = run_keyed(device, num_tiles, 4, 1, f2u(0.0), f2u(1.0)).flatten()
    b = run_keyed(device, num_tiles, 4, 2, f2u(0.0), f2u(1.0)).flatten()
    assert abs(torch.corrcoef(torch.stack([a, b]))[0, 1].item()) < 0.01
    tiles = a.reshape(num_tiles, -1)
    assert abs(torch.corrcoef(torch.stack([tiles[:-1].flatten(), tiles[1:].flatten()]))[0, 1].item()) < 0.01


@pytest.mark.parametrize("probability", [0.1, 0.5])
@counter_hash_only
def test_dropout_keyed_matches_host_reference(device, probability):
    num_tiles = 32
    scale = 1.0 / (1.0 - probability)
    int_probability = int(INT_MAX * probability)
    torch.manual_seed(0)
    data = torch.rand([1, 1, 32 * num_tiles, 32]) + 0.5
    input_tensor = ttnn.from_torch(data, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    result = run_keyed(device, num_tiles, 4, 42, int_probability, f2u(scale), input_tensor=input_tensor)

    dropped = torch.from_numpy(reference_draw31(42, range(num_tiles)) < int_probability)
    assert torch.equal(result == 0, dropped)
    # copy_tile carries the FP32 input through source register A at TF32 precision.
    expected = data.reshape(num_tiles, 32, 32) * scale
    torch.testing.assert_close(result[~dropped], expected[~dropped], rtol=2e-3, atol=0)


@counter_hash_only
def test_dropout_keyed_drops_neighbours_independently(device):
    # With the SFPU PRNG, an element below a dropped one was dropped about half the time at p = 0.1.
    num_tiles, probability = 256, 0.1
    data = torch.ones([1, 1, 32 * num_tiles, 32])
    input_tensor = ttnn.from_torch(data, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    result = run_keyed(
        device, num_tiles, 8, 3, int(INT_MAX * probability), f2u(1.0), input_tensor=input_tensor
    ).reshape(-1, 32)
    dropped = result == 0
    assert abs(dropped.float().mean().item() - probability) < 0.005
    below = dropped[1:][dropped[:-1]].float().mean().item()
    right = dropped[:, 1:][dropped[:, :-1]].float().mean().item()
    two_below = dropped[2:, 1:][dropped[:-2, :-1]].float().mean().item()
    for conditional in (below, right, two_below):
        assert abs(conditional - probability) < 0.02, (below, right, two_below)


def test_dropout_keyed_does_not_depend_on_core_split(device):
    num_tiles = 96
    data = torch.ones([1, 1, 32 * num_tiles, 32])
    input_tensor = ttnn.from_torch(data, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    args = (8, int(INT_MAX * 0.3), f2u(1.0))
    one = run_keyed(device, num_tiles, 1, *args, input_tensor=input_tensor)
    many = run_keyed(device, num_tiles, 12, *args, input_tensor=input_tensor)
    assert torch.equal(one, many)
