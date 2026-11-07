# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.nomic_embed_text_v2_moe.pirotski_matmul against ttnn.experimental.minimal_matmul.

The op runs minimal_matmul's compute kernel on minimal_matmul's split with the same config, compute kernel config,
dtypes and bias; only the data movement differs. So its output must equal minimal_matmul's bit for bit, wherever
minimal_matmul writes: a block delivered to the wrong slot, a wrong K order, a stale or missing output tile all show
as a mismatch. Torch checks the oracle itself.
"""

import re

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device]

pirotski_matmul = ttnn.experimental.nomic_embed_text_v2_moe.pirotski_matmul
TILE = 32
HIDDEN = 768
INTERMEDIATE = 3072
# The split layer norm's layout: 8 core rows of 3 hidden tiles each, M over up to 11 core columns.
NORM_ROWS = 8


def compute_config(device, fidelity):
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )


def norm_layout(m_tiles, width_tiles, columns):
    """The split layer norm's block-sharded layout: core (x, y) holds M tile rows of column x, K tiles of row y."""
    block_h = -(-m_tiles // columns)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(columns - 1, NORM_ROWS - 1))})
    shard = ttnn.ShardSpec(grid, (block_h * TILE, width_tiles // NORM_ROWS * TILE), ttnn.ShardOrientation.COL_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard)


def config(m_block, k_block, n_block, subblock, grid):
    return ttnn.MinimalMatmulConfig(
        M_block_size=m_block,
        K_block_size=k_block,
        N_block_size=n_block,
        subblock_h=subblock[0],
        subblock_w=subblock[1],
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
    )


def operands(device, shape, n, input_dtype, input_memory, seed, bias=True):
    generator = torch.Generator().manual_seed(seed)
    k = shape[-1]
    x = torch.randn(shape, generator=generator)
    w = torch.randn(k, n, generator=generator) / k**0.5
    b = torch.randn(1, n, generator=generator) if bias else None
    tensors = dict(
        input_tensor=ttnn.from_torch(
            x, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_memory
        ),
        weight=ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        bias=(ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) if bias else None),
    )
    return tensors, (x, w, b)


def reference(tensors, matmul_config, dtype, memory_config, kernel_config):
    return ttnn.experimental.minimal_matmul(
        tensors["input_tensor"],
        tensors["weight"],
        bias_tensor=tensors["bias"],
        config=matmul_config,
        memory_config=memory_config,
        dtype=dtype,
        compute_kernel_config=kernel_config,
    )


def call(
    tensors,
    matmul_config,
    dtype,
    memory_config,
    kernel_config,
    resident_weights=False,
    strip=0,
    strip_read_ahead=False,
    read_ahead=False,
):
    return pirotski_matmul(
        tensors["input_tensor"],
        tensors["weight"],
        config=matmul_config,
        dtype=dtype,
        bias=tensors["bias"],
        memory_config=memory_config,
        compute_kernel_config=kernel_config,
        resident_weights=resident_weights,
        strip_m_tiles=strip,
        strip_read_ahead=strip_read_ahead,
        read_ahead=read_ahead,
    )


def poisoned(device, shape, dtype, memory_config, run):
    """run()'s output, placed where a NaN tensor of the same spec was: an unwritten tile then shows as NaN.

    Freed buffers keep their contents and the allocator hands the same address back first; the program is built
    once beforehand, since a cold call also allocates its kernel binaries."""
    ttnn.deallocate(run())
    poison = ttnn.from_torch(
        torch.full(shape, float("nan")),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    address = poison.buffer_address()
    ttnn.synchronize_device(device)
    ttnn.deallocate(poison)
    output = run()
    assert output.buffer_address() == address, "the output did not land on the poisoned buffer"
    return output


def assert_bit_identical(expected, actual, torch_reference=None):
    expected_values = ttnn.to_torch(expected)
    actual_values = ttnn.to_torch(actual)
    assert not torch.isnan(actual_values).any(), f"{int(torch.isnan(actual_values).sum())} output elements unwritten"
    differ = expected_values.view(torch.int16) != actual_values.view(torch.int16)
    assert not differ.any(), f"{int(differ.sum())} of {differ.numel()} elements differ from minimal_matmul"
    if torch_reference is not None:
        x, w, b = torch_reference
        golden = x.float() @ w.float() + (b.float() if b is not None else 0)
        passing, pcc = comp_pcc(golden, actual_values.float(), 0.999)
        assert passing, f"PCC {pcc} against torch"


# The model's three calls at 8x512 (M = 128 tile rows): layouts, configs and fidelities as tt/matmul_config.py and
# tt/model_config.py choose them, plus the QKV blocking the model takes with this op.
MODEL_CALLS = {
    "qkv": dict(
        n=3 * HIDDEN,
        k=HIDDEN,
        fidelity=ttnn.MathFidelity.HiFi2,
        input="norm",
        output="l1",
        cfg=(12, 2, 8, (1, 4), (11, 9)),
        resident=False,
    ),
    "qkv_m5_resident": dict(
        n=3 * HIDDEN,
        k=HIDDEN,
        fidelity=ttnn.MathFidelity.HiFi2,
        input="norm",
        output="l1",
        cfg=(5, 2, 8, (1, 4), (11, 9)),
        resident=True,
    ),
    "qkv_layer0": dict(
        n=3 * HIDDEN,
        k=HIDDEN,
        fidelity=ttnn.MathFidelity.HiFi2,
        input="norm",
        output="l1",
        cfg=(12, 3, 8, (1, 4), (11, 9)),
        resident=False,
    ),
    "out_proj": dict(
        n=HIDDEN,
        k=HIDDEN,
        fidelity=ttnn.MathFidelity.HiFi3,
        input="l1",
        output="norm",
        cfg=(6, 4, 3, (1, 3), (11, 8)),
        resident=False,
    ),
    "fc2": dict(
        n=HIDDEN,
        k=INTERMEDIATE,
        fidelity=ttnn.MathFidelity.HiFi2,
        input="l1_bf8",
        output="norm",
        cfg=(12, 4, 3, (4, 1), (11, 8)),
        resident=False,
    ),
}


def layouts(case, m_tiles):
    """(input memory config, input dtype, output memory config) of a MODEL_CALLS case."""
    columns = case["cfg"][4][0]
    input_memory = (
        norm_layout(m_tiles, case["k"] // TILE, columns) if case["input"] == "norm" else ttnn.L1_MEMORY_CONFIG
    )
    input_dtype = ttnn.bfloat8_b if case["input"] == "l1_bf8" else ttnn.bfloat16
    output_memory = (
        norm_layout(m_tiles, case["n"] // TILE, columns) if case["output"] == "norm" else ttnn.L1_MEMORY_CONFIG
    )
    return input_memory, input_dtype, output_memory


@pytest.mark.parametrize("name", MODEL_CALLS.keys())
def test_model_calls_match_minimal_matmul(device, name):
    case = MODEL_CALLS[name]
    shape = (8, 1, 512, case["k"])
    input_memory, input_dtype, output_memory = layouts(case, 8 * 512 // TILE)
    tensors, torch_operands = operands(device, shape, case["n"], input_dtype, input_memory, seed=len(name))
    matmul_config = config(*case["cfg"])
    kernel_config = compute_config(device, case["fidelity"])

    expected = reference(tensors, matmul_config, ttnn.bfloat16, output_memory, kernel_config)
    expected_values = ttnn.from_torch(ttnn.to_torch(expected), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn.deallocate(expected)
    output_shape = (*shape[:-1], case["n"])
    actual = poisoned(
        device,
        output_shape,
        ttnn.bfloat16,
        output_memory,
        lambda: call(tensors, matmul_config, ttnn.bfloat16, output_memory, kernel_config, case["resident"]),
    )
    assert_bit_identical(expected_values, actual, torch_operands)


# (M tile rows, K, N, M block, K block, subblock, grid): M > N on minimal_matmul's split, with padding rows on the
# last column (33, 132, 252), partial last M blocks (5 of 12 rows), and K blocks of 2, 3, 4 and 8. Blocks are
# sized so that minimal_matmul's buffers fit beside L1 inputs and outputs.
SHAPES = [
    (33, HIDDEN, HIDDEN, 3, 4, (1, 3), (11, 8)),
    (64, HIDDEN, HIDDEN, 6, 2, (1, 3), (11, 8)),
    (96, HIDDEN, 3 * HIDDEN, 9, 8, (1, 4), (11, 9)),
    (128, INTERMEDIATE, HIDDEN, 12, 4, (4, 1), (11, 8)),
    (132, HIDDEN, HIDDEN, 5, 3, (1, 3), (11, 8)),
    (192, HIDDEN, 3 * HIDDEN, 6, 4, (1, 4), (11, 9)),
    (252, HIDDEN, HIDDEN, 6, 4, (1, 3), (11, 8)),
    (256, HIDDEN, 3 * HIDDEN, 6, 2, (1, 4), (11, 9)),
]


@pytest.mark.parametrize("m_tiles, k, n, m_block, k_block, subblock, grid", SHAPES)
@pytest.mark.parametrize("buffer", ["dram", "l1"])
@pytest.mark.parametrize("resident_weights", [False, True])
def test_shapes_match_minimal_matmul(device, m_tiles, k, n, m_block, k_block, subblock, grid, buffer, resident_weights):
    memory = ttnn.DRAM_MEMORY_CONFIG if buffer == "dram" else ttnn.L1_MEMORY_CONFIG
    shape = (1, 1, m_tiles * TILE, k)
    tensors, torch_operands = operands(device, shape, n, ttnn.bfloat16, memory, seed=m_tiles)
    matmul_config = config(m_block, k_block, n // TILE // grid[1], subblock, grid)
    kernel_config = compute_config(device, ttnn.MathFidelity.HiFi2)
    expected = ttnn.to_torch(reference(tensors, matmul_config, ttnn.bfloat16, memory, kernel_config))
    actual = poisoned(
        device,
        (*shape[:-1], n),
        ttnn.bfloat16,
        memory,
        lambda: call(tensors, matmul_config, ttnn.bfloat16, memory, kernel_config, resident_weights),
    )
    assert_bit_identical(ttnn.from_torch(expected, dtype=ttnn.bfloat16), actual, torch_operands)


# (M tile rows, K, N, M block, K block, N block, subblock, grid, bias): several N blocks a core, every second one run
# K backwards from the last input block. QKV's plans at 192 and 256 tile rows, three N blocks (an odd count, so the
# last runs K backwards), four in one K block (no input block sent after the first N block), and no bias.
N_BLOCK_SHAPES = [
    (192, HIDDEN, 3 * HIDDEN, 9, 8, 4, (1, 4), (11, 9), True),
    (256, HIDDEN, 3 * HIDDEN, 12, 4, 4, (1, 4), (11, 9), True),
    (132, HIDDEN, HIDDEN, 5, 3, 1, (1, 1), (11, 8), True),
    (128, 4 * TILE, 3 * HIDDEN, 6, 4, 2, (1, 2), (11, 9), True),
    (96, INTERMEDIATE, HIDDEN, 9, 4, 1, (3, 1), (11, 8), False),
]


@pytest.mark.parametrize("m_tiles, k, n, m_block, k_block, n_block, subblock, grid, bias", N_BLOCK_SHAPES)
@pytest.mark.parametrize("buffer", ["dram", "l1"])
@pytest.mark.parametrize("resident_weights", [False, True])
def test_n_blocks_match_minimal_matmul(
    device, m_tiles, k, n, m_block, k_block, n_block, subblock, grid, bias, buffer, resident_weights
):
    memory = ttnn.DRAM_MEMORY_CONFIG if buffer == "dram" else ttnn.L1_MEMORY_CONFIG
    shape = (1, 1, m_tiles * TILE, k)
    tensors, torch_operands = operands(device, shape, n, ttnn.bfloat16, memory, seed=m_tiles + n_block, bias=bias)
    matmul_config = config(m_block, k_block, n_block, subblock, grid)
    kernel_config = compute_config(device, ttnn.MathFidelity.HiFi2)
    expected = ttnn.to_torch(reference(tensors, matmul_config, ttnn.bfloat16, memory, kernel_config))
    actual = poisoned(
        device,
        (*shape[:-1], n),
        ttnn.bfloat16,
        memory,
        lambda: call(tensors, matmul_config, ttnn.bfloat16, memory, kernel_config, resident_weights),
    )
    assert_bit_identical(ttnn.from_torch(expected, dtype=ttnn.bfloat16), actual, torch_operands)


def test_batched_input_with_padded_sequences(device):
    """(B, 1, S, K) with S off the tile grid: M counts each sequence at its tile-padded length."""
    shape = (3, 1, 437, HIDDEN)
    tensors, torch_operands = operands(device, shape, HIDDEN, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, seed=437)
    matmul_config = config(4, 4, 3, (1, 3), (11, 8))
    kernel_config = compute_config(device, ttnn.MathFidelity.HiFi3)
    expected = reference(tensors, matmul_config, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, kernel_config)
    actual = call(tensors, matmul_config, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, kernel_config)
    assert actual.shape == expected.shape
    assert_bit_identical(expected, actual, torch_operands)


def test_logical_n_off_the_tile_grid(device):
    """N = 1000 pads to 32 tiles: the output and the bias are N wide, as minimal_matmul's."""
    shape = (1, 1, 128 * TILE, HIDDEN)
    tensors, torch_operands = operands(device, shape, 1000, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, seed=1000)
    matmul_config = config(6, 4, 4, (1, 4), (11, 8))
    kernel_config = compute_config(device, ttnn.MathFidelity.HiFi2)
    expected = reference(tensors, matmul_config, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, kernel_config)
    actual = call(tensors, matmul_config, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, kernel_config)
    assert actual.shape == expected.shape
    assert_bit_identical(expected, actual, torch_operands)


def test_bfloat8_b_output_and_no_bias(device):
    shape = (1, 1, 128 * TILE, HIDDEN)
    tensors, torch_operands = operands(device, shape, HIDDEN, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, 7, bias=False)
    matmul_config = config(6, 4, 3, (1, 3), (11, 8))
    kernel_config = compute_config(device, ttnn.MathFidelity.HiFi2)
    expected = reference(tensors, matmul_config, ttnn.bfloat8_b, ttnn.DRAM_MEMORY_CONFIG, kernel_config)
    actual = call(tensors, matmul_config, ttnn.bfloat8_b, ttnn.DRAM_MEMORY_CONFIG, kernel_config)
    assert_bit_identical(expected, actual)


@pytest.mark.parametrize("name", ["out_proj", "qkv_m5_resident"])
def test_program_cache_reuse(device, name):
    """A second call on new buffers of the same shapes, the first call's still allocated, reuses the program and
    follows the new buffers: out_proj's direct-packed output, QKV's input read from the norm's shards."""
    case = MODEL_CALLS[name]
    shape = (8, 1, 512, case["k"])
    input_memory, input_dtype, output_memory = layouts(case, 8 * 512 // TILE)
    matmul_config = config(*case["cfg"])
    kernel_config = compute_config(device, case["fidelity"])
    run = lambda tensors: call(tensors, matmul_config, ttnn.bfloat16, output_memory, kernel_config, case["resident"])
    first, first_operands = operands(device, shape, case["n"], input_dtype, input_memory, seed=1)
    first_output = run(first)
    entries = device.num_program_cache_entries()

    second, second_operands = operands(device, shape, case["n"], input_dtype, input_memory, seed=2)
    second_output = run(second)

    assert device.num_program_cache_entries() == entries
    assert second["input_tensor"].buffer_address() != first["input_tensor"].buffer_address()
    assert second_output.buffer_address() != first_output.buffer_address()
    for tensors, output, torch_operands in (
        (first, first_output, first_operands),
        (second, second_output, second_operands),
    ):
        expected = reference(tensors, matmul_config, ttnn.bfloat16, output_memory, kernel_config)
        assert_bit_identical(expected, output, torch_operands)
        ttnn.deallocate(expected)


# The strip on the device grid's rows below the call's grid: (M tile rows, K, N, M block, K block, subblock, grid,
# strip_m_tiles, output). The model's out_proj and fc2 at 8x512 packed into the norm layout, its main rows
# in one block and the last column's strip rows past M; a strip on the last of 33 rows; out_proj at 192 and fc2 at 256
# tile rows with interleaved outputs, the main rows in several blocks, one partial.
STRIP_SHAPES = [
    (128, HIDDEN, HIDDEN, 10, 3, (1, 3), (11, 8), 2, "norm"),
    (128, INTERMEDIATE, HIDDEN, 10, 4, (1, 3), (11, 8), 2, "norm"),
    (33, HIDDEN, HIDDEN, 2, 3, (1, 3), (11, 8), 1, "norm"),
    (192, HIDDEN, HIDDEN, 6, 4, (1, 3), (11, 8), 3, "l1"),
    (256, INTERMEDIATE, HIDDEN, 12, 4, (4, 1), (11, 8), 4, "dram"),
]


@pytest.mark.parametrize("m_tiles, k, n, m_block, k_block, subblock, grid, strip, output", STRIP_SHAPES)
@pytest.mark.parametrize("read_ahead", [False, True])
def test_strip_matches_minimal_matmul(
    device, m_tiles, k, n, m_block, k_block, subblock, grid, strip, output, read_ahead
):
    shape = (1, 1, m_tiles * TILE, k)
    # Above 128 tile rows the input stays out of the way of the buffers in DRAM.
    input_memory = ttnn.L1_MEMORY_CONFIG if m_tiles <= 128 else ttnn.DRAM_MEMORY_CONFIG
    tensors, torch_operands = operands(device, shape, n, ttnn.bfloat16, input_memory, seed=m_tiles + strip)
    output_memory = {
        "norm": norm_layout(m_tiles, n // TILE, grid[0]),
        "l1": ttnn.L1_MEMORY_CONFIG,
        "dram": ttnn.DRAM_MEMORY_CONFIG,
    }[output]
    matmul_config = config(m_block, k_block, n // TILE // grid[1], subblock, grid)
    kernel_config = compute_config(device, ttnn.MathFidelity.HiFi2)
    expected = reference(tensors, matmul_config, ttnn.bfloat16, output_memory, kernel_config)
    expected_values = ttnn.from_torch(ttnn.to_torch(expected), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn.deallocate(expected)
    actual = poisoned(
        device,
        (*shape[:-1], n),
        ttnn.bfloat16,
        output_memory,
        lambda: call(
            tensors,
            matmul_config,
            ttnn.bfloat16,
            output_memory,
            kernel_config,
            strip=strip,
            strip_read_ahead=read_ahead,
        ),
    )
    assert_bit_identical(expected_values, actual, torch_operands)


# (M tile rows, K, N, K block, subblock): the model's small-share calls with the main rows' read-ahead, one M block
# and one N block a core on 11 x 8: fc2 and out_proj at 32 tile rows (a share of 3), out_proj at 31 (a last column
# of one row), in the K blocks the model plans there.
READ_AHEAD_SHAPES = [
    (32, INTERMEDIATE, HIDDEN, 12, (1, 3)),
    (32, HIDDEN, HIDDEN, 6, (1, 3)),
    (31, HIDDEN, HIDDEN, 6, (3, 1)),
]


@pytest.mark.parametrize("m_tiles, k, n, k_block, subblock", READ_AHEAD_SHAPES)
@pytest.mark.parametrize("buffer", ["dram", "l1"])
def test_read_ahead_matches_minimal_matmul(device, m_tiles, k, n, k_block, subblock, buffer):
    memory = ttnn.DRAM_MEMORY_CONFIG if buffer == "dram" else ttnn.L1_MEMORY_CONFIG
    shape = (1, 1, m_tiles * TILE, k)
    tensors, torch_operands = operands(device, shape, n, ttnn.bfloat16, memory, seed=m_tiles + k_block)
    grid = (11, 8)
    matmul_config = config(-(-m_tiles // grid[0]), k_block, n // TILE // grid[1], subblock, grid)
    kernel_config = compute_config(device, ttnn.MathFidelity.HiFi2)
    expected = ttnn.to_torch(reference(tensors, matmul_config, ttnn.bfloat16, memory, kernel_config))
    actual = poisoned(
        device,
        (*shape[:-1], n),
        ttnn.bfloat16,
        memory,
        lambda: call(tensors, matmul_config, ttnn.bfloat16, memory, kernel_config, read_ahead=True),
    )
    assert_bit_identical(ttnn.from_torch(expected, dtype=ttnn.bfloat16), actual, torch_operands)


INVALID_CALLS = {
    "m_not_above_n": (dict(m_tiles=32, n=3 * HIDDEN), config(3, 2, 8, (1, 4), (11, 9)), "must exceed N"),
    "n_not_split_by_rows": (dict(), config(6, 4, 3, (1, 3), (11, 7)), "must split evenly over the grid's"),
    "n_block_not_dividing": (dict(), config(6, 4, 2, (1, 2), (11, 8)), "must divide a core's N share"),
    "k_block_not_dividing": (dict(), config(6, 5, 3, (1, 3), (11, 8)), "must divide K's"),
    "subblock_not_dividing": (dict(), config(6, 4, 3, (4, 1), (11, 8)), "must divide the"),
    "grid_too_large": (dict(), config(6, 4, 3, (1, 3), (12, 8)), "must fit the device's"),
    "strip_not_below_share": (dict(strip=12), config(6, 4, 3, (1, 3), (11, 8)), "must be below a core's M share"),
    "strip_rows_not_dividing": (dict(strip=2), config(6, 4, 4, (1, 4), (11, 6)), "that divide its"),
    "strip_read_ahead_without_strip": (dict(strip_read_ahead=True), config(6, 4, 3, (1, 3), (11, 8)), "needs a strip"),
    "read_ahead_with_two_m_blocks": (
        dict(read_ahead=True),
        config(6, 4, 3, (1, 3), (11, 8)),
        "read_ahead needs one M block and one N block",
    ),
    "strip_with_resident_weights": (
        dict(strip=2, resident=True),
        config(6, 4, 3, (1, 3), (11, 8)),
        "one N block a core and streamed weights",
    ),
}


@pytest.mark.parametrize("case", INVALID_CALLS.keys())
def test_invalid_calls_raise(device, expect_error, case):
    overrides, matmul_config, message = INVALID_CALLS[case]
    m_tiles = overrides.get("m_tiles", 128)
    n = overrides.get("n", HIDDEN)
    tensors, _ = operands(device, (1, 1, m_tiles * TILE, HIDDEN), n, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, seed=3)
    with expect_error(RuntimeError, f"pirotski_matmul: .*{re.escape(message)}"):
        call(
            tensors,
            matmul_config,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            compute_config(device, ttnn.MathFidelity.HiFi2),
            resident_weights=overrides.get("resident", False),
            strip=overrides.get("strip", 0),
            strip_read_ahead=overrides.get("strip_read_ahead", False),
            read_ahead=overrides.get("read_ahead", False),
        )
