# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""UInt32 and Int32 column AVG through the public Compute API sfpu_reduce, driven from ttnn.

Repro/guard for tenstorrent/tt-metal#57509 items 2 (Blackhole) and 4 (Wormhole B0). No ttnn op routes AVG onto the SFPU reduce, so
this drives sfpu_reduce<PoolType::AVG, format, ReduceDim::REDUCE_COL> from a small compute kernel
via ttnn.generic_op, with real ttnn.uint32 / ttnn.int32 tensors.

- UInt32: perform_int_average() used to pick its divide-by-32 arm from the load mode, and UInt32
  shares INT32 with signed Int32, so every column sum >= 2^31 came back negated
  (0x80000000 -> 0xFC000000 instead of 0x04000000).
- Int32: ttnn tensors hold two's-complement words, but Int32 AVG used to load with INT32_2S_COMP,
  which reads them as sign-magnitude, so every negative average came back as its magnitude
  (a column summing to -32 gave 1 instead of -1).

Each band is a list of 32 exact column sums, and every comparison is exact: UInt32 divides with a
logical shift, Int32 rounds toward zero. The bands mirror test_uint32_reduce_column_average_bit31 and
test_int32_reduce_column_average_exact in tt_metal/tt-llk/tests/python_tests/test_sfpu_reduce.py.
"""

import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device

TILE = 32
CB_INPUT = 0
CB_OUTPUT = 16

COMPUTE_KERNEL = "tests/ttnn/unit_tests/kernel_lib/reduce/kernels/sfpu_reduce_col_avg.cpp"

INT32_MIN = torch.iinfo(torch.int32).min
INT32_MAX = torch.iinfo(torch.int32).max

# Per-column average at which the 32-element column sum first sets bit 31.
BIT31_COLUMN_AVERAGE = 2**31 // TILE
# Largest per-column average whose 32-element column sum still fits in 32 bits.
UINT32_MAX_COLUMN_AVERAGE = (2**32 - 1) // TILE


def _seeded_column_averages(low: int, high: int) -> list[int]:
    generator = torch.Generator().manual_seed(57509)
    return torch.randint(low, high, (TILE,), generator=generator).tolist()


def _uint32_sums(column_averages: list[int]) -> list[int]:
    # Column c sums to 32 * v[c] + c: the quotient is v[c] and the remainder c is what the divide drops.
    return [TILE * v + c for c, v in enumerate(column_averages)]


UINT32_BANDS = {
    # Control: every column sum stays below 2^31.
    "below_bit31": _uint32_sums([BIT31_COLUMN_AVERAGE - TILE + c for c in range(TILE)]),
    # Lane 0 sums to exactly 0x80000000.
    "at_bit31": _uint32_sums([BIT31_COLUMN_AVERAGE] * TILE),
    # Lanes 0-15 below 2^31, lanes 16-31 at or above it.
    "straddle_bit31": _uint32_sums([BIT31_COLUMN_AVERAGE - 16 + c for c in range(TILE)]),
    # The issue's value: 32 * 100000000 == 0xBEBC2000.
    "issue_value": _uint32_sums([100_000_000] * TILE),
    # Column sums just under 2^32.
    "near_uint32_max": _uint32_sums([UINT32_MAX_COLUMN_AVERAGE - c for c in range(TILE)]),
    "random_above_bit31": _uint32_sums(_seeded_column_averages(BIT31_COLUMN_AVERAGE, UINT32_MAX_COLUMN_AVERAGE)),
    "random_full_range": _uint32_sums(_seeded_column_averages(0, UINT32_MAX_COLUMN_AVERAGE)),
}

INT32_BANDS = {
    # Control: small positive sums, which were already correct.
    "small_positive": [c + 1 for c in range(TILE)],
    # -1 .. -32: every remainder of a small negative sum; only -32 has a non-zero quotient.
    "small_negative": [-(c + 1) for c in range(TILE)],
    # -16 .. 15: both signs and zero in one tile.
    "zero_crossing": [c - TILE // 2 for c in range(TILE)],
    # Exact negative multiples of 32, then the same with the largest remainder.
    "negative_multiples": [-TILE * (c + 1) - (c % 2) * 31 for c in range(TILE)],
    # Lane 0 sums to exactly INT32_MIN.
    "int32_min": [INT32_MIN + c for c in range(TILE)],
    "int32_max": [INT32_MAX - c for c in range(TILE)],
    "random_full_range": torch.randint(
        INT32_MIN, INT32_MAX + 1, (TILE,), generator=torch.Generator().manual_seed(57660), dtype=torch.int64
    ).tolist(),
}

DTYPES = {
    "uint32": (ttnn.uint32, torch.uint32, "DataFormat::UInt32"),
    "int32": (ttnn.int32, torch.int32, "DataFormat::Int32"),
}

CASES = [("uint32", band) for band in UINT32_BANDS] + [("int32", band) for band in INT32_BANDS]


def _single_core() -> ttnn.CoreRangeSet:
    core = ttnn.CoreCoord(0, 0)
    return ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])


def _runtime_args(values: list[int]) -> ttnn.RuntimeArgs:
    args = ttnn.RuntimeArgs()
    args[0][0] = values
    return args


def _sharded_memory_config(shape: tuple[int, int]) -> ttnn.MemoryConfig:
    return ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _compute_config() -> ttnn.ComputeConfigDescriptor:
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        dst_full_sync_en=False,
    )
    # 32-bit integers must unpack straight into the 32-bit DEST; SrcA would truncate them.
    # Host descriptors use the maximum CB count so this vector covers both Wormhole (32) and Blackhole (64).
    unpack_modes = [ttnn.UnpackToDestMode.Default] * 64
    unpack_modes[CB_INPUT] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = unpack_modes
    return config


def _run_column_average(device, grid: torch.Tensor, dtype: str) -> torch.Tensor:
    ttnn_dtype, torch_dtype, reduce_format = DTYPES[dtype]
    shape = tuple(grid.shape)
    device_input = ttnn.from_torch(
        grid.to(torch_dtype),
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded_memory_config(shape),
    )
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), ttnn_dtype, ttnn.TILE_LAYOUT, device, _sharded_memory_config(shape)
    )
    num_tiles = 1
    kernel = ttnn.KernelDescriptor(
        kernel_source=COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=_single_core(),
        runtime_args=_runtime_args([num_tiles]),
        defines=[("REDUCE_FORMAT", reduce_format)],
        config=_compute_config(),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT, device_input),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT, output),
    ]
    result = ttnn.generic_op([device_input, output], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs))
    # The column reduce leaves each column's result in row 0.
    return ttnn.to_torch(result, dtype=torch_dtype).to(torch.int64)[0]


@pytest.mark.parametrize("dtype, band", CASES, ids=[f"{dtype}-{band}" for dtype, band in CASES])
def test_sfpu_reduce_integer_column_average(device, dtype, band):
    arch = str(device.arch()).upper()
    if "QUASAR" in arch:
        pytest.skip("Quasar's SFPU reduce does not support 32-bit integer formats")
    if "BLACKHOLE" in arch:
        pytest.skip(
            "Blackhole has the same integer-average defects; fixed separately in "
            "https://github.com/tenstorrent/tt-metal/pull/57661"
        )

    column_sums = torch.tensor((UINT32_BANDS if dtype == "uint32" else INT32_BANDS)[band], dtype=torch.int64)
    # Every row holds floor(sum / 32) and row 0 also carries the remainder in [0, 31], so the column
    # sums to exactly the target and every element fits in the dtype.
    base = torch.div(column_sums, TILE, rounding_mode="floor")
    grid = base.repeat(TILE, 1).clone()
    grid[0, :] += column_sums - TILE * base
    assert torch.equal(grid.sum(dim=0), column_sums), "band construction lost the column sum"
    if dtype == "uint32":
        assert int(column_sums.max()) <= 0xFFFFFFFF, "stimuli must not overflow a UInt32 column sum"
        golden = column_sums >> 5
    else:
        assert INT32_MIN <= int(column_sums.min()) and int(column_sums.max()) <= INT32_MAX
        golden = torch.div(column_sums, TILE, rounding_mode="trunc")

    actual = _run_column_average(device, grid, dtype)

    mismatch = torch.nonzero(golden != actual).flatten().tolist()
    detail = "\n".join(
        f"  col={i}: sum={int(column_sums[i])} golden={int(golden[i])} device={int(actual[i])}" for i in mismatch[:12]
    )
    assert not mismatch, f"{len(mismatch)}/{TILE} mismatched columns for {dtype} band '{band}':\n{detail}"
