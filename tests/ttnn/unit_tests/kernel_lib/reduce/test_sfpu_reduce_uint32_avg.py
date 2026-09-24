# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""UInt32 column AVG through the public Compute API sfpu_reduce, driven from ttnn.

Repro/guard for tenstorrent/tt-metal#57509 items 2 (Blackhole) and 4 (Wormhole B0). No ttnn op routes AVG onto the SFPU reduce, so
this drives sfpu_reduce<PoolType::AVG, DataFormat::UInt32, ReduceDim::REDUCE_COL> from a small
compute kernel via ttnn.generic_op, with real ttnn.uint32 tensors. perform_int_average() used to
pick its divide-by-32 arm from the load mode, and UInt32 shares INT32 with signed Int32, so every
column sum >= 2^31 came back negated (0x80000000 -> 0xFC000000 instead of 0x04000000).

The stimuli mirror test_uint32_reduce_column_average_bit31 in
tt_metal/tt-llk/tests/python_tests/test_sfpu_reduce.py: column c of each band sums to exactly
32 * v[c] + c, so the expected average is v[c] and the comparison is exact.
"""

import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device

TILE = 32
CB_INPUT = 0
CB_OUTPUT = 16

COMPUTE_KERNEL = "tests/ttnn/unit_tests/kernel_lib/reduce/kernels/sfpu_reduce_col_avg.cpp"

# Per-column average at which the 32-element column sum first sets bit 31.
BIT31_COLUMN_AVERAGE = 2**31 // TILE
# Largest per-column average whose 32-element column sum still fits in 32 bits.
UINT32_MAX_COLUMN_AVERAGE = (2**32 - 1) // TILE


def _seeded_column_averages(low: int, high: int) -> list[int]:
    generator = torch.Generator().manual_seed(57509)
    return torch.randint(low, high, (TILE,), generator=generator).tolist()


BANDS = {
    # Control: every column sum stays below 2^31.
    "below_bit31": [BIT31_COLUMN_AVERAGE - TILE + c for c in range(TILE)],
    # Lane 0 sums to exactly 0x80000000.
    "at_bit31": [BIT31_COLUMN_AVERAGE] * TILE,
    # Lanes 0-15 below 2^31, lanes 16-31 at or above it.
    "straddle_bit31": [BIT31_COLUMN_AVERAGE - 16 + c for c in range(TILE)],
    # The issue's value: 32 * 100000000 == 0xBEBC2000.
    "issue_value": [100_000_000] * TILE,
    # Column sums just under 2^32.
    "near_uint32_max": [UINT32_MAX_COLUMN_AVERAGE - c for c in range(TILE)],
    "random_above_bit31": _seeded_column_averages(BIT31_COLUMN_AVERAGE, UINT32_MAX_COLUMN_AVERAGE),
    "random_full_range": _seeded_column_averages(0, UINT32_MAX_COLUMN_AVERAGE),
}


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


def _run_column_average(device, grid: torch.Tensor) -> torch.Tensor:
    shape = tuple(grid.shape)
    device_input = ttnn.from_torch(
        grid.to(torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded_memory_config(shape),
    )
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), ttnn.uint32, ttnn.TILE_LAYOUT, device, _sharded_memory_config(shape)
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=_single_core(),
        runtime_args=_runtime_args([1]),
        defines=[("REDUCE_FORMAT", "DataFormat::UInt32")],
        config=_compute_config(),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT, device_input),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT, output),
    ]
    result = ttnn.generic_op([device_input, output], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs))
    # The column reduce leaves each column's result in row 0.
    return ttnn.to_torch(result, dtype=torch.uint32).to(torch.int64)[0]


@pytest.mark.parametrize("band", list(BANDS), ids=list(BANDS))
def test_sfpu_reduce_uint32_column_average_bit31(device, band):
    arch = str(device.arch()).upper()
    if "QUASAR" in arch:
        pytest.skip("Quasar's SFPU reduce does not support UInt32")
    if "BLACKHOLE" in arch:
        pytest.skip(
            "Blackhole has the same unsigned-average defect; fixed separately in "
            "https://github.com/tenstorrent/tt-metal/pull/57661"
        )

    averages = torch.tensor(BANDS[band], dtype=torch.int64)
    grid = averages.repeat(TILE, 1).clone()
    grid[0, :] += torch.arange(TILE, dtype=torch.int64)
    column_sums = grid.sum(dim=0)
    assert int(column_sums.max()) <= 0xFFFFFFFF, "stimuli must not overflow a UInt32 column sum"
    golden = column_sums >> 5
    assert torch.equal(golden, averages), "band construction lost the intended quotient"

    actual = _run_column_average(device, grid)

    mismatch = torch.nonzero(golden != actual).flatten().tolist()
    detail = "\n".join(
        f"  col={i}: sum=0x{int(column_sums[i]):08X} golden=0x{int(golden[i]):08X} device=0x{int(actual[i]):08X}"
        for i in mismatch[:12]
    )
    assert not mismatch, f"{len(mismatch)}/{TILE} mismatched columns for band '{band}':\n{detail}"
