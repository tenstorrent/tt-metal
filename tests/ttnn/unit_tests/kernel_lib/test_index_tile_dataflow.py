# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Bit-exactness tests for the index_tile_dataflow generator (the #58041 helper).

The helper contract (ttnn/cpp/ttnn/kernel_lib/index_tile_dataflow.hpp): the index tile at width
tile position wt holds, for every element of that width tile, the element's position along the
width — tile[r][c] = c + 32 * wt, identical on every row. The tests run the generator in a
reader-slot dataflow kernel on one core, drain the tile through the house
writer_unary_interleaved_start_id to DRAM, and compare bit-exact against the host golden —
the uint16 tile (LO16 dest, WH/BH) and the uint32 tile (INT32 dest, Quasar) forms alike.

Bit-exactness is the whole point: the generator is a store/NoC-replicate construction whose
correctness is store-visibility and face-replication order, not numerics. Any mismatch is a
byte-level defect (stale seed line, a wrong face copy), not a tolerance question.
"""

import torch
import pytest
import ttnn
from loguru import logger

GENERATOR_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/index_tile/generate_index_tile.cpp"

INDEX_DTYPES = {
    2: ttnn.uint16,
    4: ttnn.uint32,
}

# uint32 tile coverage is parametrized: wt runs past 255 exercise every seed-word position of
# the 32-bit path on Quasar (its natural home); the marker keeps a non-Quasar skip explicit.
QUASAR_U32_WTS = [0, 1, 255, 256, 257]


def index_tile_golden(wt: int, dtype: torch.dtype) -> torch.Tensor:
    """The helper contract: tile[r][c] = c + 32 * wt for every row."""
    base = torch.arange(32, dtype=torch.int64) + 32 * wt
    return base.repeat(32, 1).to(dtype)


def build_program(device, index_width_bytes: int, wt_dim: int, start_wt: int):
    """One reader-slot generator + the house 1-output writer over a single core.

    The generator owns CB c_0 (page = one tile); the writer drains it to DRAM. Depth 2 lets a
    second tile's reserve_back overlap the first tile's write — the sync the helper's
    reserve/push contract exists for.
    """
    assert index_width_bytes in INDEX_DTYPES
    dt = INDEX_DTYPES[index_width_bytes]
    shape = [1, 1, 32, 32 * wt_dim]
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    tile_bytes = 32 * 32 * index_width_bytes
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)

    fmt = ttnn.CBFormatDescriptor(buffer_index=0, data_format=dt, page_size=tile_bytes)
    cbs = [ttnn.CBDescriptor(total_size=tile_bytes * 2, core_ranges=core_grid, format_descriptors=[fmt])]

    generator = ttnn.KernelDescriptor(
        kernel_source=GENERATOR_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=[index_width_bytes],
        runtime_args=ttnn.RuntimeArgs([[wt_dim, start_wt]]),
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=[0] + ttnn.TensorAccessorArgs(tt_out).get_compile_time_args(),
        runtime_args=ttnn.RuntimeArgs([[tt_out.buffer_address(), wt_dim, start_wt]]),
        config=ttnn.WriterConfigDescriptor(),
    )
    program = ttnn.ProgramDescriptor(kernels=[generator, writer], semaphores=[], cbs=cbs)
    return program, [tt_out]


def run_case(device, index_width_bytes: int, wt_dim: int, start_wt: int = 0) -> torch.Tensor:
    program, tensors = build_program(device, index_width_bytes, wt_dim, start_wt)
    output = ttnn.generic_op(tensors, program)
    return ttnn.to_torch(output)


def assert_bit_exact(actual: torch.Tensor, index_width_bytes: int, start_wt: int, label: str) -> None:
    dtype = torch.uint32 if index_width_bytes == 4 else torch.uint16
    golden = torch.cat(
        [index_tile_golden(wt, dtype) for wt in range(start_wt, start_wt + actual.shape[3] // 32)], dim=1
    )
    identical = torch.equal(actual, golden)
    if not identical:
        mismatch = (actual != golden).sum().item()
        diff_pos = (actual != golden).nonzero()[0].tolist()
        logger.error(
            f"{label}: {mismatch} mismatched elements, first at {diff_pos}: "
            f"got {actual[tuple(diff_pos)].item()}, want {golden[tuple(diff_pos)].item()}"
        )
    assert identical, f"{label}: not bit-exact against tile[r][c] = c + 32 * wt"


# =============================================================================
# uint16 index tile (LO16 dest — WH/BH): the form every sort/topk/moe/sampling
# consumer runs on this part.
# =============================================================================
def test_u16_tile_bit_exact_single_wt(device):
    """One tile: the seed, the NoC replicate and the lower-face copy at wt=0."""
    out = run_case(device, 2, wt_dim=1)
    assert_bit_exact(out, 2, 0, "u16 single tile wt=0")


@pytest.mark.parametrize("wt", [0, 1, 7, 255, 256, 257])
def test_u16_tile_bit_exact_per_wt(device, wt):
    """Every seed-word lane position across wt: the two-packing ((v+1)<<16)|v must roll over
    cleanly at every 16-bit boundary, and every face copy must carry the right base."""
    out = run_case(device, 2, wt_dim=1, start_wt=wt)
    assert_bit_exact(out, 2, wt, f"u16 tile wt={wt}")


def test_u16_tile_bit_exact_stream(device):
    """Multiple tiles through one program: the reserve/push sync must carry 8 tiles in wt order."""
    wt_dim = 8
    out = run_case(device, 2, wt_dim=wt_dim)
    assert out.shape == (1, 1, 32, 32 * wt_dim)
    assert_bit_exact(out, 2, 0, "u16 8-tile stream")


def test_u16_tile_bit_exact_stream_past_ring_wrap(device):
    """A deliberately wrapped CB ring: 32 tiles through a 2-deep ring forces reserve/push to
    recycle every slot — the store-visibility guard must hold across ring wraps, not just the
    first pass."""
    wt_dim = 32
    out = run_case(device, 2, wt_dim=wt_dim)
    assert_bit_exact(out, 2, 0, "u16 32-tile ring wrap")


# =============================================================================
# uint32 index tile (INT32 dest — Quasar): the u32 path the #58041 NoC build targets.
# =============================================================================
@pytest.mark.skipif(not hasattr(ttnn, "uint32"), reason="uint32 index tiles unsupported on this build")
def test_u32_tile_bit_exact_single_wt(device):
    out = run_case(device, 4, wt_dim=1)
    assert_bit_exact(out, 4, 0, "u32 single tile wt=0")


@pytest.mark.skipif(not hasattr(ttnn, "uint32"), reason="uint32 index tiles unsupported on this build")
@pytest.mark.parametrize("wt", QUASAR_U32_WTS)
def test_u32_tile_bit_exact_per_wt(device, wt):
    """The 32-bit form has no pairable half-lines: each seed row is face_size stores; the
    guard site is seed + seed_words - 1. wt past 255 rolls every byte lane."""
    out = run_case(device, 4, wt_dim=1, start_wt=wt)
    assert_bit_exact(out, 4, wt, f"u32 tile wt={wt}")


@pytest.mark.skipif(not hasattr(ttnn, "uint32"), reason="uint32 index tiles unsupported on this build")
def test_u32_tile_bit_exact_stream(device):
    out = run_case(device, 4, wt_dim=4)
    assert_bit_exact(out, 4, 0, "u32 4-tile stream")
