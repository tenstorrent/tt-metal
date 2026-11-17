# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""MatmulUnifiedProgramConfig: one placement-first matmul factory (Quasar-native matmul).

Every test drives the same kernels through a different (cores, C_slice_M_tiles, C_slice_N_tiles) placement and
memory layout. Correctness is checked with allclose against an fp32 golden of the bf16-rounded inputs
(the device runs HiFi4 here), plus exact checks with structured inputs (identity / ones / zeros), which catch
indexing and edge-clipping errors that a statistical check would not.

Compute threads: on Quasar the compute kernel runs on all four NEOs of a cluster by default
(num_compute_threads auto = 4), and each C slice's subblocks are assigned round-robin to the threads.
Wormhole / Blackhole have one compute engine per core, so there every test runs with one thread and the
tests that set num_compute_threads > 1 are skipped.

DM threads and buffering: on Quasar up to four reader threads, no more than a C slice's K chunks, take each C
slice's K chunks round-robin (padded to a multiple of the reader threads with credit-only K chunks) and two writer
threads take the compute threads' entries of C; Wormhole / Blackhole have one reader and one writer. Cores with
several C slices keep two in flight. These are fixed; an auto C slice, subblock and K chunk is the combination with
the lowest estimated cycles among those that fit L1 with them. When each
core produces one C slice on a rectangle laid out like the C slices, A is multicast along rows of cores and B down
columns, each reader thread multicasting its own K chunks; when C is one row (column) of C slices, A (B) is multicast
to every core.

Bias and activation are applied to the finished sum in DST before the final pack. Without cores, a
MatmulUnifiedProgramConfig asks for its placement to be chosen by a cost estimate; on Quasar a matmul without a
program config does the same.

Run on Wormhole / Blackhole:
    pytest tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_matmul_unified.py

Run on the Quasar simulator (one config per process; a sim-side hang ignores pytest --timeout):
    TT_METAL_SIMULATOR=<path>/libttsim.so TT_SIMULATOR_LOCALHOST=1 \\
        TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \\
        pytest tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_matmul_unified.py -k <case>

    Needs a craq-sim built from the quasar branch of 2026-09-24 or later. The emulator is the place to confirm
    timing on Quasar.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# Every test in this module takes the function-scoped `device` fixture without
# parametrizing device_params, so the device is opened once per module rather than
# once per test.
pytestmark = pytest.mark.use_module_device

qsr = ttnn._ttnn.operations.experimental.quasar

TILE = 32


def _grid(device):
    g = device.compute_with_storage_grid_size()
    return g.x, g.y


def _on_quasar(device):
    return device.arch() == ttnn.device.Arch.QUASAR


def _skip_unless_threads_supported(device, num_compute_threads):
    if num_compute_threads > 1 and not _on_quasar(device):
        pytest.skip("several compute threads per core need Quasar")


def _rect(x0, y0, x1, y1):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


def _shard(grid, shape, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    return ttnn.ShardSpec(grid, shape, orientation)


def _hifi4(device, fp32_dest_acc_en=False, packer_l1_acc=False):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
        dst_full_sync_en=False,
    )


def _golden(a, b):
    # What the device computes: bf16-rounded operands, fp32 accumulation.
    return torch.matmul(a.to(torch.float32), b.to(torch.float32))


def _quantized(t, dtype):
    # Round-trips t through dtype so the golden sees the same operand the device does.
    return ttnn.to_torch(ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT)) if dtype == ttnn.bfloat8_b else t


def _check(out, golden, rtol=0.02, pcc=0.9999):
    out_f = out.to(torch.float32)
    scale = golden.abs().max().item()
    atol = rtol * scale
    assert_with_pcc(golden, out_f, pcc)
    assert torch.allclose(
        out_f, golden, rtol=rtol, atol=atol
    ), f"max abs err {(out_f - golden).abs().max().item():.4g} vs atol {atol:.4g} (scale {scale:.4g})"


def _run(
    device,
    a,
    b,
    config,
    *,
    in0_mem=ttnn.DRAM_MEMORY_CONFIG,
    in1_mem=ttnn.DRAM_MEMORY_CONFIG,
    out_mem=ttnn.DRAM_MEMORY_CONFIG,
    in0_dtype=ttnn.bfloat16,
    in1_dtype=ttnn.bfloat16,
    out_dtype=ttnn.bfloat16,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
):
    a_t = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=in0_dtype, device=device, memory_config=in0_mem)
    b_t = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=in1_dtype, device=device, memory_config=in1_mem)
    out_t = ttnn.experimental.quasar.matmul(
        a_t,
        b_t,
        program_config=config,
        memory_config=out_mem,
        dtype=out_dtype,
        compute_kernel_config=_hifi4(device, fp32_dest_acc_en, packer_l1_acc),
    )
    return ttnn.to_torch(out_t)


def _randn(*shape):
    return torch.randn(shape, dtype=torch.bfloat16)


# ----------------------------------------------------------------------------------------------------
# Placement sweep: the legacy 1D / 2D strategies and everything in between, one problem, one kernel set
# ----------------------------------------------------------------------------------------------------

ROW_MAJOR, COL_MAJOR = ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR

PLACEMENTS = [
    # name,            cores (x0,y0,x1,y1) or list, C_slice_M_tiles, C_slice_N_tiles, orientation
    ("single_core", (0, 0, 0, 0), 16, 16, ROW_MAJOR),  # 1 block
    ("row_1d_mcast_in0_shape", (0, 0, 7, 0), 16, 2, ROW_MAJOR),  # 8 blocks, one per core, all read the same in0 rows
    ("col_1d_transposed", (0, 0, 0, 7), 2, 16, ROW_MAJOR),  # 8 blocks along M on a column of cores
    ("grid_2d", (0, 0, 3, 3), 4, 4, ROW_MAJOR),  # 16 blocks on a 4x4 rectangle
    ("grid_2d_col_major", (0, 0, 3, 3), 4, 4, COL_MAJOR),  # same blocks, y-fastest core order
    ("more_blocks_than_cores", (0, 0, 1, 1), 4, 4, ROW_MAJOR),  # 16 blocks over 4 cores, 4 each
    ("fewer_blocks_than_cores", (0, 0, 7, 7), 8, 8, ROW_MAJOR),  # 4 blocks, 60 cores idle
    ("uneven_blocks_per_core", (0, 0, 2, 0), 4, 4, ROW_MAJOR),  # 16 blocks over 3 cores: 6, 5, 5
    ("non_rect_cores", [(0, 0, 3, 0), (0, 2, 1, 2)], 4, 4, ROW_MAJOR),  # two ranges, 6 cores, 16 blocks
]


@pytest.mark.parametrize(
    "name,cores,C_slice_M_tiles,C_slice_N_tiles,orientation", PLACEMENTS, ids=[p[0] for p in PLACEMENTS]
)
def test_placements(device, name, cores, C_slice_M_tiles, C_slice_N_tiles, orientation):
    gx, gy = _grid(device)
    ranges = cores if isinstance(cores, list) else [cores]
    for x0, y0, x1, y1 in ranges:
        if x1 >= gx or y1 >= gy:
            pytest.skip(f"needs a {x1 + 1}x{y1 + 1} grid, device has {gx}x{gy}")
    crs = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, y0, x1, y1 in ranges]
    )
    M = K = N = 16 * TILE
    torch.manual_seed(0)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=crs, C_slice_M_tiles=C_slice_M_tiles, C_slice_N_tiles=C_slice_N_tiles, orientation=orientation
    )
    out = _run(device, a, b, config)
    _check(out, _golden(a, b))


def test_repr_and_fields():
    cfg = qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 1, 1), C_slice_M_tiles=2, C_slice_N_tiles=3, K_chunk_tiles=4)
    assert cfg.C_slice_M_tiles == 2 and cfg.C_slice_N_tiles == 3 and cfg.K_chunk_tiles == 4
    assert cfg.subblock_M_tiles is None and cfg.subblock_N_tiles is None and cfg.orientation == ROW_MAJOR
    assert cfg.num_compute_threads is None
    assert "MatmulUnifiedProgramConfig(" in repr(cfg) and "num_compute_threads=" in repr(cfg)
    cfg.num_compute_threads = 2
    assert cfg.num_compute_threads == 2
    assert cfg.fused_activation is None
    cfg.fused_activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
    assert "fused_activation=" in repr(cfg) and cfg.fused_activation.op_type == ttnn.UnaryOpType.RELU


AUTO_C_SLICE_GRIDS = [
    ("grid_2x2", (0, 0, 1, 1)),  # M/2 x N/2 per core
    ("row_of_4", (0, 0, 3, 0)),  # M x N/4
    ("column_of_3", (0, 0, 0, 2)),  # M/3 x N
]


@pytest.mark.parametrize("name,cores", AUTO_C_SLICE_GRIDS, ids=[g[0] for g in AUTO_C_SLICE_GRIDS])
def test_auto_C_slice_from_core_grid(device, name, cores):
    """C_slice_M_tiles / C_slice_N_tiles unset: a C slice dividing M / N split over the bounding box of the cores."""
    gx, gy = _grid(device)
    x0, y0, x1, y1 = cores
    if x1 >= gx or y1 >= gy:
        pytest.skip(f"needs a {x1 + 1}x{y1 + 1} grid, device has {gx}x{gy}")
    M, K, N = 6 * TILE, 4 * TILE, 8 * TILE
    torch.manual_seed(11)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    out = _run(device, a, b, qsr.MatmulUnifiedProgramConfig(cores=_rect(*cores)))
    _check(out, _golden(a, b))


def test_auto_C_slice_from_output_shard(device):
    """With a sharded output the auto C slice is the output shard, so each core writes its own: width-sharded over a
    2x2 rectangle gives M x N/4 slices, which the grid split (M/2 x N/2) would not."""
    gx, gy = _grid(device)
    if gx < 2 or gy < 2:
        pytest.skip("needs a 2x2 grid")
    M, K, N = 4 * TILE, 3 * TILE, 8 * TILE
    cores = _rect(0, 0, 1, 1)
    out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, _shard(cores, [M, 2 * TILE]))
    torch.manual_seed(12)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    out = _run(device, a, b, qsr.MatmulUnifiedProgramConfig(cores=cores), out_mem=out_mem)
    _check(out, _golden(a, b))


# ----------------------------------------------------------------------------------------------------
# Edges and corners: ragged M / K / N, sub-tile shapes, explicit blocking knobs, subblock padding
# ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "M,K,N,C_slice_M_tiles,C_slice_N_tiles,K_chunk_tiles,subblock",
    [
        (7 * TILE, 3 * TILE, 11 * TILE, 3, 4, None, (None, None)),  # 3x3 blocks, ragged right and bottom edge
        (5 * TILE, 8 * TILE, 5 * TILE, 4, 4, 2, (2, 2)),  # 2x2 blocks with a 1-tile edge strip each way
        (2 * TILE, 100, 3 * TILE, 2, 3, None, (1, 3)),  # K=100: last K tile is 4 columns wide, must be zeroed
        (4 * TILE, 6 * TILE, 6 * TILE, 4, 6, 1, (4, 1)),  # K_chunk_tiles=1: six K steps, spill/reload every step
        (3 * TILE, 4 * TILE, 3 * TILE, 3, 3, 4, (3, 1)),  # single K step: no partials at all
        (7 * TILE, 2 * TILE, 5 * TILE, 7, 5, None, (2, 4)),  # subblock does not divide the 7x5 C slice: padded to 8x8
        (
            4 * TILE,
            3 * TILE,
            9 * TILE,
            4,
            3,
            None,
            (2, 2),
        ),  # interior slices' padded-N overshoot overlaps the neighbour
        (3 * TILE, 4 * TILE, 3 * TILE, 3, 3, 2, (2, 2)),  # non-dividing subblock with K spill/reload through partials
    ],
)
def test_edges_and_blocking(device, M, K, N, C_slice_M_tiles, C_slice_N_tiles, K_chunk_tiles, subblock):
    gx, gy = _grid(device)
    torch.manual_seed(1)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, min(gx, 4) - 1, min(gy, 2) - 1),
        C_slice_M_tiles=C_slice_M_tiles,
        C_slice_N_tiles=C_slice_N_tiles,
        K_chunk_tiles=K_chunk_tiles,
        subblock_M_tiles=subblock[0],
        subblock_N_tiles=subblock[1],
    )
    out = _run(device, a, b, config)
    _check(out, _golden(a, b))


def test_single_tile(device):
    """The smallest problem: one tile, one core, 1x1 C slice."""
    M = K = N = TILE
    torch.manual_seed(2)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 0, 0), C_slice_M_tiles=1, C_slice_N_tiles=1)
    out = _run(device, a, b, config)
    _check(out, _golden(a, b))


def test_auto_K_chunk_with_prime_K_tiles(device):
    """K_tiles = 11 is prime: the auto K chunk is all of K or one tile (11 K chunks, spill/reload between each)."""
    M, K, N = 2 * TILE, 11 * TILE, 2 * TILE
    torch.manual_seed(9)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 0, 0), C_slice_M_tiles=2, C_slice_N_tiles=2)
    out = _run(device, a, b, config)
    _check(out, _golden(a, b))


def test_auto_C_slice_shrinks_to_fit_l1(device):
    """One core, 64x64-tile C: the largest C slice (all of C) does not fit L1, so the auto C slice is a smaller one
    whose dims divide it."""
    M, K, N = 64 * TILE, 2 * TILE, 64 * TILE
    torch.manual_seed(13)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    out = _run(device, a, b, qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 0, 0)))
    _check(out, _golden(a, b))


def test_single_tile_C_slice_over_long_K(device):
    """A 1x1-tile C slice over K = 64 tiles, at the auto K chunk (fp32 accumulation, since 2048 products in 16-bit
    DST drift past the 2% bound)."""
    M, K, N = TILE, 64 * TILE, TILE
    torch.manual_seed(14)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 0, 0), C_slice_M_tiles=1, C_slice_N_tiles=1)
    out = _run(device, a, b, config, fp32_dest_acc_en=True)
    _check(out, _golden(a, b))


def test_sub_tile_dims(device):
    """M=5, K=10, N=7 elements: one partly-valid tile per operand; A's K padding must be zeroed."""
    M, K, N = 5, 10, 7
    torch.manual_seed(3)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 0, 0), C_slice_M_tiles=1, C_slice_N_tiles=1)
    out = _run(device, a, b, config)
    _check(out, _golden(a, b))


def test_auto_subblock_falls_back_when_padding_does_not_fit(device):
    """22x22-tile C slice: subblocks that pad a dim to 24 blow the DFB extent limit, so auto must take one that
    fits (2x2 pads nothing) instead of failing."""
    M = N = 22 * TILE
    K = TILE
    torch.manual_seed(4)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 0, 0), C_slice_M_tiles=22, C_slice_N_tiles=22)
    out = _run(device, a, b, config)
    _check(out, _golden(a, b))


# ----------------------------------------------------------------------------------------------------
# Structured inputs: exact answers that expose indexing, clipping and K-step errors bit for bit
# ----------------------------------------------------------------------------------------------------


def test_identity_is_exact_on_ragged_edges(device):
    """in1 = I: the output must be in0 bit for bit, on every tile of every edge block."""
    gx, gy = _grid(device)
    M, K = 7 * TILE, 5 * TILE
    torch.manual_seed(2)
    a = _randn(1, 1, M, K)
    b = torch.eye(K, dtype=torch.bfloat16).reshape(1, 1, K, K)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, min(gx, 3) - 1, min(gy, 3) - 1), C_slice_M_tiles=3, C_slice_N_tiles=2
    )
    out = _run(device, a, b, config)
    assert torch.equal(out.to(torch.bfloat16), a), "identity matmul differs from in0"


def test_ones_give_constant_k(device):
    """all-ones inputs: every output element is exactly K; a skipped or doubled K step shows up here."""
    gx, gy = _grid(device)
    M, K, N = 6 * TILE, 9 * TILE, 5 * TILE
    a = torch.ones(1, 1, M, K, dtype=torch.bfloat16)
    b = torch.ones(1, 1, K, N, dtype=torch.bfloat16)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, min(gx, 2) - 1, 0), C_slice_M_tiles=4, C_slice_N_tiles=2, K_chunk_tiles=3
    )
    out = _run(device, a, b, config)
    assert torch.equal(out.to(torch.float32), torch.full((1, 1, M, N), float(K)))


def test_zeros_give_exact_zero(device):
    """in0 = 0: every output element is exactly 0; stale DFB entries or unclipped subblock padding
    leak nonzeros here (the 2x4 subblock pads the 2x3 C slice)."""
    gx, gy = _grid(device)
    M, K, N = 5 * TILE, 6 * TILE, 7 * TILE
    torch.manual_seed(5)
    a = torch.zeros(1, 1, M, K, dtype=torch.bfloat16)
    b = _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, min(gx, 2) - 1, min(gy, 2) - 1),
        C_slice_M_tiles=2,
        C_slice_N_tiles=3,
        K_chunk_tiles=2,
        subblock_M_tiles=2,
        subblock_N_tiles=4,
    )
    out = _run(device, a, b, config)
    assert torch.equal(out.to(torch.float32), torch.zeros(1, 1, M, N))


# ----------------------------------------------------------------------------------------------------
# Batch: broadcast in1, batched in1, several leading dims, and batch interleaved with K spill
# ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("K_chunk_tiles", [None, 1], ids=["single_K_chunk", "K_spill"])
@pytest.mark.parametrize("bcast", [True, False], ids=["bcast_in1", "batched_in1"])
def test_batch(device, bcast, K_chunk_tiles):
    gx, gy = _grid(device)
    B, M, K, N = 3, 4 * TILE, 4 * TILE, 6 * TILE
    torch.manual_seed(3)
    a = _randn(1, B, M, K)
    b = _randn(1, 1, K, N) if bcast else _randn(1, B, K, N)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, min(gx, 3) - 1, 0), C_slice_M_tiles=2, C_slice_N_tiles=3, K_chunk_tiles=K_chunk_tiles
    )
    out = _run(device, a, b, config)
    _check(out, _golden(a, b))


@pytest.mark.parametrize(
    "batch,M_tiles,N_tiles,cores,C_slice,num_batch_groups,K_chunk_tiles",
    [
        # 16 cores in 4 groups of 4, 2 batches each, one C slice per core and batch.
        (8, 2, 2, (0, 0, 3, 3), (1, 1), 4, None),
        # 7 batches over 3 groups of 2 cores (3, 2 and 2 batches); 3 C slices per batch, so 2 and 1 per core.
        (7, 3, 1, (0, 0, 2, 1), (1, 1), 3, None),
        # One batch per group, ragged C slices, K over several K chunks (with several reader threads on Quasar).
        (4, 3, 3, (0, 0, 3, 1), (2, 2), 4, 1),
    ],
    ids=["even", "uneven", "one_batch_per_group_K_chunks"],
)
def test_batch_groups(device, batch, M_tiles, N_tiles, cores, C_slice, num_batch_groups, K_chunk_tiles):
    """Batches with their own B split over groups of the cores, each group taking a run of the batches."""
    gx, gy = _grid(device)
    if cores[2] >= gx or cores[3] >= gy:
        pytest.skip("grid too small")
    M, K, N = M_tiles * TILE, 4 * TILE, N_tiles * TILE
    torch.manual_seed(5)
    a, b = _randn(1, batch, M, K), _randn(1, batch, K, N)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(*cores),
        C_slice_M_tiles=C_slice[0],
        C_slice_N_tiles=C_slice[1],
        K_chunk_tiles=K_chunk_tiles,
        num_batch_groups=num_batch_groups,
    )
    _check(_run(device, a, b, config), _golden(a, b))


def test_batch_from_multiple_leading_dims(device):
    """(2, 3, M, K) x (2, 3, K, N): every leading dim folds into one batch of 6."""
    gx, gy = _grid(device)
    M, K, N = 2 * TILE, 3 * TILE, 2 * TILE
    torch.manual_seed(6)
    a, b = _randn(2, 3, M, K), _randn(2, 3, K, N)
    config = qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, min(gx, 2) - 1, 0), C_slice_M_tiles=2, C_slice_N_tiles=2)
    out = _run(device, a, b, config)
    _check(out, _golden(a, b))


# ----------------------------------------------------------------------------------------------------
# Memory layouts: the same kernels read and write interleaved, L1-sharded and DRAM-sharded tensors
# ----------------------------------------------------------------------------------------------------


def test_height_sharded_in0_and_out(device):
    """in0 height-sharded over a column of 4 cores, output height-sharded the same way (one block per core)."""
    gx, gy = _grid(device)
    if gy < 4:
        pytest.skip("needs 4 rows")
    M, K, N = 8 * TILE, 4 * TILE, 3 * TILE
    cores = _rect(0, 0, 0, 3)
    in0_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard(cores, [2 * TILE, K])
    )
    out_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard(cores, [2 * TILE, N])
    )
    torch.manual_seed(4)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(cores=cores, C_slice_M_tiles=2, C_slice_N_tiles=3)
    out = _run(device, a, b, config, in0_mem=in0_mem, out_mem=out_mem)
    _check(out, _golden(a, b))


def test_block_sharded_in0_and_out(device):
    """2D: in0 block-sharded on a 2x2 rectangle, output block-sharded on the same rectangle."""
    M, K, N = 4 * TILE, 4 * TILE, 6 * TILE
    cores = _rect(0, 0, 1, 1)
    in0_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, _shard(cores, [2 * TILE, 2 * TILE])
    )
    out_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, _shard(cores, [2 * TILE, 3 * TILE])
    )
    torch.manual_seed(5)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(cores=cores, C_slice_M_tiles=2, C_slice_N_tiles=3)
    out = _run(device, a, b, config, in0_mem=in0_mem, out_mem=out_mem)
    _check(out, _golden(a, b))


def test_width_sharded_in1_and_out(device):
    """1D along N: in1 width-sharded in L1 on a row of 4 cores, output width-sharded on the same row."""
    gx, gy = _grid(device)
    if gx < 4:
        pytest.skip("needs 4 columns")
    M, K, N = 2 * TILE, 4 * TILE, 8 * TILE
    cores = _rect(0, 0, 3, 0)
    in1_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, _shard(cores, [K, 2 * TILE]))
    out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, _shard(cores, [M, 2 * TILE]))
    torch.manual_seed(6)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(cores=cores, C_slice_M_tiles=2, C_slice_N_tiles=2)
    out = _run(device, a, b, config, in1_mem=in1_mem, out_mem=out_mem)
    _check(out, _golden(a, b))


def test_dram_sharded_in1(device):
    """in1 width-sharded across the DRAM banks (the 'DRAM-sharded' strategy's weight layout), read by page id."""
    gx, gy = _grid(device)
    dram = device.dram_grid_size()
    num_banks = dram.x * dram.y
    M, K = 1 * TILE, 4 * TILE
    N = num_banks * 2 * TILE
    dram_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))])
    in1_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, _shard(dram_grid, [K, 2 * TILE])
    )
    torch.manual_seed(7)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, min(gx, 8) - 1, 0), C_slice_M_tiles=1, C_slice_N_tiles=2)
    out = _run(device, a, b, config, in1_mem=in1_mem)
    _check(out, _golden(a, b))


def test_borrowed_B_over_several_K_chunks(device):
    """B width-sharded in L1 on the row of cores that own the C slices: its shard is bound as the B slice DFB
    (no copy) and consumed one K chunk at a time; four K chunks here."""
    gx, gy = _grid(device)
    if gx < 4:
        pytest.skip("needs 4 columns")
    M, K, N = 2 * TILE, 8 * TILE, 8 * TILE
    cores = _rect(0, 0, 3, 0)
    in1_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, _shard(cores, [K, 2 * TILE]))
    torch.manual_seed(13)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(cores=cores, C_slice_M_tiles=2, C_slice_N_tiles=2, K_chunk_tiles=2)
    out = _run(device, a, b, config, in1_mem=in1_mem)
    _check(out, _golden(a, b))


def test_borrowable_A_with_a_7_tile_C_slice_height(device):
    """A's L1 shard is borrowable, but only by a subblock whose height divides 7 (padding voids the borrow): the
    estimate weighs the A copy the borrow saves against subblock volume."""
    gx, gy = _grid(device)
    if gy < 2:
        pytest.skip("needs 2 rows")
    M, K, N = 14 * TILE, 2 * TILE, 3 * TILE
    cores = _rect(0, 0, 0, 1)
    in0_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard(cores, [7 * TILE, K])
    )
    torch.manual_seed(5)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(cores=cores, C_slice_M_tiles=7, C_slice_N_tiles=3)
    out = _run(device, a, b, config, in0_mem=in0_mem)
    _check(out, _golden(a, b))


def test_height_sharded_A_with_padded_K_takes_the_copy_path(device):
    """K = 100 is not a tile multiple, so A's padding has to be zeroed in the DFB: A cannot be borrowed and is
    copied even though its shard matches the C slices."""
    gx, gy = _grid(device)
    if gy < 4:
        pytest.skip("needs 4 rows")
    M, K, N = 8 * TILE, 100, 3 * TILE
    cores = _rect(0, 0, 0, 3)
    in0_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard(cores, [2 * TILE, 4 * TILE])
    )
    torch.manual_seed(14)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(cores=cores, C_slice_M_tiles=2, C_slice_N_tiles=3)
    out = _run(device, a, b, config, in0_mem=in0_mem)
    _check(out, _golden(a, b))


@pytest.mark.parametrize(
    "batch,M_tiles,N_tiles,cores,C_slice,extra,out_layout,shard_cores,shard_tiles,orientation",
    [
        # The C slices (2x2 on a 2x2 rectangle) are not the shards (1x4).
        (
            1,
            4,
            4,
            (0, 0, 1, 1),
            (2, 2),
            {},
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (0, 0, 1, 1),
            (1, 4),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
        # Shards of the C slices' shape, on other cores than the ones producing them.
        (
            1,
            4,
            4,
            (0, 0, 1, 0),
            (4, 2),
            {},
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (0, 1, 1, 1),
            (4, 2),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
        # Batches with their own B: each core produces its C slice in both batches, which are different shards.
        (
            2,
            4,
            4,
            (0, 0, 1, 1),
            (2, 2),
            {},
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (0, 0, 1, 1),
            (2, 4),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
        # COL_MAJOR block sharding puts shard (row, column) on core (x=row, y=column): a 2x3 rectangle for 2x3 C
        # slices. They are the shards, packed in place (one compute thread, subblocks as wide as the C slice).
        (
            1,
            4,
            6,
            (0, 0, 1, 2),
            (2, 2),
            dict(num_compute_threads=1, subblock_M_tiles=1, subblock_N_tiles=2),
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (0, 0, 1, 2),
            (2, 2),
            ttnn.ShardOrientation.COL_MAJOR,
        ),
    ],
    ids=["C_slices_not_the_shards", "shards_on_other_cores", "batched_B", "block_col_major_in_place"],
)
def test_sharded_C_takes_any_placement(
    device, batch, M_tiles, N_tiles, cores, C_slice, extra, out_layout, shard_cores, shard_tiles, orientation
):
    """A sharded C does not limit the placement: each core writes its tiles wherever C's shards are."""
    gx, gy = _grid(device)
    if max(cores[2], shard_cores[2]) >= gx or max(cores[3], shard_cores[3]) >= gy:
        pytest.skip("grid too small")
    M, K, N = M_tiles * TILE, 3 * TILE, N_tiles * TILE
    out_mem = ttnn.MemoryConfig(
        out_layout,
        ttnn.BufferType.L1,
        _shard(_rect(*shard_cores), [shard_tiles[0] * TILE, shard_tiles[1] * TILE], orientation),
    )
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(*cores), C_slice_M_tiles=C_slice[0], C_slice_N_tiles=C_slice[1], orientation=orientation, **extra
    )
    torch.manual_seed(16)
    a, b = _randn(batch, 1, M, K), _randn(batch, 1, K, N)
    _check(_run(device, a, b, config, out_mem=out_mem), _golden(a, b))


@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.L1, ttnn.BufferType.DRAM], ids=["l1", "dram"])
@pytest.mark.parametrize(
    "out_layout",
    [ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED],
    ids=["height", "width"],
)
def test_specless_sharded_C_keeps_the_layout_asked_for(device, out_layout, buffer_type):
    """C slices that tile C as blocks (2x2 on a 2x2 rectangle) and a height- or width-sharded C without a shard spec:
    C gets the layout asked for, split over the grid, and each core writes its tiles where their shards are."""
    gx, gy = _grid(device)
    if gx < 2 or gy < 2:
        pytest.skip("needs a 2x2 grid")
    M, K, N = 4 * TILE, 3 * TILE, 4 * TILE
    torch.manual_seed(17)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    a_t = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    b_t = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    out = ttnn.experimental.quasar.matmul(
        a_t,
        b_t,
        program_config=qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 1, 1), C_slice_M_tiles=2, C_slice_N_tiles=2),
        memory_config=ttnn.MemoryConfig(out_layout, buffer_type),
        compute_kernel_config=_hifi4(device),
    )
    assert out.memory_config().memory_layout == out_layout
    assert out.memory_config().buffer_type == buffer_type
    _check(ttnn.to_torch(out), _golden(a, b))


def test_sharded_C_with_narrow_subblock_takes_the_copy_path(device):
    """An explicit subblock narrower than the C slice means pack order != shard order, so C is written by the
    writer instead of packed in place (with a warning); the result must still be right."""
    gx, gy = _grid(device)
    if gy < 2:
        pytest.skip("needs 2 rows")
    M, K, N = 4 * TILE, 4 * TILE, 4 * TILE
    cores = _rect(0, 0, 0, 1)
    out_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard(cores, [2 * TILE, N])
    )
    torch.manual_seed(15)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=cores, C_slice_M_tiles=2, C_slice_N_tiles=4, subblock_M_tiles=1, subblock_N_tiles=2
    )
    out = _run(device, a, b, config, out_mem=out_mem)
    _check(out, _golden(a, b))


def test_single_core_all_operands_borrowed(device):
    """One core, A / B / C all L1-sharded on it: every DFB is a resident shard, so the reader reads nothing
    and the writer writes nothing."""
    M, K, N = 3 * TILE, 4 * TILE, 2 * TILE
    core = _rect(0, 0, 0, 0)
    in0_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard(core, [M, K]))
    in1_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, _shard(core, [K, N]))
    out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard(core, [M, N]))
    torch.manual_seed(16)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(cores=core, C_slice_M_tiles=3, C_slice_N_tiles=2)
    out = _run(device, a, b, config, in0_mem=in0_mem, in1_mem=in1_mem, out_mem=out_mem)
    _check(out, _golden(a, b))


def test_l1_interleaved_everything(device):
    M, K, N = 4 * TILE, 4 * TILE, 4 * TILE
    torch.manual_seed(8)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 1, 1), C_slice_M_tiles=2, C_slice_N_tiles=2)
    out = _run(
        device,
        a,
        b,
        config,
        in0_mem=ttnn.L1_MEMORY_CONFIG,
        in1_mem=ttnn.L1_MEMORY_CONFIG,
        out_mem=ttnn.L1_MEMORY_CONFIG,
    )
    _check(out, _golden(a, b))


# ----------------------------------------------------------------------------------------------------
# Compute options: bfp8 operands and output, fp32 accumulation, packer L1 accumulation
# ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "in0_dtype,in1_dtype,out_dtype,rtol",
    [
        (ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat16, 0.03),
        (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16, 0.03),
        (ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, 0.06),
    ],
    ids=["bfp8_inputs", "bfp8_weights_only", "bfp8_out"],
)
def test_bfp8(device, in0_dtype, in1_dtype, out_dtype, rtol):
    M, K, N = 4 * TILE, 8 * TILE, 4 * TILE
    torch.manual_seed(9)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 1, 1), C_slice_M_tiles=2, C_slice_N_tiles=2)
    out = _run(device, a, b, config, in0_dtype=in0_dtype, in1_dtype=in1_dtype, out_dtype=out_dtype)
    _check(out, _golden(_quantized(a, in0_dtype), _quantized(b, in1_dtype)), rtol=rtol)


@pytest.mark.parametrize(
    "C_slice_M_tiles,C_slice_N_tiles,subblock",
    [(4, 2, (None, None)), (3, 3, (2, 2))],
    ids=["dividing_subblock", "padded_subblock"],
)
def test_fp32_dest_acc_and_fp32_out(device, C_slice_M_tiles, C_slice_N_tiles, subblock):
    """fp32 accumulation caps the subblock at 4 tiles; a 2x2 subblock pads the 3x3 slice under that cap.
    fp32 output means partials and C differ in format."""
    M, K, N = 4 * TILE, 8 * TILE, 4 * TILE
    torch.manual_seed(10)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, 1, 0),
        C_slice_M_tiles=C_slice_M_tiles,
        C_slice_N_tiles=C_slice_N_tiles,
        K_chunk_tiles=2,
        subblock_M_tiles=subblock[0],
        subblock_N_tiles=subblock[1],
    )
    out = _run(device, a, b, config, out_dtype=ttnn.float32, fp32_dest_acc_en=True)
    _check(out, _golden(a, b), rtol=0.005)


@pytest.mark.parametrize(
    "C_slice_M_tiles,C_slice_N_tiles,subblock",
    [(2, 2, (None, None)), (3, 3, (2, 2))],
    ids=["dividing_subblock", "padded_subblock"],
)
def test_packer_l1_acc(device, C_slice_M_tiles, C_slice_N_tiles, subblock):
    """packer_l1_acc engages when there are more than 2 K steps (K_chunk_tiles=1 on K_tiles=6 gives 6);
    the padded case must keep the C_partials DFB's credits balanced over padded subblocks."""
    M, K, N = 4 * TILE, 6 * TILE, 4 * TILE
    torch.manual_seed(11)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, 1, 1),
        C_slice_M_tiles=C_slice_M_tiles,
        C_slice_N_tiles=C_slice_N_tiles,
        K_chunk_tiles=1,
        subblock_M_tiles=subblock[0],
        subblock_N_tiles=subblock[1],
    )
    out = _run(device, a, b, config, packer_l1_acc=True)
    _check(out, _golden(a, b))


def test_C_slice_single_buffered_when_two_do_not_fit_one_dfb(device):
    """Two C slices of 8x38 tiles per core: double-buffered they would exceed one DFB, so C_slice holds one."""
    M, K, N = 16 * TILE, 2 * TILE, 38 * TILE
    torch.manual_seed(13)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 0, 0), C_slice_M_tiles=8, C_slice_N_tiles=38)
    out = _run(device, a, b, config, packer_l1_acc=True)
    _check(out, _golden(a, b))


@pytest.mark.parametrize("K_tiles", [2, 4], ids=["two_K_chunks", "four_K_chunks"])
@pytest.mark.parametrize("C_slices_per_core", [1, 2], ids=["one_C_slice", "two_C_slices"])
def test_accumulate_in_C_slice_then_another_op(device, K_tiles, C_slices_per_core):
    """With packer_l1_acc and a bf16 C every K chunk adds straight onto the C slice, which goes to the writer one
    subblock at a time on the last K chunk; the next program on the same cores must not find L1 accumulation on."""
    M, K, N = 2 * TILE, K_tiles * TILE, 4 * C_slices_per_core * TILE
    torch.manual_seed(12)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, 1, 0),
        C_slice_M_tiles=2,
        C_slice_N_tiles=2,
        K_chunk_tiles=1,
        subblock_M_tiles=2,
        subblock_N_tiles=1,
    )
    a_t = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    b_t = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    out_t = ttnn.experimental.quasar.matmul(
        a_t, b_t, program_config=config, compute_kernel_config=_hifi4(device, packer_l1_acc=True)
    )
    relu_t = ttnn.relu(out_t)
    golden = _golden(a, b)
    _check(ttnn.to_torch(out_t), golden)
    _check(ttnn.to_torch(relu_t), torch.relu(golden))


# ----------------------------------------------------------------------------------------------------
# Fused bias and activation: applied to the finished sum in DST, once, before the final pack
# ----------------------------------------------------------------------------------------------------


def _run_linear(device, a, b, bias, config, *, packer_l1_acc=False, fp32_dest_acc_en=False, bias_dtype=ttnn.bfloat16):
    a_t = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    b_t = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    bias_t = None
    if bias is not None:
        bias_t = ttnn.from_torch(bias, layout=ttnn.TILE_LAYOUT, dtype=bias_dtype, device=device)
    out = ttnn.experimental.quasar.linear(
        a_t,
        b_t,
        bias=bias_t,
        program_config=config,
        compute_kernel_config=_hifi4(device, fp32_dest_acc_en, packer_l1_acc),
    )
    return ttnn.to_torch(out)


def _bias_row(N):
    # Larger than a typical A @ B entry, so a bias applied per K chunk or after the activation stands out.
    return (torch.rand(1, 1, 1, N) * 8 - 4).to(torch.bfloat16)


# name, (M, K, N) in tiles, C slice, K_chunk_tiles, cores (x1, y1), packer_l1_acc
BIAS_CASES = [
    ("single_K_chunk", (4, 4, 4), (2, 2), None, (1, 1), False),
    ("K_spill", (2, 6, 4), (2, 2), 1, (1, 0), False),
    ("K_spill_packer_l1_acc", (2, 8, 4), (2, 4), 2, (0, 0), True),
    ("several_C_slices_per_core", (8, 2, 8), (2, 2), None, (1, 0), False),
    ("several_C_slices_K_spill", (4, 4, 6), (2, 3), 1, (0, 0), False),
    ("ragged_N_padded_subblock", (3, 3, 5), (3, 5), None, (0, 0), False),
]


@pytest.mark.parametrize(
    "name,MKN,C_slice,K_chunk_tiles,cores,packer_l1_acc", BIAS_CASES, ids=[c[0] for c in BIAS_CASES]
)
def test_fused_bias(device, name, MKN, C_slice, K_chunk_tiles, cores, packer_l1_acc):
    M, K, N = (d * TILE for d in MKN)
    torch.manual_seed(12)
    a, b, bias = _randn(1, 1, M, K), _randn(1, 1, K, N), _bias_row(N)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, *cores), C_slice_M_tiles=C_slice[0], C_slice_N_tiles=C_slice[1], K_chunk_tiles=K_chunk_tiles
    )
    out = _run_linear(device, a, b, bias, config, packer_l1_acc=packer_l1_acc)
    _check(out, _golden(a, b) + bias.to(torch.float32))


def test_fused_bias_is_exact_on_zero_A(device):
    # C = 0 @ B + bias: every row of C is the bias row exactly, through two K chunks and ragged edges.
    M, K, N = 2 * TILE + 5, 2 * TILE, 3 * TILE + 7
    a, b, bias = torch.zeros(1, 1, M, K, dtype=torch.bfloat16), _randn(1, 1, K, N), _bias_row(N)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, 1, 0), C_slice_M_tiles=3, C_slice_N_tiles=2, K_chunk_tiles=1
    )
    out = _run_linear(device, a, b, bias, config)
    assert torch.equal(out.to(torch.float32), bias.to(torch.float32).expand(1, 1, M, N))


@pytest.mark.parametrize("K_chunk_tiles", [None, 1], ids=["single_K_chunk", "K_spill"])
def test_fused_bias_batched_A(device, K_chunk_tiles):
    M, K, N = 2 * TILE, 2 * TILE, 4 * TILE
    torch.manual_seed(5)
    a, b, bias = _randn(3, 1, M, K), _randn(1, 1, K, N), _bias_row(N)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, 1, 0), C_slice_M_tiles=2, C_slice_N_tiles=2, K_chunk_tiles=K_chunk_tiles
    )
    out = _run_linear(device, a, b, bias, config)
    _check(out, _golden(a, b) + bias.to(torch.float32))


def test_bias_with_fp32_dest_acc_is_added_separately(device):
    # The row broadcast into DST needs a 16-bit DST, so with fp32 accumulation ttnn.linear adds the bias after.
    M, K, N = 2 * TILE, 4 * TILE, 4 * TILE
    torch.manual_seed(6)
    a, b, bias = _randn(1, 1, M, K), _randn(1, 1, K, N), _bias_row(N)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, 0, 0), C_slice_M_tiles=2, C_slice_N_tiles=4, K_chunk_tiles=2
    )
    out = _run_linear(device, a, b, bias, config, fp32_dest_acc_en=True)
    _check(out, _golden(a, b) + bias.to(torch.float32))


def test_fused_bias_bfp8(device):
    M, K, N = 2 * TILE, 2 * TILE, 4 * TILE
    torch.manual_seed(7)
    a, b, bias = _randn(1, 1, M, K), _randn(1, 1, K, N), _bias_row(N)
    config = qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 1, 0), C_slice_M_tiles=2, C_slice_N_tiles=2)
    out = _run_linear(device, a, b, bias, config, bias_dtype=ttnn.bfloat8_b)
    _check(out, _golden(a, b) + _quantized(bias, ttnn.bfloat8_b).to(torch.float32))


@pytest.mark.parametrize(
    "bias_rows,fp32_dest_acc_en", [(1, True), (16, False)], ids=["fp32_dest_acc", "several_bias_rows"]
)
def test_activation_follows_a_bias_added_separately(device, bias_rows, fp32_dest_acc_en):
    # ttnn.linear adds this bias after the op, every row of it, and the fused activation moves after the add:
    # relu(A @ B + bias), not relu(A @ B) + bias.
    M, K, N = 16, 2 * TILE, 4 * TILE
    torch.manual_seed(9)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    bias = (torch.rand(1, 1, bias_rows, N) * 8 - 4).to(torch.bfloat16)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, 1, 0),
        C_slice_M_tiles=1,
        C_slice_N_tiles=2,
        fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
    )
    out = _run_linear(device, a, b, bias, config, fp32_dest_acc_en=fp32_dest_acc_en)
    _check(out, torch.relu(_golden(a, b) + bias.to(torch.float32)), pcc=0.999)


ACTIVATIONS = [
    ("relu", ttnn.UnaryOpType.RELU, torch.relu, 0.02),
    ("gelu", ttnn.UnaryOpType.GELU, torch.nn.functional.gelu, 0.03),
    ("silu", ttnn.UnaryOpType.SILU, torch.nn.functional.silu, 0.03),
]


@pytest.mark.parametrize("with_bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize("K_chunk_tiles", [None, 1], ids=["single_K_chunk", "K_spill"])
@pytest.mark.parametrize("name,op,torch_op,rtol", ACTIVATIONS, ids=[a[0] for a in ACTIVATIONS])
def test_fused_activation(device, name, op, torch_op, rtol, K_chunk_tiles, with_bias):
    M, K, N = 4 * TILE, 3 * TILE, 4 * TILE
    torch.manual_seed(8)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    bias = _bias_row(N) if with_bias else None
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, 1, 1),
        C_slice_M_tiles=2,
        C_slice_N_tiles=2,
        K_chunk_tiles=K_chunk_tiles,
        fused_activation=ttnn.UnaryWithParam(op),
    )
    out = _run_linear(device, a, b, bias, config)
    golden = _golden(a, b) + (bias.to(torch.float32) if with_bias else 0)
    _check(out, torch_op(golden), rtol=rtol, pcc=0.999)


# ----------------------------------------------------------------------------------------------------
# Everything at once: interactions between mechanisms show up here first
# ----------------------------------------------------------------------------------------------------


def test_ragged_batched_spill_all_at_once(device):
    """Ragged M / K / N, a batch, K spill with packer L1 accumulation, a padded subblock and an
    uneven C-slice split, all in one run."""
    gx, gy = _grid(device)
    B, M, K, N = 2, 7 * TILE + 13, 5 * TILE + 20, 9 * TILE + 5
    torch.manual_seed(17)
    a, b = _randn(1, B, M, K), _randn(1, B, K, N)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, min(gx, 3) - 1, min(gy, 2) - 1),
        C_slice_M_tiles=3,
        C_slice_N_tiles=4,
        K_chunk_tiles=2,
        subblock_M_tiles=2,
        subblock_N_tiles=4,
    )
    out = _run(device, a, b, config, packer_l1_acc=True)
    _check(out, _golden(a, b))


# ----------------------------------------------------------------------------------------------------
# Compute threads: a C slice's subblocks are assigned round-robin to the NEOs of a cluster
# ----------------------------------------------------------------------------------------------------

# Subblocks are numbered across N then down M. With T threads, round r is subblocks r*T .. r*T + T - 1, one
# per thread, so a C slice of S subblocks takes ceil(S / T) rounds and in the last round the threads past
# S mod T are idle. K_chunk_tiles = K_tiles = 2 keeps one K chunk, so only the two cases that set K_chunk_tiles below
# K_tiles spill partials through C_partials.
SUBBLOCK_SPLITS = [
    # name, C slice (M, N tiles), subblock (M, N tiles), K_tiles, K_chunk_tiles. At four threads: fewer subblocks
    # than threads, a last round with idle threads, full rounds, and a padded C slice.
    ("one_subblock", (2, 2), (2, 2), 2, 2),
    ("two_subblocks", (2, 4), (2, 2), 2, 2),
    ("three_subblocks", (3, 2), (1, 2), 2, 2),
    ("five_subblocks", (5, 2), (1, 2), 2, 2),
    ("six_subblocks", (3, 4), (1, 2), 2, 2),
    ("eight_subblocks", (4, 4), (1, 2), 2, 2),
    ("padded_slice_four_subblocks", (3, 3), (2, 2), 2, 2),
    ("three_subblocks_K_spill", (3, 2), (1, 2), 4, 2),
    ("five_subblocks_K_spill", (5, 2), (1, 2), 4, 1),
]


@pytest.mark.parametrize(
    "name,C_slice,subblock,K_tiles,K_chunk_tiles", SUBBLOCK_SPLITS, ids=[d[0] for d in SUBBLOCK_SPLITS]
)
def test_subblocks_across_threads(device, name, C_slice, subblock, K_tiles, K_chunk_tiles):
    """Four C slices over two cores (two per core, so the per-core walk runs as well), with the auto thread
    count: four on Quasar, one elsewhere."""
    C_slice_M_tiles, C_slice_N_tiles = C_slice
    M, K, N = 2 * C_slice_M_tiles * TILE, K_tiles * TILE, 2 * C_slice_N_tiles * TILE
    torch.manual_seed(5)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, 1, 0),
        C_slice_M_tiles=C_slice_M_tiles,
        C_slice_N_tiles=C_slice_N_tiles,
        K_chunk_tiles=K_chunk_tiles,
        subblock_M_tiles=subblock[0],
        subblock_N_tiles=subblock[1],
    )
    out = _run(device, a, b, config)
    _check(out, _golden(a, b))


@pytest.mark.parametrize("num_compute_threads", [1, 2, 4])
def test_explicit_compute_threads(device, num_compute_threads):
    """M = 3, K = 3, N = 2 tiles on one core: one 3x2-tile C slice of three 1x2 subblocks, one K chunk.
    One thread does all three subblocks in three rounds; with two threads, thread 0 does subblocks 0 and 2 and
    thread 1 does subblock 1 and is idle in the second round; with four threads, threads 0-2 do one subblock
    each in a single round and thread 3 is idle. Ones as input, so every element of C must equal K = 96."""
    _skip_unless_threads_supported(device, num_compute_threads)
    M, K, N = 3 * TILE, 3 * TILE, 2 * TILE
    a, b = torch.ones(1, 1, M, K, dtype=torch.bfloat16), torch.ones(1, 1, K, N, dtype=torch.bfloat16)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, 0, 0),
        C_slice_M_tiles=3,
        C_slice_N_tiles=2,
        subblock_M_tiles=1,
        subblock_N_tiles=2,
        num_compute_threads=num_compute_threads,
    )
    out = _run(device, a, b, config)
    assert torch.equal(out.to(torch.float32), torch.full((1, 1, M, N), float(K)))


@pytest.mark.parametrize("num_compute_threads", [1, 2, 4])
def test_K_spill_over_several_C_slices_per_core(device, num_compute_threads):
    """M = 4, K = 4, N = 2 tiles on one core: two 2x2-tile C slices of two 1x2 subblocks each, two K chunks
    of 2 tiles. With two C slices per core, C_partials does not reuse C_slice's L1, so the last K chunk
    lands in C only if the packer is re-pointed at C_slice (on Quasar it otherwise keeps the C_partials base).
    One thread does both subblocks of a slice, two threads do one each, and with four, threads 2 and 3 are
    idle. Ones as input, so every element of C must equal K = 128."""
    _skip_unless_threads_supported(device, num_compute_threads)
    M, K, N = 4 * TILE, 4 * TILE, 2 * TILE
    a, b = torch.ones(1, 1, M, K, dtype=torch.bfloat16), torch.ones(1, 1, K, N, dtype=torch.bfloat16)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, 0, 0),
        C_slice_M_tiles=2,
        C_slice_N_tiles=2,
        K_chunk_tiles=2,
        subblock_M_tiles=1,
        subblock_N_tiles=2,
        num_compute_threads=num_compute_threads,
    )
    out = _run(device, a, b, config)
    assert torch.equal(out.to(torch.float32), torch.full((1, 1, M, N), float(K)))


def test_compute_threads_rejections(device, expect_error):
    M = K = N = 4 * TILE
    a_t = ttnn.from_torch(_randn(1, 1, M, K), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    b_t = ttnn.from_torch(_randn(1, 1, K, N), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)

    def run(num_compute_threads):
        ttnn.experimental.quasar.matmul(
            a_t,
            b_t,
            program_config=qsr.MatmulUnifiedProgramConfig(
                cores=_rect(0, 0, 0, 0), num_compute_threads=num_compute_threads
            ),
            compute_kernel_config=_hifi4(device),
        )

    with expect_error(RuntimeError, "1, 2 or 4"):
        run(3)
    with expect_error(RuntimeError, "1, 2 or 4"):
        run(2**32 + 1)  # would narrow to 1
    if not _on_quasar(device):
        with expect_error(RuntimeError, "needs Quasar"):
            run(2)


# ----------------------------------------------------------------------------------------------------
# DM threads and buffering: on Quasar up to four reader threads take each C slice's K chunks round-robin, padded
# with credit-only K chunks to a multiple of the reader threads, and two writer threads take the compute threads'
# entries
# ----------------------------------------------------------------------------------------------------

K_CHUNK_SPLITS = [
    # name, K in elements, K_chunk_tiles: 1, 2, 3, 4 and 6 K chunks per C slice (1, 2, 3, 4 and 4 reader threads
    # on Quasar, six padded to 8); a ragged K zeroes A's padding columns in the last real K chunk.
    ("one_K_chunk", 3 * TILE, 3),
    ("two_K_chunks", 4 * TILE, 2),
    ("three_K_chunks_ragged_K", 2 * TILE + 20, 1),
    ("four_K_chunks", 4 * TILE, 1),
    ("six_K_chunks_ragged_K", 5 * TILE + 20, 1),
]


@pytest.mark.parametrize("name,K,K_chunk_tiles", K_CHUNK_SPLITS, ids=[s[0] for s in K_CHUNK_SPLITS])
def test_K_chunks_across_reader_threads(device, name, K, K_chunk_tiles):
    """Batch 2, M = 4, N = 2 tiles on one core: two 2x2-tile C slices per batch, four in all (two in flight). On
    Quasar reader thread t reads K chunks t, t + num_reader_threads, ... of every C slice and the padding K chunks
    carry credits only, so every C slice starts at reader thread 0 again; elsewhere one reader reads them all."""
    B, M, N = 2, 4 * TILE, 2 * TILE
    torch.manual_seed(23)
    a, b = _randn(1, B, M, K), _randn(1, B, K, N)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, 0, 0), C_slice_M_tiles=2, C_slice_N_tiles=2, K_chunk_tiles=K_chunk_tiles
    )
    out = _run(device, a, b, config)
    _check(out, _golden(a, b))


def test_dm_threads_with_K_spill_and_packer_l1_acc(device):
    """Ragged M / K / N, a batch, several C slices per core on two cores, nine K chunks (padded to twelve on Quasar)
    with packer L1 accumulation and a padded subblock: on Quasar all six DM cores (four readers, two writers) and
    two C slices in flight."""
    B, M, K, N = 2, 5 * TILE + 3, 8 * TILE + 7, 6 * TILE + 9
    torch.manual_seed(24)
    a, b = _randn(1, B, M, K), _randn(1, B, K, N)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, 1, 0),
        C_slice_M_tiles=3,
        C_slice_N_tiles=3,
        K_chunk_tiles=1,
        subblock_M_tiles=2,
        subblock_N_tiles=2,
    )
    out = _run(device, a, b, config, packer_l1_acc=True)
    _check(out, _golden(a, b))


MULTICAST_GRIDS = [
    # name, cores (columns, rows), K chunks: one C slice per core, so A is multicast along each row of cores and B
    # down each column (an operand with one core per row / column is read directly).
    ("row_A_multicast", (4, 1), 3),
    ("column_B_multicast", (1, 4), 3),
    ("grid_A_and_B_multicast", (3, 2), 3),
    ("grid_A_and_B_multicast_long_K", (3, 2), 9),
]


@pytest.mark.parametrize("name,grid,K_chunks", MULTICAST_GRIDS, ids=[g[0] for g in MULTICAST_GRIDS])
def test_multicast(device, name, grid, K_chunks):
    """Batch 2, ragged M / K / N, 2-tile K chunks: the first core of each row reads the A slices and multicasts
    them along the row, and the first core of each column does the same with B. On Quasar each reader thread
    multicasts its own K chunks; with 9 of them each thread multicasts several per C slice."""
    gx, gy = _grid(device)
    columns, rows = grid
    if columns > gx or rows > gy:
        pytest.skip(f"needs a {columns}x{rows} grid, device has {gx}x{gy}")
    B, M, K, N = 2, rows * 2 * TILE - 7, (2 * K_chunks - 1) * TILE + 20, columns * 3 * TILE - 11
    torch.manual_seed(26)
    a, b = _randn(1, B, M, K), _randn(1, B, K, N)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, columns - 1, rows - 1), C_slice_M_tiles=2, C_slice_N_tiles=3, K_chunk_tiles=2
    )
    out = _run(device, a, b, config)
    _check(out, _golden(a, b))


MULTICAST_TO_ALL_GRIDS = [
    # name, cores (columns, rows), C slice spans all of M (A to all cores) or all of N (B to all cores)
    ("A_to_all_of_4x2", (4, 2), "M"),
    ("B_to_all_of_2x4", (2, 4), "N"),
    ("A_to_all_of_3x3", (3, 3), "M"),
]


@pytest.mark.parametrize("name,grid,spans", MULTICAST_TO_ALL_GRIDS, ids=[g[0] for g in MULTICAST_TO_ALL_GRIDS])
def test_multicast_to_all_cores(device, name, grid, spans):
    """C is one row of C slices (one column): every core of the rectangle needs the same A (B) slices, which its
    first core multicasts to all of them, over several K chunks and with ragged edges and a batched B."""
    gx, gy = _grid(device)
    columns, rows = grid
    if columns > gx or rows > gy:
        pytest.skip(f"needs a {columns}x{rows} grid, device has {gx}x{gy}")
    num_cores = columns * rows
    B, K = 2, 7 * TILE + 9
    M, N = (3 * TILE - 5, num_cores * 2 * TILE - 13) if spans == "M" else (num_cores * 2 * TILE - 13, 3 * TILE - 5)
    torch.manual_seed(27)
    a, b = _randn(1, B, M, K), _randn(1, B, K, N)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, columns - 1, rows - 1),
        C_slice_M_tiles=3 if spans == "M" else 2,
        C_slice_N_tiles=2 if spans == "M" else 3,
        K_chunk_tiles=2,
    )
    out = _run(device, a, b, config)
    _check(out, _golden(a, b))


# ----------------------------------------------------------------------------------------------------
# Auto placement: MatmulUnifiedProgramConfig() without cores, which is also what Quasar takes without a config
# ----------------------------------------------------------------------------------------------------

AUTO_PLACEMENT_SHAPES = [
    # name, batch, M, K, N (elements), PCC and relative error bounds: each favours a different placement. A 16-bit
    # DST summing 128 K tiles is off by up to ~8% on every path.
    ("decode", 1, 32, 1024, 2048, 0.9998, 0.03),
    ("prefill", 1, 512, 512, 1024, 0.9998, 0.03),
    ("skinny_N", 1, 2048, 256, 64, 0.9998, 0.03),
    ("long_K", 1, 64, 4096, 64, 0.998, 0.1),
    ("ragged", 1, 3 * TILE + 5, 5 * TILE - 3, 7 * TILE + 9, 0.9998, 0.03),
    ("batched_A_folded_into_M", 4, 2 * TILE, 256, 512, 0.9998, 0.03),
]


def _auto_config(device, auto):
    # None takes the unified matmul only on Quasar; an empty MatmulUnifiedProgramConfig takes it everywhere.
    return None if auto == "none" else qsr.MatmulUnifiedProgramConfig()


@pytest.mark.parametrize("auto", ["empty_unified_config", "none"])
@pytest.mark.parametrize("name,batch,M,K,N,pcc,rtol", AUTO_PLACEMENT_SHAPES, ids=[s[0] for s in AUTO_PLACEMENT_SHAPES])
def test_auto_placement(device, name, batch, M, K, N, pcc, rtol, auto):
    torch.manual_seed(31)
    a, b = _randn(1, batch, M, K), _randn(1, 1, K, N)
    out = _run(device, a, b, _auto_config(device, auto))
    _check(out, _golden(a, b), rtol=rtol, pcc=pcc)


@pytest.mark.parametrize(
    "out_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
    ids=["height", "width", "block"],
)
def test_auto_placement_sharded_output(device, out_layout):
    """A sharded output without a shard spec gets the layout asked for, the C slices tiling it one per core."""
    M, K, N = 8 * TILE, 4 * TILE, 8 * TILE
    torch.manual_seed(32)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    out_mem = ttnn.MemoryConfig(out_layout, ttnn.BufferType.L1)
    a_t = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    b_t = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    out = ttnn.experimental.quasar.matmul(
        a_t,
        b_t,
        program_config=qsr.MatmulUnifiedProgramConfig(),
        memory_config=out_mem,
        compute_kernel_config=_hifi4(device),
    )
    assert out.memory_config().memory_layout == out_layout
    _check(ttnn.to_torch(out), _golden(a, b))


@pytest.mark.parametrize("auto", ["empty_unified_config", "none"])
@pytest.mark.parametrize(
    "out_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
    ids=["height", "width", "block"],
)
def test_auto_placement_sharded_output_batched_B(device, out_layout, auto):
    """Batches with their own B and a sharded output without a shard spec: each core produces its C slice in every
    batch, so the C slices cannot be the shards; C is split over the grid and each tile written where its shard is."""
    if auto == "none" and device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip("without a program config only Quasar takes the unified matmul")
    M, K, N = 4 * TILE, 4 * TILE, 8 * TILE
    torch.manual_seed(35)
    a, b = _randn(2, 1, M, K), _randn(2, 1, K, N)
    a_t = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    b_t = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    out = ttnn.experimental.quasar.matmul(
        a_t,
        b_t,
        program_config=_auto_config(device, auto),
        memory_config=ttnn.MemoryConfig(out_layout, ttnn.BufferType.L1),
        compute_kernel_config=_hifi4(device),
    )
    assert out.memory_config().memory_layout == out_layout
    _check(ttnn.to_torch(out), _golden(a, b))


@pytest.mark.parametrize("auto", ["empty_unified_config", "none"])
@pytest.mark.parametrize(
    "batch,M,K,N",
    [(16, 2 * TILE, 4 * TILE, 2 * TILE), (12, 197, 197, 64), (7, 3 * TILE, 2 * TILE, 2 * TILE)],
    ids=["heads", "ragged_attention_V", "odd_batch"],
)
def test_auto_placement_batched_B(device, batch, M, K, N, auto):
    """Batches with their own B and few C slices each: auto placement splits the cores into batch groups."""
    if auto == "none" and device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip("without a program config only Quasar takes the unified matmul")
    torch.manual_seed(36)
    a, b = _randn(1, batch, M, K), _randn(1, batch, K, N)
    _check(_run(device, a, b, _auto_config(device, auto)), _golden(a, b))


def test_auto_placement_keeps_explicit_fields(device):
    # K chunk and subblock stay as given; only the cores, C slice and orientation are chosen.
    M, K, N = 4 * TILE, 6 * TILE, 4 * TILE
    torch.manual_seed(33)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(K_chunk_tiles=2, subblock_M_tiles=1, subblock_N_tiles=2)
    _check(_run(device, a, b, config), _golden(a, b))


@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["fused_bias", "bias_added_separately"])
@pytest.mark.parametrize("auto", ["empty_unified_config", "none"])
def test_auto_placement_linear_bias_and_activation(device, auto, fp32_dest_acc_en):
    if auto == "none" and not _on_quasar(device):
        pytest.skip("without a program config only Quasar takes the unified matmul")
    # One K tile with a 32-bit DST: craq-sim hangs on a 32-bit DST over several K chunks.
    M, K, N = 2 * TILE, (TILE if fp32_dest_acc_en else 4 * TILE), 8 * TILE
    torch.manual_seed(34)
    a, b, bias = _randn(1, 1, M, K), _randn(1, 1, K, N), _bias_row(N)
    a_t = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    b_t = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    bias_t = ttnn.from_torch(bias, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    config = _auto_config(device, auto)
    if config is not None:
        config.fused_activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
    out = ttnn.experimental.quasar.linear(
        a_t,
        b_t,
        bias=bias_t,
        program_config=config,
        activation=None if config is not None else "relu",
        compute_kernel_config=_hifi4(device, fp32_dest_acc_en),
    )
    _check(ttnn.to_torch(out), torch.relu(_golden(a, b) + bias.to(torch.float32)))


def test_auto_placement_on_a_core_grid(device):
    """A core grid without a program config: on Quasar the unified matmul is placed on that grid."""
    if not _on_quasar(device):
        pytest.skip("without a program config only Quasar takes the unified matmul")
    M, K, N = 4 * TILE, 4 * TILE, 4 * TILE
    torch.manual_seed(37)
    a, b, bias = _randn(1, 1, M, K), _randn(1, 1, K, N), _bias_row(N)
    a_t = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    b_t = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    bias_t = ttnn.from_torch(bias, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    out = ttnn.experimental.quasar.linear(
        a_t,
        b_t,
        bias=bias_t,
        core_grid=ttnn.CoreGrid(y=2, x=2),
        activation="relu",
        compute_kernel_config=_hifi4(device),
    )
    _check(ttnn.to_torch(out), torch.relu(_golden(a, b) + bias.to(torch.float32)))


@pytest.mark.parametrize("auto", ["empty_unified_config", "none"])
def test_auto_placement_resnet50_fc(device, auto):
    """ResNet-50's fc as the Quasar model calls it: a width-sharded L1 activation, DRAM weights and bias, a
    width-sharded L1 output and LoFi with packer L1 accumulation."""
    gx, gy = _grid(device)
    K, N, M = 2048, 1024, 32
    num_cores = max(c for c in range(1, gx * gy + 1) if (N // TILE) % c == 0 and (K // TILE) % c == 0)
    columns = max(c for c in range(1, gx + 1) if num_cores % c == 0 and num_cores // c <= gy)
    activation_grid = _rect(0, 0, columns - 1, num_cores // columns - 1)
    torch.manual_seed(35)
    a, b, bias = _randn(1, 1, M, K), _randn(1, 1, K, N), _randn(1, 1, 1, N)
    a_t = ttnn.from_torch(
        a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            _shard(activation_grid, [M, K // num_cores]),
        ),
    )
    b_t = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    bias_t = ttnn.from_torch(bias, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    out = ttnn.experimental.quasar.linear(
        a_t,
        b_t,
        bias=bias_t,
        program_config=_auto_config(device, auto),
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        ),
    )
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    _check(ttnn.to_torch(out), _golden(a, b) + bias.to(torch.float32), rtol=0.05, pcc=0.999)


@pytest.mark.parametrize("auto", ["empty_unified_config", "none"])
def test_auto_placement_sharded_inputs(device, auto):
    """A height-sharded L1 A can be bound in place by the placement that keeps it where it is; a DRAM-sharded B is
    read by the cores next to its banks."""
    if auto == "none" and not _on_quasar(device):
        pytest.skip("without a config Wormhole / Blackhole take the legacy picker, which rejects a DRAM-sharded B")
    gx, gy = _grid(device)
    rows = min(gy, 4)
    dram = device.dram_grid_size()
    num_banks = dram.x * dram.y
    M, K, N = rows * 2 * TILE, 4 * TILE, num_banks * 2 * TILE
    torch.manual_seed(36)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    in0_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard(_rect(0, 0, 0, rows - 1), [2 * TILE, K])
    )
    dram_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))])
    in1_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, _shard(dram_grid, [K, 2 * TILE])
    )
    out = _run(device, a, b, _auto_config(device, auto), in0_mem=in0_mem, in1_mem=in1_mem)
    _check(out, _golden(a, b))


def test_auto_placement_activation_runs_once_with_vector_B(device):
    """A vector B becomes a column only after matmul picks its config, so on Quasar the op picks the unified matmul
    for the column; the activation must still run once."""
    M, K = TILE, 2 * TILE
    torch.manual_seed(37)
    a, b = (torch.randn(1, 1, M, K) * 0.1).to(torch.bfloat16), _randn(K)
    a_t = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    b_t = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    out = ttnn.experimental.quasar.matmul(a_t, b_t, activation="gelu", compute_kernel_config=_hifi4(device))
    golden = torch.nn.functional.gelu(_golden(a, b))
    _check(ttnn.to_torch(out).reshape(golden.shape), golden)


def test_activation_follows_a_bias_added_separately_on_a_core_grid(device):
    """A core grid without a program config and a bias of several rows, which ttnn.linear adds after the op: the
    activation follows the add (Wormhole / Blackhole: the legacy config for the grid; Quasar: the unified matmul)."""
    M, K, N = 2 * TILE, 2 * TILE, 4 * TILE
    torch.manual_seed(38)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    bias = (torch.rand(1, 1, M, N) * 8 - 4).to(torch.bfloat16)
    a_t = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    b_t = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    bias_t = ttnn.from_torch(bias, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    out = ttnn.experimental.quasar.linear(
        a_t,
        b_t,
        bias=bias_t,
        core_grid=ttnn.CoreGrid(y=2, x=2),
        activation="relu",
        compute_kernel_config=_hifi4(device),
    )
    _check(ttnn.to_torch(out), torch.relu(_golden(a, b) + bias.to(torch.float32)), pcc=0.999)


def test_auto_placement_rejects_a_C_slice(device, expect_error):
    a_t = ttnn.from_torch(_randn(1, 1, 64, 64), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    b_t = ttnn.from_torch(_randn(1, 1, 64, 64), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    with expect_error(RuntimeError, "the C slice is chosen with the placement"):
        ttnn.experimental.quasar.matmul(
            a_t,
            b_t,
            program_config=qsr.MatmulUnifiedProgramConfig(C_slice_M_tiles=1),
            compute_kernel_config=_hifi4(device),
        )


# ----------------------------------------------------------------------------------------------------
# Rejections: every constraint fails loudly
# ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_config,out_mem,pattern",
    [
        (
            lambda: qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 63, 63), C_slice_M_tiles=1, C_slice_N_tiles=1),
            None,
            "exceed the device compute grid",
        ),
        (
            lambda: qsr.MatmulUnifiedProgramConfig(
                cores=_rect(0, 0, 0, 0), C_slice_M_tiles=2, C_slice_N_tiles=2, K_chunk_tiles=3
            ),
            None,
            "must divide K_tiles",
        ),
        (
            lambda: qsr.MatmulUnifiedProgramConfig(
                cores=_rect(0, 0, 0, 0), C_slice_M_tiles=2, C_slice_N_tiles=2, subblock_M_tiles=2
            ),
            None,
            "both be set",
        ),
        (
            lambda: qsr.MatmulUnifiedProgramConfig(
                cores=_rect(0, 0, 0, 0), C_slice_M_tiles=4, C_slice_N_tiles=4, subblock_M_tiles=4, subblock_N_tiles=4
            ),
            None,
            "DST fits",
        ),
        (
            lambda: qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 0, 0), C_slice_M_tiles=40, C_slice_N_tiles=40),
            None,
            "do not fit for any C slice",
        ),
        (
            lambda: qsr.MatmulUnifiedProgramConfig(
                cores=_rect(0, 0, 0, 0), C_slice_M_tiles=40, C_slice_N_tiles=40, K_chunk_tiles=4
            ),
            None,
            "do not fit",
        ),
    ],
    ids=[
        "cores_off_grid",
        "K_chunk_not_divisor",
        "half_auto_subblock",
        "subblock_too_big",
        "slice_does_not_fit_l1",
        "explicit_K_chunk_does_not_fit",
    ],
)
def test_rejections(device, expect_error, make_config, out_mem, pattern):
    M = K = N = 4 * TILE
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    a_t = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    b_t = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    with expect_error(RuntimeError, pattern):
        ttnn.experimental.quasar.matmul(
            a_t,
            b_t,
            program_config=make_config(),
            memory_config=out_mem or ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=_hifi4(device),
        )


@pytest.mark.parametrize(
    "batch,num_batch_groups,pattern",
    [(1, 2, "at most the batches"), (4, 3, "exceeds the cores")],
    ids=["more_batch_groups_than_batches", "more_batch_groups_than_cores"],
)
def test_batch_group_rejections(device, expect_error, batch, num_batch_groups, pattern):
    M = K = N = 2 * TILE
    a, b = _randn(1, batch, M, K), _randn(1, batch, K, N)
    a_t = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    b_t = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    with expect_error(RuntimeError, pattern):
        ttnn.experimental.quasar.matmul(
            a_t,
            b_t,
            program_config=qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 1, 0), num_batch_groups=num_batch_groups),
            compute_kernel_config=_hifi4(device),
        )


@pytest.mark.parametrize("sharded_out", [True, False], ids=["block_sharded_out", "interleaved_out"])
def test_auto_config_batched_a_unbatched_b(device, sharded_out):
    """Default program config, batched A x unbatched B (batch must be fused for a block-sharded output)."""
    torch.manual_seed(8)
    a, b = _randn(2, 1, 1024, 1024), _randn(1, 1, 1024, 1024)
    a_t = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    b_t = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    out_mem = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG if sharded_out else ttnn.DRAM_MEMORY_CONFIG
    out = ttnn.experimental.quasar.matmul(a_t, b_t, memory_config=out_mem, compute_kernel_config=_hifi4(device))
    if sharded_out:
        # Guard against a silent fallback to an interleaved output.
        assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
        assert out.memory_config().shard_spec is not None

    # The issue's K=1024 shape reaches PCC ~0.99984 and ~4% max error (bf16 accumulation), looser than _check defaults.
    _check(ttnn.to_torch(out), _golden(a, b), rtol=0.05, pcc=0.999)


@pytest.mark.parametrize(
    "a_shape, b_shape, expected_shard_shape, fp32_dest_acc_en",
    [
        ((1, 400, 32), (1, 32, 400), (416, 416), False),
        ((1, 400, 32), (1, 32, 128), (416, 128), False),
        ((1, 512, 32), (1, 32, 512), (512, 512), False),
        # Fused M = 4 * 13 = 52 tiles keeps the per-batch cap (4 block rows) and fits every Quasar grid (<= 4 rows).
        ((4, 400, 32), (32, 400), (416, 416), False),
        # Fused M = 5 * 13 = 65 tiles needs five block rows to get the 416-row shard (skipped on 4-row grids).
        pytest.param(
            (5, 400, 32),
            (32, 400),
            (416, 416),
            False,
            marks=pytest.mark.requires_grid_size((1, 5)),
        ),
        ((1, 320, 32), (1, 32, 64), (320, 64), False),
        ((1, 128, 32), (1, 32, 128), (128, 128), False),
        ((16, 32, 32), (32, 512), (512, 512), False),
        # fp32 accumulation exercises the fp32 branch of the subblock search (h * w <= 4). It cannot run until the
        # Quasar 2D factory sets unpack_modes for the fp32 partials DFB (#61018); strict so it flags when fixed.
        pytest.param(
            (1, 128, 32),
            (1, 32, 128),
            (128, 128),
            True,
            marks=pytest.mark.xfail(
                reason="#61018: fp32_dest_acc_en fails in the 2D factory (partials DFB has no unpack_modes entry)",
                raises=RuntimeError,
                strict=True,
            ),
        ),
    ],
    ids=[
        "400x400",
        "400x128",
        "512x512",
        "batch4_broadcast_b",
        "batch5_broadcast_b",
        "320x64_10x2tiles",
        "128x128_4x4tiles",
        "batch16_small_m_exceeds_grid_rows",
        "128x128_4x4tiles_fp32",
    ],
)
def test_auto_config_block_sharded_caps_per_core_to_output(
    device, a_shape, b_shape, expected_shard_shape, fp32_dest_acc_en
):
    """BLOCK_SHARDED output, no program_config: the shard shape must follow the output size, not the L1 ladder.

    The auto-config used to start per_core_M/N at 16 tiles and never cap them to the output, so a 13x13 tile output
    came back with a 512x512 shard.
    """
    torch.manual_seed(0)
    a, b = _randn(*a_shape), _randn(*b_shape)
    a_t = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    b_t = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    out_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            expected_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    out = ttnn.experimental.quasar.matmul(
        a_t, b_t, memory_config=out_mem, compute_kernel_config=_hifi4(device, fp32_dest_acc_en=fp32_dest_acc_en)
    )

    mem = out.memory_config()
    assert mem.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
    assert tuple(mem.shard_spec.shape) == expected_shard_shape
    _check(ttnn.to_torch(out), _golden(a, b), rtol=0.05, pcc=0.999)


def test_auto_config_block_sharded_keeps_l1_sharded_b_width(device):
    """An L1 WIDTH_SHARDED B fixes per_core_N (validator: per_core_N == B shard width in tiles), so the auto-config
    must not cap it to the output's tile count (13 for N=400) when B's shard is padded to 16 tiles."""
    torch.manual_seed(0)
    a, b = _randn(1, 512, 32), _randn(1, 32, 400)
    a_t = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    b_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            (32, 512),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    b_t = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=b_mem)
    out_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            (512, 512),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    out = ttnn.experimental.quasar.matmul(a_t, b_t, memory_config=out_mem, compute_kernel_config=_hifi4(device))
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
    # per_core_N stays at B's 16-tile shard width (not capped to Nt = 13).
    assert tuple(out.memory_config().shard_spec.shape) == (512, 512)
    _check(ttnn.to_torch(out), _golden(a, b), rtol=0.05, pcc=0.999)


def test_auto_config_block_sharded_n_cap_keeps_output_tile_alignment(device):
    """With B on a narrow (32, 16) tile and a wider (16, 32) output tile, per_core_N must stay a multiple of the
    output/in1 tile width ratio (2). B=208 columns is 13 narrow tiles (odd), so capping to Nt=13 would break it."""
    torch.manual_seed(0)
    a, b = _randn(1, 128, 32), _randn(1, 32, 208)
    a_t = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, tile=ttnn.Tile((16, 32)))
    b_t = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, tile=ttnn.Tile((32, 16)))
    out_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            (128, 224),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    out = ttnn.experimental.quasar.matmul(
        a_t, b_t, memory_config=out_mem, output_tile=ttnn.Tile((16, 32)), compute_kernel_config=_hifi4(device)
    )
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
    # per_core_N is rounded up to 14 (7 wide tiles * 2), not capped to the odd Nt = 13.
    assert tuple(out.memory_config().shard_spec.shape) == (128, 224)
    _check(ttnn.to_torch(out), _golden(a, b), rtol=0.05, pcc=0.999)
