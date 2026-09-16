# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""MatmulUnifiedProgramConfig: one placement-first matmul factory (Quasar-native matmul, stage A).

Every test drives the same kernels through a different (cores, per_core_M, per_core_N) placement and
memory layout. Correctness is checked with allclose against an fp32 golden of the bf16-rounded inputs
(the device runs HiFi4 here), plus exact checks with structured inputs (identity / ones), which catch
indexing and edge-clipping errors that a statistical check would not.

Run on Wormhole / Blackhole silicon (one NEO per cluster, same as Quasar in stage A):
    pytest tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_matmul_unified.py

Run on the Quasar simulator (one config per process; a sim-side hang ignores pytest --timeout):
    TT_METAL_SIMULATOR=<path>/libttsim.so TT_SIMULATOR_LOCALHOST=1 ARCH_NAME=quasar CHIP_ARCH=quasar \\
        TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \\
        pytest tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_matmul_unified.py -k <case>
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

qsr = ttnn._ttnn.operations.experimental.quasar

TILE = 32


def _grid(device):
    g = device.compute_with_storage_grid_size()
    return g.x, g.y


def _rect(x0, y0, x1, y1):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


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


def _check(out, golden, dtype_out=ttnn.bfloat16, rtol=0.02):
    out_f = out.to(torch.float32)
    scale = golden.abs().max().item()
    atol = rtol * scale
    assert_with_pcc(golden, out_f, 0.9999)
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

PLACEMENTS = [
    # name,            cores (x0,y0,x1,y1) or list, per_core_M, per_core_N, row_major
    ("single_core", (0, 0, 0, 0), 16, 16, True),  # 1 block
    ("row_1d_mcast_in0_shape", (0, 0, 7, 0), 16, 2, True),  # 8 blocks, one per core, all share in0 rows
    ("col_1d_transposed", (0, 0, 0, 7), 2, 16, True),  # 8 blocks along M on a column of cores
    ("grid_2d", (0, 0, 3, 3), 4, 4, True),  # 16 blocks on a 4x4 rectangle
    ("grid_2d_col_major", (0, 0, 3, 3), 4, 4, False),  # same blocks, y-fastest core order
    ("more_blocks_than_cores", (0, 0, 1, 1), 4, 4, True),  # 16 blocks over 4 cores, 4 each
    ("fewer_blocks_than_cores", (0, 0, 7, 7), 8, 8, True),  # 4 blocks, 60 cores idle
    ("uneven_blocks_per_core", (0, 0, 2, 0), 4, 4, True),  # 16 blocks over 3 cores: 6, 5, 5
    ("non_rect_cores", [(0, 0, 3, 0), (0, 2, 1, 2)], 4, 4, True),  # two ranges, 6 cores, 16 blocks
]


@pytest.mark.parametrize("name,cores,per_core_M,per_core_N,row_major", PLACEMENTS, ids=[p[0] for p in PLACEMENTS])
def test_placements(device, name, cores, per_core_M, per_core_N, row_major):
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
        cores=crs, per_core_M=per_core_M, per_core_N=per_core_N, row_major_cores=row_major
    )
    out = _run(device, a, b, config)
    _check(out, _golden(a, b))


def test_repr_and_fields():
    cfg = qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 1, 1), per_core_M=2, per_core_N=3, in0_block_w=4)
    assert cfg.per_core_M == 2 and cfg.per_core_N == 3 and cfg.in0_block_w == 4
    assert cfg.out_subblock_h == 0 and cfg.out_subblock_w == 0 and cfg.row_major_cores is True
    assert "MatmulUnifiedProgramConfig(" in repr(cfg)


# ----------------------------------------------------------------------------------------------------
# Edges: M, N not multiples of the block, K not a multiple of the tile, explicit blocking knobs
# ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "M,K,N,per_core_M,per_core_N,in0_block_w,subblock",
    [
        (7 * TILE, 3 * TILE, 11 * TILE, 3, 4, 0, (0, 0)),  # 3x3 blocks, ragged right and bottom edge
        (5 * TILE, 8 * TILE, 5 * TILE, 4, 4, 2, (2, 2)),  # 2x2 blocks with a 1-tile edge strip each way
        (2 * TILE, 100, 3 * TILE, 2, 3, 0, (1, 3)),  # K=100: last K tile is 4 columns wide, must be zeroed
        (4 * TILE, 6 * TILE, 6 * TILE, 4, 6, 1, (4, 1)),  # in0_block_w=1: six K blocks, spill/reload every step
        (3 * TILE, 4 * TILE, 3 * TILE, 3, 3, 4, (3, 1)),  # single K block: no partials at all
    ],
)
def test_edges_and_blocking(device, M, K, N, per_core_M, per_core_N, in0_block_w, subblock):
    gx, gy = _grid(device)
    torch.manual_seed(1)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, min(gx, 4) - 1, min(gy, 2) - 1),
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        in0_block_w=in0_block_w,
        out_subblock_h=subblock[0],
        out_subblock_w=subblock[1],
    )
    out = _run(device, a, b, config)
    _check(out, _golden(a, b))


def test_identity_is_exact_on_ragged_edges(device):
    """in1 = I: the output must be in0 bit for bit, on every tile of every edge block."""
    gx, gy = _grid(device)
    M, K = 7 * TILE, 5 * TILE
    torch.manual_seed(2)
    a = _randn(1, 1, M, K)
    b = torch.eye(K, dtype=torch.bfloat16).reshape(1, 1, K, K)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, min(gx, 3) - 1, min(gy, 3) - 1), per_core_M=3, per_core_N=2
    )
    out = _run(device, a, b, config)
    assert torch.equal(out.to(torch.bfloat16), a), "identity matmul differs from in0"


def test_ones_give_constant_k(device):
    """all-ones inputs: every output element is exactly K; a skipped or doubled K block shows up here."""
    gx, gy = _grid(device)
    M, K, N = 6 * TILE, 9 * TILE, 5 * TILE
    a = torch.ones(1, 1, M, K, dtype=torch.bfloat16)
    b = torch.ones(1, 1, K, N, dtype=torch.bfloat16)
    config = qsr.MatmulUnifiedProgramConfig(
        cores=_rect(0, 0, min(gx, 2) - 1, 0), per_core_M=4, per_core_N=2, in0_block_w=3
    )
    out = _run(device, a, b, config)
    assert torch.equal(out.to(torch.float32), torch.full((1, 1, M, N), float(K)))


# ----------------------------------------------------------------------------------------------------
# Batch: broadcast in1 and batched in1
# ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bcast", [True, False], ids=["bcast_in1", "batched_in1"])
def test_batch(device, bcast):
    gx, gy = _grid(device)
    B, M, K, N = 3, 4 * TILE, 4 * TILE, 6 * TILE
    torch.manual_seed(3)
    a = _randn(1, B, M, K)
    b = _randn(1, 1, K, N) if bcast else _randn(1, B, K, N)
    config = qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, min(gx, 3) - 1, 0), per_core_M=2, per_core_N=3)
    out = _run(device, a, b, config)
    _check(out, _golden(a, b))


# ----------------------------------------------------------------------------------------------------
# Memory layouts: the same kernels read and write interleaved, L1-sharded and DRAM-sharded tensors
# ----------------------------------------------------------------------------------------------------


def _shard(grid, shape, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    return ttnn.ShardSpec(grid, shape, orientation)


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
    config = qsr.MatmulUnifiedProgramConfig(cores=cores, per_core_M=2, per_core_N=3)
    out = _run(device, a, b, config, in0_mem=in0_mem, out_mem=out_mem)
    _check(out, _golden(a, b))


def test_block_sharded_in0_and_out(device):
    """2D: in0 block-sharded on a 2x2 rectangle, output block-sharded on the same rectangle."""
    gx, gy = _grid(device)
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
    config = qsr.MatmulUnifiedProgramConfig(cores=cores, per_core_M=2, per_core_N=3)
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
    config = qsr.MatmulUnifiedProgramConfig(cores=cores, per_core_M=2, per_core_N=2)
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
    config = qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, min(gx, 8) - 1, 0), per_core_M=1, per_core_N=2)
    out = _run(device, a, b, config, in1_mem=in1_mem)
    _check(out, _golden(a, b))


def test_l1_interleaved_everything(device):
    gx, gy = _grid(device)
    M, K, N = 4 * TILE, 4 * TILE, 4 * TILE
    torch.manual_seed(8)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 1, 1), per_core_M=2, per_core_N=2)
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
# Compute options: bfp8 inputs, fp32 accumulation, packer L1 accumulation, fp32 output
# ----------------------------------------------------------------------------------------------------


def test_bfp8_inputs(device):
    gx, gy = _grid(device)
    M, K, N = 4 * TILE, 8 * TILE, 4 * TILE
    torch.manual_seed(9)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 1, 1), per_core_M=2, per_core_N=2)
    out = _run(device, a, b, config, in0_dtype=ttnn.bfloat8_b, in1_dtype=ttnn.bfloat8_b)
    # bfp8 quantizes the operands; compare against the same quantization.
    a_q = ttnn.to_torch(ttnn.from_torch(a, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT))
    b_q = ttnn.to_torch(ttnn.from_torch(b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT))
    _check(out, _golden(a_q, b_q), rtol=0.03)


def test_fp32_dest_acc_and_fp32_out(device):
    """fp32 accumulation caps the subblock at 4 tiles (auto picks one); fp32 output means interm != out format."""
    gx, gy = _grid(device)
    M, K, N = 4 * TILE, 8 * TILE, 4 * TILE
    torch.manual_seed(10)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 1, 0), per_core_M=4, per_core_N=2, in0_block_w=2)
    out = _run(device, a, b, config, out_dtype=ttnn.float32, fp32_dest_acc_en=True)
    _check(out, _golden(a, b), rtol=0.005)


def test_packer_l1_acc(device):
    """packer_l1_acc engages when there are more than 2 K blocks (in0_block_w=1 on Kt=6 gives 6)."""
    gx, gy = _grid(device)
    M, K, N = 4 * TILE, 6 * TILE, 4 * TILE
    torch.manual_seed(11)
    a, b = _randn(1, 1, M, K), _randn(1, 1, K, N)
    config = qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 1, 1), per_core_M=2, per_core_N=2, in0_block_w=1)
    out = _run(device, a, b, config, packer_l1_acc=True)
    _check(out, _golden(a, b))


def test_bias_is_applied_as_separate_add(device):
    gx, gy = _grid(device)
    M, K, N = 4 * TILE, 4 * TILE, 4 * TILE
    torch.manual_seed(12)
    a, b, bias = _randn(1, 1, M, K), _randn(1, 1, K, N), _randn(1, 1, 1, N)
    a_t = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    b_t = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    bias_t = ttnn.from_torch(bias, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    config = qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 1, 1), per_core_M=2, per_core_N=2)
    out = ttnn.experimental.quasar.linear(
        a_t, b_t, bias=bias_t, program_config=config, compute_kernel_config=_hifi4(device)
    )
    _check(ttnn.to_torch(out), _golden(a, b) + bias.to(torch.float32))


# ----------------------------------------------------------------------------------------------------
# Rejections: every constraint fails loudly
# ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_config,out_mem,pattern",
    [
        (
            lambda: qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 63, 63), per_core_M=1, per_core_N=1),
            None,
            "exceed the device compute grid",
        ),
        (
            lambda: qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 0, 0), per_core_M=2, per_core_N=2, in0_block_w=3),
            None,
            "must divide Kt",
        ),
        (
            lambda: qsr.MatmulUnifiedProgramConfig(
                cores=_rect(0, 0, 0, 0), per_core_M=2, per_core_N=2, out_subblock_h=2, out_subblock_w=0
            ),
            None,
            "both be set",
        ),
        (
            lambda: qsr.MatmulUnifiedProgramConfig(
                cores=_rect(0, 0, 0, 0), per_core_M=4, per_core_N=4, out_subblock_h=4, out_subblock_w=4
            ),
            None,
            "DST fits",
        ),
        (
            lambda: qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 0, 0), per_core_M=0, per_core_N=2),
            None,
            "must be > 0",
        ),
        (
            lambda: qsr.MatmulUnifiedProgramConfig(cores=_rect(0, 0, 0, 0), per_core_M=2, per_core_N=2),
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                _shard(_rect(0, 0, 0, 0), [2 * TILE, 4 * TILE]),
            ),
            "exactly one output block per core",
        ),
    ],
    ids=[
        "cores_off_grid",
        "in0_block_w_not_divisor",
        "half_auto_subblock",
        "subblock_too_big",
        "zero_block",
        "sharded_multi_block",
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
