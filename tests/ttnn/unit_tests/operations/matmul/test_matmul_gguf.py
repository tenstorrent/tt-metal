# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.matmul / ttnn.linear with a GGUF weight (ttnn.from_gguf) on Blackhole, against gguf-py's dequantization of the
same blocks times the same bf16 input in float64: PCC >= 0.9999 and the largest error within 1 % of the largest
output."""

import statistics
import time

import numpy as np
import pytest
import torch

import ttnn

from tests.ttnn.unit_tests.tensor.gguf_test_utils import SOURCE, gguf
from tests.ttnn.unit_tests.tensor.gguf_test_utils import blocks as _blocks, k_of as _k, name as _name, qtype as _qtype

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="GGUF weights need Blackhole")

PCC = 0.9999
MAX_ERROR = 0.01  # of the largest output


def _check(out, x, blocks, dtype, k, n):
    w = torch.from_numpy(gguf.quants.dequantize(blocks, _qtype(dtype)).reshape(n, k).astype(np.float64))
    want = x.to(torch.float64) @ w.T
    got = out.to(torch.float64)
    assert torch.isfinite(want).all(), "the reference must be finite"
    pcc = np.corrcoef(got.flatten().numpy(), want.flatten().numpy())[0, 1]
    error = (got - want).abs().max().item() / want.abs().max().item()
    assert pcc >= PCC and error <= MAX_ERROR, f"pcc {pcc:.6f}, max error {error:.4f} of the largest output"


def _run(device, dtype, shape, k, n, seed=0, linear=False):
    blocks = _blocks(dtype, k, n, seed)
    torch.manual_seed(seed)
    x = torch.randn(*shape, k).bfloat16()
    w = ttnn.from_gguf(blocks, (k, n), dtype, device=device)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.linear(x_tt, w) if linear else ttnn.matmul(x_tt, w)
    assert out.dtype == ttnn.bfloat16
    assert tuple(out.shape) == (*shape, n)
    _check(ttnn.to_torch(out), x, blocks, dtype, k, n)


@pytest.mark.parametrize("m", [1, 32])
@pytest.mark.parametrize("dtype", ttnn.GGUF_DTYPES, ids=_name)
def test_decode(device, dtype, m):
    # 8 N tiles, 2 K groups: one item a core, so every N tile is cut between two cores (partials).
    _run(device, dtype, (1, 1, m), _k(dtype), 256)


@pytest.mark.parametrize("dtype", [ttnn.gguf_q4_0, ttnn.gguf_q4_k, ttnn.gguf_q6_k, ttnn.gguf_iq2_xxs], ids=_name)
def test_decode_n_off_banks(device, dtype):
    # 3 N tiles, not a multiple of the DRAM banks: the row-major placement.
    _run(device, dtype, (1, 1, 7), _k(dtype, 4), 96)


@pytest.mark.parametrize("dtype", [ttnn.gguf_q4_0, ttnn.gguf_q4_k, ttnn.gguf_q6_k], ids=_name)
@pytest.mark.parametrize("k, n", [(4096, 4096), (4096, 12288)])
def test_decode_large(device, dtype, k, n):
    _run(device, dtype, (1, 1, 1), k, n)


# K = 25600: the input of all K tiles does not fit L1, so bands of grid rows take shares of K. Every format with
# --runslow; a few by default.
@pytest.mark.parametrize("m", [1, 32])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            d,
            id=_name(d),
            marks=() if d in (ttnn.gguf_q4_0, ttnn.gguf_q4_k, ttnn.gguf_iq2_xxs, ttnn.gguf_tq2_0) else pytest.mark.slow,
        )
        for d in ttnn.GGUF_DTYPES
    ],
)
def test_decode_k_bands(device, dtype, m):
    _run(device, dtype, (1, 1, m), 25600, 2048)


@pytest.mark.parametrize("m", [1, 3])
def test_decode_tq2_0_long_k(device, m):
    # 40 K groups: with one input row the scales apply per 16 K groups, and the cores' segments cross those windows.
    _run(device, ttnn.gguf_tq2_0, (1, 1, m), 10240, 2048)


# The formats of the prefill program (gguf_matmul_prefill_program_factory.cpp); the others take the input in 32-row
# blocks.
PREFILL_DTYPES = [
    ttnn.gguf_q8_0,
    ttnn.gguf_q4_0,
    ttnn.gguf_q4_1,
    ttnn.gguf_q5_0,
    ttnn.gguf_q5_1,
    ttnn.gguf_q4_k,
    ttnn.gguf_q6_k,
]


@pytest.mark.parametrize("m", [128, 512])
@pytest.mark.parametrize("dtype", ttnn.GGUF_DTYPES, ids=_name)
def test_prefill(device, dtype, m):
    _run(device, dtype, (1, m), _k(dtype), 96)


@pytest.mark.parametrize("dtype", list(ttnn.GGUF_EXPANDED.values()), ids=_name)
def test_expanded_round_trip(device, dtype):
    k, n = _k(dtype), 96
    blocks = _blocks(dtype, k, n)
    w = ttnn.from_gguf(blocks, (k, n), dtype)
    assert w.dtype == dtype
    np.testing.assert_array_equal(ttnn.to_gguf(w), blocks.reshape(-1))
    np.testing.assert_array_equal(ttnn.to_gguf(ttnn.to_device(w, device)), blocks.reshape(-1))


@pytest.mark.parametrize("m", [1, 32, 128])
@pytest.mark.parametrize("dtype", list(ttnn.GGUF_EXPANDED.values()), ids=_name)
def test_expanded_matches_compact(device, dtype, m):
    # The same weights: the outputs of both layouts must be the same bits. Expanded TQ1_0 runs TQ2_0's block-float
    # matmul, so its reference is TQ2_0 blocks of the same trits.
    source, k, n = SOURCE[dtype], _k(dtype), 256
    blocks = _blocks(source, k, n)
    reference, reference_blocks = source, blocks
    if source == ttnn.gguf_tq1_0:
        reference = ttnn.gguf_tq2_0
        values = gguf.quants.dequantize(blocks, gguf.GGMLQuantizationType.TQ1_0)
        reference_blocks = gguf.quants.quantize(values.reshape(n, k), gguf.GGMLQuantizationType.TQ2_0)
        np.testing.assert_array_equal(gguf.quants.dequantize(reference_blocks, gguf.GGMLQuantizationType.TQ2_0), values)
    torch.manual_seed(0)
    x = torch.randn(1, m, k).bfloat16()
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    compact, expanded = (
        ttnn.to_torch(ttnn.linear(x_tt, ttnn.from_gguf(b, (k, n), d, device=device)))
        for b, d in ((reference_blocks, reference), (blocks, dtype))
    )
    assert torch.equal(compact, expanded)
    _check(expanded, x, blocks, source, k, n)


def test_expanded_refusals(expect_error):
    blocks = _blocks(ttnn.gguf_tq1_0, 256, 32)
    blocks[0, 0] = 1  # TQ1_0 quantization never writes 1
    with expect_error(RuntimeError, "TQ1_0 quantization does not write"):
        ttnn.from_gguf(blocks, (256, 32), ttnn.gguf_tq1_0_expanded)
    with expect_error(RuntimeError, "bytes of blocks"):
        ttnn.from_gguf(_blocks(ttnn.gguf_iq2_xs, 256, 32), (256, 32), ttnn.gguf_iq2_xxs_expanded)


# The prefill program's blockings: a K split (few rows), a padded odd number of row blocks with N tiles that leave
# cores idle, M in chunks, K = 12288. Every format with --runslow; q4_0 by default.
@pytest.mark.parametrize(
    "dtype",
    [pytest.param(d, id=_name(d), marks=() if d == ttnn.gguf_q4_0 else pytest.mark.slow) for d in PREFILL_DTYPES],
)
@pytest.mark.parametrize("m, k, n", [(128, 4096, 4096), (200, 4096, 1056), (2048, 4096, 4096), (640, 12288, 4096)])
def test_prefill_blockings(device, dtype, m, k, n):
    _run(device, dtype, (1, m), k, n)


# 4748 N tiles: no blocking takes a core's N tiles at once, so the cores take them in passes. The prefill program must
# take it (one program, not the 32-row blocks).
@pytest.mark.parametrize("dtype", [ttnn.gguf_q4_k, pytest.param(ttnn.gguf_q6_k, marks=pytest.mark.slow)], ids=_name)
@pytest.mark.parametrize("m, k", [(512, 256), pytest.param(128, 2048, marks=pytest.mark.slow)])
def test_prefill_n_passes(device, dtype, m, k):
    n = 151936
    blocks = _blocks(dtype, k, n)
    torch.manual_seed(0)
    x = torch.randn(1, m, k).bfloat16()
    w = ttnn.from_gguf(blocks, (k, n), dtype, device=device)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    entries = device.num_program_cache_entries()
    out = ttnn.matmul(x_tt, w)
    assert device.num_program_cache_entries() == entries + 1
    _check(ttnn.to_torch(out), x, blocks, dtype, k, n)


def _check_tiles(out, x, blocks, dtype, k, n):
    """No output tile with an error above MAX_ERROR of the largest output (a few wrong tiles pass the PCC check)."""
    w = torch.from_numpy(gguf.quants.dequantize(blocks, _qtype(dtype)).reshape(n, k).astype(np.float64))
    want = (x.reshape(-1, k).to(torch.float64) @ w.T).reshape(-1, n)
    err = (out.to(torch.float64).reshape(-1, n) - want).abs()
    m = want.shape[0]
    err = torch.nn.functional.pad(err, (0, 0, 0, -m % 32)).reshape(-(-m // 32), 32, n // 32, 32).amax(dim=(1, 3))
    bad = (err > MAX_ERROR * want.abs().max()).nonzero().tolist()
    assert not bad, f"{len(bad)} of {err.numel()} output tiles wrong, first [M tile, N tile]: {bad[:8]}"


# A K split with one row per unit (ur = 1) and an odd number of N tiles on band 0's cores: their units of sums end on
# a single tile, so the pairs that follow start on odd pages of c_16 (24 pages). Controls: the same N with an even
# number of row tiles. M = 160, K = 544: the smallest shapes of this plan; the others with --runslow.
@pytest.mark.parametrize(
    "dtype",
    [pytest.param(d, id=_name(d), marks=() if d == ttnn.gguf_q4_0 else pytest.mark.slow) for d in PREFILL_DTYPES],
)
@pytest.mark.parametrize(
    "m, k, n",
    [
        (160, 544, 11200),
        (96, 544, 21600),
        (128, 544, 11200),
        pytest.param(160, 4096, 11008, marks=pytest.mark.slow),
        pytest.param(200, 4096, 12288, marks=pytest.mark.slow),
        pytest.param(352, 4096, 8800, marks=pytest.mark.slow),
        pytest.param(192, 4096, 12288, marks=pytest.mark.slow),
        pytest.param(256, 4096, 12288, marks=pytest.mark.slow),
    ],
)
def test_prefill_k_split_odd_units(device, dtype, m, k, n):
    if k % _k(dtype, 1):
        pytest.skip(f"K = {k} is not whole pages of {_name(dtype)}")
    blocks = _blocks(dtype, k, n)
    torch.manual_seed(0)
    x = torch.randn(1, m, k).bfloat16()
    w = ttnn.from_gguf(blocks, (k, n), dtype, device=device)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    _check_tiles(ttnn.to_torch(ttnn.matmul(x_tt, w)), x, blocks, dtype, k, n)


def _l1_small_canary(device, tiles):
    """A tensor of ones in L1_SMALL on every core of the grid, `tiles` tiles each, and its contents."""
    grid = device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    spec = ttnn.ShardSpec(cores, [32, 32 * tiles], ttnn.ShardOrientation.ROW_MAJOR)
    config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1_SMALL, spec)
    ones = torch.ones(1, 1, 32, 32 * tiles * grid.x * grid.y).bfloat16()
    return (
        ttnn.from_torch(ones, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config),
        ones,
    )


# The circular buffers must stay below L1_SMALL: a tensor there keeps its contents. Decode with K just below a step in
# the number of K bands (64 KiB of L1_SMALL), prefill with 256 KiB.
@pytest.mark.parametrize(
    "device_params, m, k, n",
    [
        ({"l1_small_size": 65536}, 1, 22304, 3520),
        ({"l1_small_size": 262144}, 512, 2048, 11008),
        pytest.param({"l1_small_size": 262144}, 2048, 4096, 11008, marks=pytest.mark.slow),
    ],
    indirect=["device_params"],
)
def test_l1_small_kept(device, device_params, m, k, n):
    dtype = ttnn.gguf_q4_0
    canary, ones = _l1_small_canary(device, (device_params["l1_small_size"] - 4096) // 2048)
    blocks = _blocks(dtype, k, n)
    torch.manual_seed(0)
    x = torch.randn(1, m, k).bfloat16()
    w = ttnn.from_gguf(blocks, (k, n), dtype, device=device)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(ttnn.matmul(x_tt, w))
    changed = int((ttnn.to_torch(canary) != ones).sum())
    assert changed == 0, f"{changed} of {ones.numel()} L1_SMALL elements overwritten"
    _check(out, x, blocks, dtype, k, n)


# A live L1 buffer lowers the L1 left to programs: the op plans again (a new program) instead of reusing a plan whose
# circular buffers would reach the buffer.
def test_l1_buffer_replans(device):
    dtype, k, n = ttnn.gguf_q4_0, 22304, 3520
    blocks = _blocks(dtype, k, n)
    torch.manual_seed(0)
    x = torch.randn(1, 1, k).bfloat16()
    w = ttnn.from_gguf(blocks, (k, n), dtype, device=device)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    _check(ttnn.to_torch(ttnn.matmul(x_tt, w)), x, blocks, dtype, k, n)
    entries = device.num_program_cache_entries()
    grid = device.compute_with_storage_grid_size()
    zeros = torch.zeros(1, 1, 32, 32 * 64 * grid.x * grid.y).bfloat16()
    held = ttnn.from_torch(
        zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    _check(ttnn.to_torch(ttnn.matmul(x_tt, w)), x, blocks, dtype, k, n)
    assert device.num_program_cache_entries() == entries + 1
    assert torch.equal(ttnn.to_torch(held), zeros), "the L1 buffer was overwritten"
    ttnn.deallocate(held)


def test_leading_dims_and_bias(device):
    dtype, k, n = ttnn.gguf_q4_k, 512, 128
    blocks = _blocks(dtype, k, n)
    x = torch.randn(2, 3, 20, k).bfloat16()
    bias = torch.randn(1, n).bfloat16()
    w = ttnn.from_gguf(blocks, (k, n), dtype, device=device)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.linear(x_tt, w, bias=b_tt)
    assert tuple(out.shape) == (2, 3, 20, n)
    _check(ttnn.to_torch(out) - bias.float(), x, blocks, dtype, k, n)


def test_program_cache(device):
    dtype, k, n = ttnn.gguf_q6_k, 512, 256
    _run(device, dtype, (1, 1, 1), k, n, seed=1)
    entries = device.num_program_cache_entries()
    _run(device, dtype, (1, 1, 1), k, n, seed=2)
    assert device.num_program_cache_entries() == entries


def test_refusals(device, expect_error):
    dtype = ttnn.gguf_q4_k
    blocks = _blocks(dtype, 512, 64)
    with expect_error(RuntimeError, "needs K %"):  # K = 96: whole MXFP4 blocks, not whole pages of 64 rows
        ttnn.from_gguf(_blocks(ttnn.gguf_mxfp4, 96, 64), (96, 64), ttnn.gguf_mxfp4)
    with expect_error(RuntimeError, "N % 32 == 0"):
        ttnn.from_gguf(blocks[:48], (512, 48), dtype)
    w = ttnn.from_gguf(blocks, (512, 64), dtype, device=device)
    x = ttnn.from_torch(torch.randn(1, 256).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "must equal the weight's K"):
        ttnn.matmul(x, w)
    x = ttnn.from_torch(torch.randn(1, 512).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "takes no transpose"):
        ttnn.matmul(x, w, transpose_b=True)
    with expect_error(RuntimeError, "is BFLOAT16"):
        ttnn.matmul(x, w, dtype=ttnn.float32)
    w_l1 = ttnn.from_gguf(blocks, (512, 64), dtype, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    with expect_error(RuntimeError, "must be DRAM interleaved"):
        ttnn.matmul(x, w_l1)


PERF_SHAPES = [(4096, 4096), (4096, 12288)]
PERF_ITERATIONS, PERF_REPLAYS = 100, 7


def _time_op(device, op):
    """Time per op in µs, min and median over PERF_REPLAYS replays of a trace of PERF_ITERATIONS ops; the last
    output."""
    op()
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(PERF_ITERATIONS):
        out = op()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
    times = []
    for _ in range(PERF_REPLAYS):
        start = time.perf_counter()
        ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
        times.append((time.perf_counter() - start) / PERF_ITERATIONS * 1e6)
    ttnn.release_trace(device, trace)
    return min(times), statistics.median(times), out


# Every format with --runslow; q4_0, q4_K and q6_K by default.
_PERF_DTYPES = [
    pytest.param(
        d, id=_name(d), marks=() if d in (ttnn.gguf_q4_0, ttnn.gguf_q4_k, ttnn.gguf_q6_k) else pytest.mark.slow
    )
    for d in (*ttnn.GGUF_DTYPES, *ttnn.GGUF_EXPANDED.values())
]


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 << 20}], indirect=True)
@pytest.mark.parametrize("dtype", _PERF_DTYPES)
@pytest.mark.parametrize("k, n", PERF_SHAPES)
def test_perf(device, dtype, k, n):
    """Decode (M = 1): µs per op and GGUF weight bytes / time."""
    blocks = _blocks(dtype, k, n)
    w = ttnn.from_gguf(blocks, (k, n), dtype, device=device)
    x = ttnn.from_torch(torch.randn(1, k).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    fastest, median, out = _time_op(device, lambda: ttnn.matmul(x, w))
    _check(ttnn.to_torch(out), ttnn.to_torch(x), blocks, dtype, k, n)
    print(
        f"GGUF {_name(dtype)} [K, N] = [{k}, {n}]: {fastest:.1f} µs (median {median:.1f}), "
        f"{blocks.nbytes / fastest / 1e3:.0f} GB/s"
    )


# The block-float baseline at the same shapes: the DRAM-sharded decode matmul as tt_transformers configures it on
# Blackhole for Qwen3-8B / Llama-3.1-8B (test_matmul_dram_sharded_multicore.py BH_SHAPES): activation and output
# shard grids and widths, in0_block_w, readers per DRAM bank; BFP4 LoFi, BFP8 HiFi2.
# Activation grid: cores row-major, or an 8 x 8 rectangle.
_DRAM_SHARDED = {(4096, 4096): (32, 4, 32, 4, 4, 2), (4096, 12288): ((8, 8), 2, 64, 6, 2, 4)}


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 << 20}], indirect=True)
@pytest.mark.parametrize("dtype, fidelity", [("bfloat4_b", "LoFi"), ("bfloat8_b", "HiFi2")])
@pytest.mark.parametrize("k, n", PERF_SHAPES)
def test_perf_block_float(device, dtype, fidelity, k, n):
    """ttnn.linear with a BFP4 / BFP8 DRAM-sharded weight, M = 1 (one 32-row tile): µs per op and weight GB/s."""
    x_cores, x_width, out_cores, out_width, block_w, readers = _DRAM_SHARDED[(k, n)]
    grid = device.compute_with_storage_grid_size()

    def width_sharded(cores, width):
        if isinstance(cores, tuple):
            end = ttnn.CoreCoord(cores[0] - 1, cores[1] - 1)
            shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), end)})
        else:
            shard_grid = ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True)
        spec = ttnn.ShardSpec(shard_grid, (32, width * 32), ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec)

    banks = device.dram_grid_size().x
    bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    w_spec = ttnn.ShardSpec(bank_grid, (k, n // banks), ttnn.ShardOrientation.ROW_MAJOR)
    w_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, w_spec)
    w = ttnn.from_torch(
        torch.randn(k, n), dtype=getattr(ttnn, dtype), layout=ttnn.TILE_LAYOUT, device=device, memory_config=w_config
    )
    x = ttnn.from_torch(
        torch.randn(32, k),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=width_sharded(x_cores, x_width),
    )
    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=block_w,
        per_core_M=1,
        per_core_N=out_width,
        fused_activation=None,
        num_workers_per_dram_bank=readers,
        use_multicore_pipeline=True,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=getattr(ttnn.MathFidelity, fidelity), fp32_dest_acc_en=False, packer_l1_acc=True
    )
    fastest, median, _ = _time_op(
        device,
        lambda: ttnn.linear(
            x,
            w,
            program_config=program_config,
            memory_config=width_sharded(out_cores, out_width),
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute_config,
        ),
    )
    weight_bytes = k * n * (4.5 if dtype == "bfloat4_b" else 8.5) / 8
    print(
        f"{dtype} [K, N] = [{k}, {n}]: {fastest:.1f} µs (median {median:.1f}), {weight_bytes / fastest / 1e3:.0f} GB/s"
    )


PREFILL_PERF_SHAPES = [(4096, 4096), (4096, 12288), (12288, 4096)]
PREFILL_PERF_ROWS = [128, 512, 2048]
HIFI2_PEAK_TFLOPS = 304.0  # the matmul peak of the 11 x 10 grid at HiFi2


def _print_prefill(name, m, k, n, fastest, median):
    tflops = 2 * m * k * n / fastest / 1e6
    print(
        f"{name} M = {m}, [K, N] = [{k}, {n}]: {fastest:.1f} µs (median {median:.1f}), {tflops:.1f} TFLOP/s, "
        f"{100 * tflops / HIFI2_PEAK_TFLOPS:.1f} % of the HiFi2 peak"
    )


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 << 20}], indirect=True)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(d, id=_name(d), marks=() if d in (ttnn.gguf_q4_0, ttnn.gguf_q4_k) else pytest.mark.slow)
        for d in PREFILL_DTYPES
    ],
)
@pytest.mark.parametrize("k, n", PREFILL_PERF_SHAPES)
def test_perf_prefill(device, dtype, k, n):
    """Prefill (M = 128, 512, 2048 rows): µs per op, TFLOP/s and the fraction of the HiFi2 peak."""
    blocks = _blocks(dtype, k, n)
    w = ttnn.from_gguf(blocks, (k, n), dtype, device=device)
    for m in PREFILL_PERF_ROWS:
        x = ttnn.from_torch(torch.randn(m, k).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
        fastest, median, out = _time_op(device, lambda: ttnn.matmul(x, w))
        if m == PREFILL_PERF_ROWS[0]:
            _check(ttnn.to_torch(out), ttnn.to_torch(x), blocks, dtype, k, n)
        _print_prefill(f"GGUF {_name(dtype)}", m, k, n, fastest, median)
        ttnn.deallocate(x)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 << 20}], indirect=True)
@pytest.mark.parametrize("k, n", PREFILL_PERF_SHAPES)
def test_perf_prefill_block_float(device, k, n):
    """The block-float prefill baseline: ttnn.matmul with an interleaved BFP8 weight, default program config, HiFi2."""
    w = ttnn.from_torch(torch.randn(k, n), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    for m in PREFILL_PERF_ROWS:
        x = ttnn.from_torch(torch.randn(m, k).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
        fastest, median, _ = _time_op(device, lambda: ttnn.matmul(x, w, compute_kernel_config=compute_config))
        _print_prefill("bfloat8_b", m, k, n, fastest, median)
        ttnn.deallocate(x)
