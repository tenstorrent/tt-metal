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

gguf = pytest.importorskip("gguf")

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="GGUF weights need Blackhole")

PCC = 0.9999
MAX_ERROR = 0.01  # of the largest output

# gguf-py quantizes these; the others get random bytes with sane scale fields.
QUANTIZED = {"Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "MXFP4", "TQ1_0", "TQ2_0", "Q1_0", "Q2_0"}
# Byte offsets of the fp16 scale fields in a block of each format that gguf-py cannot quantize.
FP16_FIELDS = {
    "IQ4_NL": (0,),
    "Q6_K": (208,),
    "Q4_K": (0, 2),
    "Q5_K": (0, 2),
    "Q3_K": (108,),
    "Q2_K": (80, 82),
    "IQ4_XS": (0,),
    "IQ2_XXS": (0,),
    "IQ2_XS": (0,),
    "IQ2_S": (0,),
    "IQ3_XXS": (0,),
    "IQ3_S": (0,),
    "IQ1_S": (0,),
}


# Expanded layout -> its GGUF format.
SOURCE = {expanded: source for source, expanded in ttnn.GGUF_EXPANDED.items()}


def _name(dtype):
    return dtype.name.removeprefix("GGUF_")


def _format(dtype):
    """The GGUF format of a dtype's blocks."""
    return _name(SOURCE.get(dtype, dtype))


def _qtype(dtype):
    qtype = getattr(gguf.GGMLQuantizationType, _format(dtype), None)
    if qtype is None:
        pytest.skip(f"gguf-py has no {_format(dtype)}")
    return qtype


def _blocks(dtype, k, n, seed=0):
    """The GGUF bytes of an [N, K] weight."""
    name, qtype = _format(dtype), _qtype(dtype)
    rng = np.random.default_rng(seed)
    if name in QUANTIZED:
        return gguf.quants.quantize(rng.standard_normal((n, k), dtype=np.float32), qtype)
    block_elems, block_bytes = gguf.GGML_QUANT_SIZES[qtype]
    blocks = rng.integers(0, 256, size=(n * k // block_elems, block_bytes), dtype=np.uint8)
    scales = (rng.uniform(0.5, 1.0, size=(blocks.shape[0], 4)) * 2.0**-7).astype(np.float16).view(np.uint8)
    if name == "NVFP4":  # four UE4M3 scales: exponents 5..8, any mantissa
        blocks[:, :4] = (
            rng.integers(5, 9, size=(blocks.shape[0], 4)) << 3 | rng.integers(0, 8, (blocks.shape[0], 4))
        ).astype(np.uint8)
    elif name == "IQ1_M":  # d is the top nibbles of the four 16-bit scale words at bytes 48..55
        d = scales[:, :2].copy().view(np.uint16)[:, 0]
        words = blocks[:, 48:56].copy().view(np.uint16)
        for j in range(4):
            words[:, j] = (words[:, j] & 0x0FFF) | (((d >> (4 * j)) & 0xF) << 12)
        blocks[:, 48:56] = words.view(np.uint8)
    else:
        for i, offset in enumerate(FP16_FIELDS[name]):
            blocks[:, offset : offset + 2] = scales[:, 2 * i : 2 * i + 2]
    return blocks.reshape(n, -1)


def _k(dtype, groups=2):
    """K of `groups` GGUF pages: whole pages and whole blocks."""
    page_rows = {"Q8_0": 32, "Q4_0": 32, "IQ4_NL": 32, "Q4_1": 32, "NVFP4": 64}.get(_format(dtype))
    if page_rows is None:
        page_rows = {"MXFP4": 64, "Q5_0": 64, "Q5_1": 64, "Q2_0": 64, "Q1_0": 128}.get(_format(dtype), 256)
    return groups * page_rows


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


@pytest.mark.parametrize("m", [1, 3])
def test_decode_tq2_0_long_k(device, m):
    # 40 K groups: with one input row the scales apply per 16 K groups, and the cores' segments cross those windows.
    _run(device, ttnn.gguf_tq2_0, (1, 1, m), 10240, 2048)


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
