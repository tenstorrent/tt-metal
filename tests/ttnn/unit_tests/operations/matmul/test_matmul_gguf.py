# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.matmul / ttnn.linear with a GGUF weight (ttnn.from_gguf) on Blackhole, against gguf-py's dequantization of the
same blocks times the same bf16 input in float64: PCC >= 0.9999 and the largest error within 1 % of the largest
output, the bounds of the llama.cpp backend's GGUF gate."""

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


def _name(dtype):
    return dtype.name.removeprefix("GGUF_")


def _qtype(dtype):
    qtype = getattr(gguf.GGMLQuantizationType, _name(dtype), None)
    if qtype is None:
        pytest.skip(f"gguf-py has no {_name(dtype)}")
    return qtype


def _blocks(dtype, k, n, seed=0):
    """The GGUF bytes of an [N, K] weight."""
    name, qtype = _name(dtype), _qtype(dtype)
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
    page_rows = {"Q8_0": 32, "Q4_0": 32, "IQ4_NL": 32, "Q4_1": 32, "NVFP4": 64}.get(_name(dtype))
    if page_rows is None:
        page_rows = {"MXFP4": 64, "Q5_0": 64, "Q5_1": 64, "Q2_0": 64, "Q1_0": 128}.get(_name(dtype), 256)
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


@pytest.mark.parametrize("m", [128, 512])
@pytest.mark.parametrize("dtype", ttnn.GGUF_DTYPES, ids=_name)
def test_prefill(device, dtype, m):
    _run(device, dtype, (1, m), _k(dtype), 96)


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


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 << 20}], indirect=True)
@pytest.mark.parametrize("dtype", [ttnn.gguf_q4_0, ttnn.gguf_q4_k, ttnn.gguf_q6_k], ids=_name)
@pytest.mark.parametrize("k, n", [(4096, 4096), (4096, 12288)])
def test_perf(device, dtype, k, n):
    """Decode (M = 1) time per op from a trace of 100 ops: µs and weight bytes / time."""
    iterations = 100
    blocks = _blocks(dtype, k, n)
    w = ttnn.from_gguf(blocks, (k, n), dtype, device=device)
    x = ttnn.from_torch(torch.randn(1, k).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.matmul(x, w)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(iterations):
        out = ttnn.matmul(x, w)
    ttnn.end_trace_capture(device, trace, cq_id=0)
    ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
    start = time.perf_counter()
    ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
    seconds = (time.perf_counter() - start) / iterations
    ttnn.release_trace(device, trace)
    _check(ttnn.to_torch(out), ttnn.to_torch(x), blocks, dtype, k, n)
    print(f"GGUF {_name(dtype)} [K, N] = [{k}, {n}]: {seconds * 1e6:.1f} µs, {blocks.nbytes / seconds / 1e9:.0f} GB/s")
