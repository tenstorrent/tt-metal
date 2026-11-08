# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""GGUF weights for the ttnn GGUF tests (test_gguf_dtypes.py, test_matmul_gguf.py): format names, gguf-py types,
blocks and the K of whole pages."""

import numpy as np
import pytest

import ttnn

gguf = pytest.importorskip("gguf")

# gguf-py quantizes these; the others get random bytes.
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
# Rows of K per page (lcm of the page height and the block size), where not 256.
K_UNIT = {
    "Q8_0": 32,
    "Q4_0": 32,
    "IQ4_NL": 32,
    "Q4_1": 32,
    "MXFP4": 64,
    "NVFP4": 64,
    "Q5_0": 64,
    "Q5_1": 64,
    "Q2_0": 64,
    "Q1_0": 128,
}

# Expanded layout -> its GGUF format.
SOURCE = {expanded: source for source, expanded in ttnn.GGUF_EXPANDED.items()}


def name(dtype):
    return dtype.name.removeprefix("GGUF_")


def fmt(dtype):
    """The GGUF format of a dtype's blocks."""
    return name(SOURCE.get(dtype, dtype))


def qtype(dtype):
    qtype = getattr(gguf.GGMLQuantizationType, fmt(dtype), None)
    if qtype is None:
        pytest.skip(f"gguf-py has no {fmt(dtype)}")
    return qtype


def blocks(dtype, k, n, seed=0, raw=False):
    """The GGUF bytes of an [N, K] weight: gguf-py's quantization where it has one; otherwise random bytes, with sane
    scale fields (about 2^-8 .. 2^-7) unless raw."""
    name, q = fmt(dtype), qtype(dtype)
    rng = np.random.default_rng(seed)
    if name in QUANTIZED:
        return gguf.quants.quantize(rng.standard_normal((n, k), dtype=np.float32), q)
    block_elems, block_bytes = gguf.GGML_QUANT_SIZES[q]
    data = rng.integers(0, 256, size=(n * k // block_elems, block_bytes), dtype=np.uint8)
    if raw:
        return data.reshape(n, -1)
    scales = (rng.uniform(0.5, 1.0, size=(data.shape[0], 4)) * 2.0**-7).astype(np.float16).view(np.uint8)
    if name == "NVFP4":  # four UE4M3 scales: exponents 5..8, any mantissa
        data[:, :4] = (
            rng.integers(5, 9, size=(data.shape[0], 4)) << 3 | rng.integers(0, 8, (data.shape[0], 4))
        ).astype(np.uint8)
    elif name == "IQ1_M":  # d is the top nibbles of the four 16-bit scale words at bytes 48..55
        d = scales[:, :2].copy().view(np.uint16)[:, 0]
        words = data[:, 48:56].copy().view(np.uint16)
        for j in range(4):
            words[:, j] = (words[:, j] & 0x0FFF) | (((d >> (4 * j)) & 0xF) << 12)
        data[:, 48:56] = words.view(np.uint8)
    else:
        for i, offset in enumerate(FP16_FIELDS[name]):
            data[:, offset : offset + 2] = scales[:, 2 * i : 2 * i + 2]
    return data.reshape(n, -1)


def k_of(dtype, pages=2):
    """K of `pages` GGUF pages: whole pages and whole blocks."""
    return pages * K_UNIT.get(fmt(dtype), 256)
