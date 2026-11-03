# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Independent MXFP8 (E4M3 + E8M0) reference, corpus, and exact accuracy gate.

value = E4M3(code) * 2^(scale - 127). E4M3 has at most four significant bits,
so every product below 2^128 and at or above 2^-126 is exactly representable in
BF16 and FP32: the contract is exact equality. Products >= 2^128 must be signed
infinity. Products below 2^-126 (the BF16/FP32 minimum normal) may either flush
to zero or be the correctly rounded subnormal of the output dtype. E4M3 NaN
codes (0x7F, 0xFF) and the E8M0 NaN code (0xFF) are outside the contract.
"""

import hashlib
import math
import os
from pathlib import Path

import torch

import ttnn

KERNELS = Path("ttnn/cpp/ttnn/operations/experimental/dequantize_mxfp8/device/kernels")
REPO = Path(__file__).resolve().parents[4]
E4M3_NAN = (0x7F, 0xFF)
MIN_NORMAL = 2.0**-126
OVERFLOW = 2.0**128

# The configuration the performance benchmark and the candidate screen use.
PRIMARY = dict(container="uint16", fp32_dest_acc_en=False, dtype="bfloat16")

CONTAINERS = {"uint8": (ttnn.uint8, 1), "uint16": (ttnn.uint16, 2)}
OUTPUT_DTYPES = {"bfloat16": ttnn.bfloat16, "float32": ttnn.float32}
# (fp32_dest_acc_en, output dtype) combinations the operation supports.
DEST_MODES = [(False, "bfloat16"), (True, "bfloat16"), (True, "float32")]
# UINT8 datacopy produces Integer-32 in DEST, so UINT8 input requires FP32 DEST.
CONFIGS = [
    (container, fp32_dest_acc_en, dtype)
    for container in CONTAINERS
    for fp32_dest_acc_en, dtype in DEST_MODES
    if container != "uint8" or fp32_dest_acc_en
]


def decode_e4m3(codes):
    """Decode E4M3FN codes (OCP; bias 7, no infinities) to float64, without torch's FP8 dtype."""
    codes = codes.long()
    exponent, mantissa = (codes >> 3) & 15, codes & 7
    magnitude = torch.where(
        exponent == 0, mantissa.double() * 2.0**-9, torch.ldexp(1 + mantissa.double() / 8, exponent - 7)
    )
    return torch.where((codes & 0x80) != 0, -magnitude, magnitude)


def decode_e8m0(scales):
    return torch.ldexp(torch.ones_like(scales, dtype=torch.float64), scales.long() - 127)


def reference(codes, scales):
    """Exact float64 product per element; `scales` has shape [..., width // 32]."""
    return decode_e4m3(codes) * decode_e8m0(scales).repeat_interleave(32, dim=-1)


def _round_to(values, dtype):
    # float64 -> float32 is exact for every product here that is not below
    # FP32's subnormal range; float32 -> bfloat16 then rounds once (RNE).
    as_float = values.float()
    return as_float if dtype == "float32" else as_float.bfloat16().float()


def check_exact(actual, exact, dtype):
    """Return a boolean mask of elements that satisfy the numerical contract."""
    actual, exact = actual.double().flatten(), exact.double().flatten()
    expected = torch.where(exact.abs() >= OVERFLOW, torch.copysign(torch.full_like(exact, math.inf), exact), exact)
    ok = actual == expected
    tiny = exact.abs() < MIN_NORMAL
    rounded = _round_to(exact, dtype).double()
    ok |= tiny & ((actual == 0) | (actual == rounded))
    return ok & ~torch.isnan(actual)


def _json_float(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def metrics(actual, exact, dtype, codes=None, scales=None, examples=8):
    actual, exact = actual.double().flatten(), exact.double().flatten()
    ok = check_exact(actual, exact, dtype)
    tiny = exact.abs() < MIN_NORMAL
    overflow = exact.abs() >= OVERFLOW
    finite = torch.isfinite(actual) & ~overflow & ~tiny & (exact != 0)
    _, exponent = torch.frexp(exact[finite].abs())
    quantum = torch.ldexp(torch.ones_like(exact[finite]), exponent - 8)
    ulp = ((actual[finite] - exact[finite]).abs() / quantum) if finite.any() else torch.zeros(1, dtype=torch.float64)
    bad = (~ok).nonzero().flatten()
    report = dict(
        values=int(exact.numel()),
        mismatches=int(bad.numel()),
        nan_outputs=int(torch.isnan(actual).sum()),
        overflow_cases=int(overflow.sum()),
        overflow_correct=int((ok & overflow).sum()),
        subnormal_cases=int(tiny.sum() - (exact == 0).sum()),
        subnormal_flushed=int((tiny & (exact != 0) & (actual == 0)).sum()),
        max_bf16_ulp_normal_range=_json_float(ulp.nan_to_num(math.inf).max()),
        exact_fraction=float(ok.double().mean()),
    )
    if bad.numel():
        flat_codes = None if codes is None else codes.flatten()
        flat_scales = None if scales is None else scales.repeat_interleave(32, dim=-1).flatten()
        report["mismatch_examples"] = [
            dict(
                flat_index=int(i),
                code=None if flat_codes is None else int(flat_codes[i]),
                scale=None if flat_scales is None else int(flat_scales[i]),
                expected=_json_float(exact[i]),
                actual=_json_float(actual[i]),
            )
            for i in bad[:examples]
        ]
    return report


def corpus():
    """Exhaustive (code, scale) pairs at varied positions, plus random blocks.

    Rows 0..254 use E8M0 scale r for every block and a distinct bijection of all
    256 codes, so each (code, scale) pair appears once and each code moves across
    columns, faces and planes between rows. NaN codes are replaced by the largest
    finite magnitude of the same sign. Row 255 uses scale 127. Rows 256..511 are
    random codes with random per-block scales. Shape [512, 256] (8 blocks/row).
    """
    generator = torch.Generator().manual_seed(20261103)
    columns = torch.arange(256)
    rows = []
    for row in range(256):
        multiplier = 2 * (row % 64) + 1  # Odd multipliers permute Z/256.
        rows.append((columns * multiplier + 37 * row) % 256)
    exhaustive = torch.stack(rows)
    random_codes = torch.randint(0, 256, (256, 256), generator=generator)
    codes = torch.cat([exhaustive, random_codes]).to(torch.int32)
    codes = torch.where(codes == 0x7F, 0x7E, codes)
    codes = torch.where(codes == 0xFF, 0xFE, codes)
    exhaustive_scales = torch.arange(256).clamp_max(254)
    exhaustive_scales[255] = 127
    scales = torch.cat(
        [
            exhaustive_scales[:, None].expand(256, 8),
            torch.randint(0, 255, (256, 8), generator=generator),
        ]
    ).to(torch.int32)
    return codes.contiguous(), scales.contiguous()


def kernel_hashes():
    """sha256 of every candidate kernel file, keyed as scripts/quant_harness.py keys them."""
    root = Path(os.environ.get("MXFP8_KERNEL_ROOT", REPO))
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / KERNELS).rglob("*"))
        if path.is_file()
    }


def run_decode(
    device,
    codes,
    scales,
    container="uint16",
    fp32_dest_acc_en=None,
    dtype="bfloat16",
    input_layout=ttnn.TILE_LAYOUT,
    output_layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    shape=None,
):
    """Pack, upload, decode, and download. Returns (actual float tensor, device tensors to keep alive)."""
    word_dtype, _ = CONTAINERS[container]
    if fp32_dest_acc_en is None:
        fp32_dest_acc_en = container == "uint8"
    packed, ordered_scales = ttnn.experimental.mxfp8.pack(codes, scales, dtype=word_dtype)
    if shape is not None:
        packed = packed.reshape(*shape[:-1], -1)
    data = ttnn.from_torch(packed, dtype=word_dtype, layout=input_layout, device=device, memory_config=memory_config)
    blocks = ttnn.from_torch(
        ordered_scales, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config
    )
    output = ttnn.experimental.dequantize_mxfp8(
        data,
        blocks,
        dtype=OUTPUT_DTYPES[dtype],
        output_layout=output_layout,
        fp32_dest_acc_en=fp32_dest_acc_en,
        memory_config=memory_config,
    )
    assert output.dtype == OUTPUT_DTYPES[dtype], "Output dtype changed"
    assert output.layout == output_layout, "Output layout changed"
    expected_shape = tuple(codes.shape) if shape is None else tuple(shape)
    assert tuple(output.shape) == expected_shape, f"Output shape {tuple(output.shape)} != {expected_shape}"
    actual = ttnn.to_torch(output).float()
    return actual, (data, blocks, output)


def assert_exact(actual, codes, scales, dtype):
    report = metrics(actual, reference(codes, scales), dtype, codes, scales)
    assert report["mismatches"] == 0, f"MXFP8 decode mismatches: {report}"
    return report
