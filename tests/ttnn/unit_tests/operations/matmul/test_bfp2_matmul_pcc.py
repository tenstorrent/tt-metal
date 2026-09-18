# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Accuracy/density bracket for low-bit block-float matmul on device.

For a matmul run in BF16 (the reference), this measures the PCC of the same matmul when the
operands are quantized to BFP2_B, BFP4_B and BFP8_B, across several operand distributions
(gaussian, true binary {-1,+1}, ternary {-1,0,1}).

BFP2_B is special: it exists at the ``DataFormat`` layer but is NOT an exposed ttnn tensor
``DataType`` (the enum stops at ``BFLOAT4_B``), so there is no ``ttnn.bfloat2_b``. We reach
it through the real encoder ``ttnn._ttnn.bfp_utils.pack_bfp2`` / ``unpack_bfp2``:
round-tripping an operand yields the exact values a BFP2_B tile would hold, which we then
carry in a bf16 tensor (bf16's 7 mantissa bits losslessly hold BFP2's 1-bit mantissa) into
the real on-device matmul. BFP4_B and BFP8_B are real ttnn dtypes, so those operands go to
the device natively.

``is_exp_a`` selects the BFP2 exponent-sharing / bias mode in the packer; both are swept.

Bits per element (incl. shared exponent, per 1024-elem tile): BFP2 ~2.5, BFP4 ~4.5,
BFP8 ~8.5, BF16 = 16.
"""

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

_bfp = ttnn._ttnn.bfp_utils
TILE = 32


def _to_tiles(x: np.ndarray) -> np.ndarray:
    """[H, W] (row-major) -> [num_tiles, 32, 32], tiles ordered row-major, each tile row-major."""
    H, W = x.shape
    return x.reshape(H // TILE, TILE, W // TILE, TILE).transpose(0, 2, 1, 3).reshape(-1, TILE, TILE)


def _from_tiles(t: np.ndarray, H: int, W: int) -> np.ndarray:
    """Inverse of _to_tiles."""
    return t.reshape(H // TILE, W // TILE, TILE, TILE).transpose(0, 2, 1, 3).reshape(H, W)


def bfp2b_quantize(x: torch.Tensor, is_exp_a: bool = False) -> torch.Tensor:
    """Round-trip a 2-D tensor through the real BFP2_B packer.

    Returns the exact (dequantized) values a BFP2_B tile would hold on device.
    H and W must be multiples of 32. ``is_exp_a`` picks the exponent-sharing/bias mode.
    """
    x_np = x.detach().to(torch.float32).cpu().numpy()
    H, W = x_np.shape
    flat = np.ascontiguousarray(_to_tiles(x_np).reshape(-1), dtype=np.float32)
    packed = np.asarray(_bfp.pack_bfp2(flat, row_major_input=True, is_exp_a=is_exp_a))
    deq = np.asarray(_bfp.unpack_bfp2(packed, row_major_output=True, is_exp_a=is_exp_a)).reshape(-1, TILE, TILE)
    return torch.from_numpy(_from_tiles(deq, H, W).copy())


def binarize_with_scale(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """XNOR-Net binarization of a real tensor with a single per-tensor scale.

        alpha  = mean(|x|)              (the optimal L2 scale for a sign quantizer)
        binary = sign(x)               (sign(0) mapped to +1 so values stay in {-1, +1})
        reconstructed = binary * alpha

    Using ``sign(x)`` (not ``sign(x - alpha)``) is what minimizes ||x - alpha*binary||^2 and
    keeps the binarization unbiased; subtracting alpha first would push almost every element to
    -1 and wreck the correlation with the original tensor.

    Returns ``(binary, alpha)`` where ``binary`` is in {-1, +1} (fp32) and ``alpha`` is a scalar.
    """
    xf = x.to(torch.float32)
    alpha = xf.abs().mean()
    binary = torch.sign(xf)
    binary = torch.where(binary == 0, torch.ones_like(binary), binary)
    return binary, alpha


def _binarized(x: torch.Tensor) -> torch.Tensor:
    """Reconstructed operand ``binary * alpha`` (values in {+alpha, -alpha}), as fp32."""
    binary, alpha = binarize_with_scale(x)
    return binary * alpha


def _make_operands(dist: str, m: int, k: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Two operands drawn from ``dist``, returned as bf16 (all values are exact in bf16)."""
    g = torch.Generator().manual_seed(0)
    if dist == "gaussian":
        a = torch.randn(m, k, generator=g)
        b = torch.randn(k, n, generator=g)
    elif dist == "binary":  # {-1, +1}
        a = torch.randint(0, 2, (m, k), generator=g).float() * 2 - 1
        b = torch.randint(0, 2, (k, n), generator=g).float() * 2 - 1
    elif dist == "ternary":  # {-1, 0, +1}
        a = torch.randint(-1, 2, (m, k), generator=g).float()
        b = torch.randint(-1, 2, (k, n), generator=g).float()
    else:
        raise ValueError(dist)
    return a.to(torch.bfloat16), b.to(torch.bfloat16)


def _device_matmul(device, a_t: torch.Tensor, b_t: torch.Tensor, dtype, out_dtype=ttnn.bfloat16) -> torch.Tensor:
    """Run a @ b on device with operands stored as ``dtype``; return the result as torch.

    The output dtype is forced (default bf16). Without this, ttnn.matmul defaults the output
    dtype to input A's dtype, so low-bit operands would also quantize the wide-dynamic-range
    *result* -- catastrophic for bfp2 (1 mantissa bit) -- masking the input-quantization effect
    we actually want to measure.
    """
    a_tt = ttnn.from_torch(a_t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b_t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.to_torch(ttnn.matmul(a_tt, b_tt, dtype=out_dtype))


@pytest.mark.parametrize("dist", ["gaussian", "binary", "ternary"])
@pytest.mark.parametrize(
    "m, k, n",
    [
        (512, 512, 512),
        (1024, 2048, 1024),
    ],
)
def test_low_bit_matmul_pcc(device, dist, m, k, n):
    a, b = _make_operands(dist, m, k, n)

    golden = a.to(torch.float32) @ b.to(torch.float32)

    # BF16 on-device matmul: the reference every low-bit variant is compared against.
    out_bf16 = _device_matmul(device, a, b, ttnn.bfloat16)
    _, pcc_bf16_vs_ref = comp_pcc(golden, out_bf16)

    results: dict[str, float] = {}

    # BFP2_B via the real packer (both exponent modes), carried in bf16 into the matmul.
    for is_exp_a in (False, True):
        a_q = bfp2b_quantize(a, is_exp_a=is_exp_a).to(torch.bfloat16)
        b_q = bfp2b_quantize(b, is_exp_a=is_exp_a).to(torch.bfloat16)
        out = _device_matmul(device, a_q, b_q, ttnn.bfloat16)
        _, results[f"bfp2_b (exp_a={is_exp_a})"] = comp_pcc(out_bf16, out)

    # BFP4_B and BFP8_B: native ttnn dtypes, quantized on the from_torch path.
    for name, dt in (("bfp4_b", ttnn.bfloat4_b), ("bfp8_b", ttnn.bfloat8_b)):
        out = _device_matmul(device, a, b, dt)
        _, results[name] = comp_pcc(out_bf16, out)

    tag = f"[{dist:8s} {m}x{k} @ {k}x{n}]"
    logger.info(f"{tag} PCC bf16 vs float-ref : {pcc_bf16_vs_ref:.6f}")
    for name, pcc in results.items():
        logger.info(f"{tag} PCC {name:20s} vs bf16 : {pcc:.6f}")

    # Loose bound: the deliverable is the logged PCC table, not a tight gate.
    for name, pcc in results.items():
        assert pcc > 0.0, f"{name} produced non-positive PCC ({pcc})"


@pytest.mark.parametrize(
    "m, k, n",
    [
        (512, 512, 512),
        (1024, 2048, 1024),
    ],
)
def test_binarized_matmul_pcc(device, m, k, n):
    """Binarize gaussian operands with a per-tensor scale (XNOR-Net style), then matmul.

    Each operand becomes ``binary * alpha`` with values in {+alpha, -alpha}. This is the
    realistic setting for a low-bit format: within any 32x32 face every magnitude equals the
    single scalar ``alpha``, so a BFP2_B block (shared exponent + 1 sign + 1 mantissa bit)
    represents it exactly up to a constant scale -> the only gap vs the bf16 result is
    accumulation precision (LoFi vs HiFi2), stored natively on device at ~2.5 bits/elem via
    ``ttnn.bfloat2_b``.
    """
    a, b = _make_operands("gaussian", m, k, n)

    golden = a.to(torch.float32) @ b.to(torch.float32)

    # Binarize both operands with their own single scalar.
    a_bin, alpha_a = binarize_with_scale(a)
    b_bin, alpha_b = binarize_with_scale(b)
    a_rec = a_bin * alpha_a  # fp32, values in {+/- alpha_a}
    b_rec = b_bin * alpha_b

    # Full-precision matmul of the binarized operands: the ceiling any dtype can reach here.
    binary_ref = a_rec @ b_rec

    a_bf16 = a_rec.to(torch.bfloat16)
    b_bf16 = b_rec.to(torch.bfloat16)
    out_bf16 = _device_matmul(device, a_bf16, b_bf16, ttnn.bfloat16)
    out_bfp2 = _device_matmul(device, a_bf16, b_bf16, ttnn.bfloat2_b)

    _, pcc_binary_vs_full = comp_pcc(golden, binary_ref)
    _, pcc_bf16_vs_binref = comp_pcc(binary_ref, out_bf16)
    _, pcc_bfp2_vs_bf16 = comp_pcc(out_bf16, out_bfp2)
    _, pcc_bfp2_vs_full = comp_pcc(golden, out_bfp2)

    tag = f"[binarized {m}x{k} @ {k}x{n}]"
    logger.info(f"{tag} alpha_a={alpha_a:.4f} alpha_b={alpha_b:.4f}")
    logger.info(f"{tag} PCC binarized(fp32) vs full-precision : {pcc_binary_vs_full:.6f}")
    logger.info(f"{tag} PCC bf16 device      vs binarized(fp32): {pcc_bf16_vs_binref:.6f}")
    logger.info(f"{tag} PCC bfloat2_b device vs bf16 device    : {pcc_bfp2_vs_bf16:.6f}")
    logger.info(f"{tag} PCC bfloat2_b device vs full-precision : {pcc_bfp2_vs_full:.6f}")

    # A 2-magnitude (per-block-constant) tensor is exact in BFP2_B up to a constant scale, and
    # PCC is scale-invariant, so input quantization is essentially free here. The residual gap
    # vs bf16 is accumulation precision, not quantization: the bf16 path runs at HiFi2 while the
    # all-low-precision bfp2 path is auto-selected to LoFi, so over large K they differ by ~1-2%
    # (compare the bf16-vs-fp32 reference row, which is itself only ~0.99 at K=2048).
    assert pcc_bfp2_vs_bf16 > 0.97, f"bfloat2_b diverged from bf16 for binarized operands ({pcc_bfp2_vs_bf16})"
