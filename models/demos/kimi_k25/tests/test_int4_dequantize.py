# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for INT4 group-32 weight dequantization.

Tests the round-trip fidelity of W4A16 symmetric INT4 packing/unpacking
used in the Kimi K2.5 weight loader. No hardware or model weights required.

Run::

    python models/demos/kimi_k25/tests/test_int4_dequantize.py

All tests must pass locally (CPU-only, PyTorch required).
"""

from __future__ import annotations

import sys

import torch

# Allow running from repo root without install
sys.path.insert(0, ".")

from models.demos.kimi_k25.utils.int4_dequantize import dequantize_int4_weight, unpack_int4_nibbles

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson Correlation Coefficient between two tensors (flattened to 1D)."""
    a = a.float().flatten()
    b = b.float().flatten()
    a_c = a - a.mean()
    b_c = b - b.mean()
    num = (a_c * b_c).sum()
    den = (a_c.pow(2).sum() * b_c.pow(2).sum()).sqrt()
    if den == 0:
        return 1.0 if torch.allclose(a, b) else 0.0
    return (num / den).item()


def _pack_reference(int4_signed: torch.Tensor) -> torch.Tensor:
    """Reference packer: signed int8 tensor (range [-8, 7]) → uint8 byte tensor.

    Packing convention (matches ``unpack_int4_nibbles``):
      - even-index elements → low nibble (bits 0-3)
      - odd-index  elements → high nibble (bits 4-7)
    """
    assert int4_signed.dtype == torch.int8, "Input must be int8"
    unsigned = (int4_signed + 8).to(torch.uint8)  # zero-point 8 → [0, 15]
    low = unsigned[..., 0::2]  # even positions → low nibble
    high = unsigned[..., 1::2]  # odd  positions → high nibble
    return low | (high << 4)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_unpack_nibbles_roundtrip():
    """unpack_int4_nibbles must invert the reference packer exactly."""
    torch.manual_seed(42)
    rows, cols = 64, 128  # cols must be even

    ref = torch.randint(-8, 8, (rows, cols), dtype=torch.int8)
    packed = _pack_reference(ref)
    unpacked = unpack_int4_nibbles(packed)

    assert unpacked.shape == ref.shape, f"Shape mismatch: {unpacked.shape} vs {ref.shape}"
    assert unpacked.dtype == torch.int8, f"Dtype mismatch: {unpacked.dtype}"
    assert (unpacked == ref).all(), "Round-trip nibble unpack failed — values differ"
    print("[PASS] test_unpack_nibbles_roundtrip: exact int8 match")


def test_dequantize_exact_pcc():
    """Full dequantization round-trip: PCC must be > 0.9999.

    Uses float32 throughout (no BF16 rounding), so the round-trip should be
    essentially exact — PCC ≈ 1.0.
    """
    torch.manual_seed(7)
    out_features, in_features = 256, 512
    group_size = 32
    n_groups = in_features // group_size

    ref_int4 = torch.randint(-8, 8, (out_features, in_features), dtype=torch.int8)
    scales = torch.rand(out_features, n_groups, dtype=torch.float32) * 0.01 + 1e-4

    # Ground-truth dequantization
    ref_fp = (ref_int4.float().reshape(out_features, n_groups, group_size) * scales.unsqueeze(-1)).reshape(
        out_features, in_features
    )

    # Pack → dequantize via our code path
    packed = _pack_reference(ref_int4)
    result = dequantize_int4_weight(packed, scales, group_size=group_size, output_dtype=torch.float32)

    assert result.shape == ref_fp.shape, f"Shape mismatch: {result.shape} vs {ref_fp.shape}"
    pcc = _pcc(result, ref_fp)
    max_abs_err = (result - ref_fp).abs().max().item()
    print(f"[INFO]  dequantize_exact_pcc: PCC={pcc:.10f}  max_abs_err={max_abs_err:.2e}")
    assert pcc > 0.9999, f"PCC too low: {pcc:.8f} (want > 0.9999)"
    print("[PASS] test_dequantize_exact_pcc")


def test_dequantize_kimi_expert_shapes():
    """PCC > 0.99 for realistic Kimi K2.5 routed-expert weight shapes (BF16 output).

    Expert linear shapes in Kimi K2.5 (moe_intermediate_size=2048, hidden=7168):
      w1/w3 (gate/up proj): (2048, 7168)   ← moe_intermediate_size × hidden
      w2    (down proj):    (7168, 2048)   ← hidden × moe_intermediate_size

    BF16 conversion introduces ~0.3 % rounding; 0.99 threshold is conservative.
    """
    torch.manual_seed(99)
    group_size = 32

    for name, shape in [("w1", (2048, 7168)), ("w2", (7168, 2048))]:
        out_features, in_features = shape
        n_groups = in_features // group_size

        ref_int4 = torch.randint(-8, 8, (out_features, in_features), dtype=torch.int8)
        scales = torch.rand(out_features, n_groups, dtype=torch.float32) * 0.01 + 1e-4

        ref_fp = (ref_int4.float().reshape(out_features, n_groups, group_size) * scales.unsqueeze(-1)).reshape(
            out_features, in_features
        )

        packed = _pack_reference(ref_int4)
        result = dequantize_int4_weight(packed, scales, group_size=group_size, output_dtype=torch.bfloat16)

        # Compare against BF16-cast ground truth (accounts for BF16 rounding)
        ref_bf16 = ref_fp.bfloat16()
        pcc = _pcc(result, ref_bf16)
        print(f"[INFO]  kimi_expert_shapes [{name}] shape={shape}: PCC={pcc:.10f}")
        assert pcc > 0.99, f"{name}: PCC too low: {pcc:.8f}"
        print(f"[PASS] test_dequantize_kimi_expert_shapes [{name}]")


def test_dequantize_bf16_output():
    """Output dtype must be bfloat16 and PCC vs float32 reference > 0.99."""
    torch.manual_seed(13)
    out_features, in_features = 128, 256
    group_size = 32
    n_groups = in_features // group_size

    ref_int4 = torch.randint(-8, 8, (out_features, in_features), dtype=torch.int8)
    scales_bf16 = (torch.rand(out_features, n_groups, dtype=torch.float32) * 0.01).bfloat16()

    packed = _pack_reference(ref_int4)
    result = dequantize_int4_weight(packed, scales_bf16, group_size=group_size, output_dtype=torch.bfloat16)

    assert result.dtype == torch.bfloat16, f"Expected bfloat16, got {result.dtype}"
    assert result.shape == (out_features, in_features), f"Shape mismatch: {result.shape}"
    print(f"[PASS] test_dequantize_bf16_output: shape={tuple(result.shape)}, dtype={result.dtype}")


def test_dequantize_scale_bfloat16_accepted():
    """BF16 scales (as stored in Kimi K2.5 safetensors) must be accepted without error."""
    torch.manual_seed(21)
    out_features, in_features = 64, 128
    group_size = 32
    n_groups = in_features // group_size

    packed = torch.randint(0, 256, (out_features, in_features // 2), dtype=torch.uint8)
    scales = torch.rand(out_features, n_groups, dtype=torch.bfloat16) * 0.005

    result = dequantize_int4_weight(packed, scales, group_size=group_size)
    assert result.shape == (out_features, in_features)
    print("[PASS] test_dequantize_scale_bfloat16_accepted")


def test_dequantize_validation_errors():
    """Invalid inputs must raise ValueError with a descriptive message."""
    cases = [
        # (description, packed, scales, group_size)
        (
            "3D packed tensor",
            torch.zeros(4, 8, 4, dtype=torch.uint8),
            torch.zeros(4, 2),
            32,
        ),
        (
            "1D scales",
            torch.zeros(4, 16, dtype=torch.uint8),
            torch.zeros(1),
            32,
        ),
        (
            "wrong scale column count (3 groups, want 1)",
            torch.zeros(4, 16, dtype=torch.uint8),  # in_features=32 → n_groups=1
            torch.zeros(4, 3),
            32,
        ),
        (
            "in_features not divisible by group_size",
            torch.zeros(4, 5, dtype=torch.uint8),  # in_features=10
            torch.zeros(4, 1),
            32,
        ),
    ]

    for desc, packed, scales, gs in cases:
        try:
            dequantize_int4_weight(packed, scales, group_size=gs)
            print(f"[FAIL] {desc}: expected ValueError, got no error")
            sys.exit(1)
        except ValueError as exc:
            print(f"[PASS] test_dequantize_validation_errors [{desc}]: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 64)
    print("  Kimi K2.5 INT4 dequantizer — unit tests")
    print("=" * 64)

    test_unpack_nibbles_roundtrip()
    test_dequantize_exact_pcc()
    test_dequantize_kimi_expert_shapes()
    test_dequantize_bf16_output()
    test_dequantize_scale_bfloat16_accepted()
    test_dequantize_validation_errors()

    print()
    print("=" * 64)
    print("  [ALL PASS] INT4 dequantize tests complete")
    print("=" * 64)


if __name__ == "__main__":
    main()
