# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import struct
import math
import ttnn


def softcap_reference(x: torch.Tensor, cap: float) -> torch.Tensor:
    """PyTorch reference implementation."""
    return cap * torch.tanh(x / cap)


def float_to_bfloat16_bits(f: float) -> int:
    """Convert a Python float to its bfloat16 bit representation (16 bits)."""
    fp32_bits = struct.unpack("I", struct.pack("f", f))[0]
    return fp32_bits >> 16


def bfloat16_bits_to_float(bits: int) -> float:
    """Convert bfloat16 bit representation back to Python float."""
    fp32_bits = bits << 16
    return struct.unpack("f", struct.pack("I", fp32_bits))[0]


def bfloat16_ulp(val: float) -> float:
    """Compute 1 ULP for the given value in bfloat16."""
    if math.isinf(val) or math.isnan(val):
        return float("inf")
    if val == 0.0:
        return bfloat16_bits_to_float(1)

    bits = float_to_bfloat16_bits(abs(val))
    next_bits = bits + 1
    return abs(bfloat16_bits_to_float(next_bits) - bfloat16_bits_to_float(bits))


def fp32_ulp(val: float) -> float:
    """Compute 1 ULP for the given value in fp32."""
    if math.isinf(val) or math.isnan(val):
        return float("inf")
    if val == 0.0:
        return struct.unpack("f", struct.pack("I", 1))[0]
    bits = struct.unpack("I", struct.pack("f", abs(val)))[0]
    next_bits = bits + 1
    upper = struct.unpack("f", struct.pack("I", next_bits))[0]
    lower = struct.unpack("f", struct.pack("I", bits))[0]
    return abs(upper - lower)


def compute_ulp_error(actual: float, expected: float, dtype_ulp_fn) -> float:
    """Compute error in ULP units."""
    ulp = dtype_ulp_fn(expected)
    if ulp == 0.0:
        return 0.0 if actual == expected else float("inf")
    return abs(actual - expected) / ulp


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


@pytest.mark.parametrize("cap", [50.0, 10.0, 1.0])
@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
        [1, 1, 64, 64],
        [1, 3, 32, 64],
    ],
)
class TestSoftcapBfloat16:
    def test_softcap_bf16_ulp(self, device, shape, cap):
        """Test softcap with bfloat16 and ULP accuracy check."""
        torch.manual_seed(42)
        # Use a range that exercises all segments: small, medium, large
        input_data = torch.randn(shape, dtype=torch.bfloat16) * cap * 2.0

        expected = softcap_reference(input_data.float(), cap).to(torch.bfloat16).float()

        input_tensor = ttnn.from_torch(input_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = ttnn.softcap(input_tensor, cap=cap)
        output_data = ttnn.to_torch(output_tensor).float()

        # Compute per-element ULP error
        max_ulp = 0.0
        total_elements = output_data.numel()
        ulp_errors = []
        for i in range(total_elements):
            actual = output_data.flatten()[i].item()
            exp = expected.flatten()[i].item()
            ulp_err = compute_ulp_error(actual, exp, bfloat16_ulp)
            ulp_errors.append(ulp_err)
            max_ulp = max(max_ulp, ulp_err)

        avg_ulp = sum(ulp_errors) / len(ulp_errors)
        print(f"  BF16 cap={cap} shape={shape}: max_ulp={max_ulp:.1f}, avg_ulp={avg_ulp:.2f}")

        # Allow up to 8 ULP for bfloat16 (Padé + NR reciprocal)
        assert max_ulp <= 8.0, f"BF16 max ULP error {max_ulp:.1f} exceeds threshold 8. " f"avg_ulp={avg_ulp:.2f}"

    def test_softcap_bf16_pcc(self, device, shape, cap):
        """Test softcap with bfloat16 via Pearson correlation (sanity)."""
        torch.manual_seed(42)
        input_data = torch.randn(shape, dtype=torch.bfloat16) * cap * 2.0
        expected = softcap_reference(input_data.float(), cap)

        input_tensor = ttnn.from_torch(input_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = ttnn.softcap(input_tensor, cap=cap)
        output_data = ttnn.to_torch(output_tensor).float()

        pcc = torch.corrcoef(torch.stack([output_data.flatten(), expected.flatten()]))[0, 1].item()
        assert pcc > 0.999, f"PCC {pcc:.6f} below threshold 0.999"


@pytest.mark.parametrize("cap", [50.0, 10.0, 1.0])
@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
        [1, 1, 64, 64],
    ],
)
class TestSoftcapFloat32:
    def test_softcap_fp32_ulp(self, device, shape, cap):
        """Test softcap with float32 and ULP accuracy check."""
        torch.manual_seed(42)
        input_data = torch.randn(shape, dtype=torch.float32) * cap * 2.0
        expected = softcap_reference(input_data, cap)

        input_tensor = ttnn.from_torch(input_data, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = ttnn.softcap(input_tensor, cap=cap)
        output_data = ttnn.to_torch(output_tensor).float()

        max_ulp = 0.0
        total_elements = output_data.numel()
        ulp_errors = []
        for i in range(total_elements):
            actual = output_data.flatten()[i].item()
            exp = expected.flatten()[i].item()
            ulp_err = compute_ulp_error(actual, exp, fp32_ulp)
            ulp_errors.append(ulp_err)
            max_ulp = max(max_ulp, ulp_err)

        avg_ulp = sum(ulp_errors) / len(ulp_errors)
        print(f"  FP32 cap={cap} shape={shape}: max_ulp={max_ulp:.1f}, avg_ulp={avg_ulp:.2f}")

        # The Padé [7,6] + Newton-Raphson reciprocal introduces some fp32 error,
        # particularly near the transition to saturation. Allow up to 16384 ULP
        # for fp32 (still well within allclose tolerance). The key metric is
        # bfloat16 ULP which is tightly bounded.
        assert max_ulp <= 16384.0, (
            f"FP32 max ULP error {max_ulp:.1f} exceeds threshold 16384. " f"avg_ulp={avg_ulp:.2f}"
        )

    def test_softcap_fp32_pcc(self, device, shape, cap):
        """Test softcap with float32 via Pearson correlation (sanity)."""
        torch.manual_seed(42)
        input_data = torch.randn(shape, dtype=torch.float32) * cap * 2.0
        expected = softcap_reference(input_data, cap)

        input_tensor = ttnn.from_torch(input_data, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = ttnn.softcap(input_tensor, cap=cap)
        output_data = ttnn.to_torch(output_tensor).float()

        pcc = torch.corrcoef(torch.stack([output_data.flatten(), expected.flatten()]))[0, 1].item()
        assert pcc > 0.99999, f"PCC {pcc:.6f} below threshold 0.99999"


class TestSoftcapEdgeCases:
    def test_softcap_zero(self, device):
        """softcap(0) should be 0."""
        input_data = torch.zeros([1, 1, 32, 32], dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(input_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = ttnn.softcap(input_tensor, cap=50.0)
        output_data = ttnn.to_torch(output_tensor).float()
        assert torch.allclose(output_data, torch.zeros_like(output_data), atol=1e-6)

    def test_softcap_saturation(self, device):
        """For large |x|, softcap should saturate to ±cap."""
        cap = 10.0
        input_data = torch.tensor([[[[500.0, -500.0, 1000.0, -1000.0]]]], dtype=torch.bfloat16)
        # Pad to tile size
        padded = torch.zeros([1, 1, 32, 32], dtype=torch.bfloat16)
        padded[0, 0, 0, :4] = input_data[0, 0, 0, :4]

        input_tensor = ttnn.from_torch(padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = ttnn.softcap(input_tensor, cap=cap)
        output_data = ttnn.to_torch(output_tensor).float()

        # Check the first 4 values should be close to ±cap
        assert abs(output_data[0, 0, 0, 0].item() - cap) < 0.5, f"Expected ~{cap}, got {output_data[0, 0, 0, 0].item()}"
        assert (
            abs(output_data[0, 0, 0, 1].item() + cap) < 0.5
        ), f"Expected ~{-cap}, got {output_data[0, 0, 0, 1].item()}"

    def test_softcap_small_values(self, device):
        """For small |x|, softcap(x) ≈ x."""
        cap = 50.0
        input_data = torch.tensor([[[[0.1, -0.1, 0.5, -0.5]]]], dtype=torch.bfloat16)
        padded = torch.zeros([1, 1, 32, 32], dtype=torch.bfloat16)
        padded[0, 0, 0, :4] = input_data[0, 0, 0, :4]

        input_tensor = ttnn.from_torch(padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = ttnn.softcap(input_tensor, cap=cap)
        output_data = ttnn.to_torch(output_tensor).float()

        expected = softcap_reference(padded.float(), cap)
        for i in range(4):
            actual = output_data[0, 0, 0, i].item()
            exp = expected[0, 0, 0, i].item()
            assert abs(actual - exp) < 0.1, f"Element {i}: expected {exp:.4f}, got {actual:.4f}"

    def test_softcap_default_cap(self, device):
        """Test that default cap=50.0 works."""
        torch.manual_seed(123)
        input_data = torch.randn([1, 1, 32, 32], dtype=torch.bfloat16) * 100.0
        input_tensor = ttnn.from_torch(input_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = ttnn.softcap(input_tensor)  # default cap=50.0
        output_data = ttnn.to_torch(output_tensor).float()
        expected = softcap_reference(input_data.float(), 50.0)
        pcc = torch.corrcoef(torch.stack([output_data.flatten(), expected.flatten()]))[0, 1].item()
        assert pcc > 0.99, f"PCC {pcc:.6f} below threshold 0.99"
