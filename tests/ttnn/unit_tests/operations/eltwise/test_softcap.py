# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


def torch_softcap(x, cap):
    """Reference implementation: cap * tanh(x / cap)"""
    return cap * torch.tanh(x / cap)


@pytest.mark.parametrize("cap", [1.0, 10.0, 50.0, 100.0])
@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
        [1, 1, 64, 64],
        [1, 1, 128, 128],
    ],
)
class TestSoftcap:
    def test_softcap_bfloat16(self, shape, cap, device):
        """Test softcap on bfloat16."""
        torch.manual_seed(42)
        torch_input = torch.randn(shape, dtype=torch.bfloat16) * cap * 2

        input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.softcap(input_tensor, cap=cap)
        output_torch = ttnn.to_torch(output_tensor)

        expected = torch_softcap(torch_input.float(), cap).to(torch.bfloat16)

        # Piecewise sigmoid approach gives ~0.035 max absolute error
        assert torch.allclose(output_torch, expected, atol=0.1, rtol=0.1), (
            f"softcap bfloat16 failed for cap={cap}, shape={shape}\n"
            f"max diff: {(output_torch.float() - expected.float()).abs().max().item()}"
        )

        # Check PCC (Pearson Correlation Coefficient)
        pcc = torch.corrcoef(torch.stack([output_torch.float().flatten(), expected.float().flatten()]))[0, 1].item()
        assert pcc > 0.998, f"softcap bfloat16 PCC too low: {pcc:.6f}"

    def test_softcap_float32(self, shape, cap, device):
        """Test softcap on float32."""
        torch.manual_seed(42)
        torch_input = torch.randn(shape, dtype=torch.float32) * cap * 2

        input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
        output_tensor = ttnn.softcap(input_tensor, cap=cap)
        output_torch = ttnn.to_torch(output_tensor)

        expected = torch_softcap(torch_input, cap)

        assert torch.allclose(output_torch, expected, atol=0.1, rtol=0.1), (
            f"softcap float32 failed for cap={cap}, shape={shape}\n"
            f"max diff: {(output_torch - expected).abs().max().item()}"
        )

        # Check PCC
        pcc = torch.corrcoef(torch.stack([output_torch.flatten(), expected.flatten()]))[0, 1].item()
        assert pcc > 0.998, f"softcap float32 PCC too low: {pcc:.6f}"


@pytest.mark.parametrize("cap", [50.0])
class TestSoftcapEdgeCases:
    def test_softcap_zeros(self, cap, device):
        """softcap(0, cap) = 0"""
        torch_input = torch.zeros([1, 1, 32, 32], dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.softcap(input_tensor, cap=cap)
        output_torch = ttnn.to_torch(output_tensor)
        assert torch.allclose(output_torch.float(), torch.zeros_like(output_torch.float()), atol=0.01)

    def test_softcap_saturation(self, cap, device):
        """For large |x|, softcap should saturate to +/-cap"""
        torch_input = torch.full([1, 1, 32, 32], cap * 100, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.softcap(input_tensor, cap=cap)
        output_torch = ttnn.to_torch(output_tensor)
        expected = torch.full_like(output_torch, cap)
        assert torch.allclose(output_torch.float(), expected.float(), atol=0.5)

    def test_softcap_negative_saturation(self, cap, device):
        """For large negative |x|, softcap should saturate to -cap"""
        torch_input = torch.full([1, 1, 32, 32], -cap * 100, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.softcap(input_tensor, cap=cap)
        output_torch = ttnn.to_torch(output_tensor)
        expected = torch.full_like(output_torch, -cap)
        assert torch.allclose(output_torch.float(), expected.float(), atol=0.5)
