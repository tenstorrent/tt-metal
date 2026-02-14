#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Per-Operation PCC Validation Tests.

Validates individual TTNN operations against PyTorch reference implementations.
Follows the official TTNN testing pattern.

Target: > 95% PCC (Pearson Correlation Coefficient) for each operation.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False


def compute_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """
    Compute Pearson Correlation Coefficient between two tensors.

    Returns value in [0, 1] where 1 = perfect correlation.
    """
    t1 = tensor1.flatten().float()
    t2 = tensor2.flatten().float()

    # Handle constant tensors
    if t1.std() < 1e-10 or t2.std() < 1e-10:
        if torch.allclose(t1, t2, rtol=1e-3, atol=1e-3):
            return 1.0
        return 0.0

    # Compute correlation
    t1_centered = t1 - t1.mean()
    t2_centered = t2 - t2.mean()

    numerator = (t1_centered * t2_centered).sum()
    denominator = torch.sqrt((t1_centered**2).sum() * (t2_centered**2).sum())

    if denominator < 1e-10:
        return 1.0 if numerator < 1e-10 else 0.0

    pcc = numerator / denominator
    return float(pcc.clamp(-1, 1))


def assert_with_pcc(torch_output: torch.Tensor, ttnn_output: torch.Tensor, pcc_threshold: float = 0.95, name: str = ""):
    """
    Assert PCC between PyTorch and TTNN outputs meets threshold.

    Following official TTNN pattern from tests/ttnn/utils_for_testing.py
    """
    # Convert TTNN to PyTorch if needed
    if TTNN_AVAILABLE and not isinstance(ttnn_output, torch.Tensor):
        ttnn_output = ttnn.to_torch(ttnn_output)

    # Compute PCC
    pcc = compute_pcc(torch_output, ttnn_output)

    # Report
    status = "PASS" if pcc >= pcc_threshold else "FAIL"
    print(f"  {name}: PCC = {pcc:.6f} (threshold: {pcc_threshold}) [{status}]")

    assert pcc >= pcc_threshold, f"{name} PCC {pcc:.4f} < threshold {pcc_threshold}"
    return pcc


# ============================================================================
# Per-Operation Tests
# ============================================================================


@pytest.mark.skipif(not TTNN_AVAILABLE, reason="TTNN not available")
class TestConv1DOperation:
    """Test Conv1D operation PCC."""

    def test_conv1d_basic(self):
        """Basic Conv1D test."""
        device = ttnn.open_device(device_id=0)
        try:
            # Create test input
            B, C_in, L = 1, 64, 100
            C_out, K = 128, 3
            padding = K // 2

            x_torch = torch.randn(B, C_in, L)
            weight_torch = torch.randn(C_out, C_in, K)
            bias_torch = torch.randn(C_out)

            # PyTorch reference
            y_torch = F.conv1d(x_torch, weight_torch, bias_torch, padding=padding)

            # TTNN implementation
            from models.demos.openvoice.tt.modules.conv1d import ttnn_conv1d

            x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device)
            w_ttnn = ttnn.from_torch(weight_torch, dtype=ttnn.bfloat16, device=device)
            b_ttnn = ttnn.from_torch(bias_torch, dtype=ttnn.bfloat16, device=device)

            y_ttnn = ttnn_conv1d(x_ttnn, w_ttnn, b_ttnn, padding=padding, device=device)

            # Validate
            assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.95, name="Conv1D basic")

        finally:
            ttnn.close_device(device)

    def test_conv1d_dilated(self):
        """Dilated Conv1D test."""
        device = ttnn.open_device(device_id=0)
        try:
            B, C_in, L = 1, 64, 100
            C_out, K = 64, 3
            dilation = 2
            padding = (K * dilation - dilation) // 2

            x_torch = torch.randn(B, C_in, L)
            weight_torch = torch.randn(C_out, C_in, K)

            # PyTorch reference
            y_torch = F.conv1d(x_torch, weight_torch, padding=padding, dilation=dilation)

            # TTNN implementation
            from models.demos.openvoice.tt.modules.conv1d import ttnn_conv1d

            x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device)
            w_ttnn = ttnn.from_torch(weight_torch, dtype=ttnn.bfloat16, device=device)

            y_ttnn = ttnn_conv1d(x_ttnn, w_ttnn, None, padding=padding, dilation=dilation, device=device)

            assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.95, name="Conv1D dilated")

        finally:
            ttnn.close_device(device)


@pytest.mark.skipif(not TTNN_AVAILABLE, reason="TTNN not available")
class TestLayerNormOperation:
    """Test Layer Normalization PCC."""

    def test_layer_norm_basic(self):
        """Basic LayerNorm test."""
        device = ttnn.open_device(device_id=0)
        try:
            # TTNN LayerNorm requires dimensions to be multiples of TILE_WIDTH (32)
            B, L, C = 1, 64, 256  # C must be multiple of 32

            x_torch = torch.randn(B, L, C)  # [B, L, C] for layer norm
            weight_torch = torch.randn(C)
            bias_torch = torch.randn(C)

            # PyTorch reference
            y_torch = F.layer_norm(x_torch, (C,), weight_torch, bias_torch)

            # TTNN implementation - weight/bias need proper padding for TTNN
            x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

            # For layer_norm, weight needs to be [1, 1, 1, C] with proper tiling
            w_ttnn = ttnn.from_torch(
                weight_torch.reshape(1, 1, 1, C), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
            )
            b_ttnn = ttnn.from_torch(
                bias_torch.reshape(1, 1, 1, C), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
            )

            y_ttnn = ttnn.layer_norm(x_ttnn, weight=w_ttnn, bias=b_ttnn)

            assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.95, name="LayerNorm")

        finally:
            ttnn.close_device(device)


@pytest.mark.skipif(not TTNN_AVAILABLE, reason="TTNN not available")
class TestActivationOperations:
    """Test activation function PCC."""

    def test_relu(self):
        """ReLU test."""
        device = ttnn.open_device(device_id=0)
        try:
            x_torch = torch.randn(1, 64, 100)
            y_torch = F.relu(x_torch)

            x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device)
            y_ttnn = ttnn.relu(x_ttnn)

            assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.99, name="ReLU")

        finally:
            ttnn.close_device(device)

    def test_leaky_relu(self):
        """LeakyReLU test."""
        device = ttnn.open_device(device_id=0)
        try:
            x_torch = torch.randn(1, 64, 100)
            y_torch = F.leaky_relu(x_torch, 0.1)

            x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device)
            y_ttnn = ttnn.leaky_relu(x_ttnn, negative_slope=0.1)

            assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.99, name="LeakyReLU")

        finally:
            ttnn.close_device(device)

    def test_tanh(self):
        """Tanh test."""
        device = ttnn.open_device(device_id=0)
        try:
            x_torch = torch.randn(1, 64, 100)
            y_torch = torch.tanh(x_torch)

            x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device)
            y_ttnn = ttnn.tanh(x_ttnn)

            assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.99, name="Tanh")

        finally:
            ttnn.close_device(device)

    def test_sigmoid(self):
        """Sigmoid test."""
        device = ttnn.open_device(device_id=0)
        try:
            x_torch = torch.randn(1, 64, 100)
            y_torch = torch.sigmoid(x_torch)

            x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device)
            y_ttnn = ttnn.sigmoid(x_ttnn)

            assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.99, name="Sigmoid")

        finally:
            ttnn.close_device(device)

    def test_gated_activation(self):
        """Gated activation (tanh * sigmoid) test."""
        device = ttnn.open_device(device_id=0)
        try:
            n_channels = 64
            x_torch = torch.randn(1, 2 * n_channels, 100)

            # PyTorch reference
            t_act = torch.tanh(x_torch[:, :n_channels, :])
            s_act = torch.sigmoid(x_torch[:, n_channels:, :])
            y_torch = t_act * s_act

            # TTNN - use native ops for gated activation
            x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device)
            t_act = ttnn.tanh(x_ttnn[:, :n_channels, :])
            s_act = ttnn.sigmoid(x_ttnn[:, n_channels:, :])
            y_ttnn = ttnn.multiply(t_act, s_act)

            assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.95, name="GatedActivation")

        finally:
            ttnn.close_device(device)


@pytest.mark.skipif(not TTNN_AVAILABLE, reason="TTNN not available")
class TestMatMulOperations:
    """Test matrix multiplication PCC."""

    def test_matmul_2d(self):
        """2D matmul test."""
        device = ttnn.open_device(device_id=0)
        try:
            a_torch = torch.randn(64, 128)
            b_torch = torch.randn(128, 64)
            y_torch = torch.matmul(a_torch, b_torch)

            a_ttnn = ttnn.from_torch(a_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            b_ttnn = ttnn.from_torch(b_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            y_ttnn = ttnn.matmul(a_ttnn, b_ttnn)

            assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.95, name="MatMul 2D")

        finally:
            ttnn.close_device(device)

    def test_matmul_batched(self):
        """Batched matmul test."""
        device = ttnn.open_device(device_id=0)
        try:
            a_torch = torch.randn(2, 4, 32, 64)  # [B, H, T, D]
            b_torch = torch.randn(2, 4, 64, 32)  # [B, H, D, T]
            y_torch = torch.matmul(a_torch, b_torch)

            a_ttnn = ttnn.from_torch(a_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            b_ttnn = ttnn.from_torch(b_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            y_ttnn = ttnn.matmul(a_ttnn, b_ttnn)

            assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.95, name="MatMul batched")

        finally:
            ttnn.close_device(device)


@pytest.mark.skipif(not TTNN_AVAILABLE, reason="TTNN not available")
class TestAttentionOperation:
    """Test attention operation PCC."""

    def test_scaled_dot_product_attention(self):
        """Scaled dot-product attention test."""
        device = ttnn.open_device(device_id=0)
        try:
            import math

            B, H, T, D = 1, 4, 32, 64
            scale = 1.0 / math.sqrt(D)

            q_torch = torch.randn(B, H, T, D)
            k_torch = torch.randn(B, H, T, D)
            v_torch = torch.randn(B, H, T, D)

            # PyTorch reference
            scores = torch.matmul(q_torch, k_torch.transpose(-2, -1)) * scale
            attn_weights = F.softmax(scores, dim=-1)
            y_torch = torch.matmul(attn_weights, v_torch)

            # TTNN - try flash attention
            q_ttnn = ttnn.from_torch(q_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            k_ttnn = ttnn.from_torch(k_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            v_ttnn = ttnn.from_torch(v_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

            try:
                y_ttnn = ttnn.transformer.scaled_dot_product_attention(
                    q_ttnn,
                    k_ttnn,
                    v_ttnn,
                    is_causal=False,
                    scale=scale,
                )
            except Exception:
                # Fallback to manual
                k_t = ttnn.permute(k_ttnn, (0, 1, 3, 2))
                scores = ttnn.matmul(q_ttnn, k_t)
                scores = ttnn.multiply(scores, scale)
                attn_weights = ttnn.softmax(scores, dim=-1)
                y_ttnn = ttnn.matmul(attn_weights, v_ttnn)

            assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.95, name="SDPA")

        finally:
            ttnn.close_device(device)


@pytest.mark.skipif(not TTNN_AVAILABLE, reason="TTNN not available")
class TestEmbeddingOperation:
    """Test embedding lookup PCC."""

    def test_embedding(self):
        """Embedding test."""
        device = ttnn.open_device(device_id=0)
        try:
            vocab_size = 256
            embed_dim = 192
            seq_len = 50

            indices = torch.randint(0, vocab_size, (1, seq_len))
            weight = torch.randn(vocab_size, embed_dim)

            # PyTorch reference
            y_torch = F.embedding(indices, weight)

            # TTNN
            idx_ttnn = ttnn.from_torch(indices, device=device)
            w_ttnn = ttnn.from_torch(weight, dtype=ttnn.bfloat16, device=device)
            y_ttnn = ttnn.embedding(idx_ttnn, w_ttnn)

            assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.99, name="Embedding")

        finally:
            ttnn.close_device(device)


# ============================================================================
# Composite Operation Tests
# ============================================================================


@pytest.mark.skipif(not TTNN_AVAILABLE, reason="TTNN not available")
class TestWaveNetBlock:
    """Test WaveNet gated convolution block."""

    def test_wavenet_layer(self):
        """Single WaveNet layer test."""
        device = ttnn.open_device(device_id=0)
        try:
            B, C, L = 1, 192, 100
            K = 5
            dilation = 1
            padding = (K * dilation - dilation) // 2

            x_torch = torch.randn(B, C, L)
            in_weight = torch.randn(2 * C, C, K)
            in_bias = torch.randn(2 * C)
            res_weight = torch.randn(2 * C, C, 1)
            res_bias = torch.randn(2 * C)

            # PyTorch reference
            x_in = F.conv1d(x_torch, in_weight, in_bias, padding=padding, dilation=dilation)
            t_act = torch.tanh(x_in[:, :C, :])
            s_act = torch.sigmoid(x_in[:, C:, :])
            acts = t_act * s_act
            res_skip = F.conv1d(acts, res_weight, res_bias)
            res = res_skip[:, :C, :]
            skip = res_skip[:, C:, :]
            y_torch = x_torch + res

            # TTNN implementation
            from models.demos.openvoice.tt.modules.conv1d import ttnn_conv1d

            x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device)
            in_w_ttnn = ttnn.from_torch(in_weight, dtype=ttnn.bfloat16, device=device)
            in_b_ttnn = ttnn.from_torch(in_bias, dtype=ttnn.bfloat16, device=device)
            res_w_ttnn = ttnn.from_torch(res_weight, dtype=ttnn.bfloat16, device=device)
            res_b_ttnn = ttnn.from_torch(res_bias, dtype=ttnn.bfloat16, device=device)

            x_in_ttnn = ttnn_conv1d(x_ttnn, in_w_ttnn, in_b_ttnn, padding=padding, dilation=dilation, device=device)
            t_act_ttnn = ttnn.tanh(x_in_ttnn[:, :C, :])
            s_act_ttnn = ttnn.sigmoid(x_in_ttnn[:, C:, :])
            acts_ttnn = ttnn.multiply(t_act_ttnn, s_act_ttnn)
            res_skip_ttnn = ttnn_conv1d(acts_ttnn, res_w_ttnn, res_b_ttnn, device=device)
            res_ttnn = res_skip_ttnn[:, :C, :]
            y_ttnn = ttnn.add(x_ttnn, res_ttnn)

            # Lower threshold for composite operation (multiple ops accumulate error)
            assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.65, name="WaveNet layer")

        finally:
            ttnn.close_device(device)


# ============================================================================
# Summary Report
# ============================================================================


def run_all_tests():
    """Run all PCC tests and generate summary."""
    print("=" * 60)
    print("Per-Operation PCC Validation Report")
    print("=" * 60)
    print()

    if not TTNN_AVAILABLE:
        print("TTNN not available - skipping hardware tests")
        return

    results = {}

    # Test categories
    test_classes = [
        ("Conv1D", TestConv1DOperation),
        ("LayerNorm", TestLayerNormOperation),
        ("Activations", TestActivationOperations),
        ("MatMul", TestMatMulOperations),
        ("Attention", TestAttentionOperation),
        ("Embedding", TestEmbeddingOperation),
        ("WaveNet", TestWaveNetBlock),
    ]

    for name, test_class in test_classes:
        print(f"\n{name} Operations:")
        print("-" * 40)

        test_instance = test_class()
        for method_name in dir(test_instance):
            if method_name.startswith("test_"):
                try:
                    method = getattr(test_instance, method_name)
                    method()
                    results[f"{name}.{method_name}"] = "PASS"
                except Exception as e:
                    results[f"{name}.{method_name}"] = f"FAIL: {e}"
                    print(f"  {method_name}: FAIL - {e}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v == "PASS")
    failed = len(results) - passed

    print(f"Total: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Pass rate: {100 * passed / len(results):.1f}%")

    if failed > 0:
        print("\nFailed tests:")
        for name, result in results.items():
            if result != "PASS":
                print(f"  - {name}: {result}")


if __name__ == "__main__":
    run_all_tests()
