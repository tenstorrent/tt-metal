#!/usr/bin/env python3
"""
Example usage of the validation decorator system.

This demonstrates how to use @validate_against to compare TTNN implementations
against reference PyTorch implementations with automatic metrics collection.
"""

import os

import torch
import ttnn

from tt_transformers_v2.src.testing import (
    device_validate_against,
    enable_validation,
    get_validation_registry,
    host_validate_against,
    to_torch_auto_compose,
)

# ============================================================================
# Example 1: Validating RMSNorm against PyTorch reference
# ============================================================================


def torch_rms_norm(x, weight, eps=1e-6):
    """Reference PyTorch implementation of RMS normalization"""
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x


class HostValidatedRMSNorm:
    """RMS Normalization with validation decorator using old input_map pattern"""

    def __init__(self, weight: torch.Tensor, eps: float, device):
        self.eps = eps
        self.weight = ttnn.from_torch(
            weight.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    @host_validate_against(
        reference_fn=torch_rms_norm,
        input_to_torch=lambda args, kwargs: (
            (
                to_torch_auto_compose(args[1]),
                to_torch_auto_compose(args[0].weight),
            ),
            {},
        ),
        tolerances={
            "max_abs_error": 1e-2,
            "mean_abs_error": 1e-3,
            "pcc": 0.99,
        },
    )
    def __call__(self, x):
        # x shape: [1, seq_len, hidden_size]
        x_squared = ttnn.mul(x, x)
        mean_x_squared = ttnn.mean(x_squared, dim=-1, keepdim=True)
        rms = ttnn.sqrt(ttnn.add(mean_x_squared, self.eps))
        x_normed = ttnn.mul(x, ttnn.reciprocal(rms))
        return ttnn.mul(x_normed, self.weight)


class DeviceValidatedRMSNorm:
    """RMS Normalization - ultra-clean pattern: NO conversions needed!"""

    def __init__(self, weight: torch.Tensor, eps: float, device):
        self.eps = eps
        self.weight_torch = weight  # Keep for reference
        self.weight = ttnn.from_torch(
            weight.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.device = device

    def _reference_impl(self, x):
        """Reference implementation - mocking a TTNN reference implementation for testing"""
        # Convert TTNN to torch for reference computation
        x_torch = ttnn.to_torch(x).squeeze(0)
        result_torch = torch_rms_norm(x_torch, self.weight_torch, self.eps)
        # Convert back to TTNN to match __call__ output type
        return ttnn.from_torch(
            result_torch.unsqueeze(0), device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    @device_validate_against(
        reference_fn=lambda self, x: self._reference_impl(x),
        tolerances={
            "max_abs_error": 1e-2,
            "mean_abs_error": 1e-3,
            "pcc": 0.99,
        },
    )
    def __call__(self, x):
        # x shape: [1, seq_len, hidden_size]
        x_squared = ttnn.mul(x, x)
        mean_x_squared = ttnn.mean(x_squared, dim=-1, keepdim=True)
        rms = ttnn.sqrt(ttnn.add(mean_x_squared, self.eps))
        x_normed = ttnn.mul(x, ttnn.reciprocal(rms))
        return ttnn.mul(x_normed, self.weight)


# ============================================================================
# Example 2: Validating matrix multiplication
# ============================================================================


@host_validate_against(
    reference_fn=torch.matmul,
    tolerances={
        "max_abs_error": 1e-1,
        "pcc": 0.99,
    },
)
def ttnn_matmul(a, b):
    """TTNN matrix multiplication with validation"""
    return ttnn.matmul(a, b)


# make a test case to show how to directly use auto_compose to convert ttnn to torch
@host_validate_against(
    reference_fn=torch.matmul,
    input_to_torch=lambda args, kwargs: ((to_torch_auto_compose(args[1]), to_torch_auto_compose(args[0])), {}),
    tolerances={
        "max_abs_error": 1e-1,
        "pcc": 0.99,
    },
)
def ttnn_matmul_reverse(a, b):
    """TTNN matrix multiplication with validation"""
    return ttnn.matmul(b, a)


# ============================================================================
# Example 3: Custom metrics and complex mappings
# ============================================================================


def custom_attention_reference(q, k, v, scale):
    """Reference attention computation"""
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v)


@host_validate_against(
    reference_fn=custom_attention_reference,
    input_to_torch=lambda args, kwargs: (
        (
            ttnn.to_torch(args[0]).squeeze(0),
            ttnn.to_torch(args[1]).squeeze(0),
            ttnn.to_torch(args[2]).squeeze(0),
            args[3],
        ),
        {},
    ),
    output_to_torch=lambda x: ttnn.to_torch(x).squeeze(0),
    metrics={
        "max_abs_error": lambda impl, ref: (impl - ref).abs().max().item(),
        "mean_abs_error": lambda impl, ref: (impl - ref).abs().mean().item(),
        "pearson_correlation": lambda impl, ref: torch.corrcoef(torch.stack([impl.flatten(), ref.flatten()]))[
            0, 1
        ].item(),
    },
    tolerances={
        "max_abs_error": 0.1,
        "mean_abs_error": 0.01,
        "pcc": 0.99,
    },
)
def ttnn_attention(q, k, v, scale):
    """Simplified attention with validation"""
    scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))
    scores = ttnn.mul(scores, scale)
    attn_weights = ttnn.softmax(scores, dim=-1)
    return ttnn.matmul(attn_weights, v)


# ============================================================================
# Demo function
# ============================================================================


def demo():
    """Demonstrate validation decorator usage"""

    print("Validation Decorator System Demo")
    print("=" * 80)

    # Setup TTNN device
    if os.environ.get("MESH_DEVICE") == "N150":
        device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape([1, 1]))
    elif os.environ.get("MESH_DEVICE") == "N300":
        device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape([1, 2]))
    else:
        device_ids = ttnn.get_device_ids()
        num_devices = len(device_ids)
        if num_devices >= 1:
            device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape([1, 1]))
        else:
            raise RuntimeError("No devices found")

    print(f"Using {device.get_num_devices()} device(s)")
    print()

    # Example 1: RMSNorm validation
    print("\n1. Testing RMSNorm validation...")
    hidden_size = 512
    batch_size = 1
    seq_len = 32

    weight = torch.randn(hidden_size).to(torch.bfloat16)
    rms_norm = DeviceValidatedRMSNorm(weight, eps=1e-6, device=device)

    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(x.unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    output = rms_norm(x_tt)
    print("   Device-Validated RMSNorm validation complete")

    rms_norm = HostValidatedRMSNorm(weight, eps=1e-6, device=device)

    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(x.unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Invoke and check registry
    _ = rms_norm(x_tt)
    print("   Host-Validated RMSNorm validation complete")

    # Example 2: Matrix multiplication validation
    print("\n2. Testing matrix multiplication validation...")
    m, n, k = 32, 64, 48

    a = torch.randn(1, m, k, dtype=torch.bfloat16)
    b = torch.randn(1, k, n, dtype=torch.bfloat16)

    a_tt = ttnn.from_torch(a.unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b_tt = ttnn.from_torch(b.unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    output = ttnn_matmul(a_tt, b_tt)
    output_reverse = ttnn_matmul_reverse(b_tt, a_tt)
    print("   Matmul validation complete")

    # Demonstrate enabling/disabling validation
    registry = get_validation_registry()
    print(f"   Validation registry: {len(registry.results)} results")

    print("\n3. Testing validation control...")
    enable_validation(False)
    print("   Validation disabled - functions run without validation")

    a_tt = ttnn.from_torch(a.unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b_tt = ttnn.from_torch(b.unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    output = ttnn_matmul(a_tt, b_tt)
    print(f"   No new validations recorded: {len(registry.results)} total")

    enable_validation(True)
    print("   Validation re-enabled")

    # Print validation report
    print("\n")
    registry.print_report()

    # Cleanup
    ttnn.close_mesh_device(device)


if __name__ == "__main__":
    demo()
