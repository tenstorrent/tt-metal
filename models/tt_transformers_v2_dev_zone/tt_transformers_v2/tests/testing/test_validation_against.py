"""
These tests demonstrate how to use @validate_against to compare TTNN implementations
against reference PyTorch implementations with automatic metrics collection.
"""

import pytest
import torch
from tt_transformers_v2.src.testing import (
    clear_validation_results,
    device_validate_against,
    enable_validation,
    get_validation_registry,
    host_validate_against,
    to_torch_auto_compose,
)

import ttnn

# [INFO] the purpose of this test is to validate the validation framework itself,
# which does not care about the mesh shape or tensor layout; we have other test files on those topics.
pytestmark = [
    pytest.mark.parametrize(
        "ttnn_mesh_device",
        [
            (1, 1),  # single device # [INFO] apply auto_compose on single device would incur error in c++ code
        ],
        ids=[
            "1x1",
        ],
        indirect=True,
    ),
]

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
        input_to_torch=lambda self, x: (
            # [INFO] produce input args to torch_rms_norm as a tuple
            (to_torch_auto_compose(x), to_torch_auto_compose(self.weight)),
            # [INFO] produce input kwargs to torch_rms_norm as a dict
            {"eps": self.eps},
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

    # [INFO] this decorator is useful when the reference function is a TTNN-native function.
    #        currently, it is experimental and requires the reference function has same-ordered
    #        inputs as the decorated function.
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


def test_validation_rmsnorm_host_and_device(ttnn_mesh_device: ttnn.MeshDevice):
    registry = get_validation_registry()

    hidden_size = 64
    batch_size = 1
    seq_len = 8

    weight = torch.randn(hidden_size, dtype=torch.bfloat16)

    # Device-validated RMSNorm
    rms_device = DeviceValidatedRMSNorm(weight, eps=1e-6, device=ttnn_mesh_device)
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(x.unsqueeze(0), device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    _ = rms_device(x_tt)

    # Host-validated RMSNorm
    rms_host = HostValidatedRMSNorm(weight, eps=1e-6, device=ttnn_mesh_device)
    x2 = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    x2_tt = ttnn.from_torch(x2.unsqueeze(0), device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    _ = rms_host(x2_tt)

    assert len(registry.results) >= 2
    # Expect the last two validations to fail max_abs_error and mean_abs_error checks (known issue??)
    assert not registry.results[-1].metrics["max_abs_error"].passed
    assert not registry.results[-2].metrics["max_abs_error"].passed
    assert not registry.results[-1].metrics["mean_abs_error"].passed
    assert not registry.results[-2].metrics["mean_abs_error"].passed
    # Expect the last two validations to pass pcc check
    assert registry.results[-1].metrics["pcc"].passed
    assert registry.results[-2].metrics["pcc"].passed


# ============================================================================
# Example 2: Validating matrix multiplication
# ============================================================================


@host_validate_against(
    reference_fn=torch.matmul,
    # [INFO] when reference function accepts inputs in the same order as the decorated function,
    #        we can omit input_to_torch; the mapping will be inferred automatically.
    tolerances={
        "max_abs_error": 1.5e-1,
        "pcc": 0.99,
    },
)
def ttnn_matmul(a, b):
    """TTNN matrix multiplication with validation"""
    return ttnn.matmul(a, b)


# make a test case to show how to directly use auto_compose to convert ttnn to torch
@host_validate_against(
    reference_fn=torch.matmul,
    # [INFO] this is a simple example of input remapping.
    input_to_torch=lambda a, b: (to_torch_auto_compose(b), to_torch_auto_compose(a)),
    tolerances={
        "max_abs_error": 1.5e-1,
        "pcc": 0.99,
    },
)
def ttnn_matmul_reverse(a, b):
    """TTNN matrix multiplication with validation"""
    return ttnn.matmul(b, a)


def test_validation_matmul(ttnn_mesh_device: ttnn.MeshDevice):
    registry = get_validation_registry()

    m, n, k = 16, 24, 12
    a = torch.randn(1, m, k, dtype=torch.bfloat16)
    b = torch.randn(1, k, n, dtype=torch.bfloat16)

    a_tt = ttnn.from_torch(a.unsqueeze(0), device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b_tt = ttnn.from_torch(b.unsqueeze(0), device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    _ = ttnn_matmul(a_tt, b_tt)
    _ = ttnn_matmul_reverse(b_tt, a_tt)

    # Expect two validations recorded and both passed
    assert len(registry.results) >= 2
    assert registry.results[-1].passed
    assert registry.results[-2].passed


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
    # [INFO]{ when reference function accepts inputs in the same order as the decorated function,
    # we can omit input_to_torch; it will be inferred automatically as if the following code were written:
    # input_to_torch=lambda q, k, v, scale: (
    #     to_torch_auto_compose(q),
    #     to_torch_auto_compose(k),
    #     to_torch_auto_compose(v),
    #     scale,
    # ),
    # [INFO]}
    tolerances={
        "max_abs_error": 0.1,
        "mean_abs_error": 0.02,
        "pcc": 0.99,
    },
)
def ttnn_attention(q, k, v, scale):
    """Simplified attention with validation"""
    scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))
    scores = ttnn.mul(scores, scale)
    attn_weights = ttnn.softmax(scores, dim=-1)
    return ttnn.matmul(attn_weights, v)


def test_validation_attention(ttnn_mesh_device: ttnn.MeshDevice):
    m, n, dk, dv = 8, 8, 16, 16
    q = torch.randn(1, m, dk, dtype=torch.bfloat16)
    k = torch.randn(1, n, dk, dtype=torch.bfloat16)
    v = torch.randn(1, n, dv, dtype=torch.bfloat16)

    q_tt = ttnn.from_torch(q.unsqueeze(0), device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    k_tt = ttnn.from_torch(k.unsqueeze(0), device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    v_tt = ttnn.from_torch(v.unsqueeze(0), device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    registry = get_validation_registry()
    before = len(registry.results)
    scale = 1.0 / (dk**0.5)
    _ = ttnn_attention(q_tt, k_tt, v_tt, scale)
    assert len(registry.results) == before + 1
    test_result = registry.results[-1]
    # expect the test to pass max_abs_error and pcc checks
    assert test_result.metrics["max_abs_error"].passed
    assert test_result.metrics["pcc"].passed
    assert test_result.metrics["mean_abs_error"].passed


# ============================================================================
# Additional test functions
# ============================================================================


def test_validation_enable_disable(ttnn_mesh_device: ttnn.MeshDevice):
    a = torch.randn(1, 8, 8, dtype=torch.bfloat16)
    b = torch.randn(1, 8, 8, dtype=torch.bfloat16)
    a_tt = ttnn.from_torch(a.unsqueeze(0), device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b_tt = ttnn.from_torch(b.unsqueeze(0), device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    registry = get_validation_registry()
    recorded_after_enable = len(registry.results)
    # Disable validation: should not record
    enable_validation(False)
    _ = ttnn_matmul(a_tt, b_tt)
    assert len(registry.results) == recorded_after_enable

    # Re-enable for subsequent tests
    enable_validation(True)


def test_validation_non_decorator_host(ttnn_mesh_device: ttnn.MeshDevice):
    registry = get_validation_registry()
    before = len(registry.results)

    m, n, k = 8, 10, 6
    a = torch.randn(1, m, k, dtype=torch.bfloat16)
    b = torch.randn(1, k, n, dtype=torch.bfloat16)
    a_tt = ttnn.from_torch(a.unsqueeze(0), device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b_tt = ttnn.from_torch(b.unsqueeze(0), device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def _matmul(a, b):
        return ttnn.matmul(a, b)

    validated_matmul = host_validate_against(
        reference_fn=torch.matmul,
        tolerances={
            "max_abs_error": 1.5e-1,
            "pcc": 0.99,
        },
    )(_matmul)

    _ = validated_matmul(a_tt, b_tt)

    assert len(registry.results) == before + 1
    assert registry.results[-1].passed


@pytest.fixture(scope="module", autouse=True)
def _print_validation_report_after_module(request):
    # Runs once after all tests in this module finish
    yield
    registry = get_validation_registry()
    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    reporter.write_line("Printing validation report after yield")
    registry.print_report()


@pytest.fixture(scope="module", autouse=True)
def _clear_validation_results_before_module():
    clear_validation_results()
