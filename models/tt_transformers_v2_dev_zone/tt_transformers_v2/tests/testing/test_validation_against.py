"""
These tests demonstrate how to use @validate_against to compare TTNN implementations
against reference PyTorch implementations with automatic metrics collection.
"""

import pytest
import torch
from tt_transformers_v2.src.testing import (
    Metric,
    MetricSpec,
    clear_validation_results,
    compare_to_torch,
    compare_to_ttnn,
    compute_pcc_host,
    enable_validation,
    get_validation_registry,
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

    @compare_to_torch(
        reference_fn=torch_rms_norm,
        input_to_torch=lambda self, x: (
            # [INFO] produce input args to torch_rms_norm as a tuple
            (to_torch_auto_compose(x), to_torch_auto_compose(self.weight)),
            # [INFO] produce input kwargs to torch_rms_norm as a dict
            {"eps": self.eps},
        ),
        metric_tolerances={
            Metric.MAX_ABS_ERROR: 1e-2,
            Metric.MEAN_ABS_ERROR: 1e-3,
            "pcc": 0.99,  # can use enum or their string values
        },
        enabled=True,
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
    @compare_to_ttnn(
        reference_fn=lambda self, x: self._reference_impl(x),
        # [INFO] passing `metric_tolerances` is optional; if not provided, the default tolerances will be used:
        # metric_tolerances={
        #     Metric.MAX_ABS_ERROR: 1e-2,
        #     Metric.PCC: 0.99,
        # },
    )
    def __call__(self, x):
        # x shape: [1, seq_len, hidden_size]
        x_squared = ttnn.mul(x, x)
        mean_x_squared = ttnn.mean(x_squared, dim=-1, keepdim=True)
        rms = ttnn.sqrt(ttnn.add(mean_x_squared, self.eps))
        x_normed = ttnn.mul(x, ttnn.reciprocal(rms))
        return ttnn.mul(x_normed, self.weight)

    @compare_to_ttnn(
        reference_fn=lambda self, x: self._reference_impl(x),
    )
    def _call_torch__(self, x):
        # copied __call__ code below and converted to torch tensor to mock a function under test that returns a torch tensor
        # x shape: [1, seq_len, hidden_size]
        x_squared = ttnn.mul(x, x)
        mean_x_squared = ttnn.mean(x_squared, dim=-1, keepdim=True)
        rms = ttnn.sqrt(ttnn.add(mean_x_squared, self.eps))
        x_normed = ttnn.mul(x, ttnn.reciprocal(rms))
        return to_torch_auto_compose(ttnn.mul(x_normed, self.weight))


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

    _ = rms_device._call_torch__(x_tt)

    # Host-validated RMSNorm
    rms_host = HostValidatedRMSNorm(weight, eps=1e-6, device=ttnn_mesh_device)
    x2 = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    x2_tt = ttnn.from_torch(x2.unsqueeze(0), device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    _ = rms_host(x2_tt)

    assert len(registry.results) >= 2
    # Expect the last two validations to fail max_abs_error and mean_abs_error checks (known issue??)
    assert not registry.results[0].metrics[Metric.MAX_ABS_ERROR].passed
    assert not registry.results[1].metrics[Metric.MAX_ABS_ERROR].passed
    assert not registry.results[2].metrics[Metric.MAX_ABS_ERROR].passed
    assert not registry.results[2].metrics[Metric.MEAN_ABS_ERROR].passed
    # Expect the last two validations to pass pcc check
    assert registry.results[0].metrics[Metric.PCC].passed
    assert registry.results[1].metrics[Metric.PCC].passed
    assert registry.results[2].metrics[Metric.PCC].passed


# ============================================================================
# Example 2: Validating matrix multiplication
# ============================================================================


@compare_to_torch(
    reference_fn=torch.matmul,
    # [INFO] when reference function accepts inputs in the same order as the decorated function,
    #        we can omit input_to_torch; the mapping will be inferred automatically.
    metric_tolerances={
        Metric.MAX_ABS_ERROR: 1.5e-1,
        Metric.PCC: 0.99,
    },
)
def ttnn_matmul(a, b):
    """TTNN matrix multiplication with validation"""
    return ttnn.matmul(a, b)


# make a test case to show how to directly use auto_compose to convert ttnn to torch
@compare_to_torch(
    reference_fn=torch.matmul,
    # [INFO] this is a simple example of input remapping.
    input_to_torch=lambda a, b: (to_torch_auto_compose(b), to_torch_auto_compose(a)),
    metric_tolerances={
        Metric.MAX_ABS_ERROR: 1.5e-1,
        Metric.PCC: 0.99,
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


# todo)) we need an example where both the reference and the decorated function are methods of a class and they share the same member variables!
# what should we pass to `reference_fn`? the object itself? or a lambda function that takes the object as an argument?
# idea: maybe we can say that set a rule in place that says if the two functions share the same member variables, we can automatically infer the input_to_torch function!
# take a module form the deepseek model as an example!

# todo)) add an example where the reference is a file name that we can load output tensors from!
# idea: validating against tensors -- give us a file name and we can compare our tensor under test against that!
# - a filename for the tensor


@compare_to_torch(
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
    metric_tolerances={
        Metric.MAX_ABS_ERROR: 0.1,
        Metric.MEAN_ABS_ERROR: 0.02,
        Metric.PCC: 0.99,
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
    assert test_result.metrics[Metric.MAX_ABS_ERROR].passed
    assert test_result.metrics[Metric.PCC].passed
    assert test_result.metrics[Metric.MEAN_ABS_ERROR].passed


# ============================================================================
# Example 4: Validating from_torch checkpoint
# ============================================================================


@compare_to_torch(
    reference_fn=lambda tensor, device: tensor,
    output_to_torch=lambda x: to_torch_auto_compose(x),
    metric_tolerances={
        Metric.MAX_ABS_ERROR: 0.015,
        Metric.MEAN_ABS_ERROR: 0.01,
        Metric.PCC: 0.99,
    },
)
def from_torch_checkpoint(tensor: torch.Tensor, device: ttnn.MeshDevice):
    """Return TTNN tensor created via from_torch from a checkpoint tensor."""
    return ttnn.from_torch(tensor, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


def test_validation_checkpoint_from_torch(ttnn_mesh_device: ttnn.MeshDevice):
    registry = get_validation_registry()
    before = len(registry.results)

    # Simulated checkpoint tensor (e.g., a weight matrix)
    rows, cols = 32, 128
    weight = torch.randn(rows, cols, dtype=torch.float32)

    # Validate a direct from_torch call via the decorated function
    _ = from_torch_checkpoint(weight, ttnn_mesh_device)

    # Ensure a result was recorded and it passed
    assert len(registry.results) == before + 1
    assert registry.results[-1].passed


# ============================================================================
# Example 5: Validating with MetricSpec
# ============================================================================


@compare_to_torch(
    reference_fn=torch.matmul,
    metric_tolerances={
        "pcc_host": MetricSpec(tolerance=0.99, higher_is_better=True, compute_fn=compute_pcc_host),
    },
)
def ttnn_matmul_metric_spec(a, b):
    return ttnn.matmul(a, b)


def test_validation_matmul_metric_spec(ttnn_mesh_device: ttnn.MeshDevice):
    registry = get_validation_registry()
    before = len(registry.results)

    m, n, k = 8, 10, 6
    a = torch.randn(1, m, k, dtype=torch.bfloat16)
    b = torch.randn(1, k, n, dtype=torch.bfloat16)

    a_tt = ttnn.from_torch(a.unsqueeze(0), device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b_tt = ttnn.from_torch(b.unsqueeze(0), device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    _ = ttnn_matmul_metric_spec(a_tt, b_tt)

    assert len(registry.results) == before + 1
    assert registry.results[-1].passed


# ============================================================================
# Example 6: Validating with non-decorator use of compare_to_torch
#            between a class instance and a reference class instance!
# ============================================================================


def test_validation_non_decorator_class_vs_class_torch(ttnn_mesh_device: ttnn.MeshDevice):
    """Validate a callable class against a reference class using non-decorator style."""
    registry = get_validation_registry()
    before = len(registry.results)

    # Simple linear layer implemented in TTNN (__call__) vs Torch reference (forward)
    m, n, k = 8, 10, 6
    x = torch.randn(1, m, k, dtype=torch.bfloat16)
    w = torch.randn(1, k, n, dtype=torch.bfloat16)

    class TTLinear:
        def __init__(self, weight: torch.Tensor, device: ttnn.MeshDevice):
            # Weight expected as [1, k, n]; add device batch dim for TTNN tensor
            self.weight = ttnn.from_torch(
                weight.unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )

        def __call__(self, inp):
            return ttnn.matmul(inp, self.weight)

    class TorchLinearRef:
        def __init__(self, weight: torch.Tensor):
            self.weight = weight

        def forward(self, inp: torch.Tensor):
            return torch.matmul(inp, self.weight)

    # Instantiate both implementations
    layer = TTLinear(w, ttnn_mesh_device)
    ref_layer = TorchLinearRef(w)

    # Convert input to TTNN tensor (add device batch dim)
    x_tt = ttnn.from_torch(x.unsqueeze(0), device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Non-decorator usage: wrap the unbound __call__ so we can pass (self, x)
    validated_call = compare_to_torch(
        reference_fn=lambda self, inp: ref_layer.forward(inp),
        metric_tolerances={
            Metric.MAX_ABS_ERROR: 1.5e-1,
            Metric.PCC: 0.99,
        },
    )(TTLinear.__call__)

    _ = validated_call(layer, x_tt)

    assert len(registry.results) == before + 1
    assert registry.results[-1].passed


# ============================================================================
# Example 7: Validating with non-decorator use of compare_to_ttnn
#            between a class instance (return torch tensor) and a reference class instance
#            (return TTNN tensor)
# NOTE: This use of compare_to_ttnn could come in handy in situations where a module instance
#       within torch implementation is being replaced by a TTNN module instance and
#       we want to check the output of the TTNN module instance against the output of
#       the torch module instance during end2end testing.
# ============================================================================


def test_validation_non_decorator_class_vs_class_ttnn(ttnn_mesh_device: ttnn.MeshDevice):
    """Validate a callable TTNN class against a TTNN reference class using non-decorator style."""
    registry = get_validation_registry()
    before = len(registry.results)

    # Simple linear layer implemented in TTNN (__call__) vs TTNN reference (forward)
    m, n, k = 8, 10, 6
    x = torch.randn(1, m, k, dtype=torch.bfloat16)
    w = torch.randn(1, k, n, dtype=torch.bfloat16)

    class TTLinear:
        def __init__(self, weight: torch.Tensor, device: ttnn.MeshDevice):
            # Weight expected as [1, k, n]; add device batch dim for TTNN tensor
            self.weight = ttnn.from_torch(
                weight.unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )

        def __call__(self, inp):
            return ttnn.matmul(inp, self.weight)

    class TTLinearRef:
        def __init__(self, weight: torch.Tensor, device: ttnn.MeshDevice):
            self.weight = weight

        def forward(self, inp):
            return torch.matmul(inp, self.weight)

    # Instantiate both implementations
    layer = TTLinear(w, ttnn_mesh_device)
    ref_layer = TTLinearRef(w, ttnn_mesh_device)

    # # Convert input to TTNN tensor (add device batch dim)
    # x_tt = ttnn.from_torch(x.unsqueeze(0), device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Non-decorator usage: wrap the unbound __call__ so we can pass (self, x)
    validated_call = compare_to_ttnn(
        reference_fn=lambda inp: layer(inp),
        input_to_ttnn=lambda self, inp: (
            ttnn.from_torch(inp, device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        ),
        metric_tolerances={
            Metric.MAX_ABS_ERROR: 1.5e-1,
            Metric.PCC: 0.99,
        },
    )(TTLinearRef.forward)

    ttnn.SetDefaultDevice(ttnn_mesh_device)
    _ = validated_call(ref_layer, x.unsqueeze(0))

    assert len(registry.results) == before + 1
    assert registry.results[-1].passed


def test_return_reference_output_torch(ttnn_mesh_device: ttnn.MeshDevice):
    """Demonstrate return_reference_output=True returns the reference (torch) output.

    The decorator computes torch.matmul on host for reference, then returns that
    reference result converted back to a TTNN tensor distributed like the impl output.
    """
    registry = get_validation_registry()
    before = len(registry.results)

    m, n, k = 8, 10, 6
    a = torch.randn(1, m, k, dtype=torch.bfloat16)
    b = torch.randn(1, k, n, dtype=torch.bfloat16)
    a_tt = ttnn.from_torch(a.unsqueeze(0), device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b_tt = ttnn.from_torch(b.unsqueeze(0), device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    ref_host = torch.ones(m, n, dtype=torch.bfloat16)

    @compare_to_torch(
        # mock a reference function that returns a torch tensor with the same shape as the decorated function output
        reference_fn=lambda a, b: ref_host,
        metric_tolerances={
            Metric.MAX_ABS_ERROR: 1,  # outrageous tolerance to confirm the mock
            Metric.PCC: 0.99,
        },
        return_reference_output=True,
    )
    def _impl_matmul(a, b):
        return ttnn.matmul(a, b)

    # Call impl; returned value should be the reference result (distributed as impl output)
    out_tt = _impl_matmul(a_tt, b_tt)

    # Registry records one validation
    assert len(registry.results) == before + 1
    assert not registry.results[-1].metrics[Metric.MAX_ABS_ERROR].passed
    assert registry.results[-1].metrics[Metric.PCC].passed

    # Convert both outputs to host and verify numerical equivalence
    out_host = to_torch_auto_compose(out_tt)
    assert torch.allclose(out_host, ref_host)


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

    validated_matmul = compare_to_torch(
        reference_fn=torch.matmul,
        metric_tolerances={
            Metric.MAX_ABS_ERROR: 1.5e-1,
            Metric.PCC: 0.99,
        },
    )(_matmul)

    _ = validated_matmul(a_tt, b_tt)

    assert len(registry.results) == before + 1
    assert registry.results[-1].passed


def test_validation_raises_on_reference_exception(ttnn_mesh_device: ttnn.MeshDevice):
    """When raise_exceptions=True, reference exceptions should propagate and not record results."""
    registry = get_validation_registry()
    before = len(registry.results)

    a = torch.randn(1, 8, 8, dtype=torch.bfloat16)
    b = torch.randn(1, 8, 8, dtype=torch.bfloat16)
    a_tt = ttnn.from_torch(a.unsqueeze(0), device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b_tt = ttnn.from_torch(b.unsqueeze(0), device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def _ref_raises(a, b):
        pass

    # [INFO] make a mismatched signature on reference function to force the reference function to raise an exception!
    @compare_to_torch(reference_fn=lambda a, b, c: _ref_raises(a, b), raise_exceptions=True)
    def _matmul(a, b):
        return ttnn.matmul(a, b)

    with pytest.raises(TypeError) as e:
        _ = _matmul(a_tt, b_tt)
    assert "missing 1 required positional argument: 'c'" in str(e.value)

    # [INFO] make a mismatched signature on output_to_torch to force the reference function to raise an exception!
    @compare_to_torch(reference_fn=lambda a, b: ..., output_to_torch=lambda x, y: ..., raise_exceptions=True)
    def _matmul_too(a, b):
        return ttnn.matmul(a, b)

    with pytest.raises(TypeError) as e:
        _ = _matmul_too(a_tt, b_tt)
    assert "missing 1 required positional argument: 'y'" in str(e.value)

    # [INFO] make a mismatched signature on input_to_torch to force the reference function to raise an exception!
    @compare_to_torch(reference_fn=lambda a, b: ..., input_to_torch=lambda x: ..., raise_exceptions=True)
    def _matmul_three(a, b):
        return ttnn.matmul(a, b)

    with pytest.raises(TypeError) as e:
        _ = _matmul_three(a_tt, b_tt)
    assert "takes 1 positional argument but 2 were given" in str(e.value)


@pytest.fixture(scope="module", autouse=True)
def _print_validation_report_after_module(request):
    # Runs once after all tests in this module finish
    yield
    registry = get_validation_registry()
    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    reporter.write_line("Printing validation report after yield")
    registry.print_report(verbose=True)


@pytest.fixture(scope="module", autouse=True)
def _clear_validation_results_before_module():
    clear_validation_results()
