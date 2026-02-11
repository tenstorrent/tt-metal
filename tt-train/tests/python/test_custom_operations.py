# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for custom autograd operations in TTML.

These tests verify the FunctionContext, get_links(), and Function base class
work correctly for defining custom operations with forward and backward passes.
"""

import math
import numpy as np
import pytest
import ttnn

import ttml
from ttml.autograd import Function, FunctionContext, get_links, Tensor


class TestFunctionContext:
    """Tests for the FunctionContext class."""

    def test_save_and_retrieve_tensors(self):
        """Test saving and retrieving tensors."""
        ctx = FunctionContext()

        tensor1 = object()
        tensor2 = object()
        tensor3 = None

        ctx.save_for_backward(tensor1, tensor2, tensor3)
        saved = ctx.saved_tensors

        assert len(saved) == 3
        assert saved[0] is tensor1
        assert saved[1] is tensor2
        assert saved[2] is None

    def test_arbitrary_attributes(self):
        """Test storing arbitrary attributes on context (PyTorch-style)."""
        ctx = FunctionContext()

        ctx.scale = 2.5
        ctx.shape = (32, 64)
        ctx.flag = True

        assert ctx.scale == 2.5
        assert ctx.shape == (32, 64)
        assert ctx.flag is True

    def test_overwrite_saved_tensors(self):
        """Test that save_for_backward overwrites previous saves."""
        ctx = FunctionContext()

        tensor1 = object()
        tensor2 = object()

        ctx.save_for_backward(tensor1)
        assert len(ctx.saved_tensors) == 1

        ctx.save_for_backward(tensor1, tensor2)
        assert len(ctx.saved_tensors) == 2

    def test_empty_saved_tensors(self):
        """Test that saved_tensors returns empty tuple initially."""
        ctx = FunctionContext()
        assert ctx.saved_tensors == ()


class MockTensor:
    """Mock tensor for testing get_links without requiring device."""

    def __init__(self, node=None, requires_grad=True):
        self._node = node
        self._requires_grad = requires_grad

    def get_node(self):
        return self._node

    def get_requires_grad(self):
        return self._requires_grad


class PythonLinearOp(Function):
    """Python implementation of linear operation using ttnn primitives.

    Forward: output = input @ weight.T + bias
    Backward:
        - weight_grad = grad_output.T @ input
        - input_grad = grad_output @ weight
        - bias_grad = sum(grad_output, over batch/sequence dims)
    """

    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor, bias: Tensor | None = None):
        """Forward pass of linear operation.

        Args:
            ctx: Function context for saving tensors
            input: Input tensor of shape [*, in_features]
            weight: Weight tensor of shape [1, 1, out_features, in_features]
            bias: Optional bias tensor of shape [1, 1, 1, out_features]

        Returns:
            Output tensor of shape [*, out_features]
        """
        ctx.save_for_backward(input, weight, bias)

        ttnn_input = input.get_value()
        ttnn_weight = weight.get_value()

        output = ttnn.linear(
            ttnn_input,
            ttnn_weight,
            bias=bias.get_value() if bias is not None else None,
            transpose_a=False,
            transpose_b=True,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of linear operation.

        Args:
            ctx: Function context with saved tensors
            grad_output: Gradient from upstream, shape [*, out_features]

        Returns:
            Tuple of (grad_input, grad_weight, grad_bias)
        """
        input, weight, bias = ctx.saved_tensors

        ttnn_input = input.get_value()
        ttnn_weight = weight.get_value()

        # Get shapes
        input_shape = ttnn_input.shape
        weight_shape = ttnn_weight.shape

        in_features = input_shape[-1]
        out_features = weight_shape[2]
        volume_without_features = ttnn_input.logical_volume() // in_features

        # Reshape input and grad_output to 2D
        reshaped_input = ttnn.reshape(ttnn_input, ttnn.Shape([volume_without_features, in_features]))

        reshaped_grad_output = ttnn.reshape(grad_output, ttnn.Shape([volume_without_features, out_features]))

        # Compute weight gradient: grad_output.T @ input
        # Shape: [out_features, batch*seq] @ [batch*seq, in_features] = [out_features, in_features]
        grad_weight = ttnn.matmul(
            reshaped_grad_output,
            reshaped_input,
            transpose_a=True,
            transpose_b=False,
        )

        # Reshape to weight's shape [1, 1, out_features, in_features]
        grad_weight = ttnn.reshape(grad_weight, weight_shape)

        # Compute input gradient: grad_output @ weight
        # Shape: [batch*seq, out_features] @ [out_features, in_features] = [batch*seq, in_features]
        grad_input = ttnn.matmul(
            reshaped_grad_output,
            ttnn_weight,
            transpose_a=False,
            transpose_b=False,  # weight is already [1, 1, out_features, in_features]
        )

        # Reshape back to input's shape
        grad_input = ttnn.reshape(grad_input, input_shape)

        # Compute bias gradient if bias exists
        grad_bias = None
        if bias is not None:
            # Sum over all dimensions except the last (features)
            # reshaped_grad_output is [batch*seq, out_features]
            # We need to sum over dimension 0 to get [out_features]
            grad_bias_flat = ttnn.sum(reshaped_grad_output, dim=0)
            # Reshape to bias shape [1, 1, 1, out_features]
            grad_bias = ttnn.reshape(grad_bias_flat, bias.get_value().shape)

        return grad_input, grad_weight, grad_bias


class PythonGroupedHeadsCreationOp(Function):
    """Python implementation of grouped_heads_creation operation using ttnn primitives.

    This operation splits query and key-value tensors into separate heads for
    grouped query attention (GQA).

    Forward:
        - qs shape: (B, 1, S, E)
        - kvs shape: (B, 1, S, E_kv * 2) where E_kv = E * num_groups / num_heads
        - Returns:
            - q: (B, num_heads, S, E / num_heads)
            - k: (B, num_groups, S, E_kv / num_groups)
            - v: (B, num_groups, S, E_kv / num_groups)

    Backward:
        - grad_q is concatenated back: (B, num_heads, S, E/num_heads) -> (B, 1, S, E)
        - grad_k is concatenated back: (B, num_groups, S, E_kv/num_groups) -> (B, 1, S, E_kv)
        - grad_v is concatenated back: (B, num_groups, S, E_kv/num_groups) -> (B, 1, S, E_kv)
        - qs_grad = grad_q
        - kvs_grad = concat([grad_k, grad_v], dim=3)
    """

    @staticmethod
    def forward(ctx, qs: Tensor, kvs: Tensor, num_heads: int, num_groups: int):
        """Forward pass of grouped heads creation operation.

        Args:
            ctx: Function context for saving tensors
            qs: Query tensor of shape (B, 1, S, E)
            kvs: Key-value tensor of shape (B, 1, S, E_kv * 2)
            num_heads: Number of query heads
            num_groups: Number of key-value head groups

        Returns:
            Tuple of (q, k, v) tensors
        """
        ctx.save_for_backward(qs, kvs)
        ctx.num_heads = num_heads
        ctx.num_groups = num_groups

        ttnn_qs = qs.get_value()
        ttnn_kvs = kvs.get_value()

        # nlp_create_qkv_heads splits qs into q heads and kvs into k,v heads
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            ttnn_qs,
            ttnn_kvs,
            num_heads=num_heads,
            num_kv_heads=num_groups,
            transpose_k_heads=False,
        )

        return q, k, v

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        """Backward pass of grouped heads creation operation.

        Args:
            ctx: Function context with saved tensors
            grad_q: Gradient w.r.t. q output, shape (B, num_heads, S, E/num_heads)
            grad_k: Gradient w.r.t. k output, shape (B, num_groups, S, E_kv/num_groups)
            grad_v: Gradient w.r.t. v output, shape (B, num_groups, S, E_kv/num_groups)

        Returns:
            Tuple of (grad_qs, grad_kvs)
        """
        # Concatenate heads back
        # (B, num_heads, S, E/num_heads) -> (B, 1, S, E)
        grad_qs = ttnn.experimental.nlp_concat_heads(grad_q)

        # (B, num_groups, S, E_kv/num_groups) -> (B, 1, S, E_kv)
        grad_k_concat = ttnn.experimental.nlp_concat_heads(grad_k)
        grad_v_concat = ttnn.experimental.nlp_concat_heads(grad_v)

        # Concatenate k and v gradients along dim 3
        # (B, 1, S, E_kv) + (B, 1, S, E_kv) -> (B, 1, S, E_kv * 2)
        grad_kvs = ttnn.concat([grad_k_concat, grad_v_concat], dim=3)

        return grad_qs, grad_kvs


class PythonLinearLayer(ttml.modules.AbstractModuleBase):
    """Python implementation of Linear layer using PythonLinearOp.

    This is an experimental implementation to test Python operations
    in tt-train's autograd system.
    """

    def __init__(self, in_features: int, out_features: int, has_bias: bool = True) -> None:
        """Initialize Python Linear layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            has_bias: Whether to include bias term
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias

        # Initialize weight with uniform distribution [-k, k] where k = sqrt(1/in_features)
        # Weight shape: [1, 1, out_features, in_features]
        init_k = math.sqrt(1.0 / in_features)
        weight_data = np.random.uniform(-init_k, init_k, (1, 1, out_features, in_features)).astype(np.float32)
        self.weight = Tensor.from_numpy(weight_data)

        # Initialize bias with same distribution
        # Bias shape: [1, 1, 1, out_features]
        if has_bias:
            bias_data = np.random.uniform(-init_k, init_k, (1, 1, 1, out_features)).astype(np.float32)
            self.bias = Tensor.from_numpy(bias_data)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of linear layer.

        Args:
            x: Input tensor of shape [*, in_features]

        Returns:
            Output tensor of shape [*, out_features]
        """
        return PythonLinearOp.apply(x, self.weight, self.bias)

    def __call__(self, x: Tensor) -> Tensor:
        """Make layer callable."""
        return self.forward(x)


class TestGetLinks:
    """Tests for the get_links() function."""

    def test_get_links_with_nodes(self):
        """Test extracting links from tensors with nodes."""
        node1 = object()
        node2 = object()

        tensor1 = MockTensor(node=node1)
        tensor2 = MockTensor(node=node2)

        links = get_links([tensor1, tensor2])

        assert len(links) == 2
        assert node1 in links
        assert node2 in links

    def test_get_links_with_none_nodes(self):
        """Test that tensors without nodes are skipped."""
        node1 = object()

        tensor1 = MockTensor(node=node1)
        tensor2 = MockTensor(node=None)

        links = get_links([tensor1, tensor2])

        assert len(links) == 1
        assert node1 in links

    def test_get_links_with_none_tensors(self):
        """Test that None tensors are skipped."""
        node1 = object()
        tensor1 = MockTensor(node=node1)

        links = get_links([tensor1, None, None])

        assert len(links) == 1
        assert node1 in links

    def test_get_links_empty_list(self):
        """Test get_links with empty list."""
        links = get_links([])
        assert links == []

    def test_get_links_all_none(self):
        """Test get_links when all tensors are None or have no nodes."""
        tensor1 = MockTensor(node=None)

        links = get_links([None, tensor1, None])
        assert links == []


class TestFunctionBase:
    """Tests for the Function base class."""

    def test_forward_not_implemented(self):
        """Test that forward raises NotImplementedError."""
        ctx = FunctionContext()
        with pytest.raises(NotImplementedError):
            Function.forward(ctx)

    def test_backward_not_implemented(self):
        """Test that backward raises NotImplementedError."""
        ctx = FunctionContext()
        with pytest.raises(NotImplementedError):
            Function.backward(ctx)


@pytest.mark.requires_device
class TestCustomOperationsWithDevice:
    """Integration tests for custom operations requiring a device.

    These tests require a Tenstorrent device to be available.
    """

    @pytest.fixture(autouse=True)
    def setup_device(self):
        """Set up device for tests."""
        auto_ctx = ttml.autograd.AutoContext.get_instance()
        auto_ctx.open_device()
        yield
        auto_ctx.close_device()

    def test_simple_scale_operation_forward(self):
        """Test a simple scale operation forward pass."""

        class Scale(Function):
            @staticmethod
            def forward(ctx, input, scale_factor):
                ctx.save_for_backward(input)
                ctx.scale_factor = scale_factor
                # Return ttnn tensor directly - auto-wrapped
                return ttnn.multiply(input.get_value(), scale_factor)

            @staticmethod
            def backward(ctx, grad_output):
                # Return gradient for input tensor
                return ttnn.multiply(grad_output, ctx.scale_factor)

        # Create input tensor
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        input_tensor = ttml.autograd.Tensor.from_numpy(input_data)

        # Apply scale
        output = Scale.apply(input_tensor, 2.0)

        # Verify output
        output_data = output.to_numpy()
        expected = input_data * 2.0
        np.testing.assert_array_almost_equal(output_data, expected, decimal=5)

    def test_gradient_mode_disabled(self):
        """Test that backward graph is not built when gradients disabled."""

        forward_called = [False]

        class TrackingOp(Function):
            @staticmethod
            def forward(ctx, input):
                forward_called[0] = True
                return input

            @staticmethod
            def backward(ctx, grad_output):
                return None

        # Create input tensor
        input_data = np.array([[1.0, 2.0]], dtype=np.float32)
        input_tensor = ttml.autograd.Tensor.from_numpy(input_data)

        auto_ctx = ttml.autograd.AutoContext.get_instance()

        # Disable gradients
        auto_ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)

        try:
            output = TrackingOp.apply(input_tensor)

            assert forward_called[0] is True
            # Output should not have a node set when gradients disabled
            assert output.get_node() is None

        finally:
            # Re-enable gradients
            auto_ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)

    def test_get_gradient_mode_binding(self):
        """Test that get_gradient_mode binding works."""
        auto_ctx = ttml.autograd.AutoContext.get_instance()

        # Default should be enabled
        mode = auto_ctx.get_gradient_mode()
        assert mode == ttml.autograd.GradMode.ENABLED

        # Set to disabled
        auto_ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)
        mode = auto_ctx.get_gradient_mode()
        assert mode == ttml.autograd.GradMode.DISABLED

        # Set back to enabled
        auto_ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
        mode = auto_ctx.get_gradient_mode()
        assert mode == ttml.autograd.GradMode.ENABLED

    def test_create_tensor_binding(self):
        """Test that create_tensor binding works."""
        # Create empty tensor
        empty_tensor = ttml.autograd.create_tensor()
        assert empty_tensor is not None

    def test_backward_pass_gradient_accumulation(self):
        """Test that gradients are accumulated correctly in backward pass."""

        class DoubleOp(Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                # Return ttnn tensor directly - auto-wrapped
                return ttnn.multiply(input.get_value(), 2.0)

            @staticmethod
            def backward(ctx, grad_output):
                # Gradient of 2*x w.r.t. x is 2
                return ttnn.multiply(grad_output, 2.0)

        # Create input tensor with shape suitable for tile layout
        input_data = np.ones((1, 1, 32, 32), dtype=np.float32)
        input_tensor = ttml.autograd.Tensor.from_numpy(input_data)

        # Forward pass
        output = DoubleOp.apply(input_tensor)

        # Trigger backward
        output.backward(retain_graph=False)

        # Check gradient was accumulated and has correct value
        assert input_tensor.is_grad_initialized()
        grad_tensor = input_tensor.get_grad_tensor()
        # Gradient should be 2.0 everywhere (since d/dx(2x) = 2)
        expected_grad = np.full_like(input_data, 2.0)
        np.testing.assert_array_almost_equal(grad_tensor.to_numpy(), expected_grad, decimal=5)

    def test_multiple_inputs(self):
        """Test custom operation with multiple inputs."""

        class AddOp(Function):
            @staticmethod
            def forward(ctx, a, b):
                ctx.save_for_backward(a, b)
                # Return ttnn tensor directly - auto-wrapped
                return ttnn.add(a.get_value(), b.get_value())

            @staticmethod
            def backward(ctx, grad_output):
                # Gradient of a + b is 1 for both inputs
                return grad_output, grad_output

        # Create input tensors
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b_data = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

        a_tensor = ttml.autograd.Tensor.from_numpy(a_data)
        b_tensor = ttml.autograd.Tensor.from_numpy(b_data)

        # Apply add
        output = AddOp.apply(a_tensor, b_tensor)

        # Verify output
        output_data = output.to_numpy()
        expected = a_data + b_data
        np.testing.assert_array_almost_equal(output_data, expected, decimal=5)

    def test_mixed_tensor_and_scalar_inputs(self):
        """Test operation with both tensor and scalar inputs."""

        class ScaleAndShift(Function):
            @staticmethod
            def forward(ctx, input, scale, shift):
                ctx.save_for_backward(input)
                ctx.scale = scale
                # input * scale + shift - return ttnn tensor directly
                scaled = ttnn.multiply(input.get_value(), scale)
                return ttnn.add(scaled, shift)

            @staticmethod
            def backward(ctx, grad_output):
                # d/d(input) of (input * scale + shift) = scale
                return ttnn.multiply(grad_output, ctx.scale)

        input_data = np.ones((1, 1, 32, 32), dtype=np.float32)
        input_tensor = ttml.autograd.Tensor.from_numpy(input_data)

        output = ScaleAndShift.apply(input_tensor, 3.0, 1.0)

        # Verify: 1.0 * 3.0 + 1.0 = 4.0
        output_data = output.to_numpy()
        expected = input_data * 3.0 + 1.0
        np.testing.assert_array_almost_equal(output_data, expected, decimal=5)

    def test_multiple_output_tensors(self):
        """Test custom operation that returns multiple output tensors as a tuple."""

        class SplitOp(Function):
            """Splits input into two outputs: one scaled by 2, one scaled by 3."""

            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                value = input.get_value()
                # Return tuple of ttnn tensors - both auto-wrapped
                out1 = ttnn.multiply(value, 2.0)
                out2 = ttnn.multiply(value, 3.0)
                return out1, out2

            @staticmethod
            def backward(ctx, grad_out1, grad_out2):
                # Gradient is 2 * grad_out1 + 3 * grad_out2
                grad1 = ttnn.multiply(grad_out1, 2.0)
                grad2 = ttnn.multiply(grad_out2, 3.0)
                return ttnn.add(grad1, grad2)

        # Create input tensor
        input_data = np.ones((1, 1, 32, 32), dtype=np.float32)
        input_tensor = ttml.autograd.Tensor.from_numpy(input_data)

        # Forward pass - returns tuple
        out1, out2 = SplitOp.apply(input_tensor)

        # Verify forward outputs
        out1_data = out1.to_numpy()
        out2_data = out2.to_numpy()
        np.testing.assert_array_almost_equal(out1_data, input_data * 2.0, decimal=5)
        np.testing.assert_array_almost_equal(out2_data, input_data * 3.0, decimal=5)

        # Verify both outputs have nodes set (for backward graph)
        assert out1.get_node() is not None, "First output should have node set"
        assert out2.get_node() is not None, "Second output should have node set"

    def test_case1_compose_with_autograd_ops(self):
        """Test Case 1: Compose with ttml autograd ops - backward is automatic."""

        class AddAndScale(Function):
            @staticmethod
            def forward(ctx, x, y, scale):
                # Using ttml autograd ops - graph builds automatically
                result = x + y  # Uses ttml tensor __add__ which has autograd
                return result * scale  # Uses ttml tensor __mul__ which has autograd

            # No backward needed - graph already built by autograd ops

        # Create input tensors
        x_data = np.ones((1, 1, 32, 32), dtype=np.float32) * 2.0
        y_data = np.ones((1, 1, 32, 32), dtype=np.float32) * 3.0

        x_tensor = ttml.autograd.Tensor.from_numpy(x_data)
        y_tensor = ttml.autograd.Tensor.from_numpy(y_data)

        # Forward pass: (x + y) * scale = (2 + 3) * 2 = 10
        output = AddAndScale.apply(x_tensor, y_tensor, 2.0)

        # Verify forward output
        output_data = output.to_numpy()
        expected = (x_data + y_data) * 2.0
        np.testing.assert_array_almost_equal(output_data, expected, decimal=5)

        # Output should have node set (from autograd ops)
        assert output.get_node() is not None, "Output should have node from autograd ops"

        # Backward should work automatically
        output.backward(retain_graph=False)

        # Check gradients were accumulated
        # d/dx((x+y)*s) = s = 2.0
        # d/dy((x+y)*s) = s = 2.0
        assert x_tensor.is_grad_initialized(), "x should have gradient"
        assert y_tensor.is_grad_initialized(), "y should have gradient"

    def test_case2_ttnn_primitives_explicit_backward(self):
        """Test Case 2: Build from ttnn primitives - explicit backward with add_grad."""

        class ScaleOp(Function):
            @staticmethod
            def forward(ctx, input, scale_factor):
                ctx.save_for_backward(input)
                ctx.scale_factor = scale_factor
                # Using raw ttnn ops - returns ttnn tensor (auto-wrapped)
                return ttnn.multiply(input.get_value(), scale_factor)

            @staticmethod
            def backward(ctx, grad_output):
                # Return gradient for input tensor
                return ttnn.multiply(grad_output, ctx.scale_factor)

        # Create input tensor
        input_data = np.ones((1, 1, 32, 32), dtype=np.float32) * 3.0
        input_tensor = ttml.autograd.Tensor.from_numpy(input_data)

        # Forward pass: input * scale = 3.0 * 2.0 = 6.0
        output = ScaleOp.apply(input_tensor, 2.0)

        # Verify forward output
        output_data = output.to_numpy()
        expected = input_data * 2.0
        np.testing.assert_array_almost_equal(output_data, expected, decimal=5)

        # Backward pass
        output.backward(retain_graph=False)

        # Check gradient was accumulated
        # d/dx(x * s) = s = 2.0
        assert input_tensor.is_grad_initialized(), "Input should have gradient"
        grad_tensor = input_tensor.get_grad_tensor()
        expected_grad = np.full_like(input_data, 2.0)
        np.testing.assert_array_almost_equal(grad_tensor.to_numpy(), expected_grad, decimal=5)

    def test_auto_wrap_ttnn_tensor(self):
        """Test that ttnn tensors returned from forward are auto-wrapped."""

        class IdentityOp(Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                # Return raw ttnn tensor - should be auto-wrapped
                return input.get_value()

            @staticmethod
            def backward(ctx, grad_output):
                # Return gradient as-is for identity
                return grad_output

        input_data = np.ones((1, 1, 32, 32), dtype=np.float32)
        input_tensor = ttml.autograd.Tensor.from_numpy(input_data)

        output = IdentityOp.apply(input_tensor)

        # Output should be ttml tensor (auto-wrapped), not raw ttnn
        assert hasattr(output, "get_requires_grad"), "Output should be ttml tensor"
        assert hasattr(output, "backward"), "Output should have backward method"

    def test_python_linear_op_vs_cpp_forward(self):
        """Compare Python linear op forward pass against C++ implementation."""
        # Create test tensors
        batch, seq_len, in_features, out_features = 2, 4, 32, 64

        # Input: [batch, seq_len, in_features]
        input_data = np.random.randn(1, 1, batch * seq_len, in_features).astype(np.float32)

        # Weight: [1, 1, out_features, in_features]
        weight_data = np.random.randn(1, 1, out_features, in_features).astype(np.float32) * 0.1

        # Bias: [1, 1, 1, out_features]
        bias_data = np.random.randn(1, 1, 1, out_features).astype(np.float32) * 0.1

        # Create ttml tensors for Python implementation
        input_py = ttml.autograd.Tensor.from_numpy(input_data)
        weight_py = ttml.autograd.Tensor.from_numpy(weight_data)
        bias_py = ttml.autograd.Tensor.from_numpy(bias_data)

        # Create ttml tensors for C++ implementation (same data)
        input_cpp = ttml.autograd.Tensor.from_numpy(input_data)
        weight_cpp = ttml.autograd.Tensor.from_numpy(weight_data)
        bias_cpp = ttml.autograd.Tensor.from_numpy(bias_data)

        # Python implementation forward
        output_py = PythonLinearOp.apply(input_py, weight_py, bias_py)

        # C++ implementation forward
        output_cpp = ttml.ops.linear.linear(input_cpp, weight_cpp, bias=bias_cpp)

        # Compare outputs (convert to float32 to handle dtype differences)
        output_py_np = output_py.to_numpy().astype(np.float32)
        output_cpp_np = output_cpp.to_numpy().astype(np.float32)

        np.testing.assert_allclose(
            output_py_np,
            output_cpp_np,
            rtol=1e-2,
            atol=1e-2,
            err_msg="Python and C++ linear op forward outputs differ",
        )

    def test_python_linear_op_vs_cpp_backward(self):
        """Compare Python linear op backward pass against C++ implementation."""
        # Create test tensors
        batch, seq_len, in_features, out_features = 2, 4, 32, 64

        # Input: [batch, seq_len, in_features]
        input_data = np.random.randn(1, 1, batch * seq_len, in_features).astype(np.float32)

        # Weight: [1, 1, out_features, in_features]
        weight_data = np.random.randn(1, 1, out_features, in_features).astype(np.float32) * 0.1

        # Bias: [1, 1, 1, out_features]
        bias_data = np.random.randn(1, 1, 1, out_features).astype(np.float32) * 0.1

        # Create ttml tensors for Python implementation
        input_py = ttml.autograd.Tensor.from_numpy(input_data)
        weight_py = ttml.autograd.Tensor.from_numpy(weight_data)
        bias_py = ttml.autograd.Tensor.from_numpy(bias_data)

        # Create ttml tensors for C++ implementation (same data)
        input_cpp = ttml.autograd.Tensor.from_numpy(input_data)
        weight_cpp = ttml.autograd.Tensor.from_numpy(weight_data)
        bias_cpp = ttml.autograd.Tensor.from_numpy(bias_data)

        # Python implementation forward + backward
        output_py = PythonLinearOp.apply(input_py, weight_py, bias_py)
        output_py.backward(retain_graph=False)

        # C++ implementation forward + backward
        output_cpp = ttml.ops.linear.linear(input_cpp, weight_cpp, bias=bias_cpp)
        output_cpp.backward(retain_graph=False)

        # Compare input gradients (convert to float32 to handle dtype differences)
        input_grad_py = input_py.get_grad_tensor().to_numpy().astype(np.float32)
        input_grad_cpp = input_cpp.get_grad_tensor().to_numpy().astype(np.float32)
        np.testing.assert_allclose(
            input_grad_py,
            input_grad_cpp,
            rtol=1e-2,
            atol=1e-2,
            err_msg="Python and C++ linear op input gradients differ",
        )

        # Compare weight gradients (convert to float32 to handle dtype differences)
        weight_grad_py = weight_py.get_grad_tensor().to_numpy().astype(np.float32)
        weight_grad_cpp = weight_cpp.get_grad_tensor().to_numpy().astype(np.float32)
        np.testing.assert_allclose(
            weight_grad_py,
            weight_grad_cpp,
            rtol=1e-2,
            atol=1e-2,
            err_msg="Python and C++ linear op weight gradients differ",
        )

        # Compare bias gradients (convert to float32 to handle dtype differences)
        bias_grad_py = bias_py.get_grad_tensor().to_numpy().astype(np.float32)
        bias_grad_cpp = bias_cpp.get_grad_tensor().to_numpy().astype(np.float32)
        np.testing.assert_allclose(
            bias_grad_py,
            bias_grad_cpp,
            rtol=1e-2,
            atol=1e-2,
            err_msg="Python and C++ linear op bias gradients differ",
        )

    def test_python_linear_layer_vs_cpp_forward(self):
        """Compare Python LinearLayer forward pass against C++ implementation."""
        in_features, out_features = 32, 64
        batch, seq_len = 2, 4

        # Create shared weight and bias tensors (create once to ensure same data)
        np.random.seed(42)
        weight_data = np.random.randn(1, 1, out_features, in_features).astype(np.float32) * 0.1
        bias_data = np.random.randn(1, 1, 1, out_features).astype(np.float32) * 0.1

        # Create shared ttml tensors
        weight_tensor = ttml.autograd.Tensor.from_numpy(weight_data)
        bias_tensor = ttml.autograd.Tensor.from_numpy(bias_data)

        # Create Python layer and set weights
        python_layer = PythonLinearLayer(in_features, out_features, has_bias=True)
        python_layer.weight = weight_tensor
        python_layer.bias = bias_tensor

        # Create C++ layer using constructor that takes weight and bias
        cpp_layer = ttml.modules.LinearLayer(weight_tensor, bias_tensor)

        # Create input
        input_data = np.random.randn(1, 1, batch * seq_len, in_features).astype(np.float32)

        # Python forward
        input_py = ttml.autograd.Tensor.from_numpy(input_data)
        output_py = python_layer(input_py)

        # C++ forward
        input_cpp = ttml.autograd.Tensor.from_numpy(input_data)
        output_cpp = cpp_layer(input_cpp)

        # Compare outputs (convert to float32 to handle dtype differences)
        output_py_np = output_py.to_numpy().astype(np.float32)
        output_cpp_np = output_cpp.to_numpy().astype(np.float32)

        np.testing.assert_allclose(
            output_py_np,
            output_cpp_np,
            rtol=1e-2,
            atol=1e-2,
            err_msg="Python and C++ LinearLayer forward outputs differ",
        )

    def test_python_linear_layer_vs_cpp_backward(self):
        """Compare Python LinearLayer backward pass against C++ implementation."""
        in_features, out_features = 32, 64
        batch, seq_len = 2, 4

        # Create shared weight and bias tensors (create once to ensure same data)
        np.random.seed(42)
        weight_data = np.random.randn(1, 1, out_features, in_features).astype(np.float32) * 0.1
        bias_data = np.random.randn(1, 1, 1, out_features).astype(np.float32) * 0.1

        # Create shared ttml tensors for Python layer
        weight_py = ttml.autograd.Tensor.from_numpy(weight_data)
        bias_py = ttml.autograd.Tensor.from_numpy(bias_data)

        # Create shared ttml tensors for C++ layer (separate instances to avoid shared gradients)
        weight_cpp = ttml.autograd.Tensor.from_numpy(weight_data)
        bias_cpp = ttml.autograd.Tensor.from_numpy(bias_data)

        # Create Python layer and set weights
        python_layer = PythonLinearLayer(in_features, out_features, has_bias=True)
        python_layer.weight = weight_py
        python_layer.bias = bias_py

        # Create C++ layer using constructor that takes weight and bias
        cpp_layer = ttml.modules.LinearLayer(weight_cpp, bias_cpp)

        # Create input
        input_data = np.random.randn(1, 1, batch * seq_len, in_features).astype(np.float32)

        # Python forward + backward
        input_py = ttml.autograd.Tensor.from_numpy(input_data)
        output_py = python_layer(input_py)
        output_py.backward(retain_graph=False)

        # C++ forward + backward
        input_cpp = ttml.autograd.Tensor.from_numpy(input_data)
        output_cpp = cpp_layer(input_cpp)
        output_cpp.backward(retain_graph=False)

        # Compare input gradients (convert to float32 to handle dtype differences)
        input_grad_py = input_py.get_grad_tensor().to_numpy().astype(np.float32)
        input_grad_cpp = input_cpp.get_grad_tensor().to_numpy().astype(np.float32)
        np.testing.assert_allclose(
            input_grad_py,
            input_grad_cpp,
            rtol=1e-2,
            atol=1e-2,
            err_msg="Python and C++ LinearLayer input gradients differ",
        )

        # Compare weight gradients (convert to float32 to handle dtype differences)
        weight_grad_py = python_layer.weight.get_grad_tensor().to_numpy().astype(np.float32)
        weight_grad_cpp = cpp_layer.get_weight().get_grad_tensor().to_numpy().astype(np.float32)
        np.testing.assert_allclose(
            weight_grad_py,
            weight_grad_cpp,
            rtol=1e-2,
            atol=1e-2,
            err_msg="Python and C++ LinearLayer weight gradients differ",
        )

        bias_grad_py = python_layer.bias.get_grad_tensor().to_numpy().astype(np.float32)
        bias_grad_cpp = bias_cpp.get_grad_tensor().to_numpy().astype(np.float32)
        np.testing.assert_allclose(
            bias_grad_py,
            bias_grad_cpp,
            rtol=1e-2,
            atol=1e-2,
            err_msg="Python and C++ LinearLayer bias gradients differ",
        )

    def test_python_grouped_heads_creation_forward(self):
        """Test Python grouped_heads_creation forward pass against C++ implementation."""
        # nanollama config
        batch_size = 64
        seq_len = 256
        embedding_dim = 384
        num_heads = 6  # Query heads
        num_groups = 3  # Key-value head groups (GQA: num_heads > num_groups)

        head_dim = embedding_dim // num_heads  # 16
        kv_embedding_dim = head_dim * num_groups  # 64

        # qs shape: (B, 1, S, E)
        qs_data = np.random.randn(batch_size, 1, seq_len, embedding_dim).astype(np.float32)

        # kvs shape: (B, 1, S, E_kv * 2)
        kvs_data = np.random.randn(batch_size, 1, seq_len, kv_embedding_dim * 2).astype(np.float32)

        # Create tensors for Python implementation
        qs_py = ttml.autograd.Tensor.from_numpy(qs_data, new_type=ttnn.DataType.BFLOAT16)
        kvs_py = ttml.autograd.Tensor.from_numpy(kvs_data, new_type=ttnn.DataType.BFLOAT16)

        # Create tensors for C++ implementation
        qs_cpp = ttml.autograd.Tensor.from_numpy(qs_data, new_type=ttnn.DataType.BFLOAT16)
        kvs_cpp = ttml.autograd.Tensor.from_numpy(kvs_data, new_type=ttnn.DataType.BFLOAT16)

        # Python implementation forward
        q_py, k_py, v_py = PythonGroupedHeadsCreationOp.apply(qs_py, kvs_py, num_heads, num_groups)

        # C++ implementation forward
        q_cpp, k_cpp, v_cpp = ttml.ops.multi_head_utils.grouped_heads_creation(qs_cpp, kvs_cpp, num_heads, num_groups)

        # Compare outputs
        q_py_np = q_py.to_numpy().astype(np.float32)
        q_cpp_np = q_cpp.to_numpy().astype(np.float32)
        np.testing.assert_allclose(
            q_py_np,
            q_cpp_np,
            rtol=1e-3,
            atol=1e-3,
            err_msg="Python and C++ grouped_heads_creation q outputs differ",
        )

        k_py_np = k_py.to_numpy().astype(np.float32)
        k_cpp_np = k_cpp.to_numpy().astype(np.float32)
        np.testing.assert_allclose(
            k_py_np,
            k_cpp_np,
            rtol=1e-3,
            atol=1e-3,
            err_msg="Python and C++ grouped_heads_creation k outputs differ",
        )

        v_py_np = v_py.to_numpy().astype(np.float32)
        v_cpp_np = v_cpp.to_numpy().astype(np.float32)
        np.testing.assert_allclose(
            v_py_np,
            v_cpp_np,
            rtol=1e-3,
            atol=1e-3,
            err_msg="Python and C++ grouped_heads_creation v outputs differ",
        )

        # Verify output shapes
        assert q_py_np.shape == (
            batch_size,
            num_heads,
            seq_len,
            head_dim,
        ), f"q shape mismatch: {q_py_np.shape}"
        assert k_py_np.shape == (
            batch_size,
            num_groups,
            seq_len,
            head_dim,
        ), f"k shape mismatch: {k_py_np.shape}"
        assert v_py_np.shape == (
            batch_size,
            num_groups,
            seq_len,
            head_dim,
        ), f"v shape mismatch: {v_py_np.shape}"

    def test_python_grouped_heads_creation_backward(self):
        """Test Python grouped_heads_creation backward pass against C++ implementation."""
        batch_size = 64
        seq_len = 256
        embedding_dim = 384
        num_heads = 6  # Query heads
        num_groups = 3  # Key-value head groups

        head_dim = embedding_dim // num_heads
        kv_embedding_dim = head_dim * num_groups

        # qs shape: (B, 1, S, E)
        qs_data = np.random.randn(batch_size, 1, seq_len, embedding_dim).astype(np.float32)

        # kvs shape: (B, 1, S, E_kv * 2)
        kvs_data = np.random.randn(batch_size, 1, seq_len, kv_embedding_dim * 2).astype(np.float32)

        # Create tensors for Python implementation
        qs_py = ttml.autograd.Tensor.from_numpy(qs_data, new_type=ttnn.DataType.BFLOAT16)
        kvs_py = ttml.autograd.Tensor.from_numpy(kvs_data, new_type=ttnn.DataType.BFLOAT16)

        # Create tensors for C++ implementation
        qs_cpp = ttml.autograd.Tensor.from_numpy(qs_data, new_type=ttnn.DataType.BFLOAT16)
        kvs_cpp = ttml.autograd.Tensor.from_numpy(kvs_data, new_type=ttnn.DataType.BFLOAT16)

        # Python implementation forward + create loss for backward
        q_py, k_py, v_py = PythonGroupedHeadsCreationOp.apply(qs_py, kvs_py, num_heads, num_groups)

        # Create a simple loss: mean of each output, then combine
        # (q, k, v may have different shapes when num_heads != num_groups)
        loss_q_py = ttml.ops.unary.mean(q_py)
        loss_k_py = ttml.ops.unary.mean(k_py)
        loss_v_py = ttml.ops.unary.mean(v_py)
        loss_py = loss_q_py + loss_k_py + loss_v_py
        loss_py.backward(retain_graph=False)

        # C++ implementation forward + backward
        q_cpp, k_cpp, v_cpp = ttml.ops.multi_head_utils.grouped_heads_creation(qs_cpp, kvs_cpp, num_heads, num_groups)

        loss_q_cpp = ttml.ops.unary.mean(q_cpp)
        loss_k_cpp = ttml.ops.unary.mean(k_cpp)
        loss_v_cpp = ttml.ops.unary.mean(v_cpp)
        loss_cpp = loss_q_cpp + loss_k_cpp + loss_v_cpp
        loss_cpp.backward(retain_graph=False)

        # Compare qs gradients
        qs_grad_py = qs_py.get_grad_tensor().to_numpy().astype(np.float32)
        qs_grad_cpp = qs_cpp.get_grad_tensor().to_numpy().astype(np.float32)
        np.testing.assert_allclose(
            qs_grad_py,
            qs_grad_cpp,
            rtol=1e-2,
            atol=1e-2,
            err_msg="Python and C++ grouped_heads_creation qs gradients differ",
        )

        # Compare kvs gradients
        kvs_grad_py = kvs_py.get_grad_tensor().to_numpy().astype(np.float32)
        kvs_grad_cpp = kvs_cpp.get_grad_tensor().to_numpy().astype(np.float32)
        np.testing.assert_allclose(
            kvs_grad_py,
            kvs_grad_cpp,
            rtol=1e-2,
            atol=1e-2,
            err_msg="Python and C++ grouped_heads_creation kvs gradients differ",
        )

    def test_python_grouped_heads_creation_multi_output_nodes(self):
        """Test that multi-output operation properly handles nodes in autograd graph."""
        batch_size = 64
        seq_len = 256
        embedding_dim = 384
        num_heads = 6
        num_groups = 3

        head_dim = embedding_dim // num_heads
        kv_embedding_dim = head_dim * num_groups

        qs_data = np.random.randn(batch_size, 1, seq_len, embedding_dim).astype(np.float32)
        kvs_data = np.random.randn(batch_size, 1, seq_len, kv_embedding_dim * 2).astype(np.float32)

        qs = ttml.autograd.Tensor.from_numpy(qs_data)
        kvs = ttml.autograd.Tensor.from_numpy(kvs_data)

        # Forward pass
        q, k, v = PythonGroupedHeadsCreationOp.apply(qs, kvs, num_heads, num_groups)

        # Verify all outputs have nodes set (required for backward graph)
        assert q.get_node() is not None, "q should have node set"
        assert k.get_node() is not None, "k should have node set"
        assert v.get_node() is not None, "v should have node set"

        # Verify backward works correctly when using only some outputs
        # This tests that the multi-output graph structure is correct
        loss = ttml.ops.unary.mean(q)  # Only use q
        loss.backward(retain_graph=False)

        # qs should have gradient (q depends on qs)
        assert qs.is_grad_initialized(), "qs should have gradient when q is used"


class TestCustomOperationsNoDevice:
    """Tests for custom operations that don't require a device.

    These tests verify the Python-only parts of the implementation.
    """

    def test_function_subclass_definition(self):
        """Test that Function subclass can be defined correctly."""

        class MyOp(Function):
            @staticmethod
            def forward(ctx, input, param):
                ctx.save_for_backward(input)
                ctx.param = param
                return input

            @staticmethod
            def backward(ctx, grad_output):
                ctx.saved_tensors  # Access to verify
                ctx.param  # Access to verify
                return grad_output

        # Verify the class is properly defined
        assert hasattr(MyOp, "forward")
        assert hasattr(MyOp, "backward")
        assert hasattr(MyOp, "apply")

    def test_context_in_custom_op(self):
        """Test that FunctionContext works in custom op definition."""

        class ContextTestOp(Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                ctx.test_key = "test_value"
                return input

            @staticmethod
            def backward(ctx, grad_output):
                return None

        # Create a mock tensor-like object
        mock_input = MockTensor(node=None)

        # Test the context directly (not through apply)
        ctx = FunctionContext()
        ContextTestOp.forward(ctx, mock_input)

        assert ctx.saved_tensors == (mock_input,)
        assert ctx.test_key == "test_value"

    def test_pytorch_style_api(self):
        """Verify the API matches PyTorch's style."""

        class PyTorchStyleOp(Function):
            @staticmethod
            def forward(ctx, x, y, alpha):
                # Save tensors for backward
                ctx.save_for_backward(x, y)
                # Save scalars as attributes
                ctx.alpha = alpha
                # Return result (in real code would compute something)
                return x

            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                alpha = ctx.alpha
                assert x is not None and y is not None
                assert alpha == 0.5
                # Return gradients for x and y (None would mean no gradient needed)
                return grad_output, grad_output

        # This should match PyTorch's API pattern
        ctx = FunctionContext()
        x = MockTensor()
        y = MockTensor()
        result = PyTorchStyleOp.forward(ctx, x, y, 0.5)

        assert result is x
        assert ctx.saved_tensors == (x, y)
        assert ctx.alpha == 0.5
