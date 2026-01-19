# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for custom autograd operations in TTML.

These tests verify the FunctionContext, get_links(), and Function base class
work correctly for defining custom operations with forward and backward passes.
"""

import numpy as np
import pytest

import ttml
from ttml.autograd import Function, FunctionContext, get_links


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
        import ttnn

        class Scale(Function):
            @staticmethod
            def forward(ctx, input, scale_factor):
                ctx.save_for_backward(input)
                ctx.scale_factor = scale_factor
                output_value = ttnn.multiply(input.get_value(), scale_factor)
                return ttml.autograd.create_tensor(output_value, True)

            @staticmethod
            def backward(ctx, grad_output):
                ctx.saved_tensors  # Access to verify it works
                # Return gradient w.r.t. input
                grad_value = ttnn.multiply(grad_output, ctx.scale_factor)
                return ttml.autograd.create_tensor(grad_value, False)

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
        import ttnn

        class DoubleOp(Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                output_value = ttnn.multiply(input.get_value(), 2.0)
                return ttml.autograd.create_tensor(output_value, True)

            @staticmethod
            def backward(ctx, grad_output):
                # Gradient of 2*x w.r.t. x is 2
                # grad_output is tt::tt_metal::Tensor (raw gradient)
                grad_value = ttnn.multiply(grad_output, 2.0)
                return ttml.autograd.create_tensor(grad_value, False)

        # Create input tensor with shape suitable for tile layout
        input_data = np.ones((1, 1, 32, 32), dtype=np.float32)
        input_tensor = ttml.autograd.Tensor.from_numpy(input_data)

        # Forward pass
        output = DoubleOp.apply(input_tensor)

        # Trigger backward
        output.backward(retain_graph=False)

        # Check gradient was accumulated
        assert input_tensor.is_grad_initialized()
        # Gradient should be 2.0 everywhere (since d/dx(2x) = 2)

    def test_multiple_inputs(self):
        """Test custom operation with multiple inputs."""
        import ttnn

        class AddOp(Function):
            @staticmethod
            def forward(ctx, a, b):
                ctx.save_for_backward(a, b)
                output_value = ttnn.add(a.get_value(), b.get_value())
                return ttml.autograd.create_tensor(output_value, True)

            @staticmethod
            def backward(ctx, grad_output):
                # Gradient of a + b is 1 for both inputs
                # Return tuple of gradients (one per input tensor)
                grad_a = ttml.autograd.create_tensor(grad_output, False)
                grad_b = ttml.autograd.create_tensor(grad_output, False)
                return grad_a, grad_b

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
        import ttnn

        class ScaleAndShift(Function):
            @staticmethod
            def forward(ctx, input, scale, shift):
                ctx.save_for_backward(input)
                ctx.scale = scale
                # input * scale + shift
                scaled = ttnn.multiply(input.get_value(), scale)
                output_value = ttnn.add(scaled, shift)
                return ttml.autograd.create_tensor(output_value, True)

            @staticmethod
            def backward(ctx, grad_output):
                # d/d(input) of (input * scale + shift) = scale
                grad_value = ttnn.multiply(grad_output, ctx.scale)
                return ttml.autograd.create_tensor(grad_value, False)

        input_data = np.ones((1, 1, 32, 32), dtype=np.float32)
        input_tensor = ttml.autograd.Tensor.from_numpy(input_data)

        output = ScaleAndShift.apply(input_tensor, 3.0, 1.0)

        # Verify: 1.0 * 3.0 + 1.0 = 4.0
        output_data = output.to_numpy()
        expected = input_data * 3.0 + 1.0
        np.testing.assert_array_almost_equal(output_data, expected, decimal=5)

    def test_multiple_output_tensors(self):
        """Test custom operation that returns multiple output tensors as a tuple."""
        import ttnn

        class SplitOp(Function):
            """Splits input into two outputs: one scaled by 2, one scaled by 3."""

            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                value = input.get_value()
                out1_value = ttnn.multiply(value, 2.0)
                out2_value = ttnn.multiply(value, 3.0)
                out1 = ttml.autograd.create_tensor(out1_value, True)
                out2 = ttml.autograd.create_tensor(out2_value, True)
                return out1, out2

            @staticmethod
            def backward(ctx, grad_out1, grad_out2):
                # Gradient is 2 * grad_out1 + 3 * grad_out2
                grad1 = ttnn.multiply(grad_out1, 2.0)
                grad2 = ttnn.multiply(grad_out2, 3.0)
                grad_input_value = ttnn.add(grad1, grad2)
                return ttml.autograd.create_tensor(grad_input_value, False)

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
                _x, _y = ctx.saved_tensors  # Retrieve to verify API works
                _alpha = ctx.alpha  # Retrieve to verify API works
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
