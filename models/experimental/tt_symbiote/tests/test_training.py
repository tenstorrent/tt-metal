# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for training with TTNN backend."""

import torch
from torch import nn
from collections import OrderedDict
import numpy as np
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.dispatcher import set_dispatcher
from models.experimental.tt_symbiote.core.dispatchers.dispatcher_config import register_dispatcher
from models.experimental.tt_symbiote.core.dispatchers.tensor_operations_dispatcher import func_to_ttnn_compatible
from models.experimental.tt_symbiote.core.run_config import (
    compose_transforms,
    tree_map,
    wrap_to_torch_ttnn_tensor,
    to_ttnn_wrap_keep_torch,
    set_device_wrap,
)
from models.experimental.tt_symbiote.core.run_config import DispatchManager


class TrainingDispatcher:
    device = None

    @staticmethod
    def can_dispatch_to_ttnn(func_name: str, args=None, kwargs=None):
        return func_name in func_to_ttnn_compatible

    @staticmethod
    def dispatch_to_ttnn(func_name: str, args=None, kwargs=None):
        transform = compose_transforms(
            wrap_to_torch_ttnn_tensor, to_ttnn_wrap_keep_torch, set_device_wrap(TrainingDispatcher.device)
        )
        func_args = tree_map(transform, args)
        func_kwargs = tree_map(transform, kwargs)
        result = func_to_ttnn_compatible[func_name](func_name, func_args, func_kwargs)
        return result


register_dispatcher("TRAINING", TrainingDispatcher)
set_dispatcher("TRAINING")


# First define the LinearFunction (from previous example)
class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
        # Convert to NumPy for computation
        x_np = x.cpu().numpy()
        weight_np = weight.cpu().numpy()

        # Forward pass in NumPy
        output_np = np.matmul(x_np, weight_np.T)
        if bias is not None:
            bias_np = bias.cpu().numpy()
            output_np += bias_np

        # Convert back to PyTorch tensor
        return torch.from_numpy(output_np).to(device=x.device, dtype=x.dtype)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias = inputs
        ctx.save_for_backward(input, weight, bias)

    @staticmethod
    def backward(ctx, grad):
        x, weight, bias = ctx.saved_tensors

        # Convert to NumPy for gradient computation
        grad_np = grad.cpu().numpy()
        x_np = x.cpu().numpy()
        weight_np = weight.cpu().numpy()

        # Compute gradients in NumPy
        grad_x_np = np.matmul(grad_np, weight_np)
        grad_weight_np = np.matmul(grad_np.T, x_np)

        # Convert back to PyTorch
        grad_x = torch.from_numpy(grad_x_np).to(device=x.device, dtype=x.dtype)
        grad_weight = torch.from_numpy(grad_weight_np).to(device=weight.device, dtype=weight.dtype)

        grad_bias = None
        if bias is not None:
            grad_bias_np = np.sum(grad_np, axis=0)
            grad_bias = torch.from_numpy(grad_bias_np).to(device=bias.device, dtype=bias.dtype)

        return grad_x, grad_weight, grad_bias


# Define ModuleLinear
class ModuleLinear:
    def forward(self, input, weight, bias=None):
        return LinearFunction.apply(input, weight, bias)


# Usage example
class MyModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Create parameters manually
        self.linear = nn.Linear(in_features, out_features)
        self.linear_grad = ModuleLinear()

    def forward(self, x):
        return self.linear.forward(x)


def wrap_state_dict(state_dict):
    """
    Wrap the tensors in the state_dict with Trackable_Tensor.

    Args:
        state_dict: The state_dict of the model.

    Returns:
        The wrapped state_dict.
    """

    wrapped_state_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            wrapped_tensor = TorchTTNNTensor(value)
            wrapped_state_dict[key] = wrapped_tensor
        else:
            wrapped_state_dict[key] = value
    wrapped_state_dict = OrderedDict(wrapped_state_dict)
    return wrapped_state_dict


def test_training_with_ttnn(device):
    """Test training a simple model with TTNN acceleration."""
    # Model configuration
    TrainingDispatcher.device = device
    input_dim = 10
    output_dim = 10
    batch_size = 4
    num_epochs = 100
    learning_rate = 0.001

    # Create model
    model = MyModel(input_dim, output_dim)
    state_dict = model.state_dict()
    wrapped_state_dict = wrap_state_dict(state_dict)
    model.load_state_dict(wrapped_state_dict, assign=True)

    # Set model to training mode
    model.train()
    torch.set_grad_enabled(True)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Generate synthetic training data
    X_train = TorchTTNNTensor(torch.randn(100, input_dim))
    # Create targets with some relationship to inputs for learning
    y_train = TorchTTNNTensor(torch.randn(100, output_dim))

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    DispatchManager.clear_timings()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = len(X_train) // batch_size

        for i in range(num_batches):
            # Get batch
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_x = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_x)

            # Compute loss
            loss = criterion(outputs, batch_y)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    DispatchManager.save_stats_to_file("training_stats.csv")
    print("\nTraining complete!")
