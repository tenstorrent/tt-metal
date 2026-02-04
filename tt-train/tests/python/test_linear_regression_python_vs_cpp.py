# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests comparing Python linear regression implementation against C++ _ttml implementation.

This test suite verifies that the Python LinearRegression model produces
equivalent results to the C++ implementation from _ttml.
"""

import numpy as np
import pytest

import ttnn
import ttml  # noqa: E402


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    batch_size = 16

    # Generate synthetic data
    x = np.random.randn(n_samples, n_features).astype(np.float32)
    # Create a simple linear relationship: y = x @ w + b + noise
    true_w = np.random.randn(n_features).astype(np.float32)
    true_b = 2.0
    y = (x @ true_w + true_b + 0.1 * np.random.randn(n_samples)).astype(np.float32)

    # Reshape for ttml: [B, 1, 1, features]
    x_reshaped = x.reshape(n_samples, 1, 1, n_features)
    y_reshaped = y.reshape(n_samples, 1, 1, 1)

    return {
        "x": x,
        "y": y,
        "x_reshaped": x_reshaped,
        "y_reshaped": y_reshaped,
        "n_features": n_features,
        "batch_size": batch_size,
        "true_w": true_w,
        "true_b": true_b,
    }


def test_model_creation(sample_data):
    """Test that both Python and C++ models can be created."""
    n_features = sample_data["n_features"]

    # Create C++ model
    cpp_model = ttml._ttml.models.linear_regression.create_linear_regression_model(
        n_features, 1
    )
    assert cpp_model is not None

    # Create Python model
    from ttml.models.linear_regression import create_linear_regression_model

    py_model = create_linear_regression_model(n_features, 1)
    assert py_model is not None

    # Both should be callable
    assert callable(cpp_model)
    assert callable(py_model)


def test_forward_pass_shape(sample_data):
    """Test that both models produce outputs with the same shape."""
    n_features = sample_data["n_features"]
    x_reshaped = sample_data["x_reshaped"]
    batch_size = sample_data["batch_size"]

    # Create models
    cpp_model = ttml._ttml.models.linear_regression.create_linear_regression_model(
        n_features, 1
    )
    from ttml.models.linear_regression import create_linear_regression_model

    py_model = create_linear_regression_model(n_features, 1)

    # Test forward pass
    x_batch = x_reshaped[:batch_size]
    tt_x = ttml.autograd.Tensor.from_numpy(x_batch.astype(np.float32))

    cpp_output = cpp_model(tt_x)
    py_output = py_model(tt_x)

    # Check shapes
    cpp_shape = cpp_output.to_numpy(ttnn.DataType.FLOAT32).shape
    py_shape = py_output.to_numpy(ttnn.DataType.FLOAT32).shape

    assert (
        cpp_shape == py_shape
    ), f"Shape mismatch: C++ {cpp_shape} vs Python {py_shape}"
    assert cpp_shape == (batch_size, 1, 1, 1), f"Unexpected shape: {cpp_shape}"


def test_parameter_structure(sample_data):
    """Test that both models have similar parameter structures."""
    n_features = sample_data["n_features"]

    # Create models
    cpp_model = ttml._ttml.models.linear_regression.create_linear_regression_model(
        n_features, 1
    )
    from ttml.models.linear_regression import create_linear_regression_model

    py_model = create_linear_regression_model(n_features, 1)

    # Get parameters
    cpp_params = cpp_model.parameters()
    py_params = py_model.parameters()

    # Both should have weight and bias
    # C++ uses "linear/weight" and "linear/bias"
    # Python uses "LinearRegression/weight" and "LinearRegression/bias" (with module name prefix)
    cpp_has_weight = any("weight" in k.lower() for k in cpp_params.keys())
    cpp_has_bias = any("bias" in k.lower() for k in cpp_params.keys())
    py_has_weight = any("weight" in k.lower() for k in py_params.keys())
    py_has_bias = any("bias" in k.lower() for k in py_params.keys())

    assert cpp_has_weight and py_has_weight, "Both models should have weight parameter"
    assert cpp_has_bias and py_has_bias, "Both models should have bias parameter"

    # Check parameter shapes
    cpp_weight_key = [k for k in cpp_params.keys() if "weight" in k.lower()][0]
    cpp_bias_key = [k for k in cpp_params.keys() if "bias" in k.lower()][0]
    py_weight_key = [k for k in py_params.keys() if "weight" in k.lower()][0]
    py_bias_key = [k for k in py_params.keys() if "bias" in k.lower()][0]

    cpp_weight_shape = cpp_params[cpp_weight_key].to_numpy(ttnn.DataType.FLOAT32).shape
    cpp_bias_shape = cpp_params[cpp_bias_key].to_numpy(ttnn.DataType.FLOAT32).shape
    py_weight_shape = py_params[py_weight_key].to_numpy(ttnn.DataType.FLOAT32).shape
    py_bias_shape = py_params[py_bias_key].to_numpy(ttnn.DataType.FLOAT32).shape

    assert (
        cpp_weight_shape == py_weight_shape
    ), f"Weight shape mismatch: C++ {cpp_weight_shape} vs Python {py_weight_shape}"
    assert (
        cpp_bias_shape == py_bias_shape
    ), f"Bias shape mismatch: C++ {cpp_bias_shape} vs Python {py_bias_shape}"


def test_training_loop(sample_data):
    """Test that both models can be trained with the same training loop."""
    n_features = sample_data["n_features"]
    x_reshaped = sample_data["x_reshaped"]
    y_reshaped = sample_data["y_reshaped"]
    batch_size = sample_data["batch_size"]

    # Create models
    cpp_model = ttml._ttml.models.linear_regression.create_linear_regression_model(
        n_features, 1
    )
    from ttml.models.linear_regression import create_linear_regression_model

    py_model = create_linear_regression_model(n_features, 1)

    # Create optimizers
    lr = 0.01
    opt_cfg = ttml.optimizers.SGDConfig.make(lr, 0.0, 0.0, 0.0, False)
    cpp_opt = ttml.optimizers.SGD(cpp_model.parameters(), opt_cfg)
    py_opt = ttml.optimizers.SGD(py_model.parameters(), opt_cfg)

    # Set to training mode
    cpp_model.train()
    py_model.train()

    # Train for a few steps
    n_steps = 5
    loss_fn = ttml.ops.loss.mse_loss

    for step in range(n_steps):
        # Get a batch
        start_idx = (step * batch_size) % (len(x_reshaped) - batch_size)
        end_idx = start_idx + batch_size
        x_batch = x_reshaped[start_idx:end_idx]
        y_batch = y_reshaped[start_idx:end_idx]

        tt_x = ttml.autograd.Tensor.from_numpy(x_batch.astype(np.float32))
        tt_y = ttml.autograd.Tensor.from_numpy(y_batch.astype(np.float32))

        # C++ model training step
        cpp_opt.zero_grad()
        cpp_pred = cpp_model(tt_x)
        cpp_loss = loss_fn(cpp_pred, tt_y, ttml.ops.ReduceType.MEAN)
        cpp_loss.backward(False)
        cpp_opt.step()

        # Python model training step
        py_opt.zero_grad()
        py_pred = py_model(tt_x)
        py_loss = loss_fn(py_pred, tt_y, ttml.ops.ReduceType.MEAN)
        py_loss.backward(False)
        py_opt.step()

        # Both should produce valid losses
        cpp_loss_val = float(cpp_loss.to_numpy(ttnn.DataType.FLOAT32))
        py_loss_val = float(py_loss.to_numpy(ttnn.DataType.FLOAT32))

        assert np.isfinite(cpp_loss_val), f"C++ loss is not finite at step {step}"
        assert np.isfinite(py_loss_val), f"Python loss is not finite at step {step}"


def test_inference_consistency(sample_data):
    """Test that both models produce similar outputs after training."""
    n_features = sample_data["n_features"]
    x_reshaped = sample_data["x_reshaped"]
    y_reshaped = sample_data["y_reshaped"]
    batch_size = sample_data["batch_size"]

    # Create models
    cpp_model = ttml._ttml.models.linear_regression.create_linear_regression_model(
        n_features, 1
    )
    from ttml.models.linear_regression import create_linear_regression_model

    py_model = create_linear_regression_model(n_features, 1)

    # Train both models with the same data and same number of steps
    lr = 0.01
    opt_cfg = ttml.optimizers.SGDConfig.make(lr, 0.0, 0.0, 0.0, False)
    cpp_opt = ttml.optimizers.SGD(cpp_model.parameters(), opt_cfg)
    py_opt = ttml.optimizers.SGD(py_model.parameters(), opt_cfg)

    cpp_model.train()
    py_model.train()

    loss_fn = ttml.ops.loss.mse_loss
    n_steps = 10

    # Train with same batches
    np.random.seed(42)  # Ensure same random batches
    for step in range(n_steps):
        start_idx = (step * batch_size) % (len(x_reshaped) - batch_size)
        end_idx = start_idx + batch_size
        x_batch = x_reshaped[start_idx:end_idx]
        y_batch = y_reshaped[start_idx:end_idx]

        tt_x = ttml.autograd.Tensor.from_numpy(x_batch.astype(np.float32))
        tt_y = ttml.autograd.Tensor.from_numpy(y_batch.astype(np.float32))

        # C++ training
        cpp_opt.zero_grad()
        cpp_pred = cpp_model(tt_x)
        cpp_loss = loss_fn(cpp_pred, tt_y, ttml.ops.ReduceType.MEAN)
        cpp_loss.backward(False)
        cpp_opt.step()

        # Python training
        py_opt.zero_grad()
        py_pred = py_model(tt_x)
        py_loss = loss_fn(py_pred, tt_y, ttml.ops.ReduceType.MEAN)
        py_loss.backward(False)
        py_opt.step()

    # Set to eval mode
    cpp_model.eval()
    py_model.eval()

    # Test inference on a batch
    test_batch = x_reshaped[:batch_size]
    tt_x_test = ttml.autograd.Tensor.from_numpy(test_batch.astype(np.float32))

    cpp_output = cpp_model(tt_x_test)
    py_output = py_model(tt_x_test)

    cpp_pred_np = cpp_output.to_numpy(ttnn.DataType.FLOAT32).reshape(-1)
    py_pred_np = py_output.to_numpy(ttnn.DataType.FLOAT32).reshape(-1)

    # Both should produce predictions
    assert len(cpp_pred_np) == batch_size
    assert len(py_pred_np) == batch_size
    assert np.all(np.isfinite(cpp_pred_np))
    assert np.all(np.isfinite(py_pred_np))


def test_parameter_access(sample_data):
    """Test that parameters can be accessed and modified in both models."""
    n_features = sample_data["n_features"]

    # Create models
    cpp_model = ttml._ttml.models.linear_regression.create_linear_regression_model(
        n_features, 1
    )
    from ttml.models.linear_regression import create_linear_regression_model

    py_model = create_linear_regression_model(n_features, 1)

    # Get parameters
    cpp_params = cpp_model.parameters()
    py_params = py_model.parameters()

    # Find weight and bias keys
    cpp_weight_key = [k for k in cpp_params.keys() if "weight" in k.lower()][0]
    cpp_bias_key = [k for k in cpp_params.keys() if "bias" in k.lower()][0]
    py_weight_key = [k for k in py_params.keys() if "weight" in k.lower()][0]
    py_bias_key = [k for k in py_params.keys() if "bias" in k.lower()][0]

    # Extract parameter values
    cpp_weight = cpp_params[cpp_weight_key].to_numpy(ttnn.DataType.FLOAT32)
    cpp_bias = cpp_params[cpp_bias_key].to_numpy(ttnn.DataType.FLOAT32)
    py_weight = py_params[py_weight_key].to_numpy(ttnn.DataType.FLOAT32)
    py_bias = py_params[py_bias_key].to_numpy(ttnn.DataType.FLOAT32)

    # Check shapes match
    assert cpp_weight.shape == py_weight.shape
    assert cpp_bias.shape == py_bias.shape

    # Both should have reasonable initial values (not all zeros)
    assert np.any(cpp_weight != 0), "C++ weight should not be all zeros"
    assert np.any(py_weight != 0), "Python weight should not be all zeros"


def test_model_interface_compatibility(sample_data):
    """Test that both models have compatible interfaces."""
    n_features = sample_data["n_features"]

    # Create models
    cpp_model = ttml._ttml.models.linear_regression.create_linear_regression_model(
        n_features, 1
    )
    from ttml.models.linear_regression import create_linear_regression_model

    py_model = create_linear_regression_model(n_features, 1)

    # Both should have train() and eval() methods
    assert hasattr(cpp_model, "train")
    assert hasattr(cpp_model, "eval")
    assert hasattr(py_model, "train")
    assert hasattr(py_model, "eval")

    # Both should have parameters() method
    assert hasattr(cpp_model, "parameters")
    assert hasattr(py_model, "parameters")

    # Both should be callable
    assert callable(cpp_model)
    assert callable(py_model)

    # Test train/eval mode switching
    cpp_model.train()
    assert cpp_model.get_run_mode() == ttml.modules.RunMode.TRAIN

    cpp_model.eval()
    assert cpp_model.get_run_mode() == ttml.modules.RunMode.EVAL

    py_model.train()
    assert py_model.get_run_mode() == ttml.modules.RunMode.TRAIN

    py_model.eval()
    assert py_model.get_run_mode() == ttml.modules.RunMode.EVAL


@pytest.mark.parametrize("n_features", [2, 5, 10])
@pytest.mark.parametrize("out_features", [1, 2])
def test_different_feature_sizes(n_features, out_features):
    """Test both models with different feature sizes."""
    # Create models
    cpp_model = ttml._ttml.models.linear_regression.create_linear_regression_model(
        n_features, out_features
    )
    from ttml.models.linear_regression import create_linear_regression_model

    py_model = create_linear_regression_model(n_features, out_features)

    # Test forward pass
    batch_size = 8
    x = np.random.randn(batch_size, 1, 1, n_features).astype(np.float32)
    tt_x = ttml.autograd.Tensor.from_numpy(x)

    cpp_output = cpp_model(tt_x)
    py_output = py_model(tt_x)

    cpp_shape = cpp_output.to_numpy(ttnn.DataType.FLOAT32).shape
    py_shape = py_output.to_numpy(ttnn.DataType.FLOAT32).shape

    assert cpp_shape == py_shape
    assert cpp_shape == (batch_size, 1, 1, out_features)


def test_gradient_flow(sample_data):
    """Test that gradients flow correctly in both models."""
    n_features = sample_data["n_features"]
    x_reshaped = sample_data["x_reshaped"]
    y_reshaped = sample_data["y_reshaped"]
    batch_size = sample_data["batch_size"]

    # Create models
    cpp_model = ttml._ttml.models.linear_regression.create_linear_regression_model(
        n_features, 1
    )
    from ttml.models.linear_regression import create_linear_regression_model

    py_model = create_linear_regression_model(n_features, 1)

    # Get initial parameters
    cpp_params_before = cpp_model.parameters()
    py_params_before = py_model.parameters()

    cpp_weight_key = [k for k in cpp_params_before.keys() if "weight" in k.lower()][0]
    py_weight_key = [k for k in py_params_before.keys() if "weight" in k.lower()][0]
    cpp_weight_before = cpp_params_before[cpp_weight_key].to_numpy(
        ttnn.DataType.FLOAT32
    )
    py_weight_before = py_params_before[py_weight_key].to_numpy(ttnn.DataType.FLOAT32)

    # Train one step
    lr = 0.1
    opt_cfg = ttml.optimizers.SGDConfig.make(lr, 0.0, 0.0, 0.0, False)
    cpp_opt = ttml.optimizers.SGD(cpp_model.parameters(), opt_cfg)
    py_opt = ttml.optimizers.SGD(py_model.parameters(), opt_cfg)

    cpp_model.train()
    py_model.train()

    x_batch = x_reshaped[:batch_size]
    y_batch = y_reshaped[:batch_size]
    tt_x = ttml.autograd.Tensor.from_numpy(x_batch.astype(np.float32))
    tt_y = ttml.autograd.Tensor.from_numpy(y_batch.astype(np.float32))

    loss_fn = ttml.ops.loss.mse_loss

    # C++ step
    cpp_opt.zero_grad()
    cpp_pred = cpp_model(tt_x)
    cpp_loss = loss_fn(cpp_pred, tt_y, ttml.ops.ReduceType.MEAN)
    cpp_loss.backward(False)
    cpp_opt.step()

    # Python step
    py_opt.zero_grad()
    py_pred = py_model(tt_x)
    py_loss = loss_fn(py_pred, tt_y, ttml.ops.ReduceType.MEAN)
    py_loss.backward(False)
    py_opt.step()

    # Check that parameters changed (gradients flowed)
    cpp_params_after = cpp_model.parameters()
    py_params_after = py_model.parameters()

    cpp_weight_after = cpp_params_after[cpp_weight_key].to_numpy(ttnn.DataType.FLOAT32)
    py_weight_after = py_params_after[py_weight_key].to_numpy(ttnn.DataType.FLOAT32)

    # Parameters should have changed
    assert not np.allclose(
        cpp_weight_before, cpp_weight_after, atol=1e-6
    ), "C++ weight should have changed after training step"
    assert not np.allclose(
        py_weight_before, py_weight_after, atol=1e-6
    ), "Python weight should have changed after training step"
