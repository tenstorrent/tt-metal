# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttml


def pytorch_rmsnorm(x, gamma, epsilon):
    x_float = x.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + epsilon)
    return (gamma * x_normed).to(x.dtype)


# @pytest.fixture(scope="module", autouse=True)
# def setup_ttml():
#     ttml.autograd.AutoContext.get_instance().open_device()
#     ttml.autograd.AutoContext.get_instance().set_seed(42)
#     yield
#     ttml.autograd.AutoContext.get_instance().close_device()


def run_rmsnorm_test(shape, epsilon, use_composite, seed, atol=0.05, rtol=0.05):
    batch_size, _, seq_len, features = shape
    torch.manual_seed(seed)
    x = torch.randn(shape)
    gamma = torch.randn(1, 1, 1, features)
    grad_output = torch.randn(shape)
    x_pt = x.clone().requires_grad_(True)
    gamma_pt = gamma.squeeze().clone().requires_grad_(True)
    output_pt = pytorch_rmsnorm(x_pt, gamma_pt, epsilon)
    output_pt.backward(grad_output)
    x_ttml = ttml.autograd.Tensor.from_numpy(x.numpy(), ttml.Layout.TILE, ttml.autograd.DataType.BFLOAT16)
    gamma_ttml = ttml.autograd.Tensor.from_numpy(gamma.numpy(), ttml.Layout.TILE, ttml.autograd.DataType.BFLOAT16)
    if use_composite:
        output_ttml = ttml.ops.rmsnorm.rmsnorm_composite(x_ttml, gamma_ttml, epsilon)
    else:
        output_ttml = ttml.ops.rmsnorm.rmsnorm(x_ttml, gamma_ttml, epsilon)
    output_ttml.set_grad(
        ttml.autograd.Tensor.from_numpy(
            grad_output.numpy(), ttml.Layout.TILE, ttml.autograd.DataType.BFLOAT16
        ).get_value()
    )
    output_ttml.backward(False)
    output_ttml_t = torch.from_numpy(output_ttml.to_numpy())
    x_grad_ttml_t = torch.from_numpy(x_ttml.get_grad().to_numpy())
    gamma_grad_ttml_t = torch.from_numpy(gamma_ttml.get_grad().to_numpy())
    gamma_grad_pt = gamma_pt.grad.reshape(1, 1, 1, features)
    assert output_ttml_t.shape == output_pt.shape, f"Output shape mismatch: {output_ttml_t.shape} vs {output_pt.shape}"
    assert torch.allclose(
        output_ttml_t, output_pt.detach(), atol=atol, rtol=rtol
    ), f"Forward mismatch: max diff = {(output_ttml_t - output_pt.detach()).abs().max()}"
    assert x_grad_ttml_t.shape == x_pt.grad.shape, f"X grad shape mismatch: {x_grad_ttml_t.shape} vs {x_pt.grad.shape}"
    assert torch.allclose(
        x_grad_ttml_t, x_pt.grad, atol=atol, rtol=rtol
    ), f"X grad mismatch: max diff = {(x_grad_ttml_t - x_pt.grad).abs().max()}"
    assert (
        gamma_grad_ttml_t.shape == gamma_grad_pt.shape
    ), f"Gamma grad shape mismatch: {gamma_grad_ttml_t.shape} vs {gamma_grad_pt.shape}"
    assert torch.allclose(
        gamma_grad_ttml_t, gamma_grad_pt, atol=atol, rtol=rtol
    ), f"Gamma grad mismatch: max diff = {(gamma_grad_ttml_t - gamma_grad_pt).abs().max()}"
    ttml.autograd.AutoContext.get_instance().reset_graph()


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 64),
        (1, 1, 64, 128),
        (2, 1, 32, 64),
        (2, 1, 64, 128),
    ],
)
@pytest.mark.parametrize("epsilon", [1e-5, 1e-6])
@pytest.mark.parametrize("use_composite", [False, True])
@pytest.mark.parametrize("seed", [42, 123])
def test_rmsnorm(shape, epsilon, use_composite, seed, setup_ttml):
    run_rmsnorm_test(shape, epsilon, use_composite, seed)


def test_rmsnorm_kernel_vs_composite(setup_ttml):
    shape = (1, 1, 32, 64)
    epsilon = 1e-5
    seed = 42
    features = shape[-1]
    torch.manual_seed(seed)
    x = torch.randn(shape)
    gamma = torch.randn(1, 1, 1, features)
    x_kernel = ttml.autograd.Tensor.from_numpy(x.numpy(), ttml.Layout.TILE, ttml.autograd.DataType.BFLOAT16)
    gamma_kernel = ttml.autograd.Tensor.from_numpy(gamma.numpy(), ttml.Layout.TILE, ttml.autograd.DataType.BFLOAT16)
    result_kernel = ttml.ops.rmsnorm.rmsnorm(x_kernel, gamma_kernel, epsilon)
    result_kernel_t = torch.from_numpy(result_kernel.to_numpy())
    ttml.autograd.AutoContext.get_instance().reset_graph()
    x_composite = ttml.autograd.Tensor.from_numpy(x.numpy(), ttml.Layout.TILE, ttml.autograd.DataType.BFLOAT16)
    gamma_composite = ttml.autograd.Tensor.from_numpy(gamma.numpy(), ttml.Layout.TILE, ttml.autograd.DataType.BFLOAT16)
    result_composite = ttml.ops.rmsnorm.rmsnorm_composite(x_composite, gamma_composite, epsilon)
    result_composite_t = torch.from_numpy(result_composite.to_numpy())
    assert torch.allclose(
        result_kernel_t, result_composite_t, atol=0.05, rtol=0.05
    ), f"Kernel vs Composite mismatch: max diff = {(result_kernel_t - result_composite_t).abs().max()}"
    ttml.autograd.AutoContext.get_instance().reset_graph()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
