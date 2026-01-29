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


def run_rmsnorm_test(shape, epsilon, use_composite, seed, atol=0.05, rtol=0.05):
    batch_size, _, seq_len, features = shape
    torch.manual_seed(seed)
    x = torch.randn(shape).to(torch.bfloat16).float()
    gamma = torch.randn(1, 1, 1, features).to(torch.bfloat16).float()
    grad_output = torch.randn(shape).to(torch.bfloat16).float()
    x_pt = x.clone().requires_grad_(True)
    gamma_pt = gamma.squeeze().clone().requires_grad_(True)
    output_pt = pytorch_rmsnorm(x_pt, gamma_pt, epsilon)
    output_pt.backward(grad_output)
    x_ttml = ttml.autograd.Tensor.from_numpy(
        x.float().numpy(), ttml.Layout.TILE, ttml.autograd.DataType.BFLOAT16
    )
    gamma_ttml = ttml.autograd.Tensor.from_numpy(
        gamma.float().numpy(), ttml.Layout.TILE, ttml.autograd.DataType.BFLOAT16
    )
    if use_composite:
        output_ttml = ttml.ops.rmsnorm.rmsnorm_composite(x_ttml, gamma_ttml, epsilon)
    else:
        output_ttml = ttml.ops.rmsnorm.rmsnorm(x_ttml, gamma_ttml, epsilon)
    output_ttml.set_grad_from_tensor(
        ttml.autograd.Tensor.from_numpy(
            grad_output.float().numpy(),
            ttml.Layout.TILE,
            ttml.autograd.DataType.BFLOAT16,
        )
    )
    output_ttml.backward(False)
    output_ttml_t = output_ttml.get_value().cpu().to_torch().float()
    x_grad_ttml_t = x_ttml.get_grad().cpu().to_torch().float()
    gamma_grad_ttml_t = gamma_ttml.get_grad().cpu().to_torch().float()
    gamma_grad_pt = gamma_pt.grad.reshape(1, 1, 1, features)
    assert (
        output_ttml_t.shape == output_pt.shape
    ), f"Output shape mismatch: {output_ttml_t.shape} vs {output_pt.shape}"
    assert torch.allclose(
        output_ttml_t, output_pt.float().detach(), atol=atol, rtol=rtol
    ), f"Forward mismatch: max diff = {(output_ttml_t - output_pt.detach()).abs().max()}"
    assert (
        x_grad_ttml_t.shape == x_pt.grad.shape
    ), f"X grad shape mismatch: {x_grad_ttml_t.shape} vs {x_pt.grad.shape}"
    assert torch.allclose(
        x_grad_ttml_t, x_pt.grad.float(), atol=atol, rtol=rtol
    ), f"X grad mismatch: max diff = {(x_grad_ttml_t - x_pt.grad).abs().max()}"
    assert (
        gamma_grad_ttml_t.shape == gamma_grad_pt.shape
    ), f"Gamma grad shape mismatch: {gamma_grad_ttml_t.shape} vs {gamma_grad_pt.shape}"
    assert torch.allclose(
        gamma_grad_ttml_t, gamma_grad_pt.float(), atol=atol, rtol=rtol
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
def test_rmsnorm(shape, epsilon, use_composite, seed):
    run_rmsnorm_test(shape, epsilon, use_composite, seed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
