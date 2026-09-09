# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize("hw", [(23, 23), (256, 257), (500, 500)], ids=["small", "boundary", "large"])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True])
def test_moreh_group_norm_reduce_boundaries(device, hw, groups, affine, fp32_dest_acc_en):
    """Direct coverage for the factories hidden by the older suite's unconditional skip."""
    torch.manual_seed(42)
    x = torch.randn((2, 4, *hw)).to(torch.bfloat16).float()
    weight = (torch.rand(4) + 0.5).to(torch.bfloat16).float() if affine else None
    bias = torch.randn(4).to(torch.bfloat16).float() if affine else None
    expected = F.group_norm(x, groups, weight, bias, 1e-5)
    grouped = x.reshape(2, groups, -1)
    expected_mean = grouped.mean(-1)
    expected_rstd = torch.rsqrt(grouped.var(-1, unbiased=False) + 1e-5)

    def to_device(value):
        if value is None:
            return None
        tensor = ttnn.from_torch(value.to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
        ttnn.fill_implicit_tile_padding(tensor, 42)
        return tensor

    output, mean, rstd = ttnn.operations.moreh.group_norm(
        to_device(x),
        groups,
        1e-5,
        to_device(weight.reshape(1, 1, 1, 4) if affine else None),
        to_device(bias.reshape(1, 1, 1, 4) if affine else None),
        are_required_outputs=(True, True, True),
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_dest_acc_en
        ),
    )
    torch.testing.assert_close(ttnn.to_torch(output).float(), expected, rtol=0.08, atol=0.05)
    torch.testing.assert_close(ttnn.to_torch(mean).float().reshape(2, groups), expected_mean, rtol=0.08, atol=0.01)
    torch.testing.assert_close(ttnn.to_torch(rstd).float().reshape(2, groups), expected_rstd, rtol=0.08, atol=0.01)


@pytest.mark.parametrize(
    "shape,normalized_dims",
    [((64, 769), 1), ((32, 16385), 1), ((2, 4, 129, 127), 3), ((2, 4, 500, 500), 3)],
    ids=["W-small", "W-large", "HW-small", "HW-large"],
)
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True])
def test_moreh_layer_norm_reduce_boundaries(device, shape, normalized_dims, affine, fp32_dest_acc_en):
    torch.manual_seed(42)
    x = torch.randn(shape).to(torch.bfloat16).float()
    normalized_shape = shape[-normalized_dims:]
    weight = (torch.rand(normalized_shape) + 0.5).to(torch.bfloat16).float() if affine else None
    bias = torch.randn(normalized_shape).to(torch.bfloat16).float() if affine else None
    expected = F.layer_norm(x, normalized_shape, weight, bias, 1e-5)
    dims = tuple(range(-normalized_dims, 0))
    expected_mean = x.mean(dims)
    expected_rstd = torch.rsqrt(x.var(dims, unbiased=False) + 1e-5)

    def to_device(value):
        if value is None:
            return None
        tensor = ttnn.from_torch(value.to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
        ttnn.fill_implicit_tile_padding(tensor, 42)
        return tensor

    output, mean, rstd = ttnn.operations.moreh.layer_norm(
        to_device(x),
        normalized_dims,
        1e-5,
        to_device(weight),
        to_device(bias),
        mean=to_device(torch.full_like(expected_mean, float("nan"))),
        rstd=to_device(torch.full_like(expected_rstd, float("nan"))),
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_dest_acc_en
        ),
    )
    torch.testing.assert_close(ttnn.to_torch(output).float(), expected, rtol=0.08, atol=0.05)
    torch.testing.assert_close(ttnn.to_torch(mean).float(), expected_mean, rtol=0.08, atol=0.01)
    torch.testing.assert_close(ttnn.to_torch(rstd).float(), expected_rstd, rtol=0.08, atol=0.01)


@pytest.mark.parametrize("shape", [(2, 769, 45), (2, 1025, 257)])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True])
def test_moreh_layer_norm_backward_reduce_boundaries(device, shape, fp32_dest_acc_en):
    """Repeated parameter-gradient accumulation across batches with partial H/W tiles."""
    torch.manual_seed(42)
    x = torch.randn(shape).to(torch.bfloat16).float().requires_grad_()
    dy = torch.randn(shape).to(torch.bfloat16).float()
    weight = (torch.rand(shape[-1]) + 0.5).to(torch.bfloat16).float().requires_grad_()
    bias = torch.randn(shape[-1]).to(torch.bfloat16).float().requires_grad_()
    F.layer_norm(x, shape[-1:], weight, bias, 1e-5).backward(dy)
    mean = x.detach().mean(-1)
    rstd = torch.rsqrt(x.detach().var(-1, unbiased=False) + 1e-5)

    def to_device(value):
        tensor = ttnn.from_torch(value.to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
        ttnn.fill_implicit_tile_padding(tensor, 42)
        return tensor

    dx, dgamma, dbeta = ttnn.operations.moreh.layer_norm_backward(
        to_device(dy),
        to_device(x.detach()),
        to_device(mean),
        to_device(rstd),
        1,
        gamma=to_device(weight.detach()),
        input_grad=to_device(torch.full_like(x, float("nan"))),
        gamma_grad=to_device(torch.full_like(weight, float("nan"))),
        beta_grad=to_device(torch.full_like(bias, float("nan"))),
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_dest_acc_en
        ),
    )
    torch.testing.assert_close(ttnn.to_torch(dx).float(), x.grad, rtol=0.08, atol=0.05)
    for name, actual, expected in (
        ("dgamma", ttnn.to_torch(dgamma).float(), weight.grad),
        ("dbeta", ttnn.to_torch(dbeta).float(), bias.grad),
    ):
        logger.info(
            "{} max absolute error: {} (fp32_dest_acc_en={})",
            name,
            (actual - expected).abs().max().item(),
            fp32_dest_acc_en,
        )
        if fp32_dest_acc_en:
            torch.testing.assert_close(actual, expected, rtol=0.1, atol=0.5)
        else:
            # Long BF16 sums contain cancellation. The legacy kernel also
            # exceeds the elementwise bound near zero, so check relative L2
            # error while retaining elementwise checks for dx and FP32 sums.
            relative_error = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
            assert relative_error < 0.02, f"BF16 parameter-gradient relative error: {relative_error}"
