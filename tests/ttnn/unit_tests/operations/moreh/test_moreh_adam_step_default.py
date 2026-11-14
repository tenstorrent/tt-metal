# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

pytestmark = pytest.mark.use_module_device

SHAPE = (1, 1, 32, 32)
MOREH_OPS = [ttnn.moreh_adam, ttnn.moreh_adamw]
BASE_KWARGS = dict(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)


def _inputs():
    torch.manual_seed(0)
    param = torch.randn(SHAPE).bfloat16()
    grad = torch.randn(SHAPE).bfloat16()
    exp_avg = torch.zeros(SHAPE).bfloat16()
    exp_avg_sq = torch.zeros(SHAPE).bfloat16()
    return param, grad, exp_avg, exp_avg_sq


def _to_device(device, *tensors):
    return [ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for t in tensors]


def _torch_first_step(param, grad, decoupled):
    """One torch Adam/AdamW update from zero state at step 1."""
    beta1, beta2, eps, lr = BASE_KWARGS["beta1"], BASE_KWARGS["beta2"], BASE_KWARGS["eps"], BASE_KWARGS["lr"]
    weight_decay = BASE_KWARGS["weight_decay"]
    p, g = param.float(), grad.float()
    if decoupled and weight_decay != 0:
        p = p * (1 - lr * weight_decay)
    exp_avg = (1 - beta1) * g
    exp_avg_sq = (1 - beta2) * g * g
    bias_correction1 = 1 - beta1
    bias_correction2 = 1 - beta2
    step_size = lr / bias_correction1
    denom = (exp_avg_sq / bias_correction2).sqrt() + eps
    return p - step_size * exp_avg / denom


@pytest.mark.parametrize("op", MOREH_OPS, ids=["adam", "adamw"])
def test_moreh_adam_step_zero_raises(device, op, expect_error):
    tensors = _to_device(device, *_inputs())
    with expect_error(RuntimeError, "step must be >= 1"):
        op(*tensors, **BASE_KWARGS, step=0)


@pytest.mark.parametrize("op", MOREH_OPS, ids=["adam", "adamw"])
def test_moreh_adam_default_step_matches_step_one(device, op):
    param, grad, exp_avg, exp_avg_sq = _inputs()
    for step in (None, 1):
        tensors = _to_device(device, param, grad, exp_avg, exp_avg_sq)
        kwargs = dict(BASE_KWARGS)
        if step is not None:
            kwargs["step"] = step
        outs = op(*tensors, **kwargs)
        result = ttnn.to_torch(outs[0]).float().reshape(SHAPE)
        golden = _torch_first_step(param, grad, decoupled=op is ttnn.moreh_adamw)
        assert torch.allclose(result, golden, atol=0.01, rtol=0.01), (op, step)


@pytest.mark.parametrize("op", MOREH_OPS, ids=["adam", "adamw"])
def test_moreh_adam_explicit_step_two_matches_torch(device, op):
    """Control: explicit step is unchanged by the new default."""
    param, grad, exp_avg, exp_avg_sq = _inputs()
    tensors = _to_device(device, param, grad, exp_avg, exp_avg_sq)
    outs = op(*tensors, **BASE_KWARGS, step=2)
    result = ttnn.to_torch(outs[0]).float().reshape(SHAPE)

    beta1, beta2, eps, lr = BASE_KWARGS["beta1"], BASE_KWARGS["beta2"], BASE_KWARGS["eps"], BASE_KWARGS["lr"]
    g = grad.float()
    exp_avg = (1 - beta1) * g
    exp_avg_sq = (1 - beta2) * g * g
    bc1, bc2 = 1 - beta1**2, 1 - beta2**2
    denom = (exp_avg_sq / bc2).sqrt() + eps
    golden = param.float() - (lr / bc1) * exp_avg / denom
    assert torch.allclose(result, golden, atol=0.01, rtol=0.01)
