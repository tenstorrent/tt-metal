# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

import ttnn
from ttnn.operations import moreh


OPERATIONS = {
    "moreh_abs_pow": moreh.abs,
    "moreh_adam": moreh.adam,
    "moreh_adamw": moreh.adamw,
    "moreh_arange": moreh.arange,
    "moreh_bmm": moreh.bmm,
    "moreh_bmm_backward": moreh.bmm_backward,
    "moreh_clip_grad_norm": moreh.clip_grad_norm,
    "moreh_cumsum": moreh.cumsum,
    "moreh_cumsum_backward": moreh.cumsum_backward,
    "moreh_dot": moreh.dot,
    "moreh_dot_backward": moreh.dot_backward,
    "moreh_fold": moreh.fold,
    "moreh_full": moreh.full,
    "moreh_full_like": moreh.full_like,
    "moreh_getitem": moreh.getitem,
    "moreh_group_norm": moreh.group_norm,
    "moreh_group_norm_backward": moreh.group_norm_backward,
    "moreh_layer_norm": moreh.layer_norm,
    "moreh_layer_norm_backward": moreh.layer_norm_backward,
    "moreh_linear": moreh.linear,
    "moreh_linear_backward": moreh.linear_backward,
    "moreh_logsoftmax": moreh.logsoftmax,
    "moreh_logsoftmax_backward": moreh.logsoftmax_backward,
    "moreh_matmul": moreh.matmul,
    "moreh_matmul_backward": moreh.matmul_backward,
    "moreh_mean": moreh.mean,
    "moreh_mean_backward": moreh.mean_backward,
    "moreh_nll_loss": moreh.nll_loss,
    "moreh_nll_loss_backward": moreh.nll_loss_backward,
    "moreh_nll_loss_unreduced_backward": moreh.nll_loss_unreduced_backward,
    "moreh_norm": moreh.norm,
    "moreh_norm_backward": moreh.norm_backward,
    "moreh_sgd": moreh.sgd,
    "moreh_softmax": moreh.softmax,
    "moreh_softmax_backward": moreh.softmax_backward,
    "moreh_softmin": moreh.softmin,
    "moreh_softmin_backward": moreh.softmin_backward,
    "moreh_sum": moreh.sum,
    "moreh_sum_backward": moreh.sum_backward,
}


def golden(operation):
    return ttnn.get_golden_function(operation)


def test_all_planned_moreh_operations_have_golden_functions():
    assert len(OPERATIONS) == 39
    assert all(golden(operation) is not None for operation in OPERATIONS.values())


def test_forward_golden_formulas_and_preallocated_outputs():
    x = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    y = torch.tensor([[2.0, 1.0], [-1.0, 2.0]])
    output = torch.empty_like(x)

    assert golden(moreh.abs)(x, 3, output=output) is output
    torch.testing.assert_close(output, x.abs().pow(3))
    torch.testing.assert_close(golden(moreh.cumsum)(x, 1), torch.cumsum(x, 1))
    torch.testing.assert_close(golden(moreh.sum)(x, 0, keepdim=True), torch.sum(x, 0, keepdim=True))
    torch.testing.assert_close(golden(moreh.mean)(x, dim=(0, 1)), torch.mean(x))
    torch.testing.assert_close(golden(moreh.norm)(x, 2.0, dim=1), torch.norm(x, p=2.0, dim=1))
    torch.testing.assert_close(golden(moreh.matmul)(x, y), torch.matmul(x, y))
    torch.testing.assert_close(
        golden(moreh.bmm)(x.unsqueeze(0), y.unsqueeze(0)), torch.bmm(x.unsqueeze(0), y.unsqueeze(0))
    )
    torch.testing.assert_close(golden(moreh.linear)(x, y), F.linear(x, y))
    torch.testing.assert_close(golden(moreh.softmax)(x, 1), torch.softmax(x, 1))
    torch.testing.assert_close(golden(moreh.softmin)(x, 1), F.softmin(x, 1))
    torch.testing.assert_close(golden(moreh.logsoftmax)(x, 1), torch.log_softmax(x, 1))


def test_creation_indexing_fold_and_dot_golden_contracts():
    created = golden(moreh.full)((2, 3), 7, None, dtype=ttnn.int32)
    assert created.dtype == torch.int32
    torch.testing.assert_close(created, torch.full((2, 3), 7, dtype=torch.int32))

    source = torch.arange(24.0).reshape(2, 3, 4)
    torch.testing.assert_close(golden(moreh.full_like)(source, -2), torch.full_like(source, -2))
    torch.testing.assert_close(
        golden(moreh.arange)(1, 7, 2, None, dtype=ttnn.int32),
        torch.tensor([[1, 3, 5]], dtype=torch.int32),
    )
    assert golden(moreh.arange)(1, 7, 2, None, untilize_out=True).shape == (3,)
    index = torch.tensor([2, 0])
    torch.testing.assert_close(
        golden(moreh.getitem)(source, [index], [1]),
        source[:, index],
    )

    columns = torch.arange(16.0).reshape(1, 4, 4)
    torch.testing.assert_close(
        golden(moreh.fold)(columns, None, (3, 3), (2, 2)),
        F.fold(columns, (3, 3), (2, 2)),
    )
    dot_output = golden(moreh.dot)(source, source)
    assert dot_output.shape == (1, 1, 1, 1)
    torch.testing.assert_close(dot_output.reshape(()), torch.dot(source.reshape(-1), source.reshape(-1)))


def test_backward_golden_output_order_and_optional_slots():
    left = torch.randn(2, 3, 4)
    right = torch.randn(2, 4, 5)
    grad = torch.randn(2, 3, 5)
    left_out = torch.empty_like(left)

    bmm_grads = golden(moreh.bmm_backward)(
        grad,
        left,
        right,
        are_required_outputs=(True, False),
        input_grad=left_out,
    )
    assert isinstance(bmm_grads, list)
    assert bmm_grads[0] is left_out
    assert bmm_grads[1] is None
    torch.testing.assert_close(left_out, torch.bmm(grad, right.transpose(-1, -2)))

    broadcast_left = torch.randn(1, 3, 4)
    broadcast_right = torch.randn(2, 4, 5)
    broadcast_grad = torch.randn(2, 3, 5)
    broadcast_left_out = torch.empty_like(broadcast_left)
    broadcast_right_out = torch.empty_like(broadcast_right)
    matmul_grads = golden(moreh.matmul_backward)(
        broadcast_grad,
        broadcast_left,
        broadcast_right,
        are_required_outputs=(True, True),
        input_a_grad=broadcast_left_out,
        input_b_grad=broadcast_right_out,
    )
    assert matmul_grads[0] is broadcast_left_out
    assert matmul_grads[1] is broadcast_right_out
    torch.testing.assert_close(
        matmul_grads[0],
        torch.matmul(broadcast_grad, broadcast_right.transpose(-1, -2)).sum_to_size(broadcast_left.shape),
    )
    torch.testing.assert_close(
        matmul_grads[1],
        torch.matmul(broadcast_left.transpose(-1, -2), broadcast_grad).sum_to_size(broadcast_right.shape),
    )

    vector = torch.randn(7)
    scalar_grad = torch.tensor([[[[2.0]]]])
    vector_out = torch.empty_like(vector)
    dot_grads = golden(moreh.dot_backward)(
        scalar_grad,
        vector,
        vector + 1,
        input_grad=vector_out,
    )
    assert dot_grads[0] is vector_out
    assert dot_grads[1] is None
    torch.testing.assert_close(vector_out, (vector + 1) * 2)


def test_reduction_and_cumulative_backward_formulas():
    output_grad = torch.tensor([1.0, 2.0, 3.0])
    cumsum_grad = golden(moreh.cumsum_backward)(output_grad, 0)
    torch.testing.assert_close(cumsum_grad, torch.tensor([6.0, 5.0, 3.0]))

    reduced_grad = torch.arange(6.0).reshape(2, 3)
    sum_grad = golden(moreh.sum_backward)(
        reduced_grad,
        input=torch.empty(2, 4, 3),
        dim=1,
        keepdim=False,
    )
    torch.testing.assert_close(sum_grad, reduced_grad.unsqueeze(1).expand(2, 4, 3))

    mean_grad = golden(moreh.mean_backward)(
        reduced_grad,
        dim=1,
        input_grad_shape=(2, 4, 3),
    )
    torch.testing.assert_close(mean_grad, reduced_grad.unsqueeze(1).expand(2, 4, 3) / 4)


def test_softmax_family_backward_formulas():
    x = torch.randn(2, 5, requires_grad=True)
    upstream = torch.randn_like(x)

    for operation, reference in (
        (moreh.softmax_backward, lambda value: torch.softmax(value, 1)),
        (moreh.softmin_backward, lambda value: F.softmin(value, 1)),
        (moreh.logsoftmax_backward, lambda value: torch.log_softmax(value, 1)),
    ):
        x.grad = None
        output = reference(x)
        output.backward(upstream)
        torch.testing.assert_close(golden(operation)(output.detach(), upstream, 1), x.grad)


def test_normalization_forward_optional_output_contracts():
    x = torch.randn(2, 4, 3, 3)
    gamma = torch.randn(4)
    beta = torch.randn(4)
    mean_out = torch.empty(2, 2)
    rstd_out = torch.empty(2, 2)
    group_outputs = golden(moreh.group_norm)(
        x,
        2,
        gamma=gamma,
        beta=beta,
        are_required_outputs=(True, True, True),
        mean=mean_out,
        rstd=rstd_out,
    )
    assert group_outputs[1] is mean_out
    assert group_outputs[2] is rstd_out
    torch.testing.assert_close(group_outputs[0], F.group_norm(x, 2, gamma, beta))

    layer_x = torch.randn(2, 3, 4)
    layer_outputs = golden(moreh.layer_norm)(layer_x, 1)
    assert isinstance(layer_outputs, list)
    assert layer_outputs[1:] == [None, None]
    torch.testing.assert_close(layer_outputs[0], F.layer_norm(layer_x, (4,)))


def test_group_norm_backward_matches_torch_autograd():
    x = torch.randn(2, 4, 3, 3, requires_grad=True)
    gamma = torch.randn(4, requires_grad=True)
    upstream = torch.randn_like(x)
    reference = F.group_norm(x, 2, gamma)
    reference.backward(upstream)

    grouped = x.detach().reshape(2, 2, -1)
    mean = grouped.mean(dim=-1).reshape(1, 1, 2, 2)
    rstd = (grouped.var(dim=-1, unbiased=False) + 1e-5).rsqrt().reshape(1, 1, 2, 2)
    input_grad = torch.empty_like(x)
    gamma_grad = torch.empty(1, 1, 1, 4)
    beta_grad = torch.empty(1, 1, 1, 4)
    actual = golden(moreh.group_norm_backward)(
        upstream,
        x.detach(),
        mean,
        rstd,
        2,
        are_required_outputs=(True, True, True),
        gamma=gamma.detach(),
        input_grad=input_grad,
        gamma_grad=gamma_grad,
        beta_grad=beta_grad,
    )
    assert actual[0] is input_grad
    assert actual[1] is gamma_grad
    assert actual[2] is beta_grad
    torch.testing.assert_close(input_grad, x.grad)
    torch.testing.assert_close(gamma_grad.reshape(4), gamma.grad)
    torch.testing.assert_close(beta_grad.reshape(4), upstream.sum(dim=(0, 2, 3)))


def test_linear_and_layer_norm_backward_optional_output_contracts():
    x = torch.randn(2, 3, 4)
    weight = torch.randn(5, 4)
    upstream = torch.randn(2, 3, 5)
    weight_out = torch.empty_like(weight)
    linear_grads = golden(moreh.linear_backward)(
        upstream,
        x,
        weight,
        are_required_outputs=(False, True, False),
        weight_grad=weight_out,
    )
    assert linear_grads[0] is None
    assert linear_grads[1] is weight_out
    assert linear_grads[2] is None
    torch.testing.assert_close(weight_out, upstream.reshape(-1, 5).T @ x.reshape(-1, 4))

    normalized_dims = 2
    dims = (-2, -1)
    mean = x.mean(dim=dims)
    rstd = (x.var(dim=dims, unbiased=False) + 1e-5).rsqrt()
    input_grad_out = torch.empty_like(x)
    gamma = torch.randn(3, 4)
    gamma_grad_out = torch.empty_like(gamma)
    beta_grad_out = torch.empty_like(gamma)
    layer_grads = golden(moreh.layer_norm_backward)(
        upstream[..., :4],
        x,
        mean,
        rstd,
        normalized_dims,
        gamma=gamma,
        input_grad=input_grad_out,
        gamma_grad=gamma_grad_out,
        beta_grad=beta_grad_out,
    )
    assert layer_grads[0] is input_grad_out
    assert layer_grads[1] is gamma_grad_out
    assert layer_grads[2] is beta_grad_out

    reference_x = x.detach().requires_grad_()
    reference_gamma = gamma.detach().requires_grad_()
    reference = F.layer_norm(reference_x, (3, 4), reference_gamma)
    reference.backward(upstream[..., :4])
    torch.testing.assert_close(input_grad_out, reference_x.grad)
    torch.testing.assert_close(gamma_grad_out, reference_gamma.grad)
    torch.testing.assert_close(beta_grad_out, upstream[..., :4].sum(dim=0))


def test_norm_backward_matches_torch_autograd():
    x = torch.randn(2, 3, 4, requires_grad=True)
    output = torch.norm(x, p=2.5, dim=(1, 2))
    upstream = torch.randn_like(output)
    output.backward(upstream)
    actual = golden(moreh.norm_backward)(x.detach(), output.detach(), upstream, 2.5, dim=(1, 2))
    torch.testing.assert_close(actual, x.grad)


def test_optimizer_golden_order_and_preallocated_outputs():
    param = torch.tensor([1.0, 2.0])
    grad = torch.tensor([0.5, -0.25])
    exp_avg = torch.tensor([0.1, 0.2])
    exp_avg_sq = torch.tensor([0.3, 0.4])
    outputs = [torch.empty_like(param) for _ in range(3)]
    adam_outputs = golden(moreh.adam)(
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        lr=0.01,
        step=2,
        param_out=outputs[0],
        exp_avg_out=outputs[1],
        exp_avg_sq_out=outputs[2],
    )
    assert all(actual is expected for actual, expected in zip(adam_outputs[:3], outputs))
    assert adam_outputs[3] is None
    expected_m = 0.9 * exp_avg + 0.1 * grad
    expected_v = 0.999 * exp_avg_sq + 0.001 * grad.square()
    expected_param = param - (0.01 / (1 - 0.9**2)) * (
        expected_m / (expected_v.sqrt() / ((1 - 0.999**2) ** 0.5) + 1e-8)
    )
    torch.testing.assert_close(adam_outputs[0], expected_param)
    torch.testing.assert_close(adam_outputs[1], expected_m)
    torch.testing.assert_close(adam_outputs[2], expected_v)

    adamw_outputs = golden(moreh.adamw)(
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        0.01,
        step=2,
    )
    assert len(adamw_outputs) == 4
    assert adamw_outputs[3] is None

    sgd_outputs = golden(moreh.sgd)(
        param,
        grad,
        lr=0.1,
        momentum=0,
        momentum_initialized=False,
    )
    assert sgd_outputs[1] is None
    torch.testing.assert_close(sgd_outputs[0], param - 0.1 * grad)

    momentum_out = torch.empty_like(param)
    sgd_with_momentum = golden(moreh.sgd)(
        param,
        grad,
        torch.tensor([0.2, 0.3]),
        momentum_buffer_out=momentum_out,
        lr=0.1,
        momentum=0.9,
        dampening=0.1,
        momentum_initialized=True,
    )
    expected_momentum = 0.9 * torch.tensor([0.2, 0.3]) + 0.9 * grad
    assert sgd_with_momentum[1] is momentum_out
    torch.testing.assert_close(momentum_out, expected_momentum)
    torch.testing.assert_close(sgd_with_momentum[0], param - 0.1 * expected_momentum)


def test_nll_loss_forward_and_backward_golden_contracts():
    log_probs = torch.log_softmax(torch.randn(2, 3, 2), dim=1)
    target = torch.tensor([[0, 2], [1, 0]], dtype=torch.int32)
    weight = torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(
        golden(moreh.nll_loss)(log_probs, target, "none", weight_tensor=weight),
        F.nll_loss(log_probs, target.long(), weight, reduction="none"),
    )
    divisor_out = torch.empty(1)
    loss_out = torch.empty(1)
    reduced = golden(moreh.nll_loss)(
        log_probs,
        target,
        "mean",
        weight_tensor=weight,
        divisor_tensor=divisor_out,
        output_tensor=loss_out,
    )
    assert reduced is loss_out
    torch.testing.assert_close(reduced, F.nll_loss(log_probs, target.long(), weight, reduction="mean").reshape(1))
    torch.testing.assert_close(divisor_out, weight[target.long()].sum().reshape(1))

    input_grad = torch.empty_like(log_probs)
    divisor = weight[target.long()].sum().reshape(1)
    result = golden(moreh.nll_loss_backward)(
        target,
        torch.tensor([2.0]),
        True,
        weight_tensor=weight,
        input_grad_tensor=input_grad,
        divisor_tensor=divisor,
    )
    expected = torch.zeros_like(log_probs)
    expected.scatter_(1, target.long().unsqueeze(1), (-2 * weight[target.long()] / divisor).unsqueeze(1))
    assert result is input_grad
    torch.testing.assert_close(result, expected)

    unreduced_grad = torch.empty_like(log_probs)
    unreduced_upstream = torch.randn_like(target, dtype=log_probs.dtype)
    actual_unreduced = golden(moreh.nll_loss_unreduced_backward)(
        target,
        unreduced_upstream,
        weight_tensor=weight,
        input_grad_tensor=unreduced_grad,
    )
    expected_unreduced = torch.zeros_like(log_probs)
    expected_unreduced.scatter_(
        1,
        target.long().unsqueeze(1),
        (-unreduced_upstream * weight[target.long()]).unsqueeze(1),
    )
    assert actual_unreduced is unreduced_grad
    torch.testing.assert_close(actual_unreduced, expected_unreduced)


def test_clip_grad_norm_mutates_host_inputs_and_returns_preclip_norm():
    inputs = [torch.tensor([3.0, 4.0]), torch.tensor([0.0])]
    total_norm_out = torch.empty(1)
    result = golden(moreh.clip_grad_norm)(inputs, 2.5, total_norm=total_norm_out)
    assert result is total_norm_out
    torch.testing.assert_close(result, torch.tensor([5.0]))
    torch.testing.assert_close(inputs[0], torch.tensor([1.5, 2.0]), rtol=1e-5, atol=1e-5)
