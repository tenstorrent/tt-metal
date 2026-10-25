# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Binary and ternary backward ops return each gradient in its operand's shape when an operand was broadcast,
as torch.autograd does (the contract in tests/ttnn/backward_shape_matrix.py)."""

import pytest
import torch
import ttnn

from tests.ttnn.backward_shape_matrix import broadcast_grad_shape, matrix_ids, matrix_shapes


def _to_device(t, device, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _inputs(shapes, torch_dtype, seed):
    torch.manual_seed(seed)
    return [(torch.rand(s) + 0.5).to(torch_dtype).float() for s in shapes]


def _torch_grads(fn, inputs, grad):
    """torch.autograd gradients, each paired with the summed magnitude of its unreduced terms.

    A broadcast operand's gradient is a sum over the broadcast axes; the magnitude bounds the rounding error of that
    sum, which a tolerance relative to the (possibly cancelling) total cannot.
    """
    leaves = [x.clone().requires_grad_() for x in inputs]
    fn(*leaves).backward(grad)
    full = [x.expand(grad.shape).clone().requires_grad_() for x in inputs]
    fn(*full).backward(grad)
    return [(x.grad, f.grad.abs().sum_to_size(x.shape)) for x, f in zip(leaves, full)]


def _check(outputs, expected):
    for i, entry in enumerate(expected):
        if entry is None:
            continue
        exp, magnitude = entry
        got = ttnn.to_torch(outputs[i]).float()
        assert list(got.shape) == list(exp.shape), f"gradient {i}: shape {list(got.shape)}, want {list(exp.shape)}"
        tol = 2e-2 * max(1.0, float(exp.abs().max())) + 2**-9 * magnitude
        assert bool(((got - exp).abs() <= tol).all()), f"gradient {i}: max error {(got - exp).abs().max()}"


BINARY_OPS = {
    "atan2_bw": (torch.atan2, ttnn.atan2_bw),
    "xlogy_bw": (torch.xlogy, ttnn.xlogy_bw),
    "hypot_bw": (torch.hypot, ttnn.hypot_bw),
    "ldexp_bw": (torch.ldexp, ttnn.ldexp_bw),
    "logaddexp_bw": (torch.logaddexp, ttnn.logaddexp_bw),
    "logaddexp2_bw": (torch.logaddexp2, ttnn.logaddexp2_bw),
    "fmod_bw": (torch.fmod, ttnn.fmod_bw),
    "rsub_bw": (lambda a, b: b - a, ttnn.rsub_bw),
    "add_bw": (torch.add, ttnn.add_bw),
    "sub_bw": (torch.sub, ttnn.sub_bw),
    "addalpha_bw": (lambda a, b: a + 2.5 * b, lambda g, a, b, **kw: ttnn.addalpha_bw(g, a, b, 2.5, **kw)),
    "subalpha_bw": (lambda a, b: a - 2.5 * b, lambda g, a, b, **kw: ttnn.subalpha_bw(g, a, b, 2.5, **kw)),
    "div_bw": (torch.div, ttnn.div_bw),
}

# Ops that take are_required_outputs and preallocated input_grad / other_grad.
OPTIONAL_OUTPUT_OPS = ["rsub_bw", "add_bw", "sub_bw", "addalpha_bw", "subalpha_bw", "div_bw"]
# Keyword names of the preallocated outputs, where an op does not use input_grad / other_grad.
GRAD_KWARGS = {"addalpha_bw": ("input_a_grad", "input_b_grad")}


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("input_shape,other_shape", matrix_shapes(), ids=matrix_ids())
@pytest.mark.parametrize("op", list(BINARY_OPS))
def test_binary_bw_broadcast_grad_shape(op, input_shape, other_shape, dtype, device):
    torch_fn, ttnn_fn = BINARY_OPS[op]
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    a, b = _inputs([input_shape, other_shape], torch_dtype, seed=0)
    grad = _inputs([broadcast_grad_shape(input_shape, other_shape)], torch_dtype, seed=1)[0] - 1.0
    expected = _torch_grads(torch_fn, [a, b], grad)

    outputs = ttnn_fn(*[_to_device(t, device, dtype) for t in (grad, a, b)])
    _check(outputs, expected)


@pytest.mark.parametrize("are_required_outputs", [[True, True], [True, False], [False, True]])
@pytest.mark.parametrize("preallocate", [False, True])
@pytest.mark.parametrize(
    "input_shape,other_shape", [((1, 3, 32, 64), (1, 1, 1, 64)), ((1, 3, 32, 1), (1, 1, 32, 64))], ids=["row", "both"]
)
@pytest.mark.parametrize("op", OPTIONAL_OUTPUT_OPS)
def test_binary_bw_broadcast_grad_shape_outputs(
    op, input_shape, other_shape, preallocate, are_required_outputs, device
):
    torch_fn, ttnn_fn = BINARY_OPS[op]
    a, b = _inputs([input_shape, other_shape], torch.bfloat16, seed=0)
    grad = _inputs([broadcast_grad_shape(input_shape, other_shape)], torch.bfloat16, seed=1)[0] - 1.0
    expected = _torch_grads(torch_fn, [a, b], grad)
    expected = [e if required else None for e, required in zip(expected, are_required_outputs)]

    input_kw, other_kw = GRAD_KWARGS.get(op, ("input_grad", "other_grad"))
    kwargs = {"are_required_outputs": are_required_outputs}
    if preallocate:
        if are_required_outputs[0]:
            kwargs[input_kw] = _to_device(torch.zeros(input_shape), device, ttnn.bfloat16)
        if are_required_outputs[1]:
            kwargs[other_kw] = _to_device(torch.zeros(other_shape), device, ttnn.bfloat16)
    outputs = ttnn_fn(*[_to_device(t, device, ttnn.bfloat16) for t in (grad, a, b)], **kwargs)
    for i, required in enumerate(are_required_outputs):
        if not required:
            assert outputs[i] is None
    _check(outputs, expected)
    if preallocate:
        _check([kwargs.get(input_kw), kwargs.get(other_kw)], expected)


def test_div_bw_rounding_mode_broadcast_grad_shape(device):
    a, b = _inputs([(1, 3, 32, 64), (1, 1, 1, 64)], torch.bfloat16, seed=0)
    grad = torch.ones(1, 3, 32, 64)
    outputs = ttnn.div_bw(*[_to_device(t, device, ttnn.bfloat16) for t in (grad, a, b)], rounding_mode="floor")
    _check(outputs, [(torch.zeros(1, 3, 32, 64), 0.0), (torch.zeros(1, 1, 1, 64), 0.0)])


FULL = (1, 3, 32, 64)
TERNARY_BROADCASTS = [(1, 1, 1, 64), (1, 3, 32, 1), (1, 1, 1, 1), (32, 64), (1, 1, 32, 64)]
TERNARY_CASES = [(side, shape) for side in range(3) for shape in TERNARY_BROADCASTS]


def _ternary_shapes(side, shape):
    shapes = [FULL, FULL, FULL]
    shapes[side] = shape
    return shapes


TERNARY_OPS = {
    "addcmul_bw": (
        lambda a, x, y: torch.addcmul(a, x, y, value=1.5),
        lambda g, a, x, y: ttnn.addcmul_bw(g, a, x, y, 1.5),
    ),
    "addcdiv_bw": (
        lambda a, x, y: torch.addcdiv(a, x, y, value=1.5),
        lambda g, a, x, y: ttnn.addcdiv_bw(g, a, x, y, 1.5),
    ),
    "lerp_bw": (torch.lerp, ttnn.lerp_bw),
}


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("side,shape", TERNARY_CASES, ids=[f"operand{s}-{list(p)}" for s, p in TERNARY_CASES])
@pytest.mark.parametrize("op", list(TERNARY_OPS))
def test_ternary_bw_broadcast_grad_shape(op, side, shape, dtype, device):
    torch_fn, ttnn_fn = TERNARY_OPS[op]
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    inputs = _inputs(_ternary_shapes(side, shape), torch_dtype, seed=0)
    grad = _inputs([FULL], torch_dtype, seed=1)[0] - 1.0
    expected = _torch_grads(torch_fn, inputs, grad)

    outputs = ttnn_fn(*[_to_device(t, device, dtype) for t in [grad] + inputs])
    _check(outputs, expected)


@pytest.mark.parametrize("side,shape", TERNARY_CASES[:10], ids=[f"operand{s}-{list(p)}" for s, p in TERNARY_CASES[:10]])
def test_lerp_bw_scalar_weight_broadcast_grad_shape(side, shape, device):
    inputs = _inputs(_ternary_shapes(side, shape)[:2], torch.bfloat16, seed=0)
    grad = _inputs([FULL], torch.bfloat16, seed=1)[0] - 1.0
    expected = _torch_grads(lambda a, e: torch.lerp(a, e, 0.3), inputs, grad)

    outputs = ttnn.lerp_bw(*[_to_device(t, device, ttnn.bfloat16) for t in [grad] + inputs], 0.3)
    _check(outputs, expected)


@pytest.mark.parametrize("are_required_outputs", [[True, True], [True, False], [False, True]])
@pytest.mark.parametrize("preallocate", [False, True])
@pytest.mark.parametrize("side,shape", TERNARY_CASES[:10], ids=[f"operand{s}-{list(p)}" for s, p in TERNARY_CASES[:10]])
def test_where_bw_broadcast_grad_shape(side, shape, preallocate, are_required_outputs, device):
    shapes = _ternary_shapes(side, shape)[:2]
    a, b = _inputs(shapes, torch.bfloat16, seed=0)
    torch.manual_seed(2)
    condition = (torch.rand(FULL) > 0.5).float()
    grad = _inputs([FULL], torch.bfloat16, seed=1)[0] - 1.0
    expected = _torch_grads(lambda x, y: torch.where(condition.bool(), x, y), [a, b], grad)
    expected = [e if required else None for e, required in zip(expected, are_required_outputs)]

    kwargs = {"are_required_outputs": are_required_outputs}
    if preallocate:
        if are_required_outputs[0]:
            kwargs["input_a_grad"] = _to_device(torch.zeros(shapes[0]), device, ttnn.bfloat16)
        if are_required_outputs[1]:
            kwargs["input_b_grad"] = _to_device(torch.zeros(shapes[1]), device, ttnn.bfloat16)
    outputs = ttnn.where_bw(*[_to_device(t, device, ttnn.bfloat16) for t in (grad, condition, a, b)], **kwargs)
    _check(outputs, expected)
    if preallocate:
        _check([kwargs.get("input_a_grad"), kwargs.get("input_b_grad")], expected)
