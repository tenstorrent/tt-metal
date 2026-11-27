# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Scalars of binary_ng's fused activations are runtime args, not compiled into the kernel (#62051).

Each case runs one binary op with an activation at several scalar values on one device. Only the first value may
compile a program; the others must hit it. The output for every value must match, bit for bit, what that value
gives on an empty program cache, so a scalar that is not refreshed on a cache hit fails. Outputs of distinct values
must differ, or the case could not tell a stale scalar from a fresh one.
"""

import pytest
import torch

import ttnn

pytestmark = pytest.mark.use_module_device

Op = ttnn.UnaryOpType


def P(op_type, *params):
    return ttnn.UnaryWithParam(op_type, *(float(p) for p in params))


def _input(device, shape, dtype, seed):
    torch.manual_seed(seed)
    host = torch.rand(shape, dtype=torch.float32) * 8 - 4
    return ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _bits(tensor):
    out = ttnn.to_torch(tensor)
    return out.view(torch.int16 if out.dtype == torch.bfloat16 else torch.int32)


def _value(bits):
    return bits.view(torch.bfloat16 if bits.dtype == torch.int16 else torch.float32).float()


def _check_runtime_scalars(device, calls, expected):
    device.enable_program_cache()
    device.clear_program_cache()
    warm = []
    entries_after_first = None
    for i, call in enumerate(calls):
        warm.append(_bits(call()))
        torch.testing.assert_close(_value(warm[i]), expected[i](), rtol=0.02, atol=0.05, msg=f"value {i}")
        entries = device.num_program_cache_entries()
        if i == 0:
            entries_after_first = entries
        else:
            assert entries == entries_after_first, f"value {i} compiled a new program ({entries} entries)"

    for i, call in enumerate(calls):
        device.clear_program_cache()
        assert torch.equal(warm[i], _bits(call())), f"value {i} differs between a cache hit and a fresh program"

    for i in range(1, len(calls)):
        if warm[0].shape == warm[i].shape:
            assert not torch.equal(warm[0], warm[i]), f"values 0 and {i} give the same output; the case is not testing"


# side, then (activation, torch reference) per value
ACTIVATION_CASES = {
    "post-relu_max": ("post", [(P(Op.RELU_MAX, v), lambda x, v=v: x.clamp(0, v)) for v in (0.5, 2.0, 3.5)]),
    "post-leaky_relu": (
        "post",
        [(P(Op.LEAKY_RELU, v), lambda x, v=v: torch.nn.functional.leaky_relu(x, v)) for v in (0.01, 0.2, 0.5)],
    ),
    "post-hardtanh": (
        "post",
        [
            (P(Op.HARDTANH, lo, hi), lambda x, lo=lo, hi=hi: x.clamp(lo, hi))
            for lo, hi in ((-1.0, 1.0), (-0.5, 2.0), (0.0, 0.25))
        ],
    ),
    "lhs-mul": ("lhs", [(P(Op.MUL_UNARY_SFPU, v), lambda x, v=v: x * v) for v in (0.5, 2.0, -3.0)]),
    "rhs-add": ("rhs", [(P(Op.ADD_UNARY_SFPU, v), lambda x, v=v: x + v) for v in (0.5, 2.0, -3.0)]),
}


def _kwargs(side, activation):
    return {
        "post": {"activations": [activation]},
        "lhs": {"input_tensor_a_activations": [activation]},
        "rhs": {"input_tensor_b_activations": [activation]},
    }[side]


def _reference(side, fn, a, b):
    a = ttnn.to_torch(a).float()
    b = ttnn.to_torch(b).float() if isinstance(b, ttnn.Tensor) else b
    if side == "post":
        return fn(a + b)
    if side == "lhs":
        return fn(a) + b
    return a + fn(b)


def _check_case(device, case, a, b):
    side, activations = ACTIVATION_CASES[case]
    _check_runtime_scalars(
        device,
        [lambda act=act: ttnn.add(a, b, **_kwargs(side, act)) for act, _ in activations],
        [lambda fn=fn: _reference(side, fn, a, b) for _, fn in activations],
    )


# bf16 add runs the FPU kernels and fp32 add the SFPU ones.
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("case", ACTIVATION_CASES.keys())
@pytest.mark.parametrize(
    "a_shape,b_shape",
    [((1, 1, 64, 128), (1, 1, 64, 128)), ((1, 2, 64, 128), (1, 1, 1, 128))],
    ids=["same-shape", "row-broadcast"],
)
def test_activation_scalars_reuse_program(device, dtype, case, a_shape, b_shape):
    _check_case(device, case, _input(device, a_shape, dtype, 0), _input(device, b_shape, dtype, 1))


@pytest.mark.parametrize("case", ["post-hardtanh", "lhs-mul"])
def test_scalar_operand_activation_scalars_reuse_program(device, case):
    _check_case(device, case, _input(device, (1, 1, 64, 128), ttnn.bfloat16, 0), 1.5)


def test_activation_scalars_follow_shape_change(device):
    """A cache hit with a new shape moves cores between work and idle; every work core must get the new scalar."""
    small = _input(device, (1, 1, 32, 32), ttnn.bfloat16, 0), _input(device, (1, 1, 32, 32), ttnn.bfloat16, 1)
    large = _input(device, (1, 4, 128, 256), ttnn.bfloat16, 2), _input(device, (1, 4, 128, 256), ttnn.bfloat16, 3)
    runs = [(small, 0.5), (large, 2.0), (small, 3.5)]
    _check_runtime_scalars(
        device,
        [lambda ab=ab, v=v: ttnn.add(*ab, activations=[P(Op.RELU_MAX, v)]) for ab, v in runs],
        [lambda ab=ab, v=v: _reference("post", lambda x: x.clamp(0, v), *ab) for ab, v in runs],
    )
