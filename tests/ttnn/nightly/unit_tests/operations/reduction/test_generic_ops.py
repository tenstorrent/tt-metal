# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Corner cases of the generic reductions (sum/mean/max/min/prod/std/var).
# Split from test_reduction_ops.py: corner-case tests, not exhaustive sweeps.

import contextlib

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose_and_pcc
from tests.ttnn.nightly.unit_tests.operations.reduction.utility_functions import (
    TTNN_REDUCTION_WRAPPERS,
)
from loguru import logger

# Module-scoped device: these tests all run with the default device config, so the
# device is opened once per file (one device context per test group) instead of
# once per test case.
pytestmark = pytest.mark.use_module_device


SHAPES = [(), (2,), (1, 1), (32, 1), (6, 0, 32), (3, 6, 40, 63, 20), (4, 8, 32, 64), (2, 4, 8, 32, 64)]
DIMS = [None, 0, -1, (-2, -1), (0, 2), (0, 2, 4), (0, 2, 3), (0, 3, 4), (1, 2, 3)]


def dims_valid(shape, dim):
    """torch dim-validity: rank-0 tensors accept dims {0, -1}; otherwise every
    axis must satisfy -rank <= d < rank. Invalid (shape, dim) combos raise in
    both frameworks and are covered by test_generic_ops_dim_parity in
    test_reduction_op_corners.py instead of being crossed here."""
    rank = len(shape)
    axes = dim if isinstance(dim, tuple) else (dim,)
    if dim is None:
        return True
    if rank == 0:
        return all(d in (0, -1) for d in axes)
    return all(-rank <= d < rank for d in axes)


VALID_SHAPE_DIMS = [(s, d) for s in SHAPES for d in DIMS if dims_valid(s, d)]

# correction is live only for std/var; pairing it with the op (instead of
# crossing) removes the 720 always-skipped combinations.
OP_CORRECTION = [
    ("mean", False),
    ("sum", False),
    ("max", False),
    ("min", False),
    ("prod", False),
    ("std", False),
    ("std", True),
    ("var", False),
    ("var", True),
]


@pytest.mark.parametrize("tensor_shape, dim", VALID_SHAPE_DIMS)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("op, correction", OP_CORRECTION)
def test_generic_ops(device, tensor_shape, dim, keepdim, dtype, layout, correction, op, expect_error):
    """
    Test the compatibility of the torch and ttnn output for the given operation and different
    tensor shapes, keepdim, and dim values.
    Checks that resulting tensors are within a certain tolerance of PyTorch outputs.
    Some operations raise exceptions in torch, we check if the same behavior is observed in ttnn.
    Note: We do not enforce the same exception type or message between PyTorch and ttnn.
    """
    torch.manual_seed(0)
    torch_tensor = torch.randn(tensor_shape, dtype=dtype)
    pad_value = 1.0 if op == "prod" else None
    ttnn_tensor = ttnn.from_torch(torch_tensor, layout=layout, device=device, pad_value=pad_value)

    # torch.max/min don't accept a tuple for dim; use amax/amin which do.
    torch_op_name = {"max": "amax", "min": "amin"}.get(op, op)
    torch_op = getattr(torch, torch_op_name)

    ttnn_op = TTNN_REDUCTION_WRAPPERS[op]

    # Run on both and flag exceptions
    torch_errored = False
    try:
        # tensor.size, which is called by various torch reduction ops, doesn't accept dim=None,
        # so we need to handle it separately.
        # See https://github.com/pytorch/pytorch/issues/127882
        if dim is None:
            # PyTorch supports the correction argument only for var and std.
            # ttnn supports it for all except prod, but it is ignored for all except var and std.
            if op in ("var", "std"):
                torch_result = torch_op(torch_tensor, correction=correction)
            else:
                torch_result = torch_op(torch_tensor)
            if keepdim:
                # Various torch ops don't support keepdim=True for dim=None,
                # so we need to reshape to match the input tensor.
                new_shape = [1] * torch_tensor.dim()
                torch_result = torch_result.reshape(new_shape)
        else:
            if op in ("var", "std"):
                torch_result = torch_op(torch_tensor, dim=dim, keepdim=keepdim, correction=correction)
            else:
                torch_result = torch_op(torch_tensor, dim=dim, keepdim=keepdim)
    except (IndexError, TypeError, RuntimeError) as e:
        logger.info(f"torch {op} raised: {e}")
        torch_errored = True

    # torch has already run, so its outcome is the expectation for ttnn: the two must
    # fail on exactly the same inputs. Bracketing the provoked failure marks it as
    # expected so CI log triage does not read it as a crash.
    ctx = (
        expect_error(
            (IndexError, TypeError, RuntimeError),
            # prod takes a scalar dim, so a tuple is rejected by the binding rather
            # than by the device-side zero-size check the other ops hit.
            "Expected reduction dim|incompatible function arguments",
        )
        if torch_errored
        else contextlib.nullcontext()
    )
    with ctx:
        # ttnn.prod doesn't support the correction argument.
        if op == "prod":
            ttnn_result = ttnn_op(ttnn_tensor, dim=dim, keepdim=keepdim)
        else:
            ttnn_result = ttnn_op(ttnn_tensor, dim=dim, keepdim=keepdim, correction=correction)
    if torch_errored:
        return

    # torch.min/max does not return just a tensor like ttnn.min/max.
    # It returns a small named tuple type (torch.return_types.min or torch.return_types.max) with:
    # .values – the min/max values (tensor of reduced values)
    # .indices – the indices where those values occur (equivalent to ttnn.argmin/argmax)
    # To make comparison with ttnn meaningful, extract the values only
    if isinstance(torch_result, (torch.return_types.min, torch.return_types.max)):
        torch_result = torch_result.values

    ttnn_result = ttnn.to_torch(ttnn.from_device(ttnn_result))

    if op == "sum" and tensor_shape == (3, 6, 40, 63, 20):
        # Summing large number of bfloat16 values accumulates rounding errors,
        # and results also vary from near 0 to relatively large values (in hundreds)
        # PCC should catch any significant errors.
        atol = 1.5
    else:
        atol = 0.1

    if op == "var":
        # For var/std there are cases where all output values are close to 1, and we're using bfloat16,
        # so even a rounding error of 0.5 ULP has a significant impact on PCC.
        pcc = 0.99
    elif op == "std":
        # For std, sqrtf() adds an extra rounding step on top of variance, further
        # lowering PCC when values cluster near 1.0 (e.g. 3-dim reduction on large tensors).
        # Therefore PCC threshold has to be lower. ATOL and RTOL should catch any significant errors.
        pcc = 0.98
    else:
        pcc = 0.999

    rtol = 0.05

    passing, output_pcc = comp_allclose_and_pcc(torch_result, ttnn_result, pcc=pcc, rtol=rtol, atol=atol)
    assert passing, f"{output_pcc}, torch: {torch_result}, ttnn: {ttnn_result}"
