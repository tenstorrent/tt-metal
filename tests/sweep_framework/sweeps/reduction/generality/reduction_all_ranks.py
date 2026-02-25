# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import pytest
import random
import sys
import torch
import ttnn
from loguru import logger

from tests.sweep_framework.sweep_utils.utils import gen_pytest_parametrize_args
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.sweep_framework.sweep_utils.roofline_utils import get_run_return

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)

DIM_SIZES = [0, 32]
"""Possible tensor dimensions are picked from this list"""

GENERIC_OPS = ["sum", "mean", "max", "min", "std", "var", "prod", "argmax"]
ACCUMULATION_OPS = ["cumsum", "cumprod"]

# rank_0 → 0D (scalar), rank_1 → 1D, rank_2+ with DIM_SIZES=[0,32] → 0-volume (e.g. (0,32), (32,0)).
parameters = {
    f"rank_{rank}": {
        "tensor_shape": list(itertools.product(DIM_SIZES, repeat=rank)),
        "dim": list(range(-rank, rank)) if rank > 0 else [None],  # Rank 0 has no dimensions
        "keepdim": [True, False],
        "op": GENERIC_OPS,
        "dtype": [torch.bfloat16, torch.float32],
    }
    for rank in range(5)
}

# Accumulation ops produce same-shape output; keepdim is not applicable
for rank in range(1, 5):
    parameters[f"rank_{rank}_accumulation"] = {
        "tensor_shape": list(itertools.product(DIM_SIZES, repeat=rank)),
        "dim": list(range(-rank, rank)),
        "keepdim": [False],
        "op": ACCUMULATION_OPS,
        "dtype": [torch.bfloat16, torch.float32],
    }

# Include 1D (32,) and 0-volume (0, 32) so the suite covers 0D/1D/0-volume.
parameters["topk"] = {
    "tensor_shape": [(1, 64), (1, 128), (32,), (0, 32)],
    "dim": [-1],
    "keepdim": [False],
    "op": ["topk"],
    "dtype": [torch.bfloat16],
}

# Include 1D (32,) and 0-volume (1, 2, 0, 128).
parameters["ema"] = {
    "tensor_shape": [(1, 2, 64, 128), (32,), (1, 2, 0, 128)],
    "dim": [None],
    "keepdim": [False],
    "op": ["ema"],
    "dtype": [torch.bfloat16],
}

# Include 0-volume (1, 1, 0, 64).
parameters["moe"] = {
    "tensor_shape": [(1, 1, 32, 64), (1, 1, 0, 64)],
    "dim": [None],
    "keepdim": [False],
    "op": ["moe"],
    "dtype": [torch.bfloat16],
}

# Include 0-volume (1, 1, 0, 64).
parameters["sampling"] = {
    "tensor_shape": [(1, 1, 32, 64), (1, 1, 0, 64)],
    "dim": [None],
    "keepdim": [False],
    "op": ["sampling"],
    "dtype": [torch.bfloat16],
}

parameters["manual_seed"] = {
    "tensor_shape": [(1,)],
    "dim": [None],
    "keepdim": [False],
    "op": ["manual_seed"],
    "dtype": [torch.bfloat16],
}


def _run_generic_reduction(device, tensor_shape, dim, keepdim, op, dtype) -> list:
    """
    Test generic reduction ops (sum, mean, max, min, std, var, prod, argmax)
    against their torch equivalents.
    """
    rank = len(tensor_shape)

    torch_tensor = torch.randn(*tensor_shape, dtype=dtype) if rank > 0 else torch.randn((), dtype=dtype)
    ttnn_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    torch_op, ttnn_op = getattr(torch, op), getattr(ttnn, op)

    # Run on both and flag exceptions
    torch_errored = False
    try:
        torch_result = torch_op(torch_tensor, dim=dim, keepdim=keepdim) if dim is not None else torch_op(torch_tensor)
    except (IndexError, TypeError, RuntimeError):
        torch_errored = True

    ttnn_errored = False
    start_time = start_measuring_time()
    try:
        op_output_tensor = ttnn_op(ttnn_tensor, dim=dim, keepdim=keepdim) if dim is not None else ttnn_op(ttnn_tensor)
        output_tensor = ttnn.to_torch(ttnn.from_device(op_output_tensor))
    except RuntimeError:
        ttnn_errored = True
    e2e_perf = stop_measuring_time(start_time)

    if torch_errored:
        return [(True, f"mismatch in errors raised: torch: {torch_errored}, ttnn: {ttnn_errored}"), e2e_perf]

    if ttnn_errored:
        return [(False, f"ttnn raised error but torch did not for op={op}"), e2e_perf]

    # torch's min/max double as argmin/argmax, so we need to extract the values only
    torch_result = (
        torch_result.values
        if isinstance(torch_result, (torch.return_types.min, torch.return_types.max))
        else torch_result
    )

    # For integer-output ops (e.g., argmax), use exact comparison
    if not torch_result.is_floating_point():
        output_tensor_casted = output_tensor.to(torch_result.dtype)
        exact_match = torch.equal(torch_result, output_tensor_casted)
        if not exact_match:
            return [(False, f"mismatch: torch: {torch_result}, ttnn: {output_tensor_casted}"), e2e_perf]
        tensors = [ttnn_tensor, op_output_tensor]
        return get_run_return(torch_result.float(), output_tensor_casted.float(), 0.99, tensors, e2e_perf)

    atol = rtol = 0.1
    # There is a scale factor difference between torch and ttnn for std and var
    # But for other operations, it should be close. Issue #19478
    if op == "std":
        atol, rtol = sys.maxsize, 0.1 + math.sqrt(2)
    elif op == "var":
        atol, rtol = sys.maxsize, 0.1 + 2

    allclose = torch.allclose(torch_result, output_tensor, atol=atol, rtol=rtol, equal_nan=True)
    if not allclose:
        return [(False, f"mismatch in allclose: torch: {torch_result}, ttnn: {output_tensor}"), e2e_perf]
    expected_pcc = 0.99
    tensors = [ttnn_tensor, op_output_tensor]
    return get_run_return(torch_result, output_tensor, expected_pcc, tensors, e2e_perf)


def _run_accumulation(device, tensor_shape, dim, op, dtype) -> list:
    """Test cumsum and cumprod against their torch equivalents."""
    rank = len(tensor_shape)

    torch_tensor = torch.randn(*tensor_shape, dtype=dtype) if rank > 0 else torch.randn((), dtype=dtype)
    ttnn_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    torch_op = getattr(torch, op)
    ttnn_op = getattr(ttnn, op)

    torch_errored = False
    try:
        torch_result = torch_op(torch_tensor, dim=dim)
    except (IndexError, RuntimeError):
        torch_errored = True

    ttnn_errored = False
    start_time = start_measuring_time()
    try:
        op_output_tensor = ttnn_op(ttnn_tensor, dim=dim)
        output_tensor = ttnn.to_torch(ttnn.from_device(op_output_tensor))
    except RuntimeError:
        ttnn_errored = True
    e2e_perf = stop_measuring_time(start_time)

    if torch_errored:
        return [(True, f"error(s) raised: torch: {torch_errored}, ttnn: {ttnn_errored}"), e2e_perf]

    if ttnn_errored:
        return [(False, f"ttnn raised error but torch did not for op={op}"), e2e_perf]

    expected_pcc = 0.99
    tensors = [ttnn_tensor, op_output_tensor]
    return get_run_return(torch_result, output_tensor, expected_pcc, tensors, e2e_perf)


def _run_topk(device, tensor_shape, dim, dtype) -> list:
    """Test topk operation."""
    k = 32
    torch_tensor = torch.randn(*tensor_shape, dtype=dtype)
    ttnn_tensor = ttnn.from_torch(torch_tensor, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    start_time = start_measuring_time()
    try:
        torch_values, _ = torch.topk(torch_tensor, k, dim=dim, largest=True, sorted=True)
        op_values, op_indices = ttnn.topk(ttnn_tensor, k, dim=dim, largest=True, sorted=True)
        output_values = ttnn.to_torch(op_values)
    except RuntimeError as e:
        e2e_perf = stop_measuring_time(start_time)
        return [(False, f"topk failed: {e}"), e2e_perf]
    e2e_perf = stop_measuring_time(start_time)

    expected_pcc = 0.99
    tensors = [ttnn_tensor, op_values]
    return get_run_return(torch_values, output_values, expected_pcc, tensors, e2e_perf)


def _run_ema(device, tensor_shape, dtype) -> list:
    """Test EMA operation against a golden reference."""
    alpha = 0.25
    torch_tensor = torch.rand(*tensor_shape, dtype=torch.bfloat16)
    ttnn_tensor = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    start_time = start_measuring_time()
    try:
        op_output_tensor = ttnn.ema(ttnn_tensor, alpha=alpha)
        output_tensor = ttnn.to_torch(ttnn.from_device(op_output_tensor))
    except RuntimeError as e:
        e2e_perf = stop_measuring_time(start_time)
        return [(False, f"EMA failed: {e}"), e2e_perf]
    e2e_perf = stop_measuring_time(start_time)

    T = tensor_shape[-1]
    golden = torch.empty_like(torch_tensor)
    prev = torch.zeros_like(torch_tensor[..., 0])
    for t in range(T):
        golden[..., t] = prev * alpha + (1 - alpha) * torch_tensor[..., t]
        prev = golden[..., t]

    expected_pcc = 0.99
    tensors = [ttnn_tensor, op_output_tensor]
    return get_run_return(golden, output_tensor, expected_pcc, tensors, e2e_perf)


def _run_moe(device, tensor_shape, dtype) -> list:
    """Test MOE operation against a torch reference."""
    N, C, H, W = tensor_shape
    k = 32
    E, e = 8, 2

    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)
    torch_input[:, :, :, E:] = 0

    expert_mask = torch.zeros([N, C, 1, W], dtype=torch.bfloat16)
    expert_mask[:, :, :, E:] = float("-inf")
    torch_input = torch_input + expert_mask

    topE_mask = torch.zeros([N, C, 1, k], dtype=torch.bfloat16)
    topE_mask[:, :, :, e:] = float("-inf")

    pyt_topk_values, pyt_topk_indices = torch.topk(torch_input, k, dim=-1)
    torch_result = torch.sum(
        (torch.softmax(pyt_topk_values + topE_mask, dim=-1) * (pyt_topk_indices == 0))[:, :, :, :e],
        dim=-1,
        keepdim=True,
    )

    ttnn_input = ttnn.from_torch(torch_input, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_expert_mask = ttnn.from_torch(expert_mask, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_topE_mask = ttnn.from_torch(topE_mask, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    start_time = start_measuring_time()
    try:
        op_output = ttnn.moe(ttnn_input, ttnn_expert_mask, ttnn_topE_mask, k)
        output_tensor = ttnn.to_torch(op_output)
    except RuntimeError as e:
        e2e_perf = stop_measuring_time(start_time)
        return [(False, f"MOE failed: {e}"), e2e_perf]
    e2e_perf = stop_measuring_time(start_time)

    passing, pcc = check_with_pcc(torch_result, output_tensor, pcc=0.95)
    if not passing:
        return [(False, f"MOE PCC check failed: {pcc}"), e2e_perf]
    return [(True, f"MOE passed with PCC={pcc}"), e2e_perf]


def _run_sampling(device, tensor_shape, dtype) -> list:
    """Test sampling operation for determinism and valid output."""
    torch_tensor = torch.randn(*tensor_shape, dtype=torch.bfloat16)
    W = tensor_shape[-1]

    input_values = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_indices = ttnn.from_torch(
        torch.arange(0, W, dtype=torch.int32).expand(tensor_shape),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    k_tensor = ttnn.from_torch(torch.tensor([10] * 32), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    p_tensor = ttnn.from_torch(
        torch.tensor([0.9] * 32), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    temp_tensor = ttnn.ones([32], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    start_time = start_measuring_time()
    try:
        output_1 = ttnn.sampling(input_values, input_indices, k=k_tensor, p=p_tensor, temp=temp_tensor, seed=42)
        output_2 = ttnn.sampling(input_values, input_indices, k=k_tensor, p=p_tensor, temp=temp_tensor, seed=42)
        out_1 = ttnn.to_torch(output_1)
        out_2 = ttnn.to_torch(output_2)
    except RuntimeError as e:
        e2e_perf = stop_measuring_time(start_time)
        return [(False, f"Sampling failed: {e}"), e2e_perf]
    e2e_perf = stop_measuring_time(start_time)

    deterministic = torch.allclose(out_1.float(), out_2.float())
    if not deterministic:
        return [(False, "Sampling not deterministic with same seed"), e2e_perf]
    return [(True, "Sampling passed determinism check"), e2e_perf]


def _run_manual_seed(device) -> list:
    """Test that manual_seed runs without error."""
    start_time = start_measuring_time()
    try:
        ttnn.manual_seed(seeds=42, device=device)
        ttnn.manual_seed(seeds=42, device=device, user_ids=7)
    except RuntimeError as e:
        e2e_perf = stop_measuring_time(start_time)
        return [(False, f"manual_seed failed: {e}"), e2e_perf]
    e2e_perf = stop_measuring_time(start_time)

    return [(True, "manual_seed completed successfully"), e2e_perf]


def run_reduction(device, tensor_shape, dim, keepdim, op, dtype) -> list:
    """
    Dispatch to the appropriate test runner based on the op type.
    """
    if op in GENERIC_OPS:
        return _run_generic_reduction(device, tensor_shape, dim, keepdim, op, dtype)
    elif op in ACCUMULATION_OPS:
        return _run_accumulation(device, tensor_shape, dim, op, dtype)
    elif op == "topk":
        return _run_topk(device, tensor_shape, dim, dtype)
    elif op == "ema":
        return _run_ema(device, tensor_shape, dtype)
    elif op == "moe":
        return _run_moe(device, tensor_shape, dtype)
    elif op == "sampling":
        return _run_sampling(device, tensor_shape, dtype)
    elif op == "manual_seed":
        return _run_manual_seed(device)
    else:
        return [(False, f"Unknown op: {op}"), None]


@pytest.mark.parametrize(**gen_pytest_parametrize_args(parameters))
def test_reduction(
    device,
    tensor_shape,
    dim,
    keepdim,
    op,
    dtype,
):
    (result, msg), e2e_perf = run_reduction(
        device,
        tensor_shape,
        dim,
        keepdim,
        op,
        dtype,
    )
    assert result, msg
    logger.info(msg)
    if e2e_perf:
        logger.info(f"E2E Performance: {e2e_perf}")


def run(
    tensor_shape,
    dim,
    keepdim,
    op,
    dtype,
    *,
    device,
) -> list:
    return run_reduction(
        device,
        tensor_shape,
        dim,
        keepdim,
        op,
        dtype,
    )
