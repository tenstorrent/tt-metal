# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Tests corner cases of reduction operations.
# These tests are not meant to exhaustively sweep over all parameter combinations.
# Many parameters are exposed to make it easy to add new tests, but are currently
# set to a single value.

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose_and_pcc
from loguru import logger


def _run_topk_with_preallocated(input_tensor, k, dim, device, ttnn_result):
    """
    Helper function that calls topk with preallocated output tensors, whose shapes are
    determined by the ttnn_result obtained from a previous run of topk without preallocated
    output tensors.
    """
    prealloc_values = ttnn.empty(
        ttnn_result[0].shape,
        dtype=ttnn_result[0].dtype,
        layout=ttnn_result[0].layout,
        device=device,
        memory_config=ttnn_result[0].memory_config(),
    )
    prealloc_indices = ttnn.empty(
        ttnn_result[1].shape,
        dtype=ttnn_result[1].dtype,
        layout=ttnn_result[1].layout,
        device=device,
        memory_config=ttnn_result[1].memory_config(),
    )
    ttnn.topk(input_tensor, k, dim=dim, output_tensor=(prealloc_values, prealloc_indices))
    return (
        ttnn.to_torch(ttnn.from_device(prealloc_values)),
        ttnn.to_torch(ttnn.from_device(prealloc_indices)),
    )


def _run_argmax_with_preallocated(input_tensor, dim, keepdim, device, ttnn_result):
    """
    Helper function that calls argmax with preallocated output tensor, whose shape is
    determined by the ttnn_result obtained from a previous run of argmax without
    preallocated output tensors.
    """
    prealloc_output = ttnn.empty(
        ttnn_result.shape,
        dtype=ttnn_result.dtype,
        layout=ttnn_result.layout,
        device=device,
        memory_config=ttnn_result.memory_config(),
    )
    ttnn.argmax(input_tensor, dim=dim, keepdim=keepdim, output_tensor=prealloc_output)
    return ttnn.to_torch(ttnn.from_device(prealloc_output))


def _run_accumulation_with_preallocated(ttnn_op, input_tensor, dim, device, ttnn_result_tensor):
    """
    Helper function that calls a cumulative op (cumsum/cumprod) with preallocated output tensor,
    whose shape is determined by the ttnn_result obtained from a previous run without
    preallocated output tensor.
    """
    prealloc_output = ttnn.empty(
        ttnn_result_tensor.shape,
        dtype=ttnn_result_tensor.dtype,
        layout=ttnn_result_tensor.layout,
        device=device,
        memory_config=ttnn_result_tensor.memory_config(),
    )
    ttnn_op(input_tensor, dim, out=prealloc_output)
    return ttnn.to_torch(ttnn.from_device(prealloc_output))


def _run_moe_with_preallocated(input_tensor, expert_mask_tensor, topk_mask_tensor, k, device, ttnn_result):
    """
    Helper function that calls moe with preallocated output tensor, whose shape is determined by
    the ttnn_result obtained from a previous run of moe without preallocated output.
    """
    prealloc_output = ttnn.empty(
        ttnn_result.shape,
        dtype=ttnn_result.dtype,
        layout=ttnn_result.layout,
        device=device,
        memory_config=ttnn_result.memory_config(),
    )
    ttnn.moe(input_tensor, expert_mask_tensor, topk_mask_tensor, k, output_tensor=prealloc_output)
    return ttnn.to_torch(ttnn.from_device(prealloc_output))


def _run_sampling_with_preallocated(
    input_values, input_indices, k_tensor, p_tensor, temp_tensor, seed, device, ttnn_result
):
    """
    Helper function that calls sampling with preallocated output tensor, whose shape is determined by
    the ttnn_result obtained from a previous run of sampling without preallocated output tensor.
    """
    prealloc_output = ttnn.empty(
        ttnn_result.shape,
        dtype=ttnn_result.dtype,
        layout=ttnn_result.layout,
        device=device,
        memory_config=ttnn_result.memory_config(),
    )
    ttnn.sampling(
        input_values,
        input_indices,
        k=k_tensor,
        p=p_tensor,
        temp=temp_tensor,
        seed=seed,
        output_tensor=prealloc_output,
    )
    return ttnn.to_torch(ttnn.from_device(prealloc_output))


def _torch_sampling_reference(values, indices, k, p, temp, seed):
    """
    Torch reference for ttnn.sampling: softmax -> top-k -> top-p (nucleus) -> multinomial.
    Required because there is no direct PyTorch equivalent.
    Returns tensor of shape (1, 1, 1, num_users) of sampled index values (one per user).
    This code was AI generated based on description of ttnn.sampling, since there is
    no direct PyTorch equivalent.
    """
    N, C, H, W = values.shape
    num_users = N * C * H

    # Flatten to (num_users, W) so each row is one user's logits.
    values_flat = values.reshape(num_users, W)
    temp_flat = temp.view(num_users, 1).expand(num_users, W)
    probs_flat = torch.softmax(values_flat / temp_flat, dim=-1)
    indices_flat = indices.reshape(num_users, W)

    torch.manual_seed(seed)
    out_list = []
    for u in range(num_users):
        probs_u = probs_flat[u, :].clone()
        k_u = int(k[u].item())
        p_u = float(p[u].item())
        # Top-k: zero out all but the top-k probabilities, then renormalize.
        if k_u < W:
            _, top_idx = torch.topk(probs_u, k_u, dim=-1)
            mask = torch.zeros_like(probs_u, dtype=torch.bool)
            mask[top_idx] = True
            probs_u = torch.where(mask, probs_u, torch.zeros_like(probs_u))
        probs_u_sum = probs_u.sum()
        if probs_u_sum > 0:
            probs_u = probs_u / probs_u_sum
        # Top-p (nucleus): sort descending, cumsum, keep until cumsum <= p_u.
        probs_sorted, _ = torch.sort(probs_u, descending=True)
        cumsum = torch.cumsum(probs_sorted, dim=-1)
        # Number of elements to keep: first position where cumsum > p_u (exclusive).
        keep = (cumsum <= p_u).sum().item()
        if keep < 1:
            keep = 1
        # Rebuild mask: keep only indices that are in the top-p set.
        _, sort_idx = torch.sort(probs_u, descending=True)
        mask = torch.zeros_like(probs_u, dtype=torch.bool)
        mask[sort_idx[:keep]] = True
        probs_u = torch.where(mask, probs_u, torch.zeros_like(probs_u))
        probs_u_sum = probs_u.sum()
        if probs_u_sum > 0:
            probs_u = probs_u / probs_u_sum
        # Multinomial: sample one index from the distribution.
        sampled_idx = torch.multinomial(probs_u.unsqueeze(0), num_samples=1, replacement=True).squeeze(0).item()
        # Output is the value of input_indices at that position (per API: returns input_indices_tensor[final_index]).
        out_val = indices_flat[u, sampled_idx].item()
        out_list.append(out_val)
    # Output shape (1, 1, 1, num_users): one sampled index value per user.
    out_tensor = torch.tensor(out_list, dtype=indices.dtype).view(1, 1, 1, num_users)
    return out_tensor


# Test a 0D, 1D, 5D, and a 0-volume tensor
@pytest.mark.parametrize("tensor_shape", [(), (2,), (1, 1), (32, 1), (3, 6, 40, 63, 20), (6, 0, 32)])
@pytest.mark.parametrize("dim", [None, 0, -1, (-2, -1), (0, 2), (0, 2, 4)])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("correction", [True, False])
@pytest.mark.parametrize("op", ["mean", "sum", "max", "min", "prod", "std", "var"])
def test_generic_ops(device, tensor_shape, dim, keepdim, dtype, layout, correction, op):
    """
    Test the compatibility of the torch and ttnn output for the given operation and different
    tensor shapes, keepdim, and dim values.
    Checks that resulting tensors are within a certain tolerance of PyTorch outputs.
    Some operations raise exceptions in torch, we check if the same behavior is observed in ttnn.
    Note: We do not enforce the same exception type or message.
    """
    if op not in ("var", "std") and correction:
        # PyTorch supports the correction argument only for var and std.
        return

    torch.manual_seed(0)
    torch_tensor = torch.randn(tensor_shape, dtype=dtype)
    pad_value = 1.0 if op == "prod" else None
    ttnn_tensor = ttnn.from_torch(torch_tensor, layout=layout, device=device, pad_value=pad_value)

    torch_op, ttnn_op = getattr(torch, op), getattr(ttnn, op)

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

    ttnn_errored = False
    try:
        # ttnn.prod doesn't support the correction argument.
        if op != "prod":
            ttnn_result = ttnn_op(ttnn_tensor, dim=dim, keepdim=keepdim, correction=correction)
        else:
            ttnn_result = ttnn_op(ttnn_tensor, dim=dim, keepdim=keepdim)
    except (IndexError, TypeError, RuntimeError) as e:
        ttnn_errored = True
        if not torch_errored:
            logger.error(f"torch passed and produced result: {torch_result}, but ttnn raised exception: {e}")

    assert torch_errored == ttnn_errored, f"torch_errored: {torch_errored}, ttnn_errored: {ttnn_errored}"

    # Skip the rest of the test if an exception was raised in both
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

    atol = rtol = 0.1
    if op == "var":
        pcc = 0.99
    elif op == "std":
        pcc = 0.98
    else:
        pcc = 0.999
    passing, output_pcc = comp_allclose_and_pcc(torch_result, ttnn_result, pcc=pcc, rtol=rtol, atol=atol)
    assert passing, f"{output_pcc}, torch: {torch_result}, ttnn: {ttnn_result}"


@pytest.mark.parametrize("tensor_shape", [(), (170,), (3, 6, 40, 63, 20), (60, 0, 32)])
@pytest.mark.parametrize("dim", [None, 0, -1])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("k", [50, 1, 0])
def test_topk(device, tensor_shape, dim, dtype, layout, k):
    """
    Test the compatibility of the torch and ttnn topk output for different tensor shapes.
    topk returns a tuple of (values, indices). We compare values via PCC and validate
    indices semantically by gathering from the original tensor and checking cosine similarity,
    since torch and ttnn may break ties differently in bfloat16.
    """
    torch.manual_seed(0)

    torch_tensor = torch.randn(tensor_shape, dtype=dtype)
    ttnn_tensor = ttnn.from_torch(torch_tensor, layout=layout, device=device)

    torch_errored = False
    try:
        torch_values, torch_indices = torch.topk(torch_tensor, k, dim=dim)
    except (IndexError, TypeError, RuntimeError) as e:
        logger.info(f"torch topk raised: {e}")
        torch_errored = True

    ttnn_errored = False
    try:
        ttnn_result = ttnn.topk(ttnn_tensor, k, dim=dim)
    except (IndexError, TypeError, RuntimeError) as e:
        ttnn_errored = True
        if not torch_errored:
            logger.error(f"torch passed, but ttnn raised exception: {e}")

    if torch_errored and ttnn_errored:
        logger.info(f"Both PyTorch and TTNN errored")
    assert torch_errored == ttnn_errored, f"torch_errored: {torch_errored}, ttnn_errored: {ttnn_errored}"

    # Skip the rest of the test if an exception was raised in both
    if torch_errored:
        return

    ttnn_values = ttnn.to_torch(ttnn.from_device(ttnn_result[0]))
    ttnn_indices = ttnn.to_torch(ttnn.from_device(ttnn_result[1]))

    if (
        torch_values.numel() == 0
        and ttnn_values.numel() == 0
        and torch_indices.numel() == 0
        and ttnn_indices.numel() == 0
    ):
        logger.info(f"Both PyTorch and TTNN returned 0-volume tensors")
        assert (
            torch_values.shape == ttnn_values.shape
        ), f"Shape mismatch on values: torch: {torch_values.shape}, ttnn: {ttnn_values.shape}"
        assert (
            torch_indices.shape == ttnn_indices.shape
        ), f"Shape mismatch on indices: torch: {torch_indices.shape}, ttnn: {ttnn_indices.shape}"

        # Repeat the test with preallocated output tensors.
        prealloc_values, prealloc_indices = _run_topk_with_preallocated(ttnn_tensor, k, dim, device, ttnn_result)
        # The two methods should produce identical results.
        assert torch.equal(
            prealloc_values, ttnn_values
        ), f"Preallocated values differ from non-preallocated: {prealloc_values} vs {ttnn_values}"
        assert torch.equal(
            prealloc_indices, ttnn_indices
        ), f"Preallocated indices differ from non-preallocated: {prealloc_indices} vs {ttnn_indices}"

        # Other checks are not meaningful for empty tensors.
        return

    atol = rtol = 0.1
    pcc = 0.999
    passing, output_pcc = comp_allclose_and_pcc(torch_values, ttnn_values, pcc=pcc, rtol=rtol, atol=atol)
    assert passing, f"Values: {output_pcc}, torch: {torch_values}, ttnn: {ttnn_values}"

    ttnn_indices_adjusted = ttnn_indices.to(torch.int32)
    # Indices can come back as negative values from ttnn (stored as unsigned in bfloat16).
    # This is fixed by adding 2^16 to negative values.
    ttnn_indices_adjusted = torch.where(ttnn_indices_adjusted < 0, ttnn_indices_adjusted + 65536, ttnn_indices_adjusted)

    cosine_sim_target = 0.99
    # Use ttnn's returned indices to gather values from the original input tensor.
    # The result is "the values that ttnn thinks are the top-k."
    ttnn_gather_from_indices = torch.gather(torch_tensor, dim, ttnn_indices_adjusted.to(torch.int64))
    cosine = torch.nn.CosineSimilarity(dim=dim)
    # Comparing indices directly may not be a good measure because when there are ties
    # (duplicate values), both implementations may return different but equally valid
    # index positions.
    # Compare PyTorch's top-k values against the values gathered using ttnn's indices.
    # If ttnn returned correct indices, then gathering from the original tensor at those
    # index positions should yield the same (or very similar) values as PyTorch's top-k values.
    cosine_sim = torch.mean(cosine(torch_values, ttnn_gather_from_indices)).float()
    assert (
        cosine_sim >= cosine_sim_target
    ), f"Cosine similarity between topk values and gather from indices is {cosine_sim} which is less than {cosine_sim_target}"

    # Repeat the test with preallocated output tensors.
    prealloc_values, prealloc_indices = _run_topk_with_preallocated(ttnn_tensor, k, dim, device, ttnn_result)

    # The two methods should produce identical results.
    assert torch.equal(
        prealloc_values, ttnn_values
    ), f"Preallocated values: {prealloc_values} do not match non-preallocated: {ttnn_values}"
    assert torch.equal(
        prealloc_indices, ttnn_indices
    ), f"Preallocated indices: {prealloc_indices} do not match non-preallocated: {ttnn_indices}"


@pytest.mark.parametrize("tensor_shape", [(), (2,), (3, 6, 40, 63, 20), (6, 0, 32)])
@pytest.mark.parametrize("dim", [None, 0, -1])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_argmax(device, tensor_shape, dim, keepdim, dtype, layout):
    """
    Test the compatibility of the torch and ttnn argmax output for different tensor shapes.
    argmax returns indices (UINT32). We validate semantically by checking that the values at
    the returned indices match the actual maximum values from torch.max, which is robust against
    tie-breaking differences between torch and ttnn.
    """
    torch.manual_seed(0)
    rank = len(tensor_shape)

    # Skip known ttnn.argmax limitations, but only for non-zero-volume tensors.
    # 0-volume tensors take a separate early-return path in ttnn and should be
    # tested for error parity with torch.
    is_zero_volume = 0 in tensor_shape
    if not is_zero_volume:
        if layout == ttnn.TILE_LAYOUT and dim is None:
            pytest.skip("ttnn.argmax does not support dim=None with TILE layout")
        if rank > 1 and dim is not None:
            normalized_dim = dim if dim >= 0 else dim + rank
            if normalized_dim != rank - 1:
                pytest.skip("ttnn.argmax only supports reduction on the last dimension")

    torch_tensor = torch.randn(tensor_shape, dtype=dtype)
    ttnn_tensor = ttnn.from_torch(torch_tensor, layout=layout, device=device)

    torch_errored = False
    try:
        torch_result = torch.argmax(torch_tensor, dim=dim, keepdim=keepdim)
    except (IndexError, TypeError, RuntimeError) as e:
        logger.info(f"torch argmax raised: {e}")
        torch_errored = True

    ttnn_errored = False
    try:
        ttnn_result = ttnn.argmax(ttnn_tensor, dim=dim, keepdim=keepdim)
    except (IndexError, TypeError, RuntimeError) as e:
        ttnn_errored = True
        if not torch_errored:
            logger.error(f"torch passed, but ttnn raised exception: {e}")

    assert torch_errored == ttnn_errored, f"torch_errored: {torch_errored}, ttnn_errored: {ttnn_errored}"

    # Skip the rest of the test if an exception was raised in both
    if torch_errored:
        return

    ttnn_result_in_torch = ttnn.to_torch(ttnn.from_device(ttnn_result))

    # For 0-volume results, verify shapes match
    if torch_result.numel() == 0 and ttnn_result_in_torch.numel() == 0:
        assert (
            torch_result.shape == ttnn_result_in_torch.shape
        ), f"Shape mismatch on 0-volume result: torch: {torch_result.shape}, ttnn: {ttnn_result_in_torch.shape}"

        # Repeat the test with preallocated output tensors.
        ttnn_result_prealloc = _run_argmax_with_preallocated(ttnn_tensor, dim, keepdim, device, ttnn_result)
        assert (
            torch_result.shape == ttnn_result_prealloc.shape
        ), f"Preallocated shape mismatch on 0-volume result: torch {torch_result.shape}, ttnn: {ttnn_result_prealloc.shape}"

        # Other checks are not meaningful for empty tensors.
        return

    # Secondary check: PCC on raw indices (ties are rare with random bfloat16)
    atol = rtol = 0.1
    pcc = 0.999
    passing, output_pcc = comp_allclose_and_pcc(torch_result, ttnn_result_in_torch, pcc=pcc, rtol=rtol, atol=atol)
    assert passing, f"Indices PCC: {output_pcc}, torch: {torch_result}, ttnn: {ttnn_result_in_torch}"

    ttnn_result_i64 = ttnn_result_in_torch.to(torch.int64)

    # Primary check: semantic validation - verify the values at ttnn's indices
    # match the values at torch's indices (robust against tie-breaking differences)
    torch_result_i64 = torch_result.to(torch.int64)
    if dim is not None:
        # torch.gather requires that index has the same number of dimensions as input.
        # When keepdim=False, the reduced dimension is removed, making the index tensor one
        # rank lower than the input; unsqueeze(dim) adds that dimension back.
        ttnn_gather_indices = ttnn_result_i64 if keepdim else ttnn_result_i64.unsqueeze(dim)
        torch_gather_indices = torch_result_i64 if keepdim else torch_result_i64.unsqueeze(dim)
        # Comparing indices directly may not be a good measure because when there are ties
        # (duplicate values), both implementations may return different but equally valid
        # index positions. Instead, we gather the values at the indices each implementation
        # thinks are the max, and compare them.
        ttnn_gathered = torch.gather(torch_tensor, dim, ttnn_gather_indices)
        torch_gathered = torch.gather(torch_tensor, dim, torch_gather_indices)
        if not keepdim:
            # Undo the unsqueeze that was added to make torch.gather work.
            # This helps make errors more readable in case of test failure.
            ttnn_gathered = ttnn_gathered.squeeze(dim)
            torch_gathered = torch_gathered.squeeze(dim)
        assert torch.allclose(
            ttnn_gathered.float(), torch_gathered.float(), atol=atol, rtol=rtol
        ), f"Values at ttnn indices don't match values at torch indices: ttnn={ttnn_gathered}, torch={torch_gathered}"
    else:
        # When dim is None, torch.argmax flattens the entire tensor and returns a single scalar
        # index into the flattened view. Therefore, we can't use torch.gather with a dim argument
        # on the original multi-dimensional tensor. Instead, we flatten the tensor and use the index
        # that ttnn returned to gather the value from the flattened tensor.
        ttnn_value = torch_tensor.flatten()[ttnn_result_i64.item()]
        # Do the same for torch then compare them.
        torch_value = torch_tensor.flatten()[torch_result_i64.item()]
        assert torch.allclose(
            ttnn_value.float(), torch_value.float(), atol=atol, rtol=rtol
        ), f"Value at ttnn index {ttnn_result_i64.item()} is {ttnn_value}, expected {torch_value} (at torch index {torch_result_i64.item()})"

    # Repeat the test with preallocated output tensor.
    ttnn_result_prealloc = _run_argmax_with_preallocated(ttnn_tensor, dim, keepdim, device, ttnn_result)
    # The two methods should produce identical results.
    assert torch.equal(
        ttnn_result_prealloc, ttnn_result_in_torch
    ), f"Preallocated argmax result: {ttnn_result_prealloc} does not match non-preallocated: {ttnn_result_in_torch}"


@pytest.mark.parametrize("tensor_shape", [(), (2,), (3, 6, 40, 63, 20), (6, 0, 32)])
@pytest.mark.parametrize("dim", [None, 0, -1])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("op", ["cumsum", "cumprod"])
def test_accumulation(device, tensor_shape, dim, dtype, layout, op):
    """
    Test the compatibility of the torch and ttnn output for cumsum/cumprod and different
    tensor shapes and dim values.
    Unlike standard reductions, cumsum/cumprod produce same-shape outputs (accumulations).
    Checks that resulting tensors are within a certain tolerance of PyTorch outputs.
    Some operations raise exceptions in torch, we check if the same behavior is observed in ttnn.
    Note: We do not enforce the same exception type or message.
    """
    torch.manual_seed(0)

    torch_tensor = torch.randn(tensor_shape, dtype=dtype)
    pad_value = 1.0 if op == "cumprod" else None
    ttnn_tensor = ttnn.from_torch(torch_tensor, layout=layout, device=device, pad_value=pad_value)

    torch_op, ttnn_op = getattr(torch, op), getattr(ttnn, op)

    # Run on both and flag exceptions
    torch_errored = False
    try:
        torch_result = torch_op(torch_tensor, dim)
    except (IndexError, TypeError, RuntimeError) as e:
        logger.info(f"torch {op} raised: {e}")
        torch_errored = True

    ttnn_errored = False
    try:
        ttnn_result = ttnn_op(ttnn_tensor, dim)
    except (IndexError, TypeError, RuntimeError) as e:
        ttnn_errored = True
        if not torch_errored:
            logger.error(f"torch passed, but ttnn raised exception: {e}")

    assert torch_errored == ttnn_errored, f"torch_errored: {torch_errored}, ttnn_errored: {ttnn_errored}"

    # Skip the rest of the test if an exception was raised in both
    if torch_errored:
        return

    ttnn_result_in_torch = ttnn.to_torch(ttnn.from_device(ttnn_result))

    # For 0-volume results, verify shapes match
    if torch_result.numel() == 0 and ttnn_result_in_torch.numel() == 0:
        assert (
            torch_result.shape == ttnn_result_in_torch.shape
        ), f"Shape mismatch on 0-volume result: torch: {torch_result.shape}, ttnn: {ttnn_result_in_torch.shape}"

        # Repeat the test with preallocated output tensor.
        prealloc_result = _run_accumulation_with_preallocated(ttnn_op, ttnn_tensor, dim, device, ttnn_result)
        # The two methods should produce identical results.
        assert (
            torch_result.shape == prealloc_result.shape
        ), f"Preallocated shape mismatch on 0-volume result: torch: {torch_result.shape}, ttnn: {prealloc_result.shape}"

        # Other checks are not meaningful for empty tensors.
        return

    atol = rtol = 0.1
    pcc = 0.999
    passing, output_pcc = comp_allclose_and_pcc(torch_result, ttnn_result_in_torch, pcc=pcc, rtol=rtol, atol=atol)
    assert passing, f"{output_pcc}, torch: {torch_result}, ttnn: {ttnn_result_in_torch}"

    # Repeat the test with preallocated output tensor.
    prealloc_result = _run_accumulation_with_preallocated(ttnn_op, ttnn_tensor, dim, device, ttnn_result)
    # The two methods should produce identical results.
    assert torch.equal(
        prealloc_result, ttnn_result_in_torch
    ), f"Preallocated {op} result: {prealloc_result} does not match non-preallocated: {ttnn_result_in_torch}"


# (2, 2, 32, 64) shape hangs the test. Issue #39795
# @pytest.mark.parametrize("tensor_shape", [(), (1, 1, 32, 64), (2, 2, 32, 64), (1, 1, 0, 64)])
@pytest.mark.parametrize("tensor_shape", [(), (1, 1, 32, 64), (1, 1, 0, 64)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_moe(device, tensor_shape, dtype, layout):
    """
    Test ttnn.moe against the torch reference (topk + softmax + sum) for scalar and
    4D tensor shapes.
    Some operations raise exceptions in torch, we check if the same behavior is observed in ttnn.
    Note: We do not enforce the same exception type or message.
    """
    torch.manual_seed(0)
    rank = len(tensor_shape)
    # k must be 32, per ttnn.moe documentation.
    k = 32
    # E = total number of active experts, e = number of top experts to route to
    # (top-e gating). Only the first E columns carry real expert scores; the rest
    # are masked to -inf so softmax assigns them zero weight.
    E, e = 8, 2

    # MOE requires 4D tensors with specific mask shapes; for non-4D shapes,
    # construct trivial tensors and let exception parity catch the errors.
    torch_input = torch.randn(tensor_shape, dtype=dtype)

    if rank == 0:
        # For rank 0, ops below are not applicable.
        expert_mask = torch.zeros(tensor_shape, dtype=dtype)
        topE_mask = torch.zeros(tensor_shape, dtype=dtype)
    else:
        N, C, H, W = tensor_shape
        # Zero out columns beyond the first E so only E experts have non-zero scores.
        torch_input[:, :, :, E:] = 0
        # Height is 1 so the mask broadcasts across all H rows.
        # Columns [E:] are -inf; adding this to the input ensures softmax
        # drives inactive expert probabilities to zero.
        expert_mask = torch.zeros([N, C, 1, W], dtype=dtype)
        expert_mask[:, :, :, E:] = float("-inf")
        torch_input = torch_input + expert_mask
        # topE_mask has width k (matching topk output width) and keeps only the
        # first e entries; positions [e:] are -inf so softmax zeroes them out,
        # implementing top-e expert selection after topk.
        topE_mask = torch.zeros([N, C, 1, k], dtype=dtype)
        topE_mask[:, :, :, e:] = float("-inf")

    # Run on both ttnn and torch and flag exceptions
    torch_errored = False
    try:
        pyt_topk_values, pyt_topk_indices = torch.topk(torch_input, k, dim=-1)
        # Reference MOE pipeline: apply topE_mask before softmax to zero out
        # all but the top-e experts, multiply by an indicator for expert 0
        # (pyt_topk_indices == 0) to isolate its contribution, slice to [:e],
        # then sum across the expert dimension to get the gated output.
        torch_result = torch.sum(
            (torch.softmax(pyt_topk_values + topE_mask, dim=-1) * (pyt_topk_indices == 0))[:, :, :, :e],
            dim=-1,
            keepdim=True,
        )
    except (IndexError, TypeError, RuntimeError) as e:
        logger.info(f"torch MOE reference raised: {e}")
        torch_errored = True

    ttnn_errored = False
    try:
        ttnn_input = ttnn.from_torch(torch_input, layout=layout, device=device)
        ttnn_expert_mask = ttnn.from_torch(expert_mask, layout=layout, device=device)
        ttnn_topE_mask = ttnn.from_torch(topE_mask, layout=layout, device=device)
        ttnn_result = ttnn.moe(ttnn_input, ttnn_expert_mask, ttnn_topE_mask, k)
    except (IndexError, TypeError, RuntimeError) as e:
        ttnn_errored = True
        if not torch_errored:
            logger.error(f"torch passed, but ttnn raised exception: {e}")

    assert torch_errored == ttnn_errored, f"torch_errored: {torch_errored}, ttnn_errored: {ttnn_errored}"

    # Skip the rest of the test if an exception was raised in both
    if torch_errored:
        return

    ttnn_result_in_torch = ttnn.to_torch(ttnn.from_device(ttnn_result))

    # For 0-volume results, verify shapes match.
    if torch_result.numel() == 0 and ttnn_result_in_torch.numel() == 0:
        assert (
            torch_result.shape == ttnn_result_in_torch.shape
        ), f"Shape mismatch on 0-volume result: torch: {torch_result.shape}, ttnn: {ttnn_result_in_torch.shape}"

        # Repeat the test with preallocated output tensor.
        prealloc_result = _run_moe_with_preallocated(
            ttnn_input, ttnn_expert_mask, ttnn_topE_mask, k, device, ttnn_result
        )
        assert (
            torch_result.shape == prealloc_result.shape
        ), f"Preallocated shape mismatch on 0-volume result: torch: {torch_result.shape}, ttnn: {prealloc_result.shape}"

        return

    atol = rtol = 0.1
    # Looser PCC tolerance than typical single-op tests because MOE chains
    # topk -> softmax -> multiply -> sum, and each step accumulates
    # bfloat16 rounding error.
    pcc = 0.95
    passing, output_pcc = comp_allclose_and_pcc(torch_result, ttnn_result_in_torch, pcc=pcc, rtol=rtol, atol=atol)
    assert passing, f"{output_pcc}, torch: {torch_result}, ttnn: {ttnn_result_in_torch}"

    # Repeat the test with preallocated output tensor.
    prealloc_result = _run_moe_with_preallocated(ttnn_input, ttnn_expert_mask, ttnn_topE_mask, k, device, ttnn_result)
    assert torch.allclose(
        prealloc_result.float(), ttnn_result_in_torch.float(), atol=atol, rtol=rtol
    ), f"Preallocated moe result: {prealloc_result} does not match non-preallocated: {ttnn_result_in_torch}"


@pytest.mark.parametrize("tensor_shape", [(), (1, 1, 32, 64), (1, 1, 32, 0)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_sampling(device, tensor_shape, dtype, layout):
    """
    Test ttnn.sampling against a torch reference (softmax -> top-k -> top-p -> multinomial).
    Structure mirrors test_moe: error parity (including scalar and non-4D), shape/validity checks, and prealloc path.
    We do not compare output values numerically to the torch reference (different RNG backends);
    we check that ttnn is deterministic (same seed -> same result) and that prealloc matches non-prealloc.
    """
    torch.manual_seed(0)
    rank = len(tensor_shape)

    SAMPLING_SEED = 42

    # Build input values; for non-4D shapes use trivial tensors and let exception parity catch the errors.
    torch_values = torch.randn(tensor_shape, dtype=dtype)
    if rank == 0:
        # Scalar case: indices same shape as values so from_torch works; sampling will reject.
        torch_indices = torch.zeros(tensor_shape, dtype=torch.int32)
    else:
        N, C, H, W = tensor_shape
        # Indices tensor: per-position index value (e.g. 0..W-1); same shape as values. W must be divisible by 32 per API.
        torch_indices = torch.arange(0, W, dtype=torch.int32).expand(tensor_shape)

    # Per-user params: k (top-k), p (top-p nucleus), temp (temperature). Must have 32 elements (per API).
    # 10 = keep top-10 logits per user before top-p; 0.9 = nucleus cumulative mass threshold.
    k_vals = torch.tensor([10] * 32, dtype=torch.uint32)
    p_vals = torch.tensor([0.9] * 32, dtype=dtype)
    temp_vals = torch.ones(32, dtype=dtype)

    # Run torch reference and ttnn; flag exceptions for error parity.
    # ValueError: torch reference unpacks values.shape to N,C,H,W; scalar gives no values to unpack.
    torch_errored = False
    try:
        torch_result = _torch_sampling_reference(
            torch_values, torch_indices, k_vals, p_vals, temp_vals, seed=SAMPLING_SEED
        )
    except (ValueError, IndexError, TypeError, RuntimeError) as e:
        logger.info(f"torch sampling reference raised: {e}")
        torch_errored = True

    ttnn_errored = False
    try:
        input_values = ttnn.from_torch(torch_values, layout=layout, device=device)
        input_indices = ttnn.from_torch(
            torch_indices,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        k_tensor = ttnn.from_torch(k_vals, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        p_tensor = ttnn.from_torch(p_vals, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        temp_tensor = ttnn.from_torch(temp_vals, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn_result = ttnn.sampling(
            input_values,
            input_indices,
            k=k_tensor,
            p=p_tensor,
            temp=temp_tensor,
            seed=SAMPLING_SEED,
        )
    except (ValueError, IndexError, TypeError, RuntimeError) as e:
        ttnn_errored = True
        if not torch_errored:
            logger.error(f"torch passed, but ttnn sampling raised: {e}")

    assert torch_errored == ttnn_errored, f"torch_errored: {torch_errored}, ttnn_errored: {ttnn_errored}"

    if torch_errored:
        return

    ttnn_result_in_torch = ttnn.to_torch(ttnn.from_device(ttnn_result))

    # Shape check: ttnn output is (1, 1, 1, input_shape[2]) i.e. one sampled index per user (32 users).
    assert (
        torch_result.shape == ttnn_result_in_torch.shape
    ), f"Shape mismatch: torch: {torch_result.shape}, ttnn: {ttnn_result_in_torch.shape}"

    # 0-volume path: if both returned empty output, only shape and prealloc (mirrors test_moe).
    if torch_result.numel() == 0 and ttnn_result_in_torch.numel() == 0:
        prealloc_result = _run_sampling_with_preallocated(
            input_values, input_indices, k_tensor, p_tensor, temp_tensor, SAMPLING_SEED, device, ttnn_result
        )
        assert (
            torch_result.shape == prealloc_result.shape
        ), f"Preallocated shape mismatch: torch: {torch_result.shape}, ttnn: {prealloc_result.shape}"
        return

    # Determinism: two ttnn runs with same seed must match (we cannot compare to torch; RNG differs).
    ttnn_result_2 = ttnn.sampling(
        input_values, input_indices, k=k_tensor, p=p_tensor, temp=temp_tensor, seed=SAMPLING_SEED
    )
    ttnn_result_2_torch = ttnn.to_torch(ttnn.from_device(ttnn_result_2))
    assert torch.equal(ttnn_result_in_torch, ttnn_result_2_torch), "Sampling must be deterministic with the same seed"

    # Preallocated output must match non-preallocated.
    prealloc_result = _run_sampling_with_preallocated(
        input_values, input_indices, k_tensor, p_tensor, temp_tensor, SAMPLING_SEED, device, ttnn_result
    )
    assert torch.equal(
        prealloc_result, ttnn_result_in_torch
    ), f"Preallocated sampling result does not match non-preallocated: {prealloc_result} vs {ttnn_result_in_torch}"
