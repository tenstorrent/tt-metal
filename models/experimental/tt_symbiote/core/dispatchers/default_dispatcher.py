# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TTNN operation dispatch handlers and mapping."""

from typing import Any, Optional, Tuple

import torch

import ttnn
from models.experimental.tt_symbiote.core.utils import TORCH_TO_TTNN, ensure_tile_layout, torch_dtype_to_ttnn_dtype

# ========== Helper Functions ==========


def _prepare_tensor_input(
    tensor: Any, device: Optional[Any] = None, ref_dtype: Optional[torch.dtype] = None
) -> Tuple[Any, bool, Optional[Any]]:
    """Prepare a single tensor input for TTNN operation.

    Args:
        tensor: Input tensor (may be TorchTTNNTensor, torch.Tensor, or scalar)
        device: Target device (optional)
        ref_dtype: Reference dtype for scalar conversion

    Returns:
        Tuple of (prepared_tensor, should_deallocate, device)
    """
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    should_deallocate = False

    if not isinstance(tensor, TorchTTNNTensor):
        if isinstance(tensor, (int, float)):
            tensor = torch.tensor(tensor)
        tensor = TorchTTNNTensor(tensor, dtype=ref_dtype)
        should_deallocate = True
    else:
        if tensor.ttnn_tensor is None:
            should_deallocate = True
        if device is None:
            device = tensor.to_ttnn.device()

    return tensor, should_deallocate, device


def _prepare_binary_inputs(
    tensor1: Any, tensor2: Any, device: Optional[Any] = None
) -> Tuple[Any, Any, bool, bool, Any]:
    """Prepare two tensor inputs for TTNN binary operation.

    Args:
        tensor1: First input tensor
        tensor2: Second input tensor
        device: Target device (optional)

    Returns:
        Tuple of (tensor1, tensor2, deallocate1, deallocate2, device)
    """
    tensor1, deallocate1, device = _prepare_tensor_input(tensor1, device, getattr(tensor2, "dtype", None))
    tensor2, deallocate2, device = _prepare_tensor_input(tensor2, device, tensor1.dtype)

    if device is None:
        raise RuntimeError("At least one of the inputs must be a TTNN tensor.")

    # Ensure both tensors are on the same device
    if tensor1.to_ttnn.device() != tensor2.to_ttnn.device():
        tensor1.ttnn_tensor = ttnn.to_device(tensor1.to_ttnn, device)
        tensor2.ttnn_tensor = ttnn.to_device(tensor2.to_ttnn, device)

    return tensor1, tensor2, deallocate1, deallocate2, device


def _cleanup_tensors(*tensor_deallocate_pairs):
    """Deallocate temporary tensors.

    Args:
        tensor_deallocate_pairs: Pairs of (tensor, should_deallocate)
    """
    for tensor, should_deallocate in tensor_deallocate_pairs:
        if should_deallocate and tensor.ttnn_tensor is not None:
            ttnn.deallocate(tensor.ttnn_tensor)


# ========== Operation Handlers ==========


def handle_view(func, args, kwargs):
    """Handle view operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    new_shape = args[1]
    return TorchTTNNTensor(ttnn.reshape(input_tensor.to_ttnn, new_shape))


def handle_unsafe_view(func, args, kwargs):
    """Handle view operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    new_shape = args[1]
    input_tensor.ttnn_tensor = ttnn.reshape(input_tensor.to_ttnn, new_shape)
    input_tensor.elem = None
    return input_tensor


def handle_transpose(func, args, kwargs):
    """Handle transpose operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    dim0 = args[1]
    dim1 = args[2]
    return TorchTTNNTensor(ttnn.transpose(input_tensor.to_ttnn, dim0, dim1))


def handle_mul(func, args, kwargs):
    """Handle multiplication operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor1, input_tensor2, deallocate_a, deallocate_b, device = _prepare_binary_inputs(args[0], args[1])

    ttnn_tensor1 = ensure_tile_layout(input_tensor1.to_ttnn)
    ttnn_tensor2 = ensure_tile_layout(input_tensor2.to_ttnn)

    res = TorchTTNNTensor(ttnn.multiply(ttnn_tensor1, ttnn_tensor2))
    _cleanup_tensors((input_tensor1, deallocate_a), (input_tensor2, deallocate_b))
    return res


def handle_sub(func, args, kwargs):
    """Handle subtraction operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor1, input_tensor2, deallocate_a, deallocate_b, device = _prepare_binary_inputs(args[0], args[1])

    ttnn_tensor1 = ensure_tile_layout(input_tensor1.to_ttnn)
    ttnn_tensor2 = ensure_tile_layout(input_tensor2.to_ttnn)

    res = TorchTTNNTensor(ttnn.subtract(ttnn_tensor1, ttnn_tensor2))
    _cleanup_tensors((input_tensor1, deallocate_a), (input_tensor2, deallocate_b))
    return res


def handle_div(func, args, kwargs):
    """Handle division operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor1, input_tensor2, deallocate_a, deallocate_b, device = _prepare_binary_inputs(args[0], args[1])

    res = TorchTTNNTensor(ttnn.divide(input_tensor1.to_ttnn, input_tensor2.to_ttnn))
    _cleanup_tensors((input_tensor1, deallocate_a), (input_tensor2, deallocate_b))
    return res


def handle_add(func, args, kwargs):
    """Handle addition operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor1, input_tensor2, deallocate_a, deallocate_b, device = _prepare_binary_inputs(args[0], args[1])

    ttnn_tensor1 = ensure_tile_layout(input_tensor1.to_ttnn)
    ttnn_tensor2 = ensure_tile_layout(input_tensor2.to_ttnn)

    res = TorchTTNNTensor(ttnn.add(ttnn_tensor1, ttnn_tensor2))
    _cleanup_tensors((input_tensor1, deallocate_a), (input_tensor2, deallocate_b))
    return res


def handle_add_inplace(func, args, kwargs):
    """Handle addition operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor1 = args[0]
    input_tensor2 = args[1]
    device = None
    if not isinstance(input_tensor1, TorchTTNNTensor):
        if isinstance(input_tensor1, (int, float)):
            input_tensor1 = torch.tensor(input_tensor1)
        input_tensor1 = TorchTTNNTensor(input_tensor1, dtype=input_tensor2.dtype)
    else:
        device = input_tensor1.to_ttnn.device()
    deallocate_b = False
    if not isinstance(input_tensor2, TorchTTNNTensor):
        if isinstance(input_tensor2, (int, float)):
            input_tensor2 = torch.tensor(input_tensor2)
        input_tensor2 = TorchTTNNTensor(input_tensor2, dtype=input_tensor1.dtype)
        deallocate_b = True
    else:
        if input_tensor2.ttnn_tensor is None:
            deallocate_b = True
        device = input_tensor2.to_ttnn.device() if device is None else device
    assert device is not None, "At least one of the inputs must be a TTNN tensor."
    if input_tensor1.to_ttnn.device() != input_tensor2.to_ttnn.device():
        input_tensor1.ttnn_tensor = ttnn.to_device(input_tensor1.to_ttnn, device)
        input_tensor2.ttnn_tensor = ttnn.to_device(input_tensor2.to_ttnn, device)

    ttnn_tensor1 = input_tensor1.to_ttnn
    ttnn_tensor2 = input_tensor2.to_ttnn
    if ttnn_tensor1.layout != ttnn.TILE_LAYOUT:
        ttnn_tensor1 = ttnn.to_layout(ttnn_tensor1, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if ttnn_tensor2.layout != ttnn.TILE_LAYOUT:
        ttnn_tensor2 = ttnn.to_layout(ttnn_tensor2, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    input_tensor1.ttnn_tensor = ttnn.add(ttnn_tensor1, ttnn_tensor2)
    input_tensor1.elem = None
    if deallocate_b:
        ttnn.deallocate(input_tensor2.ttnn_tensor)
    return input_tensor1


def handle_slice(func, args, kwargs):
    """Handle slice operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    input_shape = input_tensor.shape
    dim = args[1] + len(input_shape) if args[1] < 0 else args[1]
    start = [
        0 if i != dim else max(min(args[2] + input_shape[i] if args[2] < 0 else args[2], input_shape[i]), 0)
        for i in range(len(input_shape))
    ]
    end = [
        (
            input_shape[i]
            if i != dim
            else max(min(args[3] + input_shape[i] if args[3] < 0 else args[3], input_shape[i]), 0)
        )
        for i in range(len(input_shape))
    ]
    if len(args) == 5:
        steps = []
        for i in range(len(input_shape)):
            if i == dim:
                steps.append(args[4])
            else:
                steps.append(1)
        return TorchTTNNTensor(ttnn.slice(input_tensor.to_ttnn, start, end, steps))
    return TorchTTNNTensor(ttnn.slice(input_tensor.to_ttnn, start, end))


def handle_neg(func, args, kwargs):
    """Handle negation operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    return TorchTTNNTensor(ttnn.neg(input_tensor.to_ttnn))


def handle_cat(func, args, kwargs):
    """Handle concatenation operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    tensors = args[0]
    dim = args[1]
    deallocate_tensors = []
    device = None
    for index, tensor in enumerate(tensors):
        deallocate_tensor = False
        if not isinstance(tensor, TorchTTNNTensor):
            tensors[index] = TorchTTNNTensor(tensor)
        if tensors[index].ttnn_tensor is None:
            tensors[index].ttnn_tensor = tensors[index].to_ttnn
            deallocate_tensor = True
        deallocate_tensors.append(deallocate_tensor)
        device = tensors[index].to_ttnn.device() if device is None else device
    assert device is not None, "At least one of the inputs must be a TTNN tensor."
    dtype = tensors[0].to_ttnn.dtype
    for index, tensor in enumerate(tensors):
        if deallocate_tensors[index]:
            tensor.ttnn_tensor = ttnn.to_device(tensor.to_ttnn, device)
        if tensor.ttnn_tensor.layout != ttnn.TILE_LAYOUT:
            tensor.ttnn_tensor = ttnn.to_layout(tensor.to_ttnn, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if tensor.to_ttnn.dtype != dtype:
            print(
                f"Warning: TTNN concat requires all tensors to have the same dtype, but got {tensor.to_ttnn.dtype} and {dtype}. Casting to {dtype}."
            )
            tensor.ttnn_tensor = ttnn.typecast(tensor.to_ttnn, dtype)
    res = TorchTTNNTensor(ttnn.concat([tensor.to_ttnn for tensor in tensors if tensor.numel() > 0], dim))
    for index, tensor in enumerate(tensors):
        if deallocate_tensors[index]:
            ttnn.deallocate(tensor.ttnn_tensor)
    return res


def handle_unsqueeze(func, args, kwargs):
    """Handle unsqueeze operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    dim = args[1]
    if input_tensor.to_ttnn.dtype == ttnn.uint16:
        input_tensor.ttnn_tensor = ttnn.typecast(input_tensor.to_ttnn, ttnn.uint32)
    result = ttnn.unsqueeze(input_tensor.to_ttnn, dim)
    if input_tensor.to_ttnn.dtype == ttnn.uint16:
        result = ttnn.typecast(result, ttnn.uint16)
    result = TorchTTNNTensor(result)
    return result


def handle_expand(func, args, kwargs):
    """Handle expand operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    output_shape = args[1]
    return TorchTTNNTensor(ttnn.expand(input_tensor.to_ttnn, output_shape))


def handle_bmm(func, args, kwargs):
    """Handle batch matrix multiplication."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor1, input_tensor2, deallocate_a, deallocate_b, device = _prepare_binary_inputs(args[0], args[1])

    ttnn_tensor1 = ensure_tile_layout(input_tensor1.to_ttnn)
    ttnn_tensor2 = ensure_tile_layout(input_tensor2.to_ttnn)

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    res = TorchTTNNTensor(ttnn.matmul(ttnn_tensor1, ttnn_tensor2, compute_kernel_config=compute_kernel_config))
    _cleanup_tensors((input_tensor1, deallocate_a), (input_tensor2, deallocate_b))
    return res


def handle_sdpa(func, args, kwargs):
    """Handle scaled dot product attention."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    query = args[0]
    key = args[1]
    value = args[2]
    ttnn_kwargs = {}
    if "scale" in kwargs:
        ttnn_kwargs["scale"] = kwargs["scale"]
    if len(args) == 6:
        assert isinstance(args[3], (type(None), ttnn.Tensor)), "attn_mask must be None or a TTNN tensor."
        attn_mask = args[3]
        dropout_p = args[4]
        is_causal = args[5]
    elif len(args) == 5:
        if isinstance(args[3], (float, int)):
            attn_mask = None
            dropout_p = args[3]
            is_causal = args[4]
        else:
            assert isinstance(args[3], (type(None), ttnn.Tensor)), "attn_mask must be None or a TTNN tensor."
            attn_mask = args[3]
            dropout_p = 0.0
            is_causal = args[4]
    elif len(args) == 4:
        if isinstance(args[3], (bool, int)):
            attn_mask = None
            dropout_p = 0.0
            is_causal = args[3]
        else:
            assert isinstance(args[3], (type(None), ttnn.Tensor)), "attn_mask must be None or a TTNN tensor."
            attn_mask = args[3]
            dropout_p = 0.0
            is_causal = False
    else:
        attn_mask = None
        dropout_p = 0.0
        is_causal = False
    device = None
    deallocate_q = None
    if not isinstance(query, TorchTTNNTensor):
        query = TorchTTNNTensor(query)
        deallocate_q = True
    else:
        if query.ttnn_tensor is None:
            deallocate_q = True
        device = query.to_ttnn.device()
    deallocate_k = None
    if not isinstance(key, TorchTTNNTensor):
        key = TorchTTNNTensor(key)
        deallocate_k = True
    else:
        if key.ttnn_tensor is None:
            deallocate_k = True
        device = key.to_ttnn.device() if device is None else device
    deallocate_v = None
    if not isinstance(value, TorchTTNNTensor):
        value = TorchTTNNTensor(value)
        deallocate_v = True
    else:
        if value.ttnn_tensor is None:
            deallocate_v = True
        device = value.to_ttnn.device() if device is None else device
    assert device is not None, "At least one of the inputs must be a TTNN tensor."
    if deallocate_q:
        query.ttnn_tensor = ttnn.to_device(query.to_ttnn, device)
    if deallocate_k:
        key.ttnn_tensor = ttnn.to_device(key.to_ttnn, device)
    if deallocate_v:
        value.ttnn_tensor = ttnn.to_device(value.to_ttnn, device)
    if query.to_ttnn.device() != key.to_ttnn.device() or query.to_ttnn.device() != value.to_ttnn.device():
        query.ttnn_tensor = ttnn.to_device(query.to_ttnn, device)
        key.ttnn_tensor = ttnn.to_device(key.to_ttnn, device)
        value.ttnn_tensor = ttnn.to_device(value.to_ttnn, device)

    res = TorchTTNNTensor(
        ttnn.transformer.scaled_dot_product_attention(
            query.to_ttnn, key.to_ttnn, value.to_ttnn, attn_mask=attn_mask, is_causal=is_causal, **ttnn_kwargs
        )
    )
    if deallocate_q:
        ttnn.deallocate(query.ttnn_tensor)
    if deallocate_k:
        ttnn.deallocate(key.ttnn_tensor)
    if deallocate_v:
        ttnn.deallocate(value.ttnn_tensor)
    return res


def handle_softmax(func, args, kwargs):
    """Handle softmax operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    dim = args[1]
    return TorchTTNNTensor(ttnn.softmax(input_tensor.to_ttnn, dim))


def handle_silu(func, args, kwargs):
    """Handle SiLU activation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    return TorchTTNNTensor(ttnn.silu(input_tensor.to_ttnn))


def handle_power(func, args, kwargs):
    """Handle power operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    exponent = args[1]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    return TorchTTNNTensor(ttnn.pow(input_tensor.to_ttnn, exponent))


def handle_mean(func, args, kwargs):
    """Handle mean operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    dim = args[1]
    keepdim = args[2] if len(args) > 2 else False
    return TorchTTNNTensor(ttnn.mean(input_tensor.to_ttnn, dim, keepdim=keepdim))


def handle_rsqrt(func, args, kwargs):
    """Handle reciprocal square root operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    return TorchTTNNTensor(ttnn.rsqrt(input_tensor.to_ttnn))


def handle_gelu(func, args, kwargs):
    """Handle GELU activation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    return TorchTTNNTensor(ttnn.gelu(input_tensor.to_ttnn))


def handle_relu(func, args, kwargs):
    """Handle ReLU activation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    return TorchTTNNTensor(ttnn.relu(input_tensor.to_ttnn))


def handle_new_zeros(func, args, kwargs):
    """Handle new_zeros operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    return TorchTTNNTensor(
        ttnn.zeros(
            args[1],
            memory_config=input_tensor.to_ttnn.memory_config(),
            device=input_tensor.to_ttnn.device(),
            dtype=input_tensor.to_ttnn.dtype,
        )
    )


def handle_sigmoid(func, args, kwargs):
    """Handle Sigmoid activation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    return TorchTTNNTensor(ttnn.sigmoid(input_tensor.to_ttnn))


def handle_squeeze(func, args, kwargs):
    """Handle squeeze operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    dim = args[1]
    return TorchTTNNTensor(ttnn.squeeze(input_tensor.to_ttnn, dim))


def handle_stack(func, args, kwargs):
    """Handle stack operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    tensors = args[0]
    dim = args[1] if len(args) > 1 else 0
    deallocate_tensors = []
    device = None
    for index, tensor in enumerate(tensors):
        deallocate_tensor = False
        if not isinstance(tensor, TorchTTNNTensor):
            tensors[index] = TorchTTNNTensor(tensor)
        if tensors[index].ttnn_tensor is None:
            tensors[index].ttnn_tensor = tensors[index].to_ttnn
            deallocate_tensor = True
        deallocate_tensors.append(deallocate_tensor)
        device = tensor.to_ttnn.device() if device is None else device
    assert device is not None, "At least one of the inputs must be a TTNN tensor."
    for index, tensor in enumerate(tensors):
        if deallocate_tensors[index]:
            tensor.ttnn_tensor = ttnn.to_device(tensor.to_ttnn, device)
    res = TorchTTNNTensor(ttnn.stack([tensor.to_ttnn for tensor in tensors], dim))
    for index, tensor in enumerate(tensors):
        if deallocate_tensors[index]:
            ttnn.deallocate(tensor.ttnn_tensor)

    return res


def handle_sum(func, args, kwargs):
    """Handle sum operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    dim = args[1]
    keepdim = args[2] if len(args) > 2 else False
    return TorchTTNNTensor(ttnn.sum(input_tensor.to_ttnn, dim, keepdim=keepdim))


def handle_ge(func, args, kwargs):
    """Handle greater equal operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor1, input_tensor2, deallocate_a, deallocate_b, device = _prepare_binary_inputs(args[0], args[1])

    res = TorchTTNNTensor(ttnn.ge(input_tensor1.to_ttnn, input_tensor2.to_ttnn), dtype=torch.bool)
    _cleanup_tensors((input_tensor1, deallocate_a), (input_tensor2, deallocate_b))
    return res


def handle_gt(func, args, kwargs):
    """Handle greater than operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor1, input_tensor2, deallocate_a, deallocate_b, device = _prepare_binary_inputs(args[0], args[1])

    res = TorchTTNNTensor(ttnn.gt(input_tensor1.to_ttnn, input_tensor2.to_ttnn), dtype=torch.bool)
    _cleanup_tensors((input_tensor1, deallocate_a), (input_tensor2, deallocate_b))
    return res


def handle_eq(func, args, kwargs):
    """Handle equal operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor1, input_tensor2, deallocate_a, deallocate_b, device = _prepare_binary_inputs(args[0], args[1])

    res = TorchTTNNTensor(ttnn.eq(input_tensor1.to_ttnn, input_tensor2.to_ttnn), dtype=torch.bool)
    _cleanup_tensors((input_tensor1, deallocate_a), (input_tensor2, deallocate_b))
    return res


def handle_lt(func, args, kwargs):
    """Handle less than operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor1, input_tensor2, deallocate_a, deallocate_b, device = _prepare_binary_inputs(args[0], args[1])

    res = TorchTTNNTensor(ttnn.lt(input_tensor1.to_ttnn, input_tensor2.to_ttnn), dtype=torch.bool)
    _cleanup_tensors((input_tensor1, deallocate_a), (input_tensor2, deallocate_b))
    return res


def handle_select(func, args, kwargs):
    """Handle select operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    dim = args[1]
    index = args[2]
    input_shape = list(input_tensor.shape)
    if index < 0:
        index = input_shape[dim] + index

    starts = [0] * len(input_shape)
    ends = list(input_shape)

    starts[dim] = index
    ends[dim] = index + 1

    slice_step = [1] * len(input_shape)
    new_shape = [i for index, i in enumerate(input_shape) if index != dim]
    return TorchTTNNTensor(ttnn.reshape(ttnn.slice(input_tensor.to_ttnn, starts, ends, slice_step), new_shape))


def handle_bernoulli_p(func, args, kwargs):
    """Handle bernoulli.p operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    input_tensor_ttnn = input_tensor.to_ttnn
    if len(args) > 1:
        input_tensor_ttnn = ttnn.ones_like(input_tensor.to_ttnn) * args[1]
    res = TorchTTNNTensor(ttnn.bernoulli(input_tensor_ttnn))
    if len(args) > 1:
        ttnn.deallocate(input_tensor_ttnn)
    return res


def handle_repeat(func, args, kwargs):
    """Handle repeat operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)

    repeats = args[1]
    return TorchTTNNTensor(ttnn.repeat(input_tensor.to_ttnn, repeats))


def handle_where(func, args, kwargs):
    """Handle where operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    condition = args[0]
    deallocate_cond = False
    device = None
    if not isinstance(condition, TorchTTNNTensor):
        if isinstance(condition, (int, float)):
            condition = torch.tensor(condition)
        condition = TorchTTNNTensor(condition, dtype=torch.bool)
        deallocate_cond = True
    else:
        if condition.ttnn_tensor is None:
            deallocate_cond = True
        device = condition.to_ttnn.device() if device is None else device

    input_tensor1 = args[1]
    input_tensor2 = args[2]
    deallocate_a = False
    if not isinstance(input_tensor1, TorchTTNNTensor):
        if isinstance(input_tensor1, (int, float)):
            input_tensor1 = torch.tensor(input_tensor1)
        input_tensor1 = TorchTTNNTensor(input_tensor1, dtype=input_tensor2.dtype)
        deallocate_a = True
    else:
        if input_tensor1.ttnn_tensor is None:
            deallocate_a = True
        device = input_tensor1.to_ttnn.device() if device is None else device
    deallocate_b = False
    if not isinstance(input_tensor2, TorchTTNNTensor):
        if isinstance(input_tensor2, (int, float)):
            input_tensor2 = torch.tensor(input_tensor2)
        input_tensor2 = TorchTTNNTensor(input_tensor2, dtype=input_tensor1.dtype)
        deallocate_b = True
    else:
        if input_tensor2.ttnn_tensor is None:
            deallocate_b = True
        device = input_tensor2.to_ttnn.device() if device is None else device
    assert device is not None, "At least one of the inputs must be a TTNN tensor."
    if input_tensor1.to_ttnn.device() != input_tensor2.to_ttnn.device():
        input_tensor1.ttnn_tensor = ttnn.to_device(input_tensor1.to_ttnn, device)
        input_tensor2.ttnn_tensor = ttnn.to_device(input_tensor2.to_ttnn, device)
    if condition.to_ttnn.device() != input_tensor1.to_ttnn.device():
        condition.ttnn_tensor = ttnn.to_device(condition.to_ttnn, device)
    input_tensor1.ttnn_tensor = ensure_tile_layout(input_tensor1.to_ttnn)
    input_tensor2.ttnn_tensor = ensure_tile_layout(input_tensor2.to_ttnn)
    result = TorchTTNNTensor(ttnn.where(condition.to_ttnn, input_tensor1.to_ttnn, input_tensor2.to_ttnn))

    if deallocate_a:
        ttnn.deallocate(input_tensor1.ttnn_tensor)
    if deallocate_b:
        ttnn.deallocate(input_tensor2.ttnn_tensor)

    if deallocate_cond:
        ttnn.deallocate(condition.ttnn_tensor)

    return result


def handle_split(func, args, kwargs):
    """Handle split operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)

    split_size_or_sections = args[1]
    dim = args[2] if len(args) > 2 else 0
    dim = dim + len(input_tensor.shape) if dim < 0 else dim
    # running slice to get start and end indices for each split
    input_shape = input_tensor.shape
    splits = []
    if isinstance(split_size_or_sections, int):
        split_size = split_size_or_sections
        start_idx = 0
        while start_idx < input_shape[dim]:
            end_idx = min(start_idx + split_size, input_shape[dim])
            splits.append((start_idx, end_idx))
            start_idx = end_idx
    else:
        sections = split_size_or_sections
        start_idx = 0
        for section in sections:
            end_idx = start_idx + section
            splits.append((start_idx, end_idx))
            start_idx = end_idx
    ttnn_tensors = []
    for start, end in splits:
        starts = [0] * len(input_shape)
        ends = list(input_shape)
        starts[dim] = start
        ends[dim] = end
        slice_step = [1] * len(input_shape)
        ttnn_tensor = ttnn.slice(input_tensor.to_ttnn, starts, ends, slice_step)
        ttnn_tensors.append(ttnn_tensor)
    return [TorchTTNNTensor(tensor) for tensor in ttnn_tensors]


def _to_copy(
    x,
    dtype=None,
):
    """
    TTNN equivalent of aten::_to_copy operation.

    Creates a new tensor with potentially different properties while copying data.
    """
    # Input validation - only accept tensors or scalar numbers
    assert isinstance(x, (ttnn.Tensor, int, float, bool, complex))

    # Early return for no-op cases
    if dtype is None:
        assert isinstance(x, ttnn.Tensor)
        return ttnn.clone(x)  # Use ttnn.clone for tensor copying

    dtype_converted = False

    # Convert scalars to tensors
    if isinstance(x, ttnn.Tensor):
        x_tensor = x
    else:
        x_tensor = ttnn.from_torch(torch.scalar_tensor(x))

    # Handle dtype conversion
    if dtype is not None and not dtype_converted:
        x_tensor = ttnn.typecast(x_tensor, torch_dtype_to_ttnn_dtype(dtype))

    return x_tensor


def handle_to_copy(func, args, kwargs):
    """Handle _to_copy operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    return TorchTTNNTensor(_to_copy(input_tensor.to_ttnn, kwargs.get("dtype", None)))


def handle_max(func, args, kwargs):
    """Handle max operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)

    dim = args[1] + len(input_tensor.shape) if args[1] < 0 else args[1]
    keepdim = args[2] if len(args) > 2 else False
    max_res = ttnn.max(input_tensor.to_ttnn, dim, keepdim=keepdim)
    argmax = ttnn.argmax(input_tensor.to_ttnn, dim, keepdim=keepdim)
    return (TorchTTNNTensor(max_res), TorchTTNNTensor(argmax, dtype=torch.int64))


def handle_addmm(func, args, kwargs):
    """Handle addmm operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor1 = args[0]
    input_tensor2 = args[1]
    input_tensor3 = args[2]
    device = None
    deallocate_a = None
    if not isinstance(input_tensor1, TorchTTNNTensor):
        input_tensor1 = TorchTTNNTensor(input_tensor1)
        deallocate_a = True
    else:
        if input_tensor1.ttnn_tensor is None:
            deallocate_a = True
        device = input_tensor1.to_ttnn.device()
    deallocate_b = None
    if not isinstance(input_tensor2, TorchTTNNTensor):
        input_tensor2 = TorchTTNNTensor(input_tensor2)
        deallocate_b = True
    else:
        if input_tensor2.ttnn_tensor is None:
            deallocate_b = True
        device = input_tensor2.to_ttnn.device() if device is None else device
    deallocate_c = None
    if not isinstance(input_tensor3, TorchTTNNTensor):
        input_tensor3 = TorchTTNNTensor(input_tensor3)
        deallocate_c = True
    else:
        if input_tensor3.ttnn_tensor is None:
            deallocate_c = True
        device = input_tensor3.to_ttnn.device() if device is None else device
    assert device is not None, "At least one of the inputs must be a TTNN tensor."
    if deallocate_a:
        input_tensor1.ttnn_tensor = ttnn.to_device(input_tensor1.to_ttnn, device)
    if deallocate_b:
        input_tensor2.ttnn_tensor = ttnn.to_device(input_tensor2.to_ttnn, device)
    if deallocate_c:
        input_tensor3.ttnn_tensor = ttnn.to_device(input_tensor3.to_ttnn, device)

    ttnn_tensor1 = input_tensor1.to_ttnn
    ttnn_tensor2 = input_tensor2.to_ttnn
    ttnn_tensor3 = input_tensor3.to_ttnn
    if ttnn_tensor1.layout != ttnn.TILE_LAYOUT:
        ttnn_tensor1 = ttnn.to_layout(ttnn_tensor1, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if ttnn_tensor2.layout != ttnn.TILE_LAYOUT:
        ttnn_tensor2 = ttnn.to_layout(ttnn_tensor2, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if ttnn_tensor3.layout != ttnn.TILE_LAYOUT:
        ttnn_tensor3 = ttnn.to_layout(ttnn_tensor3, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    matmul_result = ttnn.matmul(ttnn_tensor2, ttnn_tensor3, compute_kernel_config=compute_kernel_config)
    result = ttnn.add(matmul_result, ttnn_tensor1)
    ttnn.deallocate(matmul_result)
    res = TorchTTNNTensor(result)
    if deallocate_a:
        ttnn.deallocate(input_tensor1.ttnn_tensor)
    if deallocate_b:
        ttnn.deallocate(input_tensor2.ttnn_tensor)
    if deallocate_c:
        ttnn.deallocate(input_tensor3.ttnn_tensor)
    return res


def handle_zeros_like(func, args, kwargs):
    """Handle zeros_like operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    result = TorchTTNNTensor(
        ttnn.zeros_like(
            input_tensor.to_ttnn,
            memory_config=input_tensor.to_ttnn.memory_config(),
            device=input_tensor.to_ttnn.device(),
        ),
        dtype=input_tensor.dtype,
    )
    return result


def handle_index(func, args, kwargs):
    """Handle index operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    indices = args[1][0]
    if not isinstance(indices, TorchTTNNTensor):
        indices = TorchTTNNTensor(indices)
    indices_list = indices.tolist()
    tensors = []
    for idx in indices_list:
        tensors.append(input_tensor.to_ttnn[idx, ...])

    result = ttnn.stack(tensors, 0)
    return TorchTTNNTensor(result)


def handle_topk_2args(func, args, kwargs):
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    k = args[1]

    if input_tensor.to_ttnn.dtype not in [ttnn.bfloat16, ttnn.bfloat8_b]:
        print(
            f"Warning: TTNN topk only supports bfloat16 and bfloat8_b, but got {input_tensor.to_ttnn.dtype}. Casting to bfloat16."
        )
        input_tensor.ttnn_tensor = ttnn.typecast(input_tensor.ttnn_tensor, ttnn.bfloat16)
    topk_res = ttnn.topk(input_tensor.to_ttnn, k)
    return (TorchTTNNTensor(topk_res[0]), TorchTTNNTensor(topk_res[1], dtype=torch.int64))


def handle_topk_5args(func, args, kwargs):
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    k = args[1]
    dim = args[2] + len(input_tensor.shape) if args[2] < 0 else args[2]
    largest = args[3]
    sorted = args[4]
    if input_tensor.to_ttnn.dtype not in [ttnn.bfloat16, ttnn.bfloat8_b]:
        print(
            f"Warning: TTNN topk only supports bfloat16 and bfloat8_b, but got {input_tensor.to_ttnn.dtype}. Casting to bfloat16."
        )
        input_tensor.ttnn_tensor = ttnn.typecast(input_tensor.ttnn_tensor, ttnn.bfloat16)
    topk_res = ttnn.topk(input_tensor.to_ttnn, k, dim=dim, largest=largest, sorted=sorted)
    return (TorchTTNNTensor(topk_res[0]), TorchTTNNTensor(topk_res[1], dtype=torch.int64))


def handle_topk(func, args, kwargs):
    """Handle topk operation."""
    if len(args) == 2:
        return handle_topk_2args(func, args, kwargs)
    elif len(args) == 5:
        return handle_topk_5args(func, args, kwargs)
    raise NotImplementedError("topk with {} arguments is not implemented.".format(len(args)))


def handle_permute(func, args, kwargs):
    """Handle permute operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)

    dims = args[1]
    return TorchTTNNTensor(ttnn.permute(input_tensor.to_ttnn, dims))


def handle_clamp(func, args, kwargs):
    """Handle clamp operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)

    min_val = args[1] if len(args) > 1 else None
    max_val = args[2] if len(args) > 2 else None
    return TorchTTNNTensor(ttnn.clamp(input_tensor.to_ttnn, min_val, max_val))


def handle_scatter_value_inplace(func, args, kwargs):
    """Handle scatter_ value operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    dim = args[1]
    index = args[2]
    src = args[3]

    device = None
    deallocate_a = None
    if not isinstance(input_tensor, TorchTTNNTensor):
        if isinstance(input_tensor, (int, float)):
            input_tensor = torch.tensor(input_tensor)
        input_tensor = TorchTTNNTensor(input_tensor, dtype=src.dtype)
        deallocate_a = True
    else:
        if input_tensor.ttnn_tensor is None:
            deallocate_a = True
        device = input_tensor.to_ttnn.device()
    deallocate_b = None
    if not isinstance(index, TorchTTNNTensor):
        if isinstance(index, (int, float)):
            index = torch.tensor(index)
        index = TorchTTNNTensor(index, dtype=torch.int64)
        deallocate_b = True
    else:
        if index.ttnn_tensor is None:
            deallocate_b = True
        device = index.to_ttnn.device() if device is None else device
    deallocate_c = None
    if not isinstance(src, TorchTTNNTensor):
        if isinstance(src, (int, float)):
            src = torch.ones(input_tensor.shape) * src
        src = TorchTTNNTensor(src, dtype=input_tensor.dtype)
        deallocate_c = True
    else:
        if src.ttnn_tensor is None:
            deallocate_c = True
        device = src.to_ttnn.device() if device is None else device
    if input_tensor.to_ttnn.device() != index.to_ttnn.device():
        input_tensor.ttnn_tensor = ttnn.to_device(input_tensor.to_ttnn, device)
        index.ttnn_tensor = ttnn.to_device(index.to_ttnn, device)
    if input_tensor.to_ttnn.device() != src.to_ttnn.device():
        input_tensor.ttnn_tensor = ttnn.to_device(input_tensor.to_ttnn, device)
        src.ttnn_tensor = ttnn.to_device(src.to_ttnn, device)

    if input_tensor.to_ttnn.layout != ttnn.TILE_LAYOUT:
        input_tensor.ttnn_tensor = ttnn.to_layout(
            input_tensor.to_ttnn, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    if index.to_ttnn.layout != ttnn.TILE_LAYOUT:
        index.ttnn_tensor = ttnn.to_layout(index.to_ttnn, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if src.to_ttnn.layout != ttnn.TILE_LAYOUT:
        src.ttnn_tensor = ttnn.to_layout(src.to_ttnn, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    if input_tensor.to_ttnn.dtype != src.to_ttnn.dtype:
        src.ttnn_tensor = ttnn.typecast(src.ttnn_tensor, input_tensor.ttnn_tensor.dtype)

    assert device is not None, "At least one of the inputs must be a TTNN tensor."
    if deallocate_a:
        input_tensor.ttnn_tensor = ttnn.to_device(input_tensor.to_ttnn, device)
    if deallocate_b:
        index.ttnn_tensor = ttnn.to_device(index.to_ttnn, device)
    if deallocate_c:
        src.ttnn_tensor = ttnn.to_device(src.to_ttnn, device)

    input_tensor.ttnn_tensor = ttnn.scatter(input_tensor.to_ttnn, dim, index.to_ttnn, src.to_ttnn)
    input_tensor.elem = None
    return input_tensor


def handle_bitwise_not(func, args, kwargs):
    """Handle bitwise_not operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    input_tensor.ttnn_tensor = ttnn.typecast(input_tensor.ttnn_tensor, ttnn.int32)
    return TorchTTNNTensor(ttnn.bitwise_not(input_tensor.to_ttnn), dtype=torch.bool)


def handle_gather(func, args, kwargs):
    """Handle gather operation."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    dim = args[1]
    index = args[2]

    device = None
    deallocate_a = None
    if not isinstance(input_tensor, TorchTTNNTensor):
        if isinstance(input_tensor, (int, float)):
            input_tensor = torch.tensor(input_tensor)
        input_tensor = TorchTTNNTensor(input_tensor)
        deallocate_a = True
    else:
        if input_tensor.ttnn_tensor is None:
            deallocate_a = True
        device = input_tensor.to_ttnn.device()
    deallocate_b = None
    if not isinstance(index, TorchTTNNTensor):
        if isinstance(index, (int, float)):
            index = torch.tensor(index)
        index = TorchTTNNTensor(index, dtype=torch.uint32)
        deallocate_b = True
    else:
        if index.ttnn_tensor is None:
            deallocate_b = True
        device = index.to_ttnn.device() if device is None else device
    if input_tensor.to_ttnn.device() != index.to_ttnn.device():
        input_tensor.ttnn_tensor = ttnn.to_device(input_tensor.to_ttnn, device)
        index.ttnn_tensor = ttnn.to_device(index.to_ttnn, device)

    if input_tensor.to_ttnn.layout != ttnn.TILE_LAYOUT:
        input_tensor.ttnn_tensor = ttnn.to_layout(
            input_tensor.to_ttnn, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    if index.to_ttnn.layout != ttnn.TILE_LAYOUT:
        index.ttnn_tensor = ttnn.to_layout(index.to_ttnn, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    if index.to_ttnn.dtype != ttnn.uint32:
        index.ttnn_tensor = ttnn.typecast(index.ttnn_tensor, ttnn.uint32)

    res = TorchTTNNTensor(ttnn.gather(input_tensor.to_ttnn, dim, index.to_ttnn))

    if deallocate_a:
        ttnn.deallocate(input_tensor.ttnn_tensor)
    if deallocate_b:
        ttnn.deallocate(index.ttnn_tensor)

    return res


# Mapping of ATen operations to TTNN handlers
func_to_ttnn_compatible = {
    "aten::view": handle_view,
    "aten::_unsafe_view": handle_view,
    "aten::transpose.int": handle_transpose,
    "aten::mul.Tensor": handle_mul,
    "aten::sub.Tensor": handle_sub,
    "aten::div.Tensor": handle_div,
    "aten::slice.Tensor": handle_slice,
    "aten::neg": handle_neg,
    "aten::cat": handle_cat,
    "aten::add.Tensor": handle_add,
    "aten::unsqueeze": handle_unsqueeze,
    "aten::squeeze.dim": handle_squeeze,
    "aten::expand": handle_expand,
    "aten::mul.Scalar": handle_mul,
    "aten::sub.Scalar": handle_sub,
    "aten::add.Scalar": handle_add,
    # "aten::add_.Tensor": handle_add_inplace,
    "aten::bmm": handle_bmm,
    "aten::_softmax": handle_softmax,
    "aten::pow.Tensor_Scalar": handle_power,
    "aten::mean.dim": handle_mean,
    "aten::rsqrt": handle_rsqrt,
    "aten::gelu": handle_gelu,
    "aten::relu": handle_relu,
    "aten::new_zeros": handle_new_zeros,
    "aten::sigmoid": handle_sigmoid,
    "aten::stack": handle_stack,
    "aten::sum.dim_IntList": handle_sum,
    "aten::ge.Scalar": handle_ge,
    "aten::gt.Scalar": handle_gt,
    "aten::select.int": handle_select,
    "aten::bernoulli.p": handle_bernoulli_p,
    "aten::repeat": handle_repeat,
    "aten::eq.Scalar": handle_eq,
    "aten::eq.Tensor": handle_eq,
    "aten::lt.Tensor": handle_lt,
    "aten::where.self": handle_where,
    "aten::split.Tensor": handle_split,
    "aten::_to_copy": handle_to_copy,
    "aten::max.dim": handle_max,
    "aten::addmm": handle_addmm,
    "aten::zeros_like": handle_zeros_like,
    "aten::index.Tensor": handle_index,
    "aten::topk": handle_topk,
    "aten::permute": handle_permute,
    "aten::clamp": handle_clamp,
    "aten::clone": handle_to_copy,
    "aten::_safe_softmax": handle_softmax,
    "aten::mm": handle_bmm,
    "aten::silu": handle_silu,
    # "aten::scatter_.value": handle_scatter_value_inplace,
    "aten::bitwise_not": handle_bitwise_not,
    "aten::gather": handle_gather,
}


def can_dispatch_to_ttnn(func_name: str, args=None, kwargs=None) -> bool:
    """Check if operation can be dispatched to TTNN."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    any_ttnn_tensor = False
    for elem in args:
        if isinstance(elem, TorchTTNNTensor) and elem.ttnn_tensor is not None and elem.ttnn_tensor.device() is not None:
            if not elem.ttnn_tensor.is_allocated():
                print("TTNN: Found deallocated TTNN tensor, cannot dispatch to TTNN.")
                return False
            any_ttnn_tensor = True
        elif (
            isinstance(elem, torch.Tensor)
            and elem.dtype not in TORCH_TO_TTNN
            and (not isinstance(elem, TorchTTNNTensor) or elem.ttnn_tensor is None)
        ):
            print(
                f"TTNN: Found unsupported dtype {elem.dtype} for TTNN tensor in list/tuple, cannot dispatch {func_name} to TTNN"
            )
            return False
        elif isinstance(elem, (list, tuple)):
            for sub_elem in elem:
                if (
                    isinstance(sub_elem, TorchTTNNTensor)
                    and sub_elem.ttnn_tensor is not None
                    and sub_elem.ttnn_tensor.device() is not None
                ):
                    if not sub_elem.ttnn_tensor.is_allocated():
                        print("TTNN: Found deallocated TTNN tensor in list/tuple, cannot dispatch to TTNN.")
                        return False
                    any_ttnn_tensor = True
                elif (
                    isinstance(sub_elem, torch.Tensor)
                    and sub_elem.dtype not in TORCH_TO_TTNN
                    and (not isinstance(sub_elem, TorchTTNNTensor) or sub_elem.ttnn_tensor is None)
                ):
                    print(
                        f"TTNN: Found unsupported dtype {sub_elem.dtype} for TTNN tensor in list/tuple, cannot dispatch {func_name} to TTNN"
                    )
                    return False
    passed = True
    if "aten::slice.Tensor" == func_name:
        if (
            not isinstance(args[1], int)
            or not isinstance(args[2], int)
            or not isinstance(args[3], int)
            or len(args) not in [4, 5]
        ):
            print("TTNN: aten::slice.Tensor only supports int arguments for start, end, step.")
            passed = False
        if len(args) == 5:
            if not isinstance(args[4], int):
                print("TTNN: aten::slice.Tensor only supports int argument for dim.")
                passed = False
    if "aten::addmm" == func_name:
        if len(kwargs) > 0:
            print("TTNN: aten::addmm does not support keyword arguments.")
            passed = False
    if "aten::index.Tensor" == func_name:
        if (
            len(kwargs) > 0
            or len(args) != 2
            or not isinstance(args[1], (list, tuple))
            or len(args[1]) != 1
            or not isinstance(args[1][0], (TorchTTNNTensor, torch.Tensor))
            or not len(args[1][0].shape) == 1
        ):
            print("TTNN: aten::index.Tensor only supports a single 1D tensor as index.")
            passed = False
    if "aten::topk" == func_name:
        if len(kwargs) > 0 or not (len(args) == 2 or len(args) == 5):
            print("TTNN: aten::topk only supports 2 or 5 positional arguments.")
            passed = False
    if "aten::bitwise_not" == func_name:
        if (
            len(kwargs) > 0
            or len(args) != 1
            or not isinstance(args[0], torch.Tensor)
            or not args[0].dtype == torch.bool
        ):
            print("TTNN: aten::bitwise_not only supports a single boolean tensor argument.")
            passed = False
    if "aten::sum.dim_IntList" == func_name:
        if args[0].to_ttnn.dtype not in [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint32]:
            print("TTNN: aten::sum.dim_IntList only supports float32, bfloat16, bfloat8_b, and uint32 dtypes.")
            passed = False
    if "aten::unsqueeze" == func_name:
        if args[0].to_ttnn.dtype not in [ttnn.float32, ttnn.bfloat16, ttnn.int32, ttnn.uint32, ttnn.uint16]:
            print("TTNN: aten::unsqueeze only supports float32, bfloat16, int32, and uint32 dtypes.")
            passed = False
    if not any_ttnn_tensor and func_name in ["aten::mm", "aten::addmm", "aten::bmm"]:
        print("Found invalid TTNN dispatch for matmul operation. Please check input dtypes and layouts.")
    if func_name in func_to_ttnn_compatible and any_ttnn_tensor:
        return passed
    if func_name != "aten::_scaled_dot_product_flash_attention_for_cpu" and passed and any_ttnn_tensor:
        print(
            f"Found Operation {func_name} that if written in ttnn would be more efficient. "
            "Please map this function to an appropriate ttnn function."
        )
    return False


def dispatch_to_ttnn(func_name, args, kwargs):
    """Dispatch operation to TTNN handler."""
    return func_to_ttnn_compatible[func_name](func_name, args, kwargs)
