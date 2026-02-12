# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TTNN operation dispatch handlers and mapping."""

import os
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


def handle_reshape(func, args, kwargs):
    """Handle aten::reshape on TTNN. When physical volume != new_shape volume (padded buffer), flatten to (1, old_vol), slice to (1, new_vol), then reshape."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    import math

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    new_shape = args[1] if len(args) > 1 else kwargs.get("shape")
    new_shape = tuple(int(s) for s in new_shape)
    t = input_tensor.to_ttnn
    ttnn_shp = t.shape
    new_vol = math.prod(new_shape)
    old_vol = math.prod(int(s) for s in ttnn_shp)
    if old_vol != new_vol:
        t_flat = ttnn.reshape(t, (1, old_vol))
        t_slice = ttnn.slice(t_flat, (0, 0), (1, new_vol), (1, 1))
        t = t_slice
    out = ttnn.reshape(t, new_shape)
    return TorchTTNNTensor(out)


def handle_to_dtype(func, args, kwargs):
    """Handle aten::to.dtype — typecast on TTNN (Tensix) instead of CPU fallback."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    dtype = kwargs.get("dtype") if kwargs else None
    if dtype is None and len(args) >= 2:
        dtype = args[1]
    return TorchTTNNTensor(_to_copy(input_tensor.to_ttnn, dtype))


def handle_dropout(func, args, kwargs):
    """Handle aten::dropout — no-op on device when eval (train=False); keeps tensor on Tensix."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    return TorchTTNNTensor(ttnn.clone(input_tensor.to_ttnn))


def handle_broadcast_tensors(func, args, kwargs):
    """Handle aten::broadcast_tensors — expand on TTNN (Tensix), return plain tensors so callers (e.g. mse_loss C++) accept the result."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    tensors = args[0] if args else ()
    if not tensors:
        return ()
    if len(tensors) == 1:
        t = tensors[0]
        return (t.to_torch if isinstance(t, TorchTTNNTensor) else t,)
    torch_refs = []
    for t in tensors:
        if isinstance(t, TorchTTNNTensor):
            torch_refs.append(t.elem if t.elem is not None else t.to_torch)
        else:
            torch_refs.append(torch.as_tensor(t) if not isinstance(t, torch.Tensor) else t)
    broadcasted = torch.broadcast_tensors(*torch_refs)
    target_shape = tuple(broadcasted[0].shape)
    out = []
    for t in tensors:
        if isinstance(t, TorchTTNNTensor):
            t_w = t
            if t_w.shape == target_shape:
                out.append(t_w.to_torch)
            else:
                out.append(ttnn.to_torch(ttnn.expand(t_w.to_ttnn, target_shape)))
        else:
            out.append(t)
    return tuple(out)


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
    """Handle addition operation. Uses CPU fallback when one operand is plain torch
    (e.g. vision tower output) to avoid SIGFPE from device add/layout on awkward shapes."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    a, b = args[0], args[1]
    a_plain = isinstance(a, torch.Tensor) and not isinstance(a, TorchTTNNTensor)
    b_plain = isinstance(b, torch.Tensor) and not isinstance(b, TorchTTNNTensor)

    if a_plain or b_plain:
        device = None
        if isinstance(a, TorchTTNNTensor) and getattr(a, "ttnn_tensor", None) is not None:
            device = a.to_ttnn.device()
        if isinstance(b, TorchTTNNTensor) and getattr(b, "ttnn_tensor", None) is not None:
            device = b.to_ttnn.device() if device is None else device
        if device is None:
            raise RuntimeError("At least one of the inputs must be a TTNN tensor.")
        if isinstance(a, TorchTTNNTensor):
            t_a_ttnn = ensure_tile_layout(a.to_ttnn)
        else:
            t_a_ttnn = ttnn.from_torch(
                a.detach().to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
            )
            t_a_ttnn = ensure_tile_layout(t_a_ttnn)
        if isinstance(b, TorchTTNNTensor):
            t_b_ttnn = ensure_tile_layout(b.to_ttnn)
        else:
            t_b_ttnn = ttnn.from_torch(
                b.detach().to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
            )
            t_b_ttnn = ensure_tile_layout(t_b_ttnn)
        return TorchTTNNTensor(ttnn.add(t_a_ttnn, t_b_ttnn))

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


def handle_narrow(func, args, kwargs):
    """Handle aten::narrow — slice along one dimension. narrow(input, dim, start, length)."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    input_shape = list(input_tensor.shape)
    dim = int(args[1])
    dim = dim + len(input_shape) if dim < 0 else dim
    start = int(args[2])
    length = int(args[3])
    start = start + input_shape[dim] if start < 0 else start
    end = start + length
    start_list = [0] * len(input_shape)
    end_list = list(input_shape)
    start_list[dim] = start
    end_list[dim] = end
    return TorchTTNNTensor(ttnn.slice(input_tensor.to_ttnn, start_list, end_list))


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
        math_approx_mode=True,
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

    _allowed = (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b)
    q_t, k_t, v_t = query.to_ttnn, key.to_ttnn, value.to_ttnn
    cast_tensors = []
    if q_t.dtype not in _allowed:
        q_t = ttnn.typecast(q_t, ttnn.bfloat16)
        cast_tensors.append(q_t)
    if k_t.dtype not in _allowed:
        k_t = ttnn.typecast(k_t, ttnn.bfloat16)
        cast_tensors.append(k_t)
    if v_t.dtype not in _allowed:
        v_t = ttnn.typecast(v_t, ttnn.bfloat16)
        cast_tensors.append(v_t)

    res = TorchTTNNTensor(
        ttnn.transformer.scaled_dot_product_attention(
            q_t, k_t, v_t, attn_mask=attn_mask, is_causal=is_causal, **ttnn_kwargs
        )
    )
    for t in cast_tensors:
        ttnn.deallocate(t)
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


def handle_log_softmax(func, args, kwargs):
    """Handle log_softmax: log(softmax(x, dim))."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    dim = args[1] if len(args) > 1 else kwargs.get("dim", -1)
    soft = ttnn.softmax(input_tensor.to_ttnn, dim)
    out = ttnn.log(soft)
    ttnn.deallocate(soft)
    return TorchTTNNTensor(out)


def handle_log(func, args, kwargs):
    """Handle aten::log — natural log on device."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    return TorchTTNNTensor(ttnn.log(input_tensor.to_ttnn))


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


def handle_native_layer_norm(func, args, kwargs):
    """Handle aten::native_layer_norm in the attention path (input, normalized_shape, weight, bias, eps).
    Used when layer_norm runs on TTNN tensors from attention; other LayerNorms should use direct
    module mapping (nn.LayerNorm -> TTNNLayerNorm)."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    if len(args) < 5:
        raise ValueError("aten::native_layer_norm expects (input, normalized_shape, weight, bias, eps)")
    input_tensor = args[0]
    weight = args[2]
    bias = args[3]
    eps = float(args[4])
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    device = input_tensor.to_ttnn.device()
    inp_ttnn = ensure_tile_layout(input_tensor.to_ttnn)
    weight_ttnn = None
    bias_ttnn = None
    if weight is not None and isinstance(weight, torch.Tensor):
        weight_ttnn = ttnn.from_torch(
            weight.to(torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )
    if bias is not None and isinstance(bias, torch.Tensor):
        bias_ttnn = ttnn.from_torch(
            bias.to(torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )
    out = ttnn.layer_norm(inp_ttnn, weight=weight_ttnn, bias=bias_ttnn, epsilon=eps)
    return TorchTTNNTensor(out)


def handle_mse_loss(func, args, kwargs):
    """Handle aten::mse_loss(input, target, reduction). Maps to ttnn.mse_loss(ref=target, pred=input)."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    if len(args) < 2:
        raise ValueError("aten::mse_loss expects (input, target) and optional reduction")
    inp = args[0]
    target = args[1]
    reduction = args[2] if len(args) > 2 else 0
    if not isinstance(inp, TorchTTNNTensor):
        inp = TorchTTNNTensor(inp)
    if not isinstance(target, TorchTTNNTensor):
        target = TorchTTNNTensor(target)
    device = inp.to_ttnn.device()
    ref_tt = ensure_tile_layout(target.to_ttnn)
    pred_tt = ensure_tile_layout(inp.to_ttnn)
    ref_tt = ttnn.to_device(ref_tt, device)
    pred_tt = ttnn.to_device(pred_tt, device)
    reduction_map = {0: ttnn.LossReductionMode.NONE, 1: ttnn.LossReductionMode.MEAN, 2: ttnn.LossReductionMode.SUM}
    mode = reduction_map.get(int(reduction), ttnn.LossReductionMode.NONE)
    out = ttnn.mse_loss(ref_tt, pred_tt, reduction=mode)
    return TorchTTNNTensor(out)


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
    """Handle sum operation (aten::sum.dim_IntList or aten::sum full-reduce)."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    ndim = len(input_tensor.shape)
    if len(args) >= 2 and isinstance(args[1], (list, tuple)):
        dim = list(args[1])
        keepdim = args[2] if len(args) > 2 else False
    else:
        dim = list(range(ndim))
        keepdim = False
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


def handle_masked_fill_Scalar(func, args, kwargs):
    """Handle aten::masked_fill.Scalar — where mask is True, fill with value. output = where(mask, value, input)."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    mask = args[1]
    value = float(args[2]) if len(args) > 2 else float(kwargs.get("value", 0))
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    deallocate_mask = False
    if not isinstance(mask, TorchTTNNTensor):
        mask = TorchTTNNTensor(mask, dtype=torch.bool)
        deallocate_mask = True
    device = input_tensor.to_ttnn.device()
    value_tensor_tt = ttnn.ones_like(input_tensor.to_ttnn) * value
    out = ttnn.where(mask.to_ttnn, value_tensor_tt, input_tensor.to_ttnn)
    ttnn.deallocate(value_tensor_tt)
    if deallocate_mask and mask.ttnn_tensor is not None:
        ttnn.deallocate(mask.to_ttnn)
    return TorchTTNNTensor(out)


def handle_masked_fill_Tensor(func, args, kwargs):
    """Handle aten::masked_fill.Tensor — where mask is True, fill with value tensor. output = where(mask, value, input)."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    mask = args[1]
    value = args[2]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    deallocate_mask = False
    if not isinstance(mask, TorchTTNNTensor):
        mask = TorchTTNNTensor(mask, dtype=torch.bool)
        deallocate_mask = True
    deallocate_value = False
    if not isinstance(value, TorchTTNNTensor):
        value = TorchTTNNTensor(value)
        deallocate_value = True
    device = input_tensor.to_ttnn.device()
    for t, name in [(mask, "mask"), (value, "value")]:
        if t.ttnn_tensor is not None and t.ttnn_tensor.device() != device:
            t.ttnn_tensor = ttnn.to_device(t.to_ttnn, device)
    out = ttnn.where(mask.to_ttnn, value.to_ttnn, input_tensor.to_ttnn)
    if deallocate_mask and mask.ttnn_tensor is not None:
        ttnn.deallocate(mask.to_ttnn)
    if deallocate_value and value.ttnn_tensor is not None:
        ttnn.deallocate(value.to_ttnn)
    return TorchTTNNTensor(out)


def handle_copy_(func, args, kwargs):
    """Handle aten::copy_ (in-place copy): copy src into self on device. Returns self."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    self_tensor = args[0]
    src = args[1]
    if not isinstance(self_tensor, TorchTTNNTensor):
        self_tensor = TorchTTNNTensor(self_tensor)
    device = self_tensor.to_ttnn.device() if self_tensor.ttnn_tensor is not None else None
    if device is None and isinstance(src, TorchTTNNTensor) and src.ttnn_tensor is not None:
        device = src.to_ttnn.device()
    if device is None:
        src_ttnn = ttnn.from_torch(src.cpu() if isinstance(src, torch.Tensor) else src)
        device = src_ttnn.device()
    else:
        if isinstance(src, TorchTTNNTensor):
            src_ttnn = src.to_ttnn
            if src_ttnn.device() != device:
                src_ttnn = ttnn.to_device(src_ttnn, device)
        else:
            src_ttnn = ttnn.from_torch(
                src.cpu() if isinstance(src, torch.Tensor) else src,
                device=device,
                dtype=torch_dtype_to_ttnn_dtype(src.dtype if isinstance(src, torch.Tensor) else torch.float32),
                layout=ttnn.TILE_LAYOUT,
            )
    cloned = ttnn.clone(src_ttnn)
    if self_tensor.ttnn_tensor is not None and self_tensor.ttnn_tensor.is_allocated():
        ttnn.deallocate(self_tensor.ttnn_tensor)
    self_tensor.ttnn_tensor = cloned
    self_tensor.elem = None
    return self_tensor


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
    kwargs = kwargs or {}
    dim = kwargs.get("dim", args[2] if len(args) > 2 else 0)
    dim = dim + len(input_tensor.shape) if dim < 0 else dim
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


def handle_unbind(func, args, kwargs):
    """Handle unbind: split tensor along dim into a tuple of views (one per index along dim)."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    kwargs = kwargs or {}
    dim = int(kwargs.get("dim", args[1] if len(args) > 1 else 0))
    input_shape = list(input_tensor.shape)
    dim = dim + len(input_shape) if dim < 0 else dim
    size = input_shape[dim]
    ttnn_tensors = []
    for i in range(size):
        starts = [0] * len(input_shape)
        ends = list(input_shape)
        starts[dim] = i
        ends[dim] = i + 1
        slice_step = [1] * len(input_shape)
        ttnn_tensor = ttnn.slice(input_tensor.to_ttnn, starts, ends, slice_step)
        ttnn_tensors.append(ttnn.squeeze(ttnn_tensor, dim))
    return tuple(TorchTTNNTensor(t) for t in ttnn_tensors)


def handle_pixel_unshuffle(func, args, kwargs):
    """Handle pixel_unshuffle: [N, C, H, W] with downscale_factor r -> [N, C*r*r, H//r, W//r].
    Implemented as reshape -> permute -> reshape (no dedicated ttnn op)."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    downscale_factor = int(args[1] if len(args) > 1 else kwargs.get("downscale_factor", 2))
    if downscale_factor <= 0:
        raise ValueError("pixel_unshuffle downscale_factor must be positive")
    shp = input_tensor.shape
    if len(shp) != 4:
        raise ValueError("pixel_unshuffle expects 4D input [N, C, H, W]")
    N, C, H, W = shp
    r = downscale_factor
    if H % r != 0 or W % r != 0:
        raise ValueError(f"pixel_unshuffle: H and W must be divisible by downscale_factor {r}")
    tt = input_tensor.to_ttnn
    mid = ttnn.reshape(tt, (N, C, H // r, r, W // r, r))
    mid = ttnn.permute(mid, (0, 1, 3, 5, 2, 4))
    out = ttnn.reshape(mid, (N, C * r * r, H // r, W // r))
    return TorchTTNNTensor(out)


def handle_chunk(func, args, kwargs):
    """Handle chunk operation: split tensor into n chunks along dim (like split with equal-sized parts)."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)

    n_chunks = int(args[1])
    dim = int(kwargs.get("dim", args[2] if len(args) > 2 else 0))
    input_shape = list(input_tensor.shape)
    dim = dim + len(input_shape) if dim < 0 else dim
    size = input_shape[dim]
    if n_chunks <= 0 or size == 0:
        return tuple()
    base = size // n_chunks
    remainder = size % n_chunks
    chunk_sizes = [base + 1] * remainder + [base] * (n_chunks - remainder)
    chunk_sizes = [s for s in chunk_sizes if s > 0]
    if not chunk_sizes:
        return tuple()

    ttnn_tensors = []
    start = 0
    for ch_size in chunk_sizes:
        starts = [0] * len(input_shape)
        ends = list(input_shape)
        starts[dim] = start
        ends[dim] = start + ch_size
        start += ch_size
        slice_step = [1] * len(input_shape)
        t = ttnn.slice(input_tensor.to_ttnn, starts, ends, slice_step)
        ttnn_tensors.append(TorchTTNNTensor(t))
    return tuple(ttnn_tensors)


def handle_contiguous(func, args, kwargs):
    """Handle contiguous: return a contiguous copy on device (clone)."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    t = ensure_tile_layout(input_tensor.to_ttnn)
    return TorchTTNNTensor(ttnn.clone(t))


def _to_copy(
    x,
    dtype=None,
):
    """
    TTNN equivalent of aten::_to_copy operation.

    Creates a new tensor with potentially different properties while copying data.
    """
    assert isinstance(x, (ttnn.Tensor, int, float, bool, complex))

    if dtype is None:
        assert isinstance(x, ttnn.Tensor)
        return ttnn.clone(x)

    dtype_converted = False

    if isinstance(x, ttnn.Tensor):
        x_tensor = x
    else:
        x_tensor = ttnn.from_torch(torch.scalar_tensor(x))

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
        math_approx_mode=True,
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
    """Handle index operation. Supports single 1D int/bool index on any dimension (e.g. x[:, idx, :]).
    Uses ttnn.gather when index_dim >= 1 for better performance; slice+stack for dim 0 or fallback."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    index_elems = args[1]
    tensor_positions = [i for i, e in enumerate(index_elems) if isinstance(e, (TorchTTNNTensor, torch.Tensor))]
    if not tensor_positions:
        for i, e in enumerate(index_elems):
            if hasattr(e, "shape") and getattr(e, "shape", ()) and len(getattr(e, "shape", ())) == 1:
                tensor_positions = [i]
                break
    assert len(tensor_positions) == 1
    index_dim = tensor_positions[0]
    indices = index_elems[index_dim]
    index_torch = indices.to_torch if hasattr(indices, "to_torch") else indices
    if not isinstance(index_torch, torch.Tensor):
        index_torch = torch.as_tensor(index_torch, dtype=torch.long)
    if index_torch.dtype == torch.bool:
        indices_list = torch.where(index_torch)[0].tolist()
    else:
        indices_list = index_torch.tolist()

    t = input_tensor.to_ttnn
    shape = list(t.shape)
    rank = len(shape)
    L = len(indices_list)

    dim_size = shape[index_dim]
    if dim_size > 0 and any(idx >= dim_size for idx in indices_list):
        indices_list = [min(max(int(idx), 0), dim_size - 1) for idx in indices_list]

    if index_dim == 0:
        tensors = [t[idx, ...] for idx in indices_list]
        result = ttnn.stack(tensors, 0)
    else:
        try:
            device = t.device()
            index_shape = list(shape)
            index_shape[index_dim] = L
            view_shape = [1] * rank
            view_shape[index_dim] = L
            index_torch_nd = (
                torch.tensor(indices_list, dtype=torch.long, device="cpu").reshape(view_shape).expand(index_shape)
            )
            index_ttnn = ttnn.from_torch(
                index_torch_nd,
                device=device,
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
            )
            result = ttnn.gather(t, index_dim, index=index_ttnn)
            ttnn.deallocate(index_ttnn)
        except Exception:
            slices = []
            for idx in indices_list:
                start = [0] * rank
                end = list(shape)
                start[index_dim] = idx
                end[index_dim] = idx + 1
                s = ttnn.slice(t, start, end)
                s = ttnn.squeeze(s, index_dim)
                slices.append(s)
            result = ttnn.stack(slices, index_dim)
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


def handle_constant_pad_nd(func, args, kwargs):
    """Handle constant_pad_nd (F.pad mode='constant') for patch_embedding and similar.
    Converts PyTorch pad list (last dim first, left/right per dim) to ttnn.pad format
    (list of [before, after] per dimension, first dim first). Only end padding is supported on device."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)

    pad_list = list(args[1])
    value = float(args[2]) if len(args) > 2 else 0.0

    t = input_tensor.to_ttnn
    t = ensure_tile_layout(t)
    rank = len(t.shape)
    if rank != 4:
        raise NotImplementedError(
            f"handle_constant_pad_nd: ttnn.pad on device requires rank 4, got {rank}. "
            "Use CPU fallback for this tensor."
        )

    original_pad_len = len(pad_list)
    if len(pad_list) < 2 * rank:
        if len(pad_list) % 2 != 0:
            raise ValueError(f"handle_constant_pad_nd: pad_list length must be even, got {len(pad_list)}")
        num_padded_dims = len(pad_list) // 2
        pad_list = [0, 0] * (rank - num_padded_dims) + pad_list
    elif len(pad_list) != 2 * rank:
        raise ValueError(f"handle_constant_pad_nd: pad_list length must be <= 2*rank={2*rank}, got {len(pad_list)}")

    use_exact_padding = original_pad_len == 4

    padding_config = []
    for i in range(rank):
        j = (rank - 1 - i) * 2
        left, right = int(pad_list[j]), int(pad_list[j + 1])
        if left != 0:
            raise NotImplementedError(
                "handle_constant_pad_nd: ttnn.pad on device does not support front (left) padding. Use CPU fallback."
            )
        if not use_exact_padding and i >= rank - 2 and right > 0:
            right = ((right + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        padding_config.append([left, right])

    out = ttnn.pad(t, padding=padding_config, value=value)
    return TorchTTNNTensor(out)


def handle_flatten_using_ints(func, args, kwargs):
    """Handle aten::flatten.using_ints — flatten dims [start_dim, end_dim] on TTNN via reshape."""
    import math
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    start_dim = int(args[1]) if len(args) > 1 else 0
    end_dim = int(args[2]) if len(args) > 2 else -1
    shape = list(input_tensor.shape)
    rank = len(shape)
    start_dim = start_dim if start_dim >= 0 else start_dim + rank
    end_dim = end_dim if end_dim >= 0 else end_dim + rank
    start_dim = max(0, min(start_dim, rank - 1))
    end_dim = max(0, min(end_dim, rank - 1))
    if start_dim > end_dim:
        start_dim, end_dim = end_dim, start_dim
    flat_size = math.prod(shape[start_dim : end_dim + 1])
    new_shape = tuple(shape[:start_dim]) + (flat_size,) + tuple(shape[end_dim + 1 :])
    return TorchTTNNTensor(ttnn.reshape(input_tensor.to_ttnn, new_shape))


def handle_im2col(func, args, kwargs):
    """Handle aten::im2col (F.unfold) on Tensix. Supports dilation=(1,1), padding=(0,0), stride==kernel_size.
    Uses slice+reshape+cat per window to avoid a single large 6D permute (which can OOM on device).
    """
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    kernel_size = args[1] if len(args) > 1 else kwargs.get("kernel_size")
    dilation = args[2] if len(args) > 2 else kwargs.get("dilation", (1, 1))
    padding = args[3] if len(args) > 3 else kwargs.get("padding", (0, 0))
    stride = args[4] if len(args) > 4 else kwargs.get("stride")
    if isinstance(kernel_size, (list, tuple)):
        kH, kW = int(kernel_size[0]), int(kernel_size[1])
    else:
        kH = kW = int(kernel_size)
    if isinstance(dilation, (list, tuple)):
        dil_h, dil_w = int(dilation[0]), int(dilation[1])
    else:
        dil_h = dil_w = int(dilation)
    if isinstance(padding, (list, tuple)):
        pad_h, pad_w = int(padding[0]), int(padding[1])
    else:
        pad_h = pad_w = int(padding)
    if isinstance(stride, (list, tuple)):
        sH, sW = int(stride[0]), int(stride[1])
    else:
        sH = sW = int(stride)
    if dil_h != 1 or dil_w != 1 or pad_h != 0 or pad_w != 0:
        raise NotImplementedError("handle_im2col: only dilation=(1,1) and padding=(0,0) are supported on TTNN.")
    if sH != kH or sW != kW:
        raise NotImplementedError("handle_im2col: only stride == kernel_size (non-overlapping) is supported on TTNN.")
    N, C, H, W = input_tensor.shape
    H_logical = (H // kH) * kH
    W_logical = (W // kW) * kW
    if H_logical < kH or W_logical < kW:
        raise NotImplementedError("handle_im2col: spatial size too small for kernel on TTNN.")
    out_h = H_logical // kH
    out_w = W_logical // kW
    t = input_tensor.to_ttnn
    if H_logical != H or W_logical != W:
        t = ttnn.slice(t, (0, 0, 0, 0), (N, C, H_logical, W_logical), (1, 1, 1, 1))
    blocks = []
    for i in range(out_h):
        for j in range(out_w):
            r0, c0 = i * sH, j * sW
            block = ttnn.slice(t, (0, 0, r0, c0), (N, C, r0 + kH, c0 + kW), (1, 1, 1, 1))
            block_flat = ttnn.reshape(block, (N, C * kH * kW, 1))
            blocks.append(block_flat)
    out = ttnn.concat(blocks, dim=2)
    return TorchTTNNTensor(out)


def handle_linear(func, args, kwargs):
    """Handle aten::linear: output = input @ weight.T + bias (or no bias). Maps to addmm on TTNN."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    if not isinstance(weight, TorchTTNNTensor):
        weight = TorchTTNNTensor(weight)
    device = input_tensor.to_ttnn.device()
    weight_tt = weight.to_ttnn
    if weight_tt.layout != ttnn.TILE_LAYOUT:
        weight_tt = ensure_tile_layout(ttnn.to_device(weight_tt, device))
    weight_t = TorchTTNNTensor(ttnn.transpose(weight_tt, -2, -1))
    if bias is not None:
        if not isinstance(bias, TorchTTNNTensor):
            bias = TorchTTNNTensor(bias)
        return handle_addmm(func, (bias, input_tensor, weight_t), kwargs)
    out_shape = tuple(input_tensor.shape[:-1]) + (weight.shape[0],)
    zeros_tt = TorchTTNNTensor(
        ttnn.zeros(
            out_shape,
            memory_config=input_tensor.to_ttnn.memory_config(),
            device=device,
            dtype=input_tensor.to_ttnn.dtype,
        )
    )
    return handle_addmm(func, (zeros_tt, input_tensor, weight_t), kwargs)


def handle_pad(func, args, kwargs):
    """Handle aten::pad when mode is constant — delegate to constant_pad_nd logic."""
    mode = kwargs.get("mode", "constant")
    value = float(kwargs.get("value", 0))
    if len(args) > 2:
        a2 = args[2]
        if isinstance(a2, str):
            mode = a2
            value = float(args[3]) if len(args) > 3 else 0.0
        else:
            value = float(a2)
    if mode != "constant":
        raise NotImplementedError(f"handle_pad: only mode='constant' is supported on TTNN, got mode={mode!r}")
    return handle_constant_pad_nd(func, [args[0], args[1], value], kwargs)


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


def _get_func_to_ttnn_compatible():
    from models.experimental.tt_symbiote.core.dispatchers.tensor_operations_dispatcher import (
        func_to_ttnn_compatible,
    )

    return func_to_ttnn_compatible


def _log_fallback_op(func_name: str) -> None:
    """Log once per op the same message used when an op runs on PyTorch instead of TTNN."""
    if not hasattr(can_dispatch_to_ttnn, "_reported_fallback_ops"):
        can_dispatch_to_ttnn._reported_fallback_ops = set()
    if func_name not in can_dispatch_to_ttnn._reported_fallback_ops:
        can_dispatch_to_ttnn._reported_fallback_ops.add(func_name)
        print(
            f"Found Operation {func_name} that if written in ttnn would be more efficient. "
            "Please map this function to an appropriate ttnn function."
        )


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
        if len(kwargs) > 0 or len(args) != 2 or not isinstance(args[1], (list, tuple)):
            passed = False
        else:
            index_elems = args[1]
            tensor_indices = [
                i
                for i, e in enumerate(index_elems)
                if isinstance(e, (TorchTTNNTensor, torch.Tensor))
                or (hasattr(e, "shape") and hasattr(e, "__array__") and not isinstance(e, type(None)))
            ]
            if len(tensor_indices) != 1:
                passed = False
            else:
                idx_tensor = index_elems[tensor_indices[0]]
                if not hasattr(idx_tensor, "shape") or len(getattr(idx_tensor, "shape", ())) != 1:
                    passed = False
    if "aten::unbind.int" == func_name and passed and any_ttnn_tensor:
        try:
            thresh = int(os.environ.get("TT_SYMBIOTE_UNBIND_FALLBACK_THRESHOLD", "0"))
        except ValueError:
            thresh = 0
        if thresh > 0 and len(args) >= 1:
            shp = getattr(args[0], "shape", None)
            if shp is not None:
                _kw = kwargs or {}
                dim = int(_kw.get("dim", args[1] if len(args) > 1 else 0))
                dim = dim + len(shp) if dim < 0 else dim
                if dim < len(shp) and shp[dim] > thresh:
                    passed = False
                    _log_fallback_op(func_name)
    if func_name in ("aten::split.Tensor", "aten::split_with_sizes") and passed and any_ttnn_tensor:
        try:
            thresh = int(os.environ.get("TT_SYMBIOTE_SPLIT_FALLBACK_THRESHOLD", "0"))
        except ValueError:
            thresh = 0
        if thresh > 0 and len(args) >= 2:
            sections = args[1]
            if isinstance(sections, (list, tuple)):
                n_sections = len(sections)
            elif isinstance(sections, int) and sections > 0 and hasattr(args[0], "shape"):
                import math

                dim = (kwargs or {}).get("dim", 0)
                shp = args[0].shape
                dim = dim + len(shp) if dim < 0 else dim
                size = shp[dim] if dim < len(shp) else 0
                n_sections = math.ceil(size / sections) if size else 0
            else:
                n_sections = 0
            if n_sections > thresh:
                passed = False
                _log_fallback_op(func_name)
    if "aten::broadcast_tensors" == func_name and passed and any_ttnn_tensor:
        tensors = args[0] if args else ()
        if tensors and not all(isinstance(t, TorchTTNNTensor) for t in tensors):
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
    if func_name.startswith("aten::sum"):
        if any_ttnn_tensor and args[0].to_ttnn.dtype not in [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint32]:
            print("TTNN: aten::sum only supports float32, bfloat16, bfloat8_b, and uint32 dtypes.")
            passed = False
    if "aten::unsqueeze" == func_name:
        if args[0].to_ttnn.dtype not in [ttnn.float32, ttnn.bfloat16, ttnn.int32, ttnn.uint32, ttnn.uint16]:
            print("TTNN: aten::unsqueeze only supports float32, bfloat16, int32, and uint32 dtypes.")
            passed = False
    if "aten::scatter_.value" == func_name:
        if not any_ttnn_tensor or len(args) < 4:
            passed = False
        elif not isinstance(args[1], int):
            print("TTNN: aten::scatter_.value requires int dim.")
            passed = False
    if func_name in ("aten::chunk", "aten::chunk.Tensor", "aten::chunk.default"):
        if not any_ttnn_tensor or len(args) < 2:
            passed = False
        elif not isinstance(args[1], int) or (len(args) > 2 and not isinstance(args[2], int)):
            print("TTNN: aten::chunk requires chunks (int) and optional dim (int).")
            passed = False
    if "aten::contiguous" == func_name:
        if not any_ttnn_tensor or len(args) < 1:
            passed = False
    if "aten::constant_pad_nd" == func_name:
        if not any_ttnn_tensor or len(args) < 2:
            passed = False
        else:
            inp = args[0]
            rank = len(inp.shape)
            pad_list = args[1]
            if (
                not isinstance(pad_list, (list, tuple))
                or len(pad_list) % 2 != 0
                or len(pad_list) > 2 * rank
                or len(pad_list) < 2
            ):
                passed = False
            elif rank != 4:
                print("TTNN: aten::constant_pad_nd on device only supports rank 4 tensors.")
                passed = False
            else:
                for k in range(0, len(pad_list), 2):
                    if int(pad_list[k]) != 0:
                        print("TTNN: aten::constant_pad_nd on device only supports end (right) padding.")
                        passed = False
                        break
    if func_name in ("aten::native_layer_norm", "aten::layer_norm"):
        if not any_ttnn_tensor or len(args) < 5:
            passed = False
    if "aten::copy_" == func_name:
        if not any_ttnn_tensor or len(args) < 2:
            passed = False
        else:
            has_device = False
            for i in (0, 1):
                if (
                    i < len(args)
                    and isinstance(args[i], TorchTTNNTensor)
                    and getattr(args[i], "ttnn_tensor", None) is not None
                ):
                    if args[i].ttnn_tensor.is_allocated() and args[i].ttnn_tensor.device() is not None:
                        has_device = True
                        break
            if not has_device:
                passed = False
    if "aten::masked_fill.Tensor" == func_name:
        if not any_ttnn_tensor or len(args) < 3:
            passed = False
    if "aten::mse_loss" == func_name:
        if not any_ttnn_tensor or len(args) < 2:
            passed = False
    if func_name.startswith("aten::im2col") and len(args) >= 5:
        _shp = getattr(args[0], "shape", None) if args else None
        _in_block = passed and any_ttnn_tensor
        if _in_block:
            try:
                inp = args[0]
                shp = getattr(inp, "shape", None)
                if shp is not None and len(shp) == 4:
                    N, C = int(shp[0]), int(shp[1])
                    kr = args[1]
                    kH = int(kr[0]) if isinstance(kr, (list, tuple)) else int(kr)
                    kW = int(kr[1]) if isinstance(kr, (list, tuple)) else int(kr)
                    sr = args[4]
                    sH = int(sr[0]) if isinstance(sr, (list, tuple)) else int(sr)
                    sW = int(sr[1]) if isinstance(sr, (list, tuple)) else int(sr)
                    if kH > 0 and kW > 0 and sH > 0 and sW > 0:
                        H, W = int(shp[2]), int(shp[3])
                        out_h, out_w = (H // sH), (W // sW)
                        out_numel = N * C * kH * kW * out_h * out_w
                        if os.environ.get("TT_SYMBIOTE_IM2COL_DEBUG", "0") == "1":
                            print(
                                f"[TTNN im2col debug] func_name={func_name!r} any_ttnn_tensor={any_ttnn_tensor} shape={shp} out_numel={out_numel} -> CPU={out_numel > 500_000}"
                            )
                        if out_numel > 500_000:
                            passed = False
                            _log_fallback_op(func_name)
            except (TypeError, IndexError, ValueError, Exception) as e:
                if os.environ.get("TT_SYMBIOTE_IM2COL_DEBUG", "0") == "1":
                    print(f"[TTNN im2col debug] exception {type(e).__name__}: {e} -> force CPU")
                passed = False
                _log_fallback_op(func_name)
        if os.environ.get("TT_SYMBIOTE_IM2COL_DEBUG", "0") == "1":
            print(f"[TTNN im2col debug] after block: passed={passed} any_ttnn_tensor={any_ttnn_tensor} shape={_shp}")
    if func_name in _get_func_to_ttnn_compatible() and any_ttnn_tensor:
        return passed
    if func_name.startswith("aten::sum") and any_ttnn_tensor and passed:
        return True
    if func_name != "aten::_scaled_dot_product_flash_attention_for_cpu" and passed and any_ttnn_tensor:
        _log_fallback_op(func_name)
    return False


def dispatch_to_ttnn(func_name, args, kwargs):
    """Dispatch operation to TTNN handler."""
    func_to_ttnn_compatible = _get_func_to_ttnn_compatible()
    if func_name in func_to_ttnn_compatible:
        return func_to_ttnn_compatible[func_name](func_name, args, kwargs)
    if func_name.startswith("aten::sum"):
        return handle_sum(func_name, args, kwargs)
    raise KeyError(f"No TTNN handler for {func_name}")
