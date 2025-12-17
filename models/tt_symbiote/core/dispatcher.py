"""TTNN operation dispatch handlers and mapping."""

import torch

import ttnn


def handle_view(func, args, kwargs):
    """Handle view operation."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    new_shape = args[1]
    return TorchTTNNTensor(ttnn.reshape(input_tensor.to_ttnn, new_shape))


def handle_transpose(func, args, kwargs):
    """Handle transpose operation."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    dim0 = args[1]
    dim1 = args[2]
    return TorchTTNNTensor(ttnn.transpose(input_tensor.to_ttnn, dim0, dim1))


def handle_mul(func, args, kwargs):
    """Handle multiplication operation."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor1 = args[0]
    input_tensor2 = args[1]
    device = None
    deallocate_a = None
    if not isinstance(input_tensor1, TorchTTNNTensor):
        if isinstance(input_tensor1, (int, float)):
            input_tensor1 = torch.tensor(input_tensor1)
        input_tensor1 = TorchTTNNTensor(input_tensor1)
        deallocate_a = True
    else:
        if input_tensor1.ttnn_tensor is None:
            deallocate_a = True
        device = input_tensor1.to_ttnn.device()
    deallocate_b = False
    if not isinstance(input_tensor2, TorchTTNNTensor):
        if isinstance(input_tensor2, (int, float)):
            input_tensor2 = torch.tensor(input_tensor2)
        input_tensor2 = TorchTTNNTensor(input_tensor2)
        deallocate_b = True
    else:
        if input_tensor2.ttnn_tensor is None:
            deallocate_b = True
        device = input_tensor2.to_ttnn.device() if device is None else device
    assert device is not None, "At least one of the inputs must be a TTNN tensor."
    if input_tensor1.to_ttnn.device() != input_tensor2.to_ttnn.device():
        input_tensor1.ttnn_tensor = ttnn.to_device(input_tensor1.to_ttnn, device)
        input_tensor2.ttnn_tensor = ttnn.to_device(input_tensor2.to_ttnn, device)
    res = TorchTTNNTensor(ttnn.multiply(input_tensor1.to_ttnn, input_tensor2.to_ttnn))
    if deallocate_a:
        ttnn.deallocate(input_tensor1.ttnn_tensor)
    if deallocate_b:
        ttnn.deallocate(input_tensor2.ttnn_tensor)
    return res


def handle_sub(func, args, kwargs):
    """Handle subtraction operation."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor1 = args[0]
    input_tensor2 = args[1]
    device = None
    deallocate_a = False
    if not isinstance(input_tensor1, TorchTTNNTensor):
        if isinstance(input_tensor1, (int, float)):
            input_tensor1 = torch.tensor(input_tensor1)
        input_tensor1 = TorchTTNNTensor(input_tensor1)
        deallocate_a = True
    else:
        if input_tensor1.ttnn_tensor is None:
            deallocate_a = True
        device = input_tensor1.to_ttnn.device()
    deallocate_b = False
    if not isinstance(input_tensor2, TorchTTNNTensor):
        if isinstance(input_tensor2, (int, float)):
            input_tensor2 = torch.tensor(input_tensor2)
        input_tensor2 = TorchTTNNTensor(input_tensor2)
        deallocate_b = True
    else:
        if input_tensor2.ttnn_tensor is None:
            deallocate_b = True
        device = input_tensor2.to_ttnn.device() if device is None else device
    assert device is not None, "At least one of the inputs must be a TTNN tensor."
    if input_tensor1.to_ttnn.device() != input_tensor2.to_ttnn.device():
        input_tensor1.ttnn_tensor = ttnn.to_device(input_tensor1.to_ttnn, device)
        input_tensor2.ttnn_tensor = ttnn.to_device(input_tensor2.to_ttnn, device)

    res = TorchTTNNTensor(ttnn.subtract(input_tensor1.to_ttnn, input_tensor2.to_ttnn))
    if deallocate_a:
        ttnn.deallocate(input_tensor1.ttnn_tensor)
    if deallocate_b:
        ttnn.deallocate(input_tensor2.ttnn_tensor)
    return res


def handle_div(func, args, kwargs):
    """Handle division operation."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor1 = args[0]
    input_tensor2 = args[1]
    device = None
    deallocate_a = False
    if not isinstance(input_tensor1, TorchTTNNTensor):
        if isinstance(input_tensor1, (int, float)):
            input_tensor1 = torch.tensor(input_tensor1)
        input_tensor1 = TorchTTNNTensor(input_tensor1)
        deallocate_a = True
    else:
        if input_tensor1.ttnn_tensor is None:
            deallocate_a = True
        device = input_tensor1.to_ttnn.device()
    deallocate_b = False
    if not isinstance(input_tensor2, TorchTTNNTensor):
        if isinstance(input_tensor2, (int, float)):
            input_tensor2 = torch.tensor(input_tensor2)
        input_tensor2 = TorchTTNNTensor(input_tensor2)
        deallocate_b = True
    else:
        if input_tensor2.ttnn_tensor is None:
            deallocate_b = True
        device = input_tensor2.to_ttnn.device() if device is None else device
    assert device is not None, "At least one of the inputs must be a TTNN tensor."
    if input_tensor1.to_ttnn.device() != input_tensor2.to_ttnn.device():
        input_tensor1.ttnn_tensor = ttnn.to_device(input_tensor1.to_ttnn, device)
        input_tensor2.ttnn_tensor = ttnn.to_device(input_tensor2.to_ttnn, device)

    res = TorchTTNNTensor(ttnn.divide(input_tensor1.to_ttnn, input_tensor2.to_ttnn))
    if deallocate_a:
        ttnn.deallocate(input_tensor1.ttnn_tensor)
    if deallocate_b:
        ttnn.deallocate(input_tensor2.ttnn_tensor)
    return res


def handle_add(func, args, kwargs):
    """Handle addition operation."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor1 = args[0]
    input_tensor2 = args[1]
    device = None
    deallocate_a = False
    if not isinstance(input_tensor1, TorchTTNNTensor):
        if isinstance(input_tensor1, (int, float)):
            input_tensor1 = torch.tensor(input_tensor1)
        input_tensor1 = TorchTTNNTensor(input_tensor1)
        deallocate_a = True
    else:
        if input_tensor1.ttnn_tensor is None:
            deallocate_a = True
        device = input_tensor1.to_ttnn.device()
    deallocate_b = False
    if not isinstance(input_tensor2, TorchTTNNTensor):
        if isinstance(input_tensor2, (int, float)):
            input_tensor2 = torch.tensor(input_tensor2)
        input_tensor2 = TorchTTNNTensor(input_tensor2)
        deallocate_b = True
    else:
        if input_tensor2.ttnn_tensor is None:
            deallocate_b = True
        device = input_tensor2.to_ttnn.device() if device is None else device
    assert device is not None, "At least one of the inputs must be a TTNN tensor."
    if input_tensor1.to_ttnn.device() != input_tensor2.to_ttnn.device():
        input_tensor1.ttnn_tensor = ttnn.to_device(input_tensor1.to_ttnn, device)
        input_tensor2.ttnn_tensor = ttnn.to_device(input_tensor2.to_ttnn, device)

    res = TorchTTNNTensor(ttnn.add(input_tensor1.to_ttnn, input_tensor2.to_ttnn))
    if deallocate_a:
        ttnn.deallocate(input_tensor1.ttnn_tensor)
    if deallocate_b:
        ttnn.deallocate(input_tensor2.ttnn_tensor)
    return res


def handle_slice(func, args, kwargs):
    """Handle slice operation."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    input_shape = input_tensor.shape
    dim = args[1] + len(input_shape) if args[1] < 0 else args[1]
    start = [
        0 if i != dim else min(args[2] + input_shape[i] if args[2] < 0 else args[2], input_shape[i])
        for i in range(len(input_shape))
    ]
    end = [
        input_shape[i] if i != dim else min(args[3] + input_shape[i] if args[3] < 0 else args[3], input_shape[i])
        for i in range(len(input_shape))
    ]
    return TorchTTNNTensor(ttnn.slice(input_tensor.to_ttnn, start, end))


def handle_neg(func, args, kwargs):
    """Handle negation operation."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    return TorchTTNNTensor(ttnn.neg(input_tensor.to_ttnn))


def handle_cat(func, args, kwargs):
    """Handle concatenation operation."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

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
    for index, tensor in enumerate(tensors):
        if deallocate_tensors[index]:
            tensor.ttnn_tensor = ttnn.to_device(tensor.to_ttnn, device)
        if tensor.ttnn_tensor.layout != ttnn.TILE_LAYOUT:
            tensor.ttnn_tensor = ttnn.to_layout(tensor.to_ttnn, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    res = TorchTTNNTensor(ttnn.concat([tensor.to_ttnn for tensor in tensors if tensor.numel() > 0], dim))
    for index, tensor in enumerate(tensors):
        if deallocate_tensors[index]:
            ttnn.deallocate(tensor.ttnn_tensor)
    return res


def handle_unsqueeze(func, args, kwargs):
    """Handle unsqueeze operation."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    dim = args[1]
    return TorchTTNNTensor(ttnn.unsqueeze(input_tensor.to_ttnn, dim))


def handle_expand(func, args, kwargs):
    """Handle expand operation."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    output_shape = args[1]
    return TorchTTNNTensor(ttnn.expand(input_tensor.to_ttnn, output_shape))


def handle_bmm(func, args, kwargs):
    """Handle batch matrix multiplication."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor1 = args[0]
    input_tensor2 = args[1]
    device = None
    deallocate_a = None
    if not isinstance(input_tensor1, TorchTTNNTensor):
        input_tensor1 = TorchTTNNTensor(input_tensor1)
        deallocate_a = True
    else:
        if input_tensor1.ttnn_tensor is None:
            deallocate_a = True
        device = input_tensor1.to_ttnn.device()
    deallocate_b = False
    if not isinstance(input_tensor2, TorchTTNNTensor):
        input_tensor2 = TorchTTNNTensor(input_tensor2)
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
    res = TorchTTNNTensor(ttnn.matmul(ttnn_tensor1, ttnn_tensor2))
    if deallocate_a:
        ttnn.deallocate(input_tensor1.ttnn_tensor)
    if deallocate_b:
        ttnn.deallocate(input_tensor2.ttnn_tensor)
    return res


def handle_sdpa(func, args, kwargs):
    """Handle scaled dot product attention."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

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
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    dim = args[1]
    return TorchTTNNTensor(ttnn.softmax(input_tensor.to_ttnn, dim))


def handle_power(func, args, kwargs):
    """Handle power operation."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    exponent = args[1]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    return TorchTTNNTensor(ttnn.pow(input_tensor.to_ttnn, exponent))


def handle_mean(func, args, kwargs):
    """Handle mean operation."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    dim = args[1]
    keepdim = args[2] if len(args) > 2 else False
    return TorchTTNNTensor(ttnn.mean(input_tensor.to_ttnn, dim, keepdim=keepdim))


def handle_rsqrt(func, args, kwargs):
    """Handle reciprocal square root operation."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    return TorchTTNNTensor(ttnn.rsqrt(input_tensor.to_ttnn))


def handle_gelu(func, args, kwargs):
    """Handle GELU activation."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    return TorchTTNNTensor(ttnn.gelu(input_tensor.to_ttnn))


def handle_relu(func, args, kwargs):
    """Handle ReLU activation."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    return TorchTTNNTensor(ttnn.relu(input_tensor.to_ttnn))


def handle_new_zeros(func, args, kwargs):
    """Handle new_zeros operation."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

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
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    return TorchTTNNTensor(ttnn.sigmoid(input_tensor.to_ttnn))


def handle_squeeze(func, args, kwargs):
    """Handle squeeze operation."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    dim = args[1]
    return TorchTTNNTensor(ttnn.squeeze(input_tensor.to_ttnn, dim))


def handle_stack(func, args, kwargs):
    """Handle stack operation."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

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
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)
    dim = args[1]
    keepdim = args[2] if len(args) > 2 else False
    return TorchTTNNTensor(ttnn.sum(input_tensor.to_ttnn, dim, keepdim=keepdim))


def handle_ge(func, args, kwargs):
    """Handle greater equal operation."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor1 = args[0]
    input_tensor2 = args[1]
    device = None
    deallocate_a = False
    if not isinstance(input_tensor1, TorchTTNNTensor):
        if isinstance(input_tensor1, (int, float)):
            input_tensor1 = torch.tensor(input_tensor1)
        input_tensor1 = TorchTTNNTensor(input_tensor1)
        deallocate_a = True
    else:
        if input_tensor1.ttnn_tensor is None:
            deallocate_a = True
        device = input_tensor1.to_ttnn.device()
    deallocate_b = False
    if not isinstance(input_tensor2, TorchTTNNTensor):
        if isinstance(input_tensor2, (int, float)):
            input_tensor2 = torch.tensor(input_tensor2)
        input_tensor2 = TorchTTNNTensor(input_tensor2)
        deallocate_b = True
    else:
        if input_tensor2.ttnn_tensor is None:
            deallocate_b = True
        device = input_tensor2.to_ttnn.device() if device is None else device
    assert device is not None, "At least one of the inputs must be a TTNN tensor."
    if input_tensor1.to_ttnn.device() != input_tensor2.to_ttnn.device():
        input_tensor1.ttnn_tensor = ttnn.to_device(input_tensor1.to_ttnn, device)
        input_tensor2.ttnn_tensor = ttnn.to_device(input_tensor2.to_ttnn, device)

    res = TorchTTNNTensor(ttnn.ge(input_tensor1.to_ttnn, input_tensor2.to_ttnn), dtype=torch.bool)
    if deallocate_a:
        ttnn.deallocate(input_tensor1.ttnn_tensor)
    if deallocate_b:
        ttnn.deallocate(input_tensor2.ttnn_tensor)
    return res


def handle_eq(func, args, kwargs):
    """Handle equal operation."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor1 = args[0]
    input_tensor2 = args[1]
    device = None
    deallocate_a = False
    if not isinstance(input_tensor1, TorchTTNNTensor):
        if isinstance(input_tensor1, (int, float)):
            input_tensor1 = torch.tensor(input_tensor1)
        input_tensor1 = TorchTTNNTensor(input_tensor1)
        deallocate_a = True
    else:
        if input_tensor1.ttnn_tensor is None:
            deallocate_a = True
        device = input_tensor1.to_ttnn.device()
    deallocate_b = False
    if not isinstance(input_tensor2, TorchTTNNTensor):
        if isinstance(input_tensor2, (int, float)):
            input_tensor2 = torch.tensor(input_tensor2)
        input_tensor2 = TorchTTNNTensor(input_tensor2)
        deallocate_b = True
    else:
        if input_tensor2.ttnn_tensor is None:
            deallocate_b = True
        device = input_tensor2.to_ttnn.device() if device is None else device
    assert device is not None, "At least one of the inputs must be a TTNN tensor."
    if input_tensor1.to_ttnn.device() != input_tensor2.to_ttnn.device():
        input_tensor1.ttnn_tensor = ttnn.to_device(input_tensor1.to_ttnn, device)
        input_tensor2.ttnn_tensor = ttnn.to_device(input_tensor2.to_ttnn, device)

    res = TorchTTNNTensor(ttnn.eq(input_tensor1.to_ttnn, input_tensor2.to_ttnn), dtype=torch.bool)

    if deallocate_a:
        ttnn.deallocate(input_tensor1.ttnn_tensor)
    if deallocate_b:
        ttnn.deallocate(input_tensor2.ttnn_tensor)
    return res


def handle_select(func, args, kwargs):
    """Handle select operation."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

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
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

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
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, TorchTTNNTensor):
        input_tensor = TorchTTNNTensor(input_tensor)

    repeats = args[1]
    return TorchTTNNTensor(ttnn.repeat(input_tensor.to_ttnn, repeats))


def handle_where(func, args, kwargs):
    """Handle where operation."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    condition = args[0]
    deallocate_cond = False
    device = None
    if not isinstance(condition, TorchTTNNTensor):
        if isinstance(condition, (int, float)):
            condition = torch.tensor(condition)
        condition = TorchTTNNTensor(condition)
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
        input_tensor1 = TorchTTNNTensor(input_tensor1)
        deallocate_a = True
    else:
        if input_tensor1.ttnn_tensor is None:
            deallocate_a = True
        device = input_tensor1.to_ttnn.device() if device is None else device
    deallocate_b = False
    if not isinstance(input_tensor2, TorchTTNNTensor):
        if isinstance(input_tensor2, (int, float)):
            input_tensor2 = torch.tensor(input_tensor2)
        input_tensor2 = TorchTTNNTensor(input_tensor2)
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

    result = TorchTTNNTensor(ttnn.where(condition.to_ttnn, input_tensor1.to_ttnn, input_tensor2.to_ttnn))

    if deallocate_a:
        ttnn.deallocate(input_tensor1.ttnn_tensor)
    if deallocate_b:
        ttnn.deallocate(input_tensor2.ttnn_tensor)

    if deallocate_cond:
        ttnn.deallocate(condition.ttnn_tensor)

    return result


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
    "aten::add_.Tensor": handle_add,
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
    "aten::select.int": handle_select,
    "aten::bernoulli.p": handle_bernoulli_p,
    "aten::repeat": handle_repeat,
    "aten::eq.Scalar": handle_eq,
    "aten::where.self": handle_where,
}


def can_dispatch_to_ttnn(func_name: str, args=None, kwargs=None) -> bool:
    """Check if operation can be dispatched to TTNN."""
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    any_ttnn_tensor = False
    for elem in args:
        if isinstance(elem, TorchTTNNTensor) and elem.ttnn_tensor is not None and elem.ttnn_tensor.device() is not None:
            any_ttnn_tensor = True
        elif isinstance(elem, (list, tuple)):
            for sub_elem in elem:
                if (
                    isinstance(sub_elem, TorchTTNNTensor)
                    and sub_elem.ttnn_tensor is not None
                    and sub_elem.ttnn_tensor.device() is not None
                ):
                    any_ttnn_tensor = True
                    break
    if not any_ttnn_tensor:
        return False
    if "aten::slice.Tensor" == func_name:
        return isinstance(args[1], int) and isinstance(args[2], int) and isinstance(args[3], int) and len(args) == 4
    return func_name in func_to_ttnn_compatible


def dispatch_to_ttnn(func_name, args, kwargs):
    """Dispatch operation to TTNN handler."""
    return func_to_ttnn_compatible[func_name](func_name, args, kwargs)
