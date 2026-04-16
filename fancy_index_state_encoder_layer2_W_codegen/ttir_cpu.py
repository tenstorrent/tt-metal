# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pure-torch CPU implementations of TTIR ops.

CPU-hoisting lowers selected TTIR ops to run on the host CPU instead of
the TT hardware.  This improves numerical precision (host operates on
full 32-bit integers/floats) and reduces peak DRAM/L1 usage by keeping
intermediate tensors in host memory.

Each public function in this module provides a standalone torch
implementation that mirrors the semantics of its TTIR counterpart
(e.g. ``ttir_cpu.add``, ``ttir_cpu.sum``).
"""

import builtins
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dim_list_to_int(dim):
    """Convert a single-element list/tuple dim to int."""
    if isinstance(dim, (list, tuple)) and len(dim) == 1:
        return dim[0]
    return dim


def _dim_list_to_tuple(dim):
    """Ensure dim is a tuple (torch ops that accept multi-dim want tuple)."""
    if isinstance(dim, list):
        return tuple(dim)
    return dim


def _reduce_iterative(torch_fn, input_tensor, dim, keepdim):
    """Reduce one dim at a time (for ops that only accept scalar dim)."""
    dims = sorted(dim, reverse=True)
    result = input_tensor
    for d in dims:
        result = torch_fn(result, dim=d, keepdim=keepdim)
    return result


def _reduce_values(torch_fn, t, dim, keepdim):
    """Reduce op whose per-dim result is a (values, indices) namedtuple."""
    if dim is None and not keepdim:
        return torch_fn(t)
    if dim is None:
        dim = list(range(t.dim()))
    if isinstance(dim, (list, tuple)):
        return _reduce_iterative(lambda x, **kw: torch_fn(x, **kw).values, t, dim, keepdim)
    return torch_fn(t, dim=dim, keepdim=keepdim).values


def _pool_to_nchw(t, batch_size, input_h, input_w, channels):
    """Reshape NHWC pooling input to NCHW."""
    return t.reshape(batch_size, input_h, input_w, channels).permute(0, 3, 1, 2)


def _pool_padding(x, padding):
    """Normalize 4-element asymmetric padding for pooling ops.

    Returns (x, torch_padding) where x may have been explicitly padded.
    """
    if isinstance(padding, (list, tuple)) and len(padding) == 4:
        pad_t, pad_b, pad_l, pad_r = padding
        pad_val = float("-inf") if x.is_floating_point() else torch.iinfo(x.dtype).min
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode="constant", value=pad_val)
        return x, 0
    if isinstance(padding, (list, tuple)) and len(padding) == 2:
        return x, tuple(padding)
    return x, padding


# ===========================================================================
# Elementwise unary
# ===========================================================================


# NOTE: function names like `abs`, `max`, `min`, `sum` intentionally shadow
# Python builtins at module scope.  The function bodies only use torch.*
# and internal helpers already reference `builtins.*` where needed.


def abs(t, **_):
    return torch.abs(t)


def acos(t, **_):
    return torch.acos(t)


def asin(t, **_):
    return torch.asin(t)


def atan(t, **_):
    return torch.atan(t)


def bitwise_not(t, **_):
    return torch.bitwise_not(t)


def cbrt(t, **_):
    return torch.sign(t) * torch.pow(torch.abs(t), 1.0 / 3.0)


def ceil(t, **_):
    return torch.ceil(t)


def cos(t, **_):
    return torch.cos(t)


def exp(t, **_):
    return torch.exp(t)


def expm1(t, **_):
    return torch.expm1(t)


def erf(t, **_):
    return torch.erf(t)


def erfc(t, **_):
    return torch.erfc(t)


def floor(t, **_):
    return torch.floor(t)


def gelu(t, **_):
    return F.gelu(t)


def hardsigmoid(t, **_):
    return F.hardsigmoid(t)


def isfinite(t, **_):
    return torch.isfinite(t).to(t.dtype)


def log(t, **_):
    return torch.log(t)


def log1p(t, **_):
    return torch.log1p(t)


def logical_not(t, **_):
    return torch.logical_not(t).to(t.dtype)


def mish(t, **_):
    return t * torch.tanh(F.softplus(t))


def neg(t, **_):
    return torch.neg(t)


def reciprocal(t, **_):
    return torch.reciprocal(t)


def relu(t, **_):
    return F.relu(t)


def relu6(t, **_):
    return F.relu6(t)


def leaky_relu(t, negative_slope=0.01, **_):
    return F.leaky_relu(t, negative_slope=negative_slope)


def rsqrt(t, **_):
    return torch.rsqrt(t)


def sigmoid(t, **_):
    return torch.sigmoid(t)


def sign(t, **_):
    return torch.sign(t)


def silu(t, **_):
    return F.silu(t)


def sin(t, **_):
    return torch.sin(t)


def sqrt(t, **_):
    return torch.sqrt(t)


def tan(t, **_):
    return torch.tan(t)


def tanh(t, **_):
    return torch.tanh(t)


# ===========================================================================
# Elementwise binary
# ===========================================================================


def add(a, b, **_):
    return torch.add(a, b)


def subtract(a, b, **_):
    return torch.subtract(a, b)


def multiply(a, b, **_):
    return torch.multiply(a, b)


def div(a, b, rounding_mode=None, **_):
    result = torch.div(a, b, rounding_mode=rounding_mode)
    return result.to(a.dtype)


def eq(a, b, **_):
    return torch.eq(a, b).to(a.dtype)


def ne(a, b, **_):
    return torch.ne(a, b).to(a.dtype)


def gt(a, b, **_):
    return torch.gt(a, b).to(a.dtype)


def ge(a, b, **_):
    return torch.ge(a, b).to(a.dtype)


def lt(a, b, **_):
    return torch.lt(a, b).to(a.dtype)


def le(a, b, **_):
    return torch.le(a, b).to(a.dtype)


def logical_and(a, b, **_):
    return torch.logical_and(a, b).to(a.dtype)


def logical_or(a, b, **_):
    return torch.logical_or(a, b).to(a.dtype)


def logical_xor(a, b, **_):
    return torch.logical_xor(a, b).to(a.dtype)


def maximum(a, b, **_):
    return torch.maximum(a, b)


def minimum(a, b, **_):
    return torch.minimum(a, b)


def atan2(a, b, **_):
    return torch.atan2(a, b)


def remainder(a, b, **_):
    return torch.remainder(a, b)


def pow(a, b, **_):
    return torch.pow(a, b)


def bitwise_and(a, b, **_):
    return torch.bitwise_and(a, b)


def bitwise_or(a, b, **_):
    return torch.bitwise_or(a, b)


def bitwise_xor(a, b, **_):
    return torch.bitwise_xor(a, b)


def logical_left_shift(a, b, **_):
    return torch.bitwise_left_shift(a, b)


def logical_right_shift(a, b, **_):
    # torch.bitwise_right_shift is arithmetic for signed ints (sign-extends).
    # Logical right shift zero-fills. Reinterpret as unsigned via masking.
    if a.dtype == torch.int32:
        mask = torch.tensor(0xFFFFFFFF, dtype=torch.int64)
        result = ((a.to(torch.int64) & mask) >> b.to(torch.int64)) & mask
        return result.to(torch.int32)
    return torch.bitwise_right_shift(a, b)


def gelu_bw(grad, input_tensor, approximate="none", **_):
    broadcast_shape = torch.broadcast_shapes(grad.shape, input_tensor.shape)
    input_tensor = input_tensor.expand(broadcast_shape).clone().detach().requires_grad_(True)
    grad = grad.expand(broadcast_shape).detach()
    approx = approximate if isinstance(approximate, str) else "none"
    y = F.gelu(input_tensor, approximate=approx)
    y.backward(gradient=grad)
    return input_tensor.grad


# ===========================================================================
# Elementwise ternary
# ===========================================================================


def where(condition, x, y, **_):
    # tt-metal where treats condition as > 0 (positive = true).
    return torch.where(condition > 0, x, y)


def clamp_scalar(t, min_val, max_val, **_):
    return torch.clamp(t, min=min_val, max=max_val)


def clamp_tensor(t, min_tensor, max_tensor, **_):
    return torch.minimum(torch.maximum(t, min_tensor), max_tensor)


# ===========================================================================
# Reductions
# ===========================================================================


def sum(t, dim=None, keepdim=False, **_):
    if dim is None and not keepdim:
        return torch.sum(t).to(t.dtype)
    if dim is None:
        dim = tuple(range(t.dim()))
    dim = _dim_list_to_tuple(dim)
    return torch.sum(t, dim=dim, keepdim=keepdim).to(t.dtype)


def mean(t, dim=None, keepdim=False, **_):
    inp = t.float() if not t.is_floating_point() else t
    if dim is None and not keepdim:
        return torch.mean(inp)
    if dim is None:
        dim = tuple(range(inp.dim()))
    dim = _dim_list_to_tuple(dim)
    return torch.mean(inp, dim=dim, keepdim=keepdim)


def max(t, dim=None, keepdim=False, **_):
    return _reduce_values(torch.max, t, dim, keepdim)


def min(t, dim=None, keepdim=False, **_):
    return _reduce_values(torch.min, t, dim, keepdim)


def prod(t, dim=None, keepdim=False, **_):
    if dim is None and not keepdim:
        return torch.prod(t).to(t.dtype)
    if dim is None:
        dim = list(range(t.dim()))
    if isinstance(dim, (list, tuple)):
        return _reduce_iterative(torch.prod, t, dim, keepdim).to(t.dtype)
    return torch.prod(t, dim=dim, keepdim=keepdim).to(t.dtype)


def argmax(t, dim=None, keepdim=False, **_):
    if dim is None:
        if keepdim:
            idx = torch.argmax(t.flatten())
            return idx.reshape([1] * t.dim()).to(torch.int32)
        return torch.argmax(t).to(torch.int32)
    dim = _dim_list_to_int(dim)
    if isinstance(dim, (list, tuple)):
        # Multi-dim argmax: flatten target dims, argmax over flattened.
        dims = sorted(dim)
        perm = [i for i in range(t.dim()) if i not in dims] + dims
        x = t.permute(perm)
        x = x.reshape(list(x.shape[: -len(dims)]) + [-1])
        result = torch.argmax(x, dim=-1, keepdim=keepdim)
        if keepdim:
            shape = list(t.shape)
            for d in dims:
                shape[d] = 1
            result = result.reshape(shape)
        return result.to(torch.int32)
    return torch.argmax(t, dim=dim, keepdim=keepdim).to(torch.int32)


def reduce_or(t, dim=None, keepdim=False, **_):
    if dim is None:
        return torch.any(t.bool()).to(t.dtype)
    return torch.any(t.bool(), dim=dim, keepdim=keepdim).to(t.dtype)


def cumsum(t, dim=0, **_):
    return torch.cumsum(t, dim=dim).to(t.dtype)


# ===========================================================================
# Data movement / layout
# ===========================================================================


def reshape(t, shape, **_):
    return torch.reshape(t, shape)


def permute(t, dims, **_):
    return t.permute(dims)


def repeat(t, repeats, **_):
    return t.repeat(*repeats)


def concat(tensors, dim=0, **_):
    return torch.cat(tensors, dim=dim)


# TTIR emits flat [dim0_low, dim0_high, dim1_low, dim1_high, ...].
# PyTorch F.pad expects reversed order: (last_low, last_high, ..., first_low, first_high).
def pad(t, padding, value=0.0, **_):
    if isinstance(padding, (list, tuple)):
        pairs = [(padding[i], padding[i + 1]) for i in range(0, len(padding), 2)]
        flat = []
        for lo, hi in reversed(pairs):
            flat.extend([lo, hi])
        padding = tuple(flat)
    return F.pad(t, padding, value=value)


def squeeze(t, dim, **_):
    return torch.squeeze(t, dim=dim)


def unsqueeze(t, dim, **_):
    return torch.unsqueeze(t, dim=dim)


def transpose(t, dim0, dim1, **_):
    return torch.transpose(t, dim0, dim1)


def broadcast(t, shape, **_):
    return t.expand(shape)


def slice_static(t, begins, ends, step, **_):
    slices = tuple(builtins.slice(b, e, s) for b, e, s in zip(begins, ends, step))
    return t[slices]


def typecast(t, dtype=None, **_):
    if dtype is not None:
        return t.to(dtype)
    return t


# ===========================================================================
# Matmul
# ===========================================================================


def matmul(a, b, **_):
    return torch.matmul(a, b)


def linear(a, b, bias=None, transpose_a=False, transpose_b=False, **_):
    if transpose_a:
        a = a.transpose(-2, -1)
    if transpose_b:
        b = b.transpose(-2, -1)
    result = torch.matmul(a, b)
    if bias is not None:
        result = result + bias
    return result


def dot_general(
    lhs,
    rhs,
    batch_dims_lhs=(),
    contract_dims_lhs=(),
    batch_dims_rhs=(),
    contract_dims_rhs=(),
    **_,
):
    """Generalized dot product (StableHLO semantics)."""
    lhs_batch = list(batch_dims_lhs)
    lhs_contract = list(contract_dims_lhs)
    rhs_batch = list(batch_dims_rhs)
    rhs_contract = list(contract_dims_rhs)

    lhs_rank = lhs.ndim
    rhs_rank = rhs.ndim

    lhs_free = [i for i in range(lhs_rank) if i not in lhs_batch and i not in lhs_contract]
    rhs_free = [i for i in range(rhs_rank) if i not in rhs_batch and i not in rhs_contract]

    lhs_perm = lhs_batch + lhs_free + lhs_contract
    rhs_perm = rhs_batch + rhs_contract + rhs_free
    lhs_t = lhs.permute(lhs_perm)
    rhs_t = rhs.permute(rhs_perm)

    batch_size = 1
    for d in lhs_batch:
        batch_size *= lhs.shape[d]
    lhs_free_size = 1
    for d in lhs_free:
        lhs_free_size *= lhs.shape[d]
    contract_size = 1
    for d in lhs_contract:
        contract_size *= lhs.shape[d]
    rhs_free_size = 1
    for d in rhs_free:
        rhs_free_size *= rhs.shape[d]

    lhs_3d = lhs_t.reshape(batch_size, lhs_free_size, contract_size)
    rhs_3d = rhs_t.reshape(batch_size, contract_size, rhs_free_size)

    result_3d = torch.bmm(lhs_3d, rhs_3d)

    out_shape = [lhs.shape[d] for d in lhs_batch] + [lhs.shape[d] for d in lhs_free] + [rhs.shape[d] for d in rhs_free]
    return result_3d.reshape(out_shape)


# ===========================================================================
# Normalization
# ===========================================================================


def softmax(t, dim=-1, **_):
    return F.softmax(t, dim=dim)


def layer_norm(t, epsilon=1e-5, weight=None, bias=None, **_):
    if weight is not None:
        normalized_shape = weight.shape
    else:
        normalized_shape = (t.shape[-1],)
    return F.layer_norm(t, normalized_shape, weight=weight, bias=bias, eps=epsilon)


# ===========================================================================
# Embedding
# ===========================================================================


def embedding(indices, weight, **_):
    indices = indices.long()
    while weight.dim() > 2 and weight.shape[0] == 1:
        weight = weight.squeeze(0)
    return F.embedding(indices, weight)


# ===========================================================================
# Pooling (NHWC I/O)
# ===========================================================================


def _max_pool2d_core(
    t,
    batch_size,
    input_h,
    input_w,
    channels,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
    return_indices,
):
    """Shared implementation for max_pool2d and max_pool2d_with_indices."""
    x = _pool_to_nchw(t, batch_size, input_h, input_w, channels)
    x, torch_padding = _pool_padding(x, padding)
    values, indices = F.max_pool2d(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=torch_padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=True,
    )
    values = values.permute(0, 2, 3, 1)  # NCHW -> NHWC
    if not return_indices:
        return values
    return values, indices.permute(0, 2, 3, 1).to(torch.int32)


def max_pool2d(
    t,
    batch_size,
    input_h,
    input_w,
    channels,
    kernel_size,
    stride,
    padding,
    dilation=(1, 1),
    ceil_mode=False,
    **_,
):
    return _max_pool2d_core(
        t,
        batch_size,
        input_h,
        input_w,
        channels,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        return_indices=False,
    )


def max_pool2d_with_indices(
    t,
    batch_size,
    input_h,
    input_w,
    channels,
    kernel_size,
    stride,
    padding,
    dilation=(1, 1),
    ceil_mode=False,
    **_,
):
    return _max_pool2d_core(
        t,
        batch_size,
        input_h,
        input_w,
        channels,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        return_indices=True,
    )


def avg_pool2d(
    t,
    batch_size,
    input_h,
    input_w,
    channels,
    kernel_size,
    stride,
    padding,
    ceil_mode=False,
    count_include_pad=True,
    **_,
):
    x = _pool_to_nchw(t, batch_size, input_h, input_w, channels)

    # Resolve padding to (pad_t, pad_b, pad_l, pad_r) or a torch-compatible
    # symmetric tuple.  PyTorch enforces padding <= kernel_size // 2, so
    # large symmetric padding must be applied manually via F.pad.
    ks = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)
    if isinstance(padding, (list, tuple)) and len(padding) == 4:
        pad_t, pad_b, pad_l, pad_r = padding
    elif isinstance(padding, (list, tuple)) and len(padding) == 2:
        pad_t, pad_b, pad_l, pad_r = padding[0], padding[0], padding[1], padding[1]
    else:
        pad_t = pad_b = pad_l = pad_r = padding

    symmetric = pad_t == pad_b and pad_l == pad_r
    exceeds_limit = pad_t > ks[0] // 2 or pad_l > ks[1] // 2

    if symmetric and not exceeds_limit:
        torch_padding = (pad_t, pad_l)
    else:
        # Must apply padding manually.
        if not count_include_pad and not symmetric:
            # Asymmetric + count_include_pad=False: compensate for padded zeros.
            ones = torch.ones_like(x)
            ones_padded = F.pad(ones, (pad_l, pad_r, pad_t, pad_b))
            x_padded = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
            sum_pool = F.avg_pool2d(x_padded, ks, tuple(stride), 0, ceil_mode, True)
            cnt_pool = F.avg_pool2d(ones_padded, ks, tuple(stride), 0, ceil_mode, True)
            result = sum_pool / cnt_pool.clamp(min=1e-8)
            return result.permute(0, 2, 3, 1)
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        torch_padding = 0
    result = F.avg_pool2d(
        x,
        kernel_size=tuple(kernel_size),
        stride=tuple(stride),
        padding=torch_padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
    )
    return result.permute(0, 2, 3, 1)  # NCHW -> NHWC


def global_avg_pool2d(input_tensor, **_):
    # Input is NHWC.
    t = input_tensor.permute(0, 3, 1, 2)
    t = F.adaptive_avg_pool2d(t, output_size=(1, 1))
    return t.permute(0, 2, 3, 1)


# ===========================================================================
# Convolution
# ===========================================================================


def conv2d(
    input,
    weight,
    bias=None,
    stride=(1, 1),
    padding=(0, 0, 0, 0),
    dilation=(1, 1),
    groups=1,
    batch_dim=0,
    height_dim=1,
    width_dim=2,
    channel_dim=3,
    **_,
):
    """2D convolution supporting arbitrary NHWC/NCHW layouts."""
    src_order = [batch_dim, channel_dim, height_dim, width_dim]
    inv_perm = [0] * 4
    for dst, src in enumerate(src_order):
        inv_perm[src] = dst
    x = input.permute(src_order)  # -> NCHW
    w = weight

    if isinstance(padding, (list, tuple)) and len(padding) == 4:
        pad_t, pad_l, pad_b, pad_r = padding
        if pad_t != pad_b or pad_l != pad_r:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
            torch_padding = 0
        else:
            torch_padding = (pad_t, pad_l)
    elif isinstance(padding, (list, tuple)) and len(padding) == 2:
        torch_padding = tuple(padding)
    else:
        torch_padding = padding

    b = bias.squeeze() if bias is not None else None
    result = F.conv2d(
        x,
        w,
        bias=b,
        stride=tuple(stride),
        padding=torch_padding,
        dilation=tuple(dilation),
        groups=groups,
    )
    return result.permute(inv_perm)


# ===========================================================================
# Attention
# ===========================================================================


def split_query_key_value_and_split_heads(
    input_tensor,
    kv_input_tensor=None,
    num_heads=1,
    num_kv_heads=None,
    transpose_key=False,
    **_,
):
    """Split fused QKV tensor and reshape for multi-head attention."""
    if num_kv_heads is None:
        num_kv_heads = num_heads

    batch, seq = input_tensor.shape[0], input_tensor.shape[1]
    head_size = (
        input_tensor.shape[-1] // (num_heads + 2 * num_kv_heads)
        if kv_input_tensor is None
        else input_tensor.shape[-1] // num_heads
    )

    if kv_input_tensor is None:
        # MHA: input is [batch, seq, (num_heads + 2*num_kv_heads) * head_size]
        q_size = num_heads * head_size
        kv_size = num_kv_heads * head_size
        q = input_tensor[:, :, :q_size]
        k = input_tensor[:, :, q_size : q_size + kv_size]
        v = input_tensor[:, :, q_size + kv_size :]
    else:
        # GQA: q from input_tensor, k/v from kv_input_tensor
        q = input_tensor
        kv_size = num_kv_heads * head_size
        k = kv_input_tensor[:, :, :kv_size]
        v = kv_input_tensor[:, :, kv_size:]

    q = q.reshape(batch, seq, num_heads, head_size).permute(0, 2, 1, 3)
    k = k.reshape(batch, seq, num_kv_heads, head_size).permute(0, 2, 1, 3)
    v = v.reshape(batch, seq, num_kv_heads, head_size).permute(0, 2, 1, 3)

    if transpose_key:
        k = k.transpose(-2, -1)

    return q, k, v


def concatenate_heads(t, **_):
    # Input: [batch_size, num_heads, sequence_size, head_size]
    # Output: [batch_size, sequence_size, num_heads * head_size]
    batch, num_heads, seq, head_size = t.shape
    return t.permute(0, 2, 1, 3).reshape(batch, seq, num_heads * head_size)


# ===========================================================================
# Creation ops
# ===========================================================================


def zeros(shape=None, dtype=None, **_):
    return torch.zeros(shape, dtype=dtype)


def ones(shape=None, dtype=None, **_):
    return torch.ones(shape, dtype=dtype)


def full(shape=None, fill_value=0, dtype=None, **_):
    return torch.full(shape, fill_value, dtype=dtype)


def empty(shape=None, dtype=None, **_):
    return torch.empty(shape, dtype=dtype)


def arange(start, end, step, arange_dimension=0, shape=None, dtype=None, **_):
    seq = torch.arange(start, end, step, dtype=dtype)
    if shape is not None and len(shape) > 1:
        view_shape = [1] * len(shape)
        view_shape[arange_dimension] = -1
        return seq.view(view_shape).expand(shape).contiguous()
    return seq


def constant(shape=None, dtype=None, fill_value=None, data=None, **_):
    if data is not None:
        return torch.tensor(data, dtype=dtype).reshape(shape)
    return torch.full(shape, fill_value if fill_value is not None else 0, dtype=dtype)
