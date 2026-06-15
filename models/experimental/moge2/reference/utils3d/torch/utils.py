from typing import *
from numbers import Number
from itertools import chain
from numbers import Integral

import torch
from torch import Tensor
import torch.nn.functional as F

from .helpers import batched
from ..helpers import no_warnings


__all__ = [
    'sliding_window',
    'masked_min',
    'masked_max',
    'lookup',
    'lookup_get',
    'lookup_set',
    'csr_matrix_from_dense_indices',
    'csr_eliminate_zeros',
    'group',
    'lexsort',
    'index_reduce',
    'index_reduce_',
    'scatter_argmax',
    'scatter_argmin',
    'reverse_permutation',
    'large_multinomial',
    'matrix_trace',
    'vector_outer'
]


def sliding_window(
    x: Tensor, 
    window_size: Union[int, Tuple[int, ...]], 
    stride: Optional[Union[int, Tuple[int, ...]]] = None, 
    dilation: Optional[Union[int, Tuple[int, ...]]] = None,
    pad_size: Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]] = None, 
    pad_mode: str = 'constant',
    pad_value: Number = 0,
    dim: Tuple[int, ...] = None
) -> Tensor:
    """
    Get a sliding window of the input array.
    This function is a wrapper of `torch.nn.functional.unfold` with additional support for padding and stride.

    ## Parameters
    - `x` (Tensor): Input tensor.
    - `window_size` (int or Tuple[int,...]): Size of the sliding window. If int
        is provided, the same size is used for all specified axes.
    - `stride` (Optional[Tuple[int,...]]): Stride between the sliding windows. If None,
        no stride is applied. If int is provided, the same stride is used for all specified axes.
    - `dilation` (Optional[Tuple[int,...]]): Dilation in each sliding window. If None,
        no dilation is applied. If int is provided, the same dilation is used for all specified axes.
    - `pad_size` (Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]]): Size of padding to apply before sliding window.
        Corresponding to `axis`.
        - General format is `((before_1, after_1), (before_2, after_2), ...)`.
        - Shortcut formats: 
            - `int` -> same padding before and after for all axes;
            - `(int, int)` -> same padding before and after for each axis;
            - `((int,), (int,) ...)` -> specify padding for each axis, same before and after.
    - `pad_mode` (str): Padding mode to use. Refer to `torch.nn.functional.pad` for more details.
    - `pad_value` (Union[int, float]): Value to use for constant padding. Only used
        when `pad_mode` is 'constant'.
    - `axis` (Optional[Tuple[int,...]]): Axes to apply the sliding window. If None, all axes are used.

    ## Returns
    - (Tensor): Sliding window of the input array. 
        - If no padding, the output is a view of the input array with zero copy.
        - Otherwise, the output is no longer a view but a copy of the padded array.
    """
    if dim is None:
        dim = tuple(range(x.ndim))
    if isinstance(dim, Integral):
        dim = (dim,)
    dim = [dim[i] % x.ndim for i in range(len(dim))]
    if isinstance(window_size, Integral):
        window_size = (window_size,) * len(dim)
    if stride is None:
        stride = (1,) * len(dim)
    elif isinstance(stride, Integral):
        stride = (stride,) * len(dim)
    if dilation is None:
        dilation = (1,) * len(dim)
    elif isinstance(dilation, Integral):
        dilation = (dilation,) * len(dim)
    assert len(window_size) == len(stride) == len(dim)

    # Pad the input array if needed
    if pad_size is not None:
        if isinstance(pad_size, Integral):
            pad_size = ((pad_size, pad_size),) * len(dim)
        elif isinstance(pad_size, tuple) and len(pad_size) == 2 and all(isinstance(p, Integral) for p in pad_size):
            pad_size = (pad_size,) * len(dim)
        elif isinstance(pad_size, tuple) and all(isinstance(p, tuple) and 1 <= len(p) <= 2 for p in pad_size):
            if len(pad_size) == 1:
                pad_size = pad_size * len(dim)
            else:
                assert len(pad_size) == len(dim), f"pad_size {pad_size} must match the number of axes {len(dim)}"
            pad_size = tuple(p * 2 if len(p) == 1 else p for p in pad_size)
        else:
            raise ValueError(f"Invalid pad_size {pad_size}")
        full_pad = [(0, 0) if i not in dim else pad_size[dim.index(i)] for i in range(x.ndim)]
        full_pad = tuple(chain(*reversed(full_pad)))
        x = F.pad(x, full_pad, mode=pad_mode, value=pad_value)
    
    for i in range(len(window_size)):
        x = x.unfold(dim[i], (window_size[i] - 1) * dilation[i] + 1, stride[i])[..., ::dilation[i]]
    return x


def masked_min(input: Tensor, mask: torch.BoolTensor, dim: int = None, keepdim: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Similar to torch.min, but with mask
    """
    if dim is None:
        return torch.where(mask, input, torch.tensor(torch.inf, dtype=input.dtype, device=input.device)).min()
    else:
        return torch.where(mask, input, torch.tensor(torch.inf, dtype=input.dtype, device=input.device)).min(dim=dim, keepdim=keepdim)


def masked_max(input: Tensor, mask: torch.BoolTensor, dim: int = None, keepdim: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Similar to torch.max, but with mask
    """
    if dim is None:
        return torch.where(mask, input, torch.tensor(-torch.inf, dtype=input.dtype, device=input.device)).max()
    else:
        return torch.where(mask, input, torch.tensor(-torch.inf, dtype=input.dtype, device=input.device)).max(dim=dim, keepdim=keepdim)
    

def _lookup_pytorch(key: Tensor, query: Tensor) -> torch.LongTensor:
    """Look up `query` in `key` like a dictionary. Useful for COO indexing.

    Parameters
    ----
    - `key` (Tensor): shape `(K, *key_shape)`, the array to search in
    - `query` (Tensor): shape `(..., *key_shape)`, the array to search for. `...` represents any number of batch dimensions.

    Returns
    ----
    - `indices` (Tensor): shape `(...,)` shape `(...,)` indices in `key` for each `query`. If a query is not found in key, the corresponding index will be -1.

    Notes
    ----
    `O((Q + K) * log(Q + K))` complexity, where `Q` is the number of queries and `K` is the number of keys.
    """
    num_keys, *key_shape = key.shape
    query_batch_shape = query.shape[:query.ndim - key.ndim + 1]

    unique, inverse = torch.unique(
        torch.cat([key, query.reshape(-1, *key_shape)], dim=0),
        dim=0,
        return_inverse=True
    )
    index = torch.full((unique.shape[0],), -1, dtype=torch.long, device=key.device)
    index.scatter_(0, inverse[:num_keys], torch.arange(num_keys, device=key.device))
    result = index.index_select(0, inverse[num_keys:]).reshape(query_batch_shape)
    return torch.where(result < num_keys, result, -1)


def lookup(key: Tensor, query: Tensor) -> torch.LongTensor:
    """Look up `query` in `key` like a dictionary. Useful for COO indexing.

    Parameters
    ----
    - `key` (Tensor): shape `(K, *key_shape)`, the array to search in
    - `query` (Tensor): shape `(..., *key_shape)`, the array to search for. `...` represents any number of batch dimensions.

    Returns
    ----
    - `indices` (Tensor): shape `(...,)` shape `(...,)` indices in `key` for each `query`. If a query is not found in key, the corresponding index will be -1.

    Notes
    ----
    - If using pytorch implementation (based on `torch.unique`), the complexity is `O((Q + K) * log(Q + K))` where `Q` is the number of queries and `K` is the number of keys.
    - If using triton implementation (based on hashmap), the average complexity `O(Q + K)`. Much faster for large `Q` and `K`.
    """
    device_type = key.device.type
    if device_type == 'cpu':
        # Use PyTorch implementation
        return _lookup_pytorch(key, query)
    elif device_type == 'cuda':
        # Use triton implementation
        from ._triton.hashmap import hashmap_build_lookup_triton
        return hashmap_build_lookup_triton(key, query)


def lookup_get(key: Tensor, value: Tensor, get_key: Tensor, default_value: Union[Number, Tensor] = 0) -> Tensor:
    """Dictionary-like get for arrays

    ## Parameters
    - `key` (Tensor): shape `(N, *key_shape)`, the key array of the dictionary to get from
    - `value` (Tensor): shape `(N, *value_shape)`, the value array of the dictionary to get from
    - `get_key` (Tensor): shape `(M, *key_shape)`, the key array to get for
    - `default_value` (Union[Number, Tensor]): value to return if a key in `get_key` is not found in `key`. A scalar or tensor broadcastable to shape `(..., *value_shape)`

    ## Returns
        `get_value` (Tensor): shape `(M, *value_shape)`, result values corresponding to `get_key`
    """
    indices = lookup(key, get_key)
    if key.shape[0] == 0:
        return torch.broadcast_to(
            torch.as_tensor(default_value, dtype=value.dtype, device=value.device), 
            get_key.shape[:get_key.ndim - key.ndim + 1] + value.shape[1:]
        )
    return torch.where(
        (indices >= 0)[(slice(None), *((None,) * (value.ndim - 1)))], 
        value[indices.clip(0, key.shape[0] - 1)], 
        default_value
    )


def lookup_set(key: Tensor, value: Tensor, set_key: Tensor, set_value: Tensor, append: bool = False, inplace: bool = False) -> Tuple[Tensor, Tensor]:
    """Dictionary-like set for arrays.

    ## Parameters
    - `key` (Tensor): shape `(N, *key_shape)`, the key array of the dictionary to set
    - `value` (Tensor): shape `(N, *value_shape)`, the value array of the dictionary to set
    - `set_key` (Tensor): shape `(M, *key_shape)`, the key array to set for
    - `set_value` (Tensor): shape `(M, *value_shape)`, the value array to set as
    - `append` (bool): If True, append the (key, value) pairs in (set_key, set_value) that are not in (key, value) to the result.
    - `inplace` (bool): If True, modify the input `value` array

    ## Returns
    - `result_key` (Tensor): shape `(N_new, *value_shape)`. N_new = N + number of new keys added if append is True, else N.
    - `result_value (Tensor): shape `(N_new, *value_shape)` 
    """
    set_indices = lookup(key, set_key)
    if inplace:
        assert append is False, "Cannot append when inplace is True"
    else:
        value = value.clone()
    hit = torch.where(set_indices >= 0)
    value[set_indices[hit]] = set_value[hit]
    if append:
        missing = torch.where(set_indices < 0)
        key = torch.cat([key, set_key[missing]], axis=0)
        value = torch.cat([value, set_value[missing]], axis=0)
    return key, value


def csr_matrix_from_dense_indices(indices: Tensor, n_cols: int) -> Tensor:
    """Convert a regular indices array to a sparse CSR adjacency matrix format

    ## Parameters
        - `indices` (Tensor): shape (N, M) dense tensor. Each one in `N` has `M` connections.
        - `values` (Tensor): shape (N, M) values of the connections
        - `n_cols` (int): total number of columns in the adjacency matrix

    ## Returns
        Tensor: shape `(N, n_cols)` sparse CSR adjacency matrix
    """
    return torch.sparse_csr_tensor(
        crow_indices=torch.arange(0, indices.numel() + 1, indices.shape[1], device=indices.device),
        col_indices=indices.view(-1),
        values=torch.ones_like(indices, dtype=torch.bool).view(-1),
        size=(indices.shape[0], n_cols)
    )


def csr_eliminate_zeros(input: Tensor):
    """Remove zero elements from a sparse CSR tensor.
    """
    nonzero = input.values() != 0
    nonzero_element_indices = nonzero.nonzero(as_tuple=False).flatten()
    row_nonzero_count = torch.sparse_csr_tensor(
        input.crow_indices(), 
        input.col_indices(), 
        nonzero, 
        input.size()
    ).long().sum(dim=-1, keepdim=True).to_dense().flatten()
    crow_indices = torch.cat([torch.tensor([0], device=input.device), torch.cumsum(row_nonzero_count, dim=0)])
    col_indices = input.col_indices()[nonzero_element_indices]
    values = input.values()[nonzero_element_indices]
    return torch.sparse_csr_tensor(crow_indices, col_indices, values, input.size())

def csr_roll_col_indices(input: Tensor, shift: int):
    """Roll the order of column indices of a sparse CSR tensor.
    The result is mathematically equivalent to the original matrix, but with a different column order.

    ## Parameters
        - `input` (Tensor): shape `(N, M)`, sparse CSR tensor
        - `shift` (int): number of positions to shift the column indices

    ## Returns
        Tensor: shape `(N, M)` sparse CSR tensor with rolled column indices
    """
    lengths = input.crow_indices()[1:] - input.crow_indices()[:-1]
    start = input.crow_indices()[:-1].repeat_interleave(lengths)
    elem_indices = start + (torch.arange(input.col_indices().shape[0], dtype=input.col_indices().dtype) - start - shift) % lengths.repeat_interleave(lengths)
    col_indices = input.col_indices().gather(0, elem_indices)
    values = input.values().gather(0, elem_indices)
    return torch.sparse_csr_tensor(input.crow_indices(), col_indices, values, input.size())


def group(labels: Tensor, data: Optional[Tensor] = None) -> List[Tuple[Tensor, Tensor]]:
    """
    Split the data into groups based on the provided labels.

    ## Parameters
    - `labels` (Tensor): shape `(N, *label_dims)` array of labels for each data point. Labels can be multi-dimensional.
    - `data` (Tensor, optional): shape `(N, *data_dims)` dense tensor. Each one in `N` has `D` features.
        If None, return the indices in each group instead.

    ## Returns
    - `groups` (List[Tuple[Tensor, Tensor]]): List of each group, a tuple of `(label, data_in_group)`.
        - `label` (Tensor): shape (*label_dims,) the label of the group.
        - `data_in_group` (Tensor): shape (M, *data_dims) the data points in the group.
        If `data` is None, `data_in_group` will be the indices of the data points in the original array.
    """
    group_labels, inv, counts = torch.unique(labels, return_inverse=True, return_counts=True, dim=0)
    if data is None:
        data = torch.arange(labels.shape[0], device=labels.device)
    data_groups = torch.split(data[torch.argsort(inv)], counts.tolist())
    return list(zip(group_labels, data_groups))


def lexsort(keys: Union[Sequence[Tensor], Tensor], dim: int = -1) -> Tensor:
    """Perform lexicographical sort on multiple keys. Like `numpy.lexsort`. 
    
    Given multiple sorting keys, lexsort returns an array of integer indices that describes the sort order by multiple keys. 
    The last key in the sequence is used for the primary sort order, ties are broken by the second-to-last key, and so on.

    Parameters
    ----
    - `keys`: (Sequence[Tensor]) sequence of Tensors to sort by, or a single Tensor with shape `(num_keys, ...)`.
    - `dim`: (int) the dimension to sort along. Note that if `keys` is a single Tensor, `dim=0` refers to the second dimension of `keys`.

    Returns
    ----
    - `indices`: (Tensor) the indices that would sort the keys lexicographically along the specified dimension.

    Notes
    -----
    Sorting is always stable.
    """
    if isinstance(keys, Tensor):
        keys = torch.unbind(keys, dim=0)
    else:
        torch.broadcast_shapes(*[key.shape for key in keys])

    assert len(keys) > 0, "At least one key is required for lexsort"
    
    dim = dim % keys[0].ndim
    for i, key in enumerate(keys):
        if i == 0:
            sorted_indices = torch.argsort(key, dim=dim, stable=True)
        else:
            key = torch.take_along_dim(key, sorted_indices, dim=dim)
            sorted_indices = torch.take_along_dim(sorted_indices, torch.argsort(key, dim=dim, stable=True), dim=dim)
    
    return sorted_indices


def index_reduce(input: Tensor, indices: Union[Tuple[Tensor], List[Tensor]], values: Tensor, reduce: Literal['amin', 'amax', 'sum', 'prod', 'mean'], include_self: bool = True) -> Tensor:
    """Put values into the input tensor at the specified indices (like `index_put`), with reduction support.
    Behaves like `numpy.ufunc.at`.

    Parameters
    ----
    - `input`: (Tensor) the input tensor to modify.
    - `indices`: (Tensor) the indices at which to put the values.
    - `values`: (Tensor) the values to put into the input tensor.
    - `reduce`: (str) the reduction method to use when multiple values are put at the same index. Options are 'amin', 'amax', 'sum', 'prod', and 'mean'.

    Returns
    ----
    - (Tensor) the modified tensor after putting the values.
    """
    flat_idx = (torch.stack(indices, dim=1) * torch.tensor(input.stride(), dtype=torch.long, device=input.device)).sum(dim=1)
    output = input.reshape(-1).scatter_reduce(0, flat_idx, values.view(-1), reduce=reduce, include_self=include_self)
    output = output.reshape(input.shape)
    return output


def index_reduce_(input: Tensor, indices: Union[Tuple[Tensor], List[Tensor]], values: Tensor, reduce: Literal['amin', 'amax', 'sum', 'prod', 'mean'], include_self: bool = True) -> Tensor:
    """In-place put values into the input tensor at the specified indices (like `index_put_`), with reduction support.
    Behaves like `numpy.ufunc.at`.

    Parameters
    ----
    - `input`: (Tensor) the input tensor to modify.
    - `indices`: (Tensor) the indices at which to put the values.
    - `values`: (Tensor) the values to put into the input tensor.
    - `reduce`: (str) the reduction method to use when multiple values are put at the same index. Options are 'amin', 'amax', 'sum', 'prod', and 'mean'.

    Returns
    ----
    - (Tensor) the modified tensor after putting the values.
    """
    assert input.is_contiguous(), "Input tensor must be contiguous for in-place index_reduce_"
    flat_idx = (torch.stack(indices, dim=1) * torch.tensor(input.stride(), dtype=torch.long, device=input.device)).sum(dim=1)
    input.view(-1).scatter_reduce_(0, flat_idx, values.view(-1), reduce=reduce, include_self=include_self)
    return input


def scatter_argmin(input: Tensor, dim: int, index: Tensor, src: Tensor, include_self: bool = True) -> Tensor:
    """Scatter src into input at index along dim with min reduction. Return the indices of the winners in src.
    
    Parameters
    ----
    - `input`: (Tensor) the input tensor to scatter into.
    - `dim`: (int) the dimension along which to index.
    - `index`: (LongTensor) the indices at which to scatter.
    - `src`: (Tensor) the source tensor to scatter from.
    - `include_self`: (bool) whether to include the original values in `input` when computing the min.

    Returns
    ----
    - `argmin`: (LongTensor) shape same as `input`, the indices of the min values in `src`.
    
    Notes
    ----
    - If multiple values in `src` are equal to the min value at a position, the one with the smallest index in `src` will be chosen.
    - If none of src was scattered to a position (i.e., not presented, or the min value is from the original input), the index will be -1.
    """
    dim = dim % input.ndim
    min_values = input.scatter_reduce(dim=dim, index=index, src=src, reduce='amin', include_self=include_self)
    min_where_in_src = torch.where(src == torch.gather(min_values, dim=dim, index=index))
    min_indices = torch.full_like(min_values, -1, dtype=torch.long)
    index_reduce_(min_indices, (*min_where_in_src[:dim], index[min_where_in_src[dim]], *min_where_in_src[dim + 1:]), include_self=False)
    return min_indices


def scatter_argmax(input: Tensor, dim: int, index: Tensor, src: Tensor, include_self: bool = True) -> Tensor:
    """Scatter src into input at index along dim with min reduction. Return the indices of the winners in src.
    
    Parameters
    ----
    - `input`: (Tensor) the input tensor to scatter into.
    - `dim`: (int) the dimension along which to index.
    - `index`: (LongTensor) the indices at which to scatter.
    - `src`: (Tensor) the source tensor to scatter from.
    - `include_self`: (bool) whether to include the original values in `input` when computing the min.

    Returns
    ----
    - `argmin`: (LongTensor) shape same as `input`, the indices of the min values in `src`.
    
    Notes
    ----
    - If multiple values in `src` are equal to the min value at a position, the one with the smallest index in `src` will be chosen.
    - If none of src was scattered to a position (i.e., not presented, or the min value is from the original input), the index will be -1.
    """
    dim = dim % input.ndim
    max_values = input.scatter_reduce(dim=dim, index=index, src=src, reduce='amax', include_self=include_self)
    max_where_in_src = torch.where(src == torch.gather(max_values, dim=dim, index=index))
    max_indices = torch.full_like(max_values, -1, dtype=torch.long)
    index_reduce_(max_indices, (*max_where_in_src[:dim], index[max_where_in_src[dim]], *max_where_in_src[dim + 1:]), include_self=False)
    return max_indices


def reverse_permutation(perm: Tensor, dim: int = 0) -> Tensor:
    """Reverse a permutation tensor along a specified dimension. 
    Parameters
    ----
    - `perm`: (LongTensor) the permutation tensor to reverse.
    - `dim`: (int) the dimension of permutation indices. Other dimensions are treated as batch dimensions.

    Returns
    ----
    - (LongTensor) the reversed permutation tensor, such that `reversed_perm[perm] == torch.arange(perm.shape[dim])`

    Notes
    -----
    Equivalent to `torch.argsort(perm, dim=dim)`, but more efficient.
    """
    dim = dim % perm.ndim
    reversed_perm = torch.empty_like(perm)
    indices = torch.arange(perm.shape[dim], device=perm.device)[(None,) * dim + (slice(None),) + (None,) * (perm.ndim - dim - 1)]
    reversed_perm.scatter_(dim, perm, indices.expand_as(perm))
    return reversed_perm


@torch.no_grad()
def large_multinomial(weights: Tensor, num_samples: int, replacement: bool = False) -> Tensor:
    weights = weights.double()
    weights = weights / weights.sum()

    if replacement:
        cum_weights = torch.cumsum(weights, dim=0)
        rand = torch.rand(num_samples, dtype=torch.float64, device=weights.device)
        indices = torch.searchsorted(cum_weights, rand)
    else:
        scores = weights.log() - torch.empty_like(weights).exponential_().log()
        indices = torch.topk(scores, num_samples).indices
    return indices


def matrix_trace(input: Tensor, dim1: int = -2, dim2: int = -1) -> Tensor:
    """Compute the trace of a batch of matrices"""
    return torch.diagonal(input, dim1=dim1, dim2=dim2).sum(dim=-1)


def vector_outer(x: Tensor, y: Optional[Tensor] = None) -> Tensor:
    """
    Compute the outer product of two arrays.

    Parameters
    ----
    - `x` (Tensor): shape `(..., M)` first array.
    - `y` (Tensor, optional): shape `(..., N)` second array. If None, compute the outer product of `x` with itself.

    Returns
    ----
    - `outer` (Tensor): shape `(..., M, N)` outer product of `x` and `y`.
    """
    if y is None:
        return x[..., :, None] * x[..., None, :]
    return x[..., :, None] * y[..., None, :]