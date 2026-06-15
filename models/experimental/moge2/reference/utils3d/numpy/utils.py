import numpy as np
from numpy import ndarray
from typing import *
from numbers import Number, Integral
import warnings
import functools
import math

if TYPE_CHECKING:
    from scipy.sparse import csr_array


__all__ = [
    'sliding_window',
    'pooling',
    'max_pool_2d',
    'lookup',
    'lookup_get',
    'lookup_set',
    'group',
    'csr_matrix_from_dense_indices',
    'reverse_permutation',
    'vector_outer'
]


def sliding_window(
    x: ndarray, 
    window_size: Union[int, Tuple[int, ...]], 
    stride: Optional[Union[int, Tuple[int, ...]]] = None, 
    dilation: Optional[Union[int, Tuple[int, ...]]] = None,
    pad_size: Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]] = None, 
    pad_mode: str = 'constant',
    pad_value: Number = 0,
    axis: Optional[Tuple[int,...]] = None
) -> ndarray:
    """
    Get a sliding window of the input array. Window axis(axes) will be appended as the last dimension(s).
    This function is a wrapper of `numpy.lib.stride_tricks.sliding_window_view` with additional support for padding and stride.

    ## Parameters
    - `x` (ndarray): Input array.
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
    - `pad_mode` (str): Padding mode to use. Refer to `numpy.pad` for more details.
    - `pad_value` (Union[int, float]): Value to use for constant padding. Only used
        when `pad_mode` is 'constant'.
    - `axis` (Optional[Tuple[int,...]]): Axes to apply the sliding window. If None, all axes are used.

    ## Returns
    - (ndarray): Sliding window of the input array. 
        - If no padding, the output is a view of the input array with zero copy.
        - Otherwise, the output is no longer a view but a copy of the padded array.
    """
    # Process axis
    if axis is None:
        axis = tuple(range(x.ndim))
    if isinstance(axis, Integral):
        axis = (axis,)
    axis = [axis[i] % x.ndim for i in range(len(axis))]
    if isinstance(window_size, Integral):
        window_size = (window_size,) * len(axis)
    if dilation is not None:
        if isinstance(dilation, Integral):
            dilation = (dilation,) * len(axis)
    if stride is not None:
        if isinstance(stride, Integral):
            stride = (stride,) * len(axis)

    # Pad the input array if needed
    if pad_size is not None:
        if isinstance(pad_size, Integral):
            pad_size = ((pad_size, pad_size),) * len(axis)
        elif isinstance(pad_size, tuple) and len(pad_size) == 2 and all(isinstance(p, Integral) for p in pad_size):
            pad_size = (pad_size,) * len(axis)
        elif isinstance(pad_size, tuple) and all(isinstance(p, tuple) and 1 <= len(p) <= 2 for p in pad_size):
            if len(pad_size) == 1:
                pad_size = pad_size * len(axis)
            else:
                assert len(pad_size) == len(axis), f"pad_size {pad_size} must match the number of axes {len(axis)}"
        else:
            raise ValueError(f"Invalid pad_size {pad_size}")
        full_pad = [(0, 0) if i not in axis else pad_size[axis.index(i)] for i in range(x.ndim)]
        if pad_mode == 'constant':
            x = np.pad(x, full_pad, mode=pad_mode, constant_values=pad_value)
        else:
            x = np.pad(x, full_pad, mode=pad_mode)
    
    # Apply sliding window
    if dilation is None:
        x = np.lib.stride_tricks.sliding_window_view(x, window_size, axis=axis)
    else:
        window_size_dilated = tuple((window_size[i] - 1) * dilation[i] + 1 for i in range(len(window_size)))
        x = np.lib.stride_tricks.sliding_window_view(x, window_size_dilated, axis=axis)

    # Apply stride if needed
    if stride is not None:
        stride_slice = tuple(slice(None) if i not in axis else slice(None, None, stride[axis.index(i)]) for i in range(x.ndim - len(axis)))
        x = x[stride_slice]
    
    # Apply dilation if needed
    if dilation is not None:
        dilation_slice = tuple(slice(None, None, dilation[i]) for i in range(len(axis)))
        x = x[(..., *dilation_slice)]

    return x


def pooling(
    x: ndarray, 
    kernel_size: Union[int, Tuple[int, ...]], 
    stride: Optional[Union[int, Tuple[int, ...]]] = None, 
    padding: Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]] = None, 
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    mode: Literal['min', 'max', 'sum', 'mean'] = 'max'
) -> ndarray:
    """Compute the pooling of the input array. 
    NOTE: NaNs will be ignored.

    ## Parameters
        - `x` (ndarray): Input array.
        - `kernel_size` (int or Tuple[int,...]): Size of the pooling window.
        - `stride` (Optional[Tuple[int,...]]): Stride of the pooling window. If None,
            no stride is applied. If int is provided, the same stride is used for all specified axes.
        - `padding` (Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]]): Size of padding to apply before pooling.
            Corresponding to `axis`.
            - General format is `((before_1, after_1), (before_2, after_2), ...)`.
            - Shortcut formats: 
                - `int` -> same padding before and after for all axes;
                - `(int, int)` -> same padding before and after for each axis;
                - `((int,), (int,) ...)` -> specify padding for each axis, same before and after.
        - `axis` (Optional[Tuple[int,...]]): Axes to apply the pooling. If None, all axes are used.
        - `mode` (str): Pooling mode. One of 'min', 'max', 'sum', 'mean'.

    ## Returns
        - (ndarray): Pooled array with the same number of dimensions as input array.
    """
    if axis is None:
        axis = tuple(range(x.ndim))
    if isinstance(axis, Integral):
        axis = (axis,)
    axis = [axis[i] % x.ndim for i in range(len(axis))]
    if isinstance(kernel_size, Integral):
        kernel_size = (kernel_size,) * len(axis)
    if not isinstance(stride, tuple):
        stride = (stride,) * len(axis)
    if padding is not None:
        if isinstance(padding, Integral):
            padding = ((padding, padding),) * len(axis)
        elif isinstance(padding, tuple) and len(padding) == 2 and all(isinstance(p, Integral) for p in padding):
            padding = (padding,) * len(axis)
        elif isinstance(padding, tuple) and all(isinstance(p, tuple) and 1 <= len(p) <= 2 for p in padding):
            if len(padding) == 1:
                padding = padding * len(axis)
            else:
                assert len(padding) == len(axis), f"padding {padding} must match the number of axes {len(axis)}"
        else:
            raise ValueError(f"Invalid padding {padding}")
    else:
        padding = ((0, 0),) * len(axis)

    if mode == 'max':
        pad_mode = 'constant'
        pad_value = -np.inf if x.dtype.kind == 'f' else np.iinfo(x.dtype).min
        pool_fn = np.nanmax
    elif mode == 'min':
        pad_mode = 'constant'
        pad_value = np.inf if x.dtype.kind == 'f' else np.iinfo(x.dtype).max
        pool_fn = np.nanmin
    elif mode == 'sum':
        pad_mode = 'constant'
        pad_value = 0
        pool_fn = np.sum
        x = np.where(np.isnan(x), 0, x)
    elif mode == 'mean':
        mask = ~np.isnan(x)
        full_pad = [(0, 0) if i not in axis else padding[axis.index(i)] for i in range(x.ndim)]
        x = pooling(np.pad(x, full_pad, mode='edge'), kernel_size, stride, axis=axis, mode='sum')
        x /= pooling(np.pad(mask, full_pad, mode='edge'), kernel_size, stride, axis=axis, mode='sum')
        return x
    else:
        raise ValueError(f"Invalid pooling mode {mode}. Supported modes are 'min', 'max', 'sum', 'mean'.")

    for i in range(len(axis)):
        x = pool_fn(
            sliding_window(x, kernel_size[i], stride[i], 
                           pad_size=padding[i], pad_mode=pad_mode, pad_value=pad_value, 
                           axis=axis[i]), 
            axis=-1
        )
    return x


def max_pool_2d(x: ndarray, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]], axis: Tuple[int, int] = (-2, -1)):
    if isinstance(kernel_size, Number):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, Number):
        stride = (stride, stride)
    if isinstance(padding, Number):
        padding = (padding, padding)
    axis = tuple(axis)
    return pooling(x, kernel_size, stride, padding, axis, 'max')


def lookup(key: ndarray, query: ndarray) -> ndarray:
    """Look up `query` in `key` like a dictionary. Useful for COO indexing.

    Parameters
    ----
    - `key` (ndarray): shape `(num_keys, *key_shape)`, the array to search in
    - `query` (ndarray): shape `(..., *key_shape)`, the array to search for. `...` represents any number of batch dimensions.

    Returns
    ----
    - `indices` (ndarray): shape `(...,)` indices in `key` for each `query`. If a query is not found in key, the corresponding index will be -1.

    Notes
    ----
    `O((Q + K) * log(Q + K))` complexity, where `Q` is the number of queries and `K` is the number of keys.
    """
    assert key.dtype == query.dtype, "Key and query must have the same dtype"
    assert key.shape[1:] == query.shape[query.ndim - key.ndim + 1:], f"Key shape {key.shape} and query shape {query.shape} are not compatible."

    num_keys, *key_shape = key.shape
    query_batch_shape = query.shape[:query.ndim - key.ndim + 1]

    key_item_nbytes = math.prod(key_shape) * key.dtype.itemsize
    if key.ndim == 1:
        # Fast path 1: 1D keys, can directly sort and search
        sorted_indices = np.argsort(key)
        key_sorted = key[sorted_indices]
        result = np.searchsorted(key_sorted, query, side='left')
        mask = (result < num_keys) & (key_sorted[result.clip(0, num_keys - 1)] == query)

        result = result.astype(np.int64, copy=False)
        result[mask] = sorted_indices[result[mask]]
        result[~mask] = -1
        return result.reshape(query_batch_shape)
    
    elif key_item_nbytes <= 8:
        # Fast path 2: small keys, can view as int64 and sort/search
        query_flat = query.reshape(-1, *key_shape)

        key_bytes = np.ascontiguousarray(key).view(np.uint8).reshape(num_keys, key_item_nbytes)
        query_bytes = np.ascontiguousarray(query_flat).view(np.uint8).reshape(query_flat.shape[0], key_item_nbytes)

        if key_item_nbytes < 8:
            pad_width = ((0, 0), (0, 8 - key_item_nbytes))
            key_bytes = np.pad(key_bytes, pad_width, mode='constant')
            query_bytes = np.pad(query_bytes, pad_width, mode='constant')

        key_i64 = key_bytes.view(np.int64).reshape(-1)
        query_i64 = query_bytes.view(np.int64).reshape(-1)

        sorted_indices = np.argsort(key_i64)
        key_sorted = key_i64[sorted_indices]
        result = np.searchsorted(key_sorted, query_i64, side='left')
        mask = (result < num_keys) & (key_sorted[result.clip(0, num_keys - 1)] == query_i64)

        result = result.astype(np.int64, copy=False)
        result[mask] = sorted_indices[result[mask]]
        result[~mask] = -1
        return result.reshape(query_batch_shape)
    else:
        query_flat = query.reshape(-1, *key_shape)
        _, index, inverse = np.unique(
            np.concatenate([key, query_flat], axis=0),
            axis=0,
            return_index=True,
            return_inverse=True
        )
        result = index[inverse[num_keys:]]
        result[result >= num_keys] = -1
        return result.reshape(query_batch_shape)


def lookup_get(key: ndarray, value: ndarray, get_key: ndarray, default_value: Union[Number, ndarray] = 0) -> ndarray:
    """Dictionary-like get for arrays

    ## Parameters
    - `key` (ndarray): shape `(N, *key_shape)`, the key array of the dictionary to get from
    - `value` (ndarray): shape `(N, *value_shape)`, the value array of the dictionary to get from
    - `get_key` (ndarray): shape `(..., *key_shape)`, the key array to get for. `...` represents any number of batch dimensions.
    - `default_value` (Union[Number, ndarray]): a scalar or an array broadcastable to shape `(..., *value_shape)`. Value to return if a key in `get_key` is not found in `key`.

    ## Returns
        `get_value` (ndarray): shape `(..., *value_shape)`, result values corresponding to `get_key`
    """
    indices = lookup(key, get_key)
    if key.shape[0] == 0:
        return np.broadcast_to(np.asarray(default_value, dtype=value.dtype), get_key.shape[:get_key.ndim - key.ndim + 1] + value.shape[1:])
    return np.where(
        (indices >= 0)[(..., *((None,) * (value.ndim - 1)))], 
        value[indices.clip(0, key.shape[0] - 1)], 
        default_value
    )


def lookup_set(key: ndarray, value: ndarray, set_key: ndarray, set_value: ndarray, append: bool = False, inplace: bool = False) -> Tuple[ndarray, ndarray]:
    """Dictionary-like set for arrays.

    ## Parameters
    - `key` (ndarray): shape `(N, *key_shape)`, the key array of the dictionary to set
    - `value` (ndarray): shape `(N, *value_shape)`, the value array of the dictionary to set
    - `set_key` (ndarray): shape `(M, *key_shape)`, the key array to set for
    - `set_value` (ndarray): shape `(M, *value_shape)`, the value array to set as
    - `append` (bool): If True, append the (key, value) pairs in (set_key, set_value) that are not in (key, value) to the result.
    - `inplace` (bool): If True, modify the input `value` array

    ## Returns
    - `result_key` (ndarray): shape `(N_new, *value_shape)`. N_new = N + number of new keys added if append is True, else N.
    - `result_value (ndarray): shape `(N_new, *value_shape)` 
    """
    set_indices = lookup(key, set_key)
    if inplace:
        assert append is False, "Cannot append when inplace is True"
    else:
        value = value.copy()
    hit = np.where(set_indices >= 0)
    value[set_indices[hit]] = set_value[hit]
    if append:
        missing = np.where(set_indices < 0)
        key = np.concatenate([key, set_key[missing]], axis=0)
        value = np.concatenate([value, set_value[missing]], axis=0)
    return key, value


def take_view(a: ndarray, i: Union[int, slice], axis: int = 0) -> ndarray:
    """Take a view of the input array at the specified index along the given axis."""
    return a[(slice(None),) * (axis % a.ndim) + (i,)]


def lite_sum(a: ndarray, axis: int = -1) -> ndarray:
    """Compute the sum of the input array along the specified small axis.
    """
    result_dtype = np.result_type(a.dtype, 0)
    if a.shape[axis] == 0:
        return np.zeros(a.shape[:axis] + a.shape[axis + 1:], dtype=result_dtype)
    elif a.shape[axis] <= 4:    # Sweet point for python loop vs einsum
        s = take_view(a, 0, axis=axis).astype(result_dtype, copy=True)
        for i in range(1, a.shape[axis]):
            s += take_view(a, i, axis=axis)
        return s
    else:   # Einsum is faster than np.sum in most cases
        return np.einsum('...i->...', np.moveaxis(a, axis, -1), optimize=False)


def lite_prod(a: ndarray, axis: int = -1) -> ndarray:
    """Compute the product of the input array along the specified small axis.
    """
    result_dtype = np.result_type(a.dtype, 1)
    if a.shape[axis] == 0:
        return np.ones(a.shape[:axis] + a.shape[axis + 1:], dtype=result_dtype)
    elif a.shape[axis] <= 8:
        p = take_view(a, 0, axis=axis).astype(result_dtype, copy=True)
        for i in range(1, a.shape[axis]):
            p *= take_view(a, i, axis=axis)
        return p
    else:
        return np.prod(a, axis=axis)


def lite_dot(a: ndarray, b: ndarray, axis: int = -1) -> ndarray:
    """Compute the dot product of two input arrays along the specified small axis.
    """
    if a.shape[axis] == 0:
        return np.zeros(a.shape[:axis] + a.shape[axis + 1:], dtype=np.result_type(a.dtype, b.dtype))
    elif a.shape[axis] <= 3:
        return lite_sum(a * b, axis=axis)
    else:
        return np.einsum('...i,...i->...', np.moveaxis(a, axis, -1), np.moveaxis(b, axis, -1), optimize=False)


def lite_norm(a: ndarray, ord: int = 2, axis: int = -1) -> ndarray:
    """Compute the norm of the input array along the specified small axis.
    """
    if ord == 1:
        return lite_sum(np.abs(a), axis=axis)
    elif ord == 2:
        return np.sqrt(lite_sum(a * a, axis=axis))
    elif ord == np.inf:
        return np.max(np.abs(a), axis=axis)
    else:
        raise ValueError(f"Unsupported norm order {ord}. Supported orders are 1, 2, and inf.")
    

def safe_inv(mat: ndarray, max_retries: int = 4) -> ndarray:
    """Compute the inverse of a matrix, no matter it is singular or not. If the matrix is singular, use pseudo-inverse instead.
    If both inverse and pseudo-inverse fail, return a matrix filled with NaNs.

    ## Parameters
    - `mat` (ndarray): shape `(..., M, M)` input square matrix/matrices to invert.

    ## Returns
    - `inv_mat` (ndarray): shape `(..., M, M)` inverse of the input matrix/matrices.
    """
    for i in range(max_retries):
        try:
            return np.linalg.inv(mat)
        except np.linalg.LinAlgError:
            eps = 10 ** i * np.finfo(mat.dtype).eps * np.linalg.norm(mat, ord='fro', axis=(-2, -1), keepdims=True) 
            mat = mat + eps * np.eye(mat.shape[-1])
    try:
        return np.linalg.pinv(mat)
    except np.linalg.LinAlgError:
        warnings.warn("Matrix inversion and pseudo-inversion both failed. Returning NaN matrix.")
        return np.full_like(mat, np.nan)


def group(labels: ndarray, data: Optional[np.ndarray] = None) -> List[Tuple[ndarray, ndarray]]:
    """
    Split the data into groups based on the provided labels.

    ## Parameters
    - `labels` `(ndarray)` shape `(N, *label_dims)` array of labels for each data point. Labels can be multi-dimensional.
    - `data`: `(ndarray, optional)` shape `(N, *data_dims)` dense tensor. Each one in `N` has `D` features.
        If None, return the indices in each group instead.

    ## Returns
    - `groups` `(List[Tuple[ndarray, ndarray]])`: List of each group, a tuple of `(label, data_in_group)`.
        - `label` (ndarray): shape `(*label_dims,)` the label of the group.
        - `data_in_group` (ndarray): shape `(length_of_group, *data_dims)` the data points in the group.
        If `data` is None, `data_in_group` will be the indices of the data points in the original array.
    """
    group_labels, inv, counts = np.unique(labels, return_inverse=True, return_counts=True, axis=0)
    if data is None:
        data = np.arange(labels.shape[0])
    sections = np.cumsum(counts, axis=0)[:-1]
    data_groups = np.split(data[np.argsort(inv)], sections)
    return list(zip(group_labels, data_groups))


def csr_matrix_from_dense_indices(indices: ndarray, n_cols: int) -> 'csr_array':
    """Convert a regular indices array to a sparse CSR adjacency matrix format

    ## Parameters
        - `indices` (ndarray): shape (N, M) dense tensor. Each one in `N` has `M` connections.
        - `n_cols` (int): total number of columns in the adjacency matrix

    ## Returns
        Tensor: shape `(N, n_cols)` sparse CSR adjacency matrix
    """
    from scipy.sparse import csr_array
    return csr_array((
        np.ones_like(indices, dtype=bool).ravel(), 
        indices.ravel(),
        np.arange(0, indices.size + 1, indices.shape[1])
    ), shape=(indices.shape[0], n_cols))


def reverse_permutation(perm: ndarray, axis: int = 0) -> ndarray:
    """Compute the reverse of a permutation array. 

    Parameters
    ----
    - `perm` (ndarray): shape `(..., N, ...)` permutation array.
    - `axis` (int): axis of the permutation array. Other axes are treated as batch dimensions.

    Returns
    ----
    - `rev_perm` (ndarray): shape `(N,)` reverse permutation array.

    Notes
    -----
    Equivalent to `np.argsort(perm, axis=axis)`, but more efficient.
    """
    axis = axis % perm.ndim
    rev_perm = np.empty_like(perm)
    indices = np.arange(perm.shape[axis], dtype=perm.dtype)[(None,) * axis + (slice(None),) + (None,) * (perm.ndim - axis - 1)]
    np.put_along_axis(rev_perm, perm, indices, axis=axis)
    return rev_perm


def vector_outer(x: ndarray, y: Optional[ndarray] = None) -> ndarray:
    """
    Compute the outer product of two arrays.

    Parameters
    ----
    - `x` (ndarray): shape `(..., M)` first array.
    - `y` (ndarray, optional): shape `(..., N)` second array. If None, compute the outer product of `x` with itself.

    Returns
    ----
    - `outer` (ndarray): shape `(..., M, N)` outer product of `x` and `y`.
    """
    if y is None:
        return x[..., :, None] * x[..., None, :]
    return x[..., :, None] * y[..., None, :]