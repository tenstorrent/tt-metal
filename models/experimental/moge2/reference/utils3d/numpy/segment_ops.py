import numpy as np
from numpy import ndarray
from typing import *
from numbers import Number, Integral
import warnings
import functools

from .utils import reverse_permutation


__all__ = [
    'segment_roll',
    'segment_take',
    'segment_argmax',
    'segment_argmin',
    'segment_concatenate',
    'segment_concat',
    'segment_chain',
    'group_as_segments'
]


def segment_roll(data: ndarray, offsets: ndarray, shift: int, axis: int = 0) -> ndarray:
    """Roll the data within each segment.

    Parameters
    ------
    - `data`: (ndarray).
    - `offsets`: (ndarray) shape `(M + 1,)` the offsets of the segmented data. `M` is the number of segments. Starts with 0 and end with `data.shape[axis]`.
    - `shift`: (int) the number of places by which elements are shifted. If negative, shift to left.
    - `axis`: (int) the segment axis to roll along. Default is 0.

    Returns
    -------
    - `data`: (ndarray) the rolled data, same shape as input.
    """
    lengths = np.diff(offsets)
    start = np.repeat(offsets[:-1], lengths)
    elem_indices = start + (np.arange(data.shape[axis], dtype=offsets.dtype) - start - shift) % np.repeat(lengths, lengths)
    data = np.take(data, elem_indices, axis=axis)
    return data


def segment_take(data: ndarray, offsets: ndarray, taking: ndarray, axis: int = 0) -> Tuple[ndarray, ndarray]:
    """Take some segments from a segmented array

    Parameters
    ------
    - `data`: (ndarray) the segmented data.
    - `offsets`: (ndarray) shape `(M + 1,)` the offsets of the segmented data. `M` is the number of segments. Starts with 0 and end with `data.shape[axis]`.
    - `taking`: (ndarray) the indices of segments to take of shape `(K,)`, or boolean mask of shape `(M,)`
    - `axis`: (int) the segment axis to take along. Default is 0. Other axes are treated as batch dimensions.

    Returns
    -------
    - `new_data`: (ndarray) the new segmented data.
    - `new_offsets`: (ndarray) shape `(K + 1,)` the offsets of the new segmented data. `K` is the number of taken segments.
    """
    if taking.dtype == np.bool_:
        taking = np.where(taking)[0]
    lengths = np.diff(offsets)
    new_lengths = lengths[taking]
    new_offsets = np.concatenate([[0], np.cumsum(new_lengths)])
    indices = np.arange(new_offsets[-1]) + np.repeat(offsets[taking] - new_offsets[:-1], new_lengths)
    new_data = np.take(data, indices, axis=axis)
    return new_data, new_offsets


def segment_concatenate(segments: Sequence[Tuple[ndarray, ndarray]], axis: int = 0) -> Tuple[ndarray, ndarray]:
    """Concatenate segmented arrays within each segment. All numbers of segments remain the same.

    Parameters
    ------
    - `segments`: (Sequence[Tuple[ndarray, ndarray]]) A sequence of segmented arrays:
        - `data`: (ndarray) shape `(..., N_i, ...)`
        - `offsets`: (ndarray) shape `(M + 1,)` segment offsets.
    - `axis`: (int) the segment axis.

    Returns
    -------
    - `data`: (ndarray) shape `(..., sum(N_i), ...)` the concatenated data
    - `offsets`: (ndarray) shape `(M + 1,)` the offsets of the concatenated segmented data.
    """
    if len(segments) == 0:
        return np.array([]), np.array([0])
    M, K = len(segments[0][1]) - 1, len(segments)       # number of segments, number of inputs
    assert all(M + 1 == len(seg[1]) for seg in segments), "All segments must have the same number of segments when concatenating along axis 1."
    data_list, offsets_list = zip(*segments)
    input_data = np.concatenate(data_list, axis=axis)                               # (..., sum(N_i), ...)
    lengths_concat_0 = np.diff(np.stack(offsets_list, axis=0), axis=1).reshape(-1)  # (K, M)
    lengths_concat_1 = lengths_concat_0.reshape(K, M).swapaxes(0, 1).reshape(-1)    # (M, K)
    offsets_concat_0 = np.cumsum(lengths_concat_0.reshape(-1), axis=0)              # (K * M,)
    offsets_concat_1 = np.cumsum(lengths_concat_1.reshape(-1), axis=0)              # (M * K,)

    indices = np.arange(input_data.shape[0]) + np.repeat(offsets_concat_0.reshape(K, M).swapaxes(0, 1).reshape(-1) - offsets_concat_1, lengths_concat_1.reshape(-1))
    new_data = np.take(input_data, indices, axis=axis)
    new_lengths = np.sum(lengths_concat_0.reshape(K, M), axis=0)
    new_offsets = np.concatenate([[0], np.cumsum(new_lengths)])
        
    return new_data, new_offsets


def segment_concat(segments: Sequence[Tuple[ndarray, ndarray]], axis: int = 0) -> Tuple[ndarray, ndarray]:
    """(Alias for segment_concatenate).
    Concatenate segmented arrays within each segment.

    Parameters
    ------
    - `segments`: (Sequence[Tuple[ndarray, ndarray]]) A sequence of segmented arrays:
        - `data`: (ndarray) shape `(..., N_i, ...)`
        - `offsets`: (ndarray) shape `(M + 1,)` segment offsets.
    - `axis`: (int) the segment axis.

    Returns
    -------
    - `data`: (ndarray) shape `(N, *data_dims)` the concatenated data
    - `offsets`: (ndarray) shape `(M + 1,)` the offsets of the concatenated segmented data.
    """
    return segment_concatenate(segments, axis=axis)


def segment_chain(segments: Sequence[Tuple[ndarray, ndarray]], axis: int = 0) -> Tuple[ndarray, ndarray]:
    """Concatenate segmented arrays in sequence. The number of segments are summed.

    Parameters
    ------
    - `segments`: (Sequence[Tuple[ndarray, ndarray]]) A sequence of segmente arrays:
        - `data`: (ndarray) shape `(..., N_i, ...)`
        - `offsets`: (ndarray) shape `(M + 1,)` segment offsets.
    - `axis`: (int) the segment axis.

    Returns
    -------
    - `data`: (ndarray) shape `(..., sum(N_i), ...)` the chain-concatenated data
    - `offsets`: (ndarray) shape `(sum(M_i) + 1,)` the offsets of the concatenated segmented data.
    """

    data_list = []
    offsets_list = [np.array([0])]
    for data, offsets in segments:
        if len(offsets) > 1:
            data_list.append(data)
            offsets_list.append(offsets[1:] + offsets_list[-1][-1])
    new_data = np.concatenate(data_list, axis=axis)
    new_offsets = np.concatenate(offsets_list, axis=0)

    return new_data, new_offsets


def group_as_segments(labels: ndarray, data: Optional[np.ndarray] = None, return_inverse: bool = False, return_group_ids: bool = False) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Group as segments by labels

    Parameters
    -----
    - `labels` (ndarray): shape `(N, *label_dims)` array of labels for each data point. Labels can be multi-dimensional.
    - `data` (ndarray, optional): shape `(N, *data_dims)` array.
        If None, return the indices in each group instead.

    Returns
    -------
    Assuming there are `M` difference labels:

    - `segment_labels`: `(ndarray)` shape `(M, *label_dims)` labels of of each segment
    - `rearranged_data`: `(ndarray)` shape `(N,)` or `(N, *data_dims)` the rearranged data (or indices) where the same labels are grouped as a continous segment.
    - `offsets`: `(ndarray)` shape `(M + 1,)`
    
    `rearranged_data[offsets[i]:offsets[i + 1]]` corresponding to the i-th segment whose label is `segment_labels[i]`
    """
    group_labels, group_ids, counts = np.unique(labels, return_inverse=True, return_counts=True, axis=0)
    offsets = np.concatenate([[0], np.cumsum(counts, axis=0)])
    indices = np.argsort(group_ids)
    if data is None:
        data = indices
    else:
        data = data[indices]
    ret = (group_labels, data, offsets)
    if return_inverse:
        inverse_indices = reverse_permutation(indices)
        ret += (inverse_indices,)
    if return_group_ids:
        ret += (group_ids,)
    return group_labels, data, offsets


def segment_argmax(data: ndarray, offsets: ndarray, axis: int = 0) -> ndarray:
    """Compute the argmax of each segment in the segmented data.

    Parameters
    -----
    - `data`: (ndarray). shape `(..., N, ...)`. `N` is the segment dimension. If `data` may have multiple dimensionsm, extra dimensions are treated as batch dimensions.
    - `offsets`: (ndarray) shape `(M + 1,)` the offsets of the segmented data
    - `axis`: (int) the segment axis to compute along. Default is 0.

    Returns
    -------
    - `argmax_indices`: (ndarray) shape `(..., M, ...)` the argmax indices of each segment along the first dimension.
    
    Notes
    -----
    If there are multiple maximum values in a segment, the index of the first one is returned. If a segment is empty, -1 is returned.
    """
    axis = axis % data.ndim
    lengths = np.diff(offsets)
    seg_maxs = np.maximum.reduceat(data, offsets[:-1], axis=axis)
    seg_ids = np.repeat(np.arange(len(offsets) - 1), lengths)
    where_in_data = np.where(data == seg_maxs.take(seg_ids, axis=axis))
    where_in_argmax = (*where_in_data[:axis], seg_ids[where_in_data[axis]], *where_in_data[axis + 1:])
    value_in_argmax = where_in_data[axis]
    argmax = np.full(data.shape[:axis] + (len(offsets) - 1,) + data.shape[axis + 1:], fill_value=np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(argmax, where_in_argmax, value_in_argmax)
    argmax[argmax == np.iinfo(np.int64).max] = -1
    return argmax


def segment_argmin(data: ndarray, offsets: ndarray, axis: int = 0) -> ndarray:
    """Compute the argmin of each segment in the segmented data.

    Parameters
    -----
    - `data`: (ndarray) shape `(..., N, ...)` the data to compute argmin from. If `data` may have multiple dimensionsm, extra dimensions are treated as batch dimensions.
    - `offsets`: (ndarray) shape `(M + 1,)` the offsets of the segmented data
    - `axis`: (int) the segment axis to compute along. Default is 0.

    Returns
    -----
    - `argmin_indices`: (ndarray) shape `(..., M, ...)` the argmin indices of each segment along the first dimension.
    
    Notes
    -----
    If there are multiple minimum values in a segment, the index of the first one is returned. If a segment is empty, -1 is returned.
    """
    axis = axis % data.ndim
    lengths = np.diff(offsets)
    seg_mins = np.minimum.reduceat(data, offsets[:-1], axis=axis)
    seg_ids = np.repeat(np.arange(len(offsets) - 1), lengths)
    where_in_data = np.where(data == seg_mins.take(seg_ids, axis=axis))
    where_in_argmin = (*where_in_data[:axis], seg_ids[where_in_data[axis]], *where_in_data[axis + 1:])
    value_in_argmin = where_in_data[axis]
    argmin = np.full(data.shape[:axis] + (len(offsets) - 1,) + data.shape[axis + 1:], fill_value=np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(argmin, where_in_argmin, value_in_argmin)
    argmin[argmin == np.iinfo(np.int64).max] = -1
    return argmin

