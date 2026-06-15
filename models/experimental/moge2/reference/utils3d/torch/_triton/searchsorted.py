from typing import *
import math

import torch
from torch import Tensor
import triton
import triton.language as tl


@triton.jit
def _segment_searchsorted_side_left_kernel(
    sorted_ptr,        # *fp32
    query_ptr,         # *fp32
    seg_start_ptr,     # *int32
    seg_end_ptr,       # *int32
    out_ptr,           # *int32
    Q,                 # total number of queries
    LOG_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < Q

    # load query and segment range
    q = tl.load(query_ptr + offs, mask=mask, other=0.0)
    lo = tl.load(seg_start_ptr + offs, mask=mask, other=0)
    hi = tl.load(seg_end_ptr + offs, mask=mask, other=0)

    # binary search
    for _ in range(LOG_N):
        mid = (lo + hi) >> 1
        mid_val = tl.load(sorted_ptr + mid, mask=mask, other=0.0)
        go_right = mid_val < q
        lo = tl.where(go_right, mid + 1, lo)
        hi = tl.where(go_right, hi, mid)

    tl.store(out_ptr + offs, lo, mask=mask)


@triton.jit
def _segment_searchsorted_side_right_kernel(
    sorted_ptr,        # *fp32
    query_ptr,         # *fp32
    seg_start_ptr,     # *int32
    seg_end_ptr,       # *int32
    out_ptr,           # *int32
    Q,                 # total number of queries
    LOG_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < Q

    # load query and segment range
    q = tl.load(query_ptr + offs, mask=mask, other=0.0)
    lo = tl.load(seg_start_ptr + offs, mask=mask, other=0)
    hi = tl.load(seg_end_ptr + offs, mask=mask, other=0)

    # binary search
    for _ in range(LOG_N):
        mid = (lo + hi) >> 1
        mid_val = tl.load(sorted_ptr + mid, mask=mask, other=0.0)
        go_right = mid_val <= q
        lo = tl.where(go_right, mid + 1, lo)
        hi = tl.where(go_right, hi, mid)

    tl.store(out_ptr + offs, lo, mask=mask)


def segment_searchsorted_1d_triton(
    sorted_sequence_flat: Tensor,
    input_flat: Tensor,
    seg_start_flat: Tensor,
    seg_end_flat: Tensor,
    side: Literal['left', 'right'] = 'left',
    *,
    max_length: Optional[int] = None,
):
    Q = input_flat.numel()
    BLOCK = 256

    # ensure dtypes
    sorted_sequence_flat = sorted_sequence_flat.contiguous()
    input_flat = input_flat.contiguous()
    seg_start_flat = seg_start_flat.int().contiguous()
    seg_end_flat = seg_end_flat.int().contiguous()

    out = torch.empty_like(seg_start_flat, dtype=torch.int32)

    if max_length is None:
        max_length = (seg_end_flat - seg_start_flat).max().item() + 1
    LOG_N = math.ceil(math.log2(max_length + 1) / 4) * 4

    grid = (triton.cdiv(Q, BLOCK),)
    if side == 'left':
        _segment_searchsorted_side_left_kernel[grid](
            sorted_sequence_flat,
            input_flat,
            seg_start_flat,
            seg_end_flat,
            out,
            Q,
            LOG_N=LOG_N,
            BLOCK=BLOCK,
        )
    else:
        _segment_searchsorted_side_right_kernel[grid](
            sorted_sequence_flat,
            input_flat,
            seg_start_flat,
            seg_end_flat,
            out,
            Q,
            LOG_N=LOG_N,
            BLOCK=BLOCK,
        )
    out = out.long()

    return out
