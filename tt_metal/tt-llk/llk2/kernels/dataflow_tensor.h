// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Kernel side: an L1 Tensor at a dataflow buffer's read or write pointer. LLK 2.0 itself (llk2/inc) takes only
// Tensors and knows nothing of buffers; kernels wait, pop, reserve and push themselves, as in 1.0:
//
//   in0.wait_front(n);
//   const auto x = llk2::front<x_spec>(in0);  // at in0's read pointer
//   ...
//   in0.pop_front(n);
//
// Buffer is tt-metal's DataflowBuffer, or the harness's stand-in (tests/helpers/test_dataflow_buffer.h). On a TRISC its
// pointers count 16-byte words (tt-metal: cb_addr_shift = CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT = 4; L1_UNCACHED_OFFSET
// is 0 off Quasar's data movement cores), and they are valid only on the thread that owns that side of the buffer:
// the read pointer on unpack, the write pointer on pack. On math they are 0. So front() is valid on unpack only and
// back() on pack only, which is where llk2 reads those addresses (tensor/tensor.h).

#pragma once

#include <cstdint>

#include "tensor/tensor.h"

namespace llk2
{

/// The tensor with spec S at Buffer's read pointer: the oldest pages, after wait_front. Valid on unpack.
template <const auto& S, class Buffer>
inline __attribute__((always_inline)) Tensor<S> front(const Buffer& buffer)
{
    static_assert(S.storage == Storage::L1, "a buffer's pages are in L1");
    return Tensor<S>(buffer.get_read_ptr() * l1_word_bytes);
}

/// The tensor with spec S at Buffer's write pointer: the free pages, after reserve_back. Valid on pack.
template <const auto& S, class Buffer>
inline __attribute__((always_inline)) Tensor<S> back(const Buffer& buffer)
{
    static_assert(S.storage == Storage::L1, "a buffer's pages are in L1");
    return Tensor<S>(buffer.get_write_ptr() * l1_word_bytes);
}

} // namespace llk2
