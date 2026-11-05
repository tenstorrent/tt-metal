// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: Tensors from a kernel's dataflow buffers. Single thread, arch-independent.
//
//   const auto x    = llk2::wait_front<x_spec>(in0);     // unpack waits for x's pages; x at the read pointer
//   const auto bias = llk2::peek_front<bias_spec>(b);    // the same, but release() does not pop: pop_front later
//   const auto slot = llk2::reserve_back<out_spec>(out); // pack waits for free pages; slot at the write pointer
//   ...
//   llk2::release(x);                                    // unpack pops x's pages
//   llk2::release(out);                                  // pack pushes the pages the op wrote (pack returns slot's)
//   llk2::pop_front<bias_spec>(b);                       // unpack pops a peeked tensor's pages
//
// The buffers stay the kernel's (tt-metal's DataflowBuffer, or the tt-llk harness's stand-in, tests/helpers/): these
// calls only call their wait_front / pop_front / reserve_back / push_back and read their pointers. Buffer is any type
// with those methods, as tt-metal's DataflowBuffer has them. The pop or push is in the Tensor's type
// (tensor/tensor.h), with the page count taken from the spec, so a release cannot free a wrong count.
//
// Ownership (compute_api/sync.h): a Tensor from wait_front or peek_front is the unpackers', one from reserve_back the
// packers', as the read and write pointers are. The thread that owns a side of the buffer is the only one that waits
// on it or moves it, and the only one where its pointer is valid (unpack: read, pack: write; 0 on math); llk2 reads an
// input's address only on unpack and an output's only on pack, so that is enough.
//
// Assumes pages of one tile (as 1.0 counts them): a spec's page count is its tile count. Buffer pointers count 16-byte
// words on a TRISC (tt-metal: cb_addr_shift = CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT = 4).

#pragma once

#include <cstdint>

#include "tensor/tensor.h"

namespace llk2
{

/// Pages a tensor with spec S takes in a dataflow buffer: one tile per page.
template <const TensorSpec<tensor_rank>& S>
inline constexpr std::uint16_t buffer_pages = tile_count(S.shape);

/// The release action of a Tensor from wait_front: pop its pages.
template <class Buffer, std::uint16_t Pages>
struct BufferPop
{
    Buffer* buffer; ///< the buffer the pages are in

    /// Unpack: frees the pages once the unpackers have read them.
    void operator()() const
    {
        buffer->pop_front(Pages);
    }
};

/// The release action of a Tensor from reserve_back: push its pages.
template <class Buffer, std::uint16_t Pages>
struct BufferPush
{
    Buffer* buffer; ///< the buffer the pages are in

    /// Pack: hands the pages on once the packer has written them.
    void operator()() const
    {
        buffer->push_back(Pages);
    }
};

/// Waits for S's pages at the front of buffer (unpack) and returns the tensor there, the unpackers'. release() pops.
template <const TensorSpec<tensor_rank>& S, class Buffer>
inline __attribute__((always_inline)) auto wait_front(Buffer& buffer)
{
    static_assert(S.storage == Storage::L1, "a buffer's pages are in L1");
    constexpr std::uint16_t pages = buffer_pages<S>;
    buffer.wait_front(pages);
    return Tensor<owned_by<S, Thread::Unpack, Exu::Unpacker>, BufferPop<Buffer, pages>>(
        buffer.get_read_ptr() * l1_word_bytes, BufferPop<Buffer, pages> {&buffer});
}

/// As wait_front, but release() does not pop: the pages stay for later calls until pop_front<S>(buffer).
template <const TensorSpec<tensor_rank>& S, class Buffer>
inline __attribute__((always_inline)) auto peek_front(Buffer& buffer)
{
    static_assert(S.storage == Storage::L1, "a buffer's pages are in L1");
    buffer.wait_front(buffer_pages<S>);
    return Tensor<owned_by<S, Thread::Unpack, Exu::Unpacker>>(buffer.get_read_ptr() * l1_word_bytes);
}

/// Pops S's pages from the front of buffer (unpack): the end of a peek_front.
template <const TensorSpec<tensor_rank>& S, class Buffer>
inline __attribute__((always_inline)) void pop_front(Buffer& buffer)
{
    buffer.pop_front(buffer_pages<S>);
}

/// Waits for S's pages free at the back of buffer (pack) and returns the tensor there, the packers'. release() pushes.
template <const TensorSpec<tensor_rank>& S, class Buffer>
inline __attribute__((always_inline)) auto reserve_back(Buffer& buffer)
{
    static_assert(S.storage == Storage::L1, "a buffer's pages are in L1");
    constexpr std::uint16_t pages = buffer_pages<S>;
    buffer.reserve_back(pages);
    return Tensor<owned_by<S, Thread::Pack, Exu::Packer>, BufferPush<Buffer, pages>>(
        buffer.get_write_ptr() * l1_word_bytes, BufferPush<Buffer, pages> {&buffer});
}

} // namespace llk2
