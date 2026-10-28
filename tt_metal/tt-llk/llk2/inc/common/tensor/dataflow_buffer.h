// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Dataflow buffer (DM) API: Tensors from dataflow (circular) buffers. Arch-independent, single thread. Replaces
// cb_wait_front, cb_pop_front, cb_reserve_back and cb_push_back (DataflowBuffer::wait_front, ... in tt-metal).
// From tt-llk-lab lpremovic/llk-tensor, report 2, section 6:
// https://github.com/tenstorrent/tt-llk-lab/tree/lpremovic/llk-tensor/experiments/llk-tensor
//
//   inline constexpr DataflowBufferSpec in0_buf {.id = 0, .format = ..., .page_bytes = ..., .pages = ...};
//   setup_dataflow_buffer<in0_buf>(base);        // every thread, once (tt-metal: the firmware does this)
//   const auto x = wait_front<in0_buf, 1, 4>();  // [32, 128]: 4 of the buffer's tiles at the read pointer
//   const auto y = reserve_back<out_buf, 1, 4>(); // 4 free tiles at the write pointer
//   release(x);                                  // pops x's pages
//   release(y);                                  // pushes y's pages
//
// A buffer is compile time: an id, its tiles' format and shape, a page size and a capacity in pages. Only its base
// address is run time. wait_front<B, TilesY, TilesX>() takes TilesY x TilesX of the buffer's tiles, row-major, as
// one L1 Tensor; reserve_back the same. The tensor must be a whole number of pages, and its pages must divide the
// capacity, so that a tensor never wraps around the end of the buffer.
//
// The release is in the Tensor's type: wait_front returns a Tensor whose release pops its pages, reserve_back one
// whose release pushes them. release(t) (tensor/tensor.h) runs it. Nothing is stored in the Tensor but its address.
//
// Per thread. Every TRISC runs this same code and keeps its own copy of each buffer's read and write pointers.
//   wait_front     unpack thread: waits for the pages (the producer's count). Every thread: the Tensor at the read
//                  pointer, owned by the unpacker
//   reserve_back   pack thread: waits for free pages (the consumer's count). Every thread: the Tensor at the write
//                  pointer, owned by the packer
//   release, pop   every thread: read pointer on by the Tensor's pages. Unpack thread: tells the producer, once its
//                  unpackers have read the pages
//   release, push  every thread: write pointer on. Pack thread: tells the consumer, once its packer has written them
// The pointers move on every thread on every call, and only by compile-time page counts, from the same setup, so
// they are equal on all three threads after every call: a Tensor has the same address on all of them. Only the
// owner waits, so the math thread (and the non-owner of each buffer) may run ahead of it; it never reads the pages
// through that address. How the pages and counters are read and signalled is the arch's: llk_lib/dataflow_buffer.h.
//
// Without data movement cores (the tt-llk test harness), nothing fills or drains a buffer. preload_dataflow_buffer
// sets a buffer up over pages the harness already wrote to L1 and marks them pushed, so wait_front finds them
// there; an output buffer is set up empty and big enough for everything the kernel pushes, since nothing pops it.

#pragma once

#include <cstdint>

#include "llk_lib/dataflow_buffer.h"
#include "llk_lib/thread.h"
#include "tensor/tensor.h"

namespace llk2
{

/// A dataflow buffer, compile time. The host's CB config holds the same facts in tt-metal.
struct DataflowBufferSpec
{
    std::uint32_t id;                           ///< buffer id (tt-metal's CB index), 0 to 63
    DataFormat format;                          ///< format of the buffer's tiles
    std::uint32_t page_bytes;                   ///< one page in bytes: a whole number of tiles
    std::uint32_t pages;                        ///< capacity in pages
    std::uint32_t tile_rows = default_tile_dim; ///< tile height
    std::uint32_t tile_cols = default_tile_dim; ///< tile width
};

/// Bytes of one of B's tiles.
template <const DataflowBufferSpec& B>
inline constexpr std::uint32_t buffer_tile_bytes = tile_bytes(B.format, B.tile_rows* B.tile_cols);

/// The spec of TilesY x TilesX of B's tiles, dense in L1.
template <const DataflowBufferSpec& B, std::uint32_t TilesY, std::uint32_t TilesX>
inline constexpr TensorSpec<tensor_rank> buffer_tensor_spec =
    make_spec<TilesY * B.tile_rows, TilesX * B.tile_cols, B.format, B.tile_rows, B.tile_cols>(Storage::L1);

/// B as llk-lib's parameters.
template <const DataflowBufferSpec& B>
inline constexpr llk_lib::dataflow_buffer_params buffer_params = llk_lib::dataflow_buffer_params_from(B);

/// Pages of B that TilesY x TilesX tiles take. Checks the tensor fits the buffer without wrapping.
template <const DataflowBufferSpec& B, std::uint32_t TilesY, std::uint32_t TilesX>
constexpr std::uint32_t buffer_pages()
{
    static_assert(B.tile_rows == default_tile_dim && B.tile_cols == default_tile_dim, "dataflow buffers take 32x32 tiles only, for now");
    static_assert(B.page_bytes % buffer_tile_bytes<B> == 0, "a page must be a whole number of the buffer's tiles");
    constexpr std::uint32_t bytes = TilesY * TilesX * buffer_tile_bytes<B>;
    static_assert(bytes % B.page_bytes == 0, "the tensor must be a whole number of the buffer's pages");
    static_assert(B.pages % (bytes / B.page_bytes) == 0, "the tensor's pages must divide the buffer's, so it never wraps");
    return bytes / B.page_bytes;
}

/// Release action of a Tensor from wait_front: pops its pages.
template <const DataflowBufferSpec& B, std::uint32_t Pages>
struct PopFront
{
    void operator()() const
    {
        llk_lib::dataflow_buffer_pop_front<Thread::Unpack, buffer_params<B>, Pages>();
    }
};

/// Release action of a Tensor from reserve_back: pushes its pages.
template <const DataflowBufferSpec& B, std::uint32_t Pages>
struct PushBack
{
    void operator()() const
    {
        llk_lib::dataflow_buffer_push_back<Thread::Pack, buffer_params<B>, Pages>();
    }
};

/// Every thread, once before B is used: B's pages start at base_address (bytes). In tt-metal the firmware does this
/// from the host's CB config; the harness calls it.
template <const DataflowBufferSpec& B>
inline __attribute__((always_inline)) void setup_dataflow_buffer(std::uint32_t base_address)
{
    llk_lib::dataflow_buffer_setup<buffer_params<B>>(base_address);
    llk_lib::dataflow_buffer_set_counters<Thread::Pack, buffer_params<B>>(
        0 /* acked */, 0 /* received */); // harness only: empty; tt-metal's firmware zeroes them
}

/// Harness only, every thread: B is set up at base_address and its first Pages pages, already in L1, count as pushed
/// by a producer (in tt-metal, a data movement kernel). The unpack thread, B's consumer, writes the counters.
template <const DataflowBufferSpec& B>
inline __attribute__((always_inline)) void preload_dataflow_buffer(std::uint32_t base_address, std::uint16_t pages)
{
    llk_lib::dataflow_buffer_setup<buffer_params<B>>(base_address);
    llk_lib::dataflow_buffer_set_counters<Thread::Unpack, buffer_params<B>>(0 /* acked */, pages /* received */);
}

/// TilesY x TilesX of B's tiles at the read pointer, once the producer has pushed them. The unpacker's; release pops.
template <const DataflowBufferSpec& B, std::uint32_t TilesY = 1, std::uint32_t TilesX = B.page_bytes / buffer_tile_bytes<B>>
inline __attribute__((always_inline)) auto wait_front()
{
    constexpr std::uint32_t pages = buffer_pages<B, TilesY, TilesX>();
    const std::uint32_t address   = llk_lib::dataflow_buffer_wait_front<Thread::Unpack, buffer_params<B>, pages>();
    return Tensor<owned_by<buffer_tensor_spec<B, TilesY, TilesX>, Thread::Unpack, Exu::Unpacker>, PopFront<B, pages>>(address);
}

/// TilesY x TilesX free tiles of B at the write pointer, once the consumer has freed them. The packer's; release pushes.
template <const DataflowBufferSpec& B, std::uint32_t TilesY = 1, std::uint32_t TilesX = B.page_bytes / buffer_tile_bytes<B>>
inline __attribute__((always_inline)) auto reserve_back()
{
    constexpr std::uint32_t pages = buffer_pages<B, TilesY, TilesX>();
    const std::uint32_t address   = llk_lib::dataflow_buffer_reserve_back<Thread::Pack, buffer_params<B>, pages>();
    return Tensor<owned_by<buffer_tensor_spec<B, TilesY, TilesX>, Thread::Pack, Exu::Packer>, PushBack<B, pages>>(address);
}

} // namespace llk2
