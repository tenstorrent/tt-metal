// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: Tensors from a kernel's dataflow buffers. Every TRISC runs these calls; there is no llk-lib part,
// because the buffer itself does each call only on the thread that owns that side of it (see "Ownership" below).
//
//   const auto x    = llk2::wait_front<x_spec>(in0);     // unpack waits for x's pages; x at the read pointer
//   const auto bias = llk2::peek_front<bias_spec>(b);    // the same, but release() does not pop: pop_front later
//   const auto slot = llk2::reserve_back<out_spec>(out); // pack waits for free pages; slot at the write pointer
//   ...
//   llk2::release(x);                                    // unpack pops x's pages
//   llk2::release(out);                                  // pack pushes the pages the op wrote (pack returns slot's)
//   llk2::pop_front<bias_spec>(b);                       // unpack pops a peeked tensor's pages
//
// A wait or reservation that may not be taken: wait_front, peek_front, peek_front_rows and reserve_back with a last
// argument `when`. With when false nothing is waited for, release() does nothing, and the Tensor must not be used: for
// a tensor that only some passes of a loop have (as pack's run-time accumulate argument is for the accumulation only
// some passes do). In the two peeks `when` is a default argument (their release does nothing either way); wait_front
// and reserve_back are two overloads each, because the release of the conditional form has to remember whether it
// was taken (BufferPopIf / BufferPushIf), and the unconditional form's Tensor should not carry that. `when` is a bool
// and nothing else: an integer there is a count (below), and does not compile for a spec without a run-time size.
//
//   const auto next = llk2::reserve_back<row_spec>(scores, has_next);   // the last pass has no next row
//   if (has_next) { ... pack(s, next); }
//   llk2::release(next);                                               // pushes only if it was reserved
//
// A buffer only some builds of a kernel have (the mask's, for an op that may have none): buffer_if<Buffer, Has> is
// Buffer with Has true and otherwise an AbsentBuffer, an object with nothing in it, made from the same argument. The
// kernel declares every buffer in one place either way, and no code is made for the ones its build has no use for
// (constructing a tt-metal DataflowBuffer is not free). Every call of this file takes an AbsentBuffer: the Tensor it
// returns is absent_front<S>() or absent_back<S>(), nothing is waited for, release() and pop_front do nothing, and
// the Tensor must not be used. It is what the kernel hands a helper in place of the operand of a step that
// `if constexpr (Has)` leaves out.
//
//   llk2::buffer_if<DataflowBuffer, has_mask> mask_in(mask_buffer_id);
//   const auto mask = llk2::wait_front<mask_spec>(mask_in);              // absent without has_mask
//
// A tensor read as it arrives: peek_front_rows<S, N>(buffer, r0) waits only for S's first r0 + N tile rows and returns
// those N rows; pop_front<S>(buffer) frees all of S at the end.
//
//   const auto q_rows = llk2::peek_front_rows<q_spec, G>(q_in, r0);     // the reader sends Q a row at a time
//
// A tensor whose size is known only at run time (tensor/tensor.h, "run-time sizes"): a spec that marks its tile rows or
// columns run time (runtime_rows<S> / runtime_cols<S>, tensor/tensor.h) takes the count as the call's argument, and the
// pages waited for, reserved, popped and pushed follow it. The count is an integer and nothing else: a bool there is
// a `when`, and does not compile for a spec with a run-time size (the conditional form is not ported for one).
//
//   const auto x    = llk2::wait_front<runtime_cols<x_spec>>(in0, n);      // n tiles of a row of at most x_spec's
//   const auto slot = llk2::reserve_back<runtime_cols<out_spec>>(out, n);
//
// A scalar the kernel's control flow needs from a buffer (a chunk's start index, a range of blocks) is read from a
// Tensor at the buffer's front with read_value (compute_api/sync.h): it has to reach all three threads. Such a page
// holds words, not a tile: its spec is words_spec<N, F>, N words of format F in one page. No op takes it.
//
//   const auto range = llk2::wait_front<llk2::words_spec<4, DataFormat::Int32>>(range_in);
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
#include <type_traits>

#include "llk_assert.h"
#include "tensor/tensor.h"

namespace llk2
{

// ---- pages and addresses -----------------------------------------------------------------------------------------

/// Pages a tensor with spec S takes in a dataflow buffer: one tile per page.
template <const Spec& S>
inline constexpr std::uint16_t buffer_pages = tile_count(S.shape);

/// The bound of the size S marks run time.
template <const Spec& S>
inline constexpr std::uint32_t runtime_bound = is_runtime_size(S, TileY) ? S.shape.sizes[TileY] : S.shape.sizes[TileX];

/// Pages of a tensor of spec S, which marks one tile-grid size run time, holding `count` along that dim: `count` tile
/// rows or tile columns, checked against S's bound.
template <const Spec& S, class Count>
inline __attribute__((always_inline)) std::uint16_t checked_runtime_pages(Count count)
{
    static_assert(
        is_runtime_size(S, TileY) || S.shape.sizes[TileY] == 1,
        "a run-time number of tile columns: of one tile row (with several, the rows' step in the buffer would be the run-time width)");
    LLK_ASSERT(count >= 1 && static_cast<std::uint32_t>(count) <= runtime_bound<S>, "a run-time size is outside 1..its bound");
    return static_cast<std::uint16_t>(static_cast<std::uint32_t>(count) * (is_runtime_size(S, TileY) ? S.shape.sizes[TileX] : S.shape.sizes[TileY]));
}

/// The byte address of the pages at the front of buffer: its read pointer, which counts 16-byte words. Valid on the
/// unpack thread only.
template <class Buffer>
inline __attribute__((always_inline)) std::uint32_t front_address(Buffer& buffer)
{
    return buffer.get_read_ptr() * l1_word_bytes;
}

/// The byte address of the pages at the back of buffer: its write pointer, which counts 16-byte words. Valid on the
/// pack thread only.
template <class Buffer>
inline __attribute__((always_inline)) std::uint32_t back_address(Buffer& buffer)
{
    return buffer.get_write_ptr() * l1_word_bytes;
}

/// The spec of one page of N words of format F (32-bit integers), for read_value: a 1 x N grid of datums that is one
/// "tile", so the page is the spec's one page and its bytes the spec's tile bytes.
template <std::uint32_t N, DataFormat F>
inline constexpr Spec words_spec = spec_object<laid_out(Dims {1, 1, 1, 1, 1, N}, F, Storage::L1)>;

// ---- buffers ---------------------------------------------------------------------------------------------------------

/// What buffer_if is without the buffer: made from a buffer's id, holding nothing.
struct AbsentBuffer
{
    constexpr explicit AbsentBuffer(std::uint32_t)
    {
    }
};

/// Buffer in a build of the kernel that has it (Has), AbsentBuffer in one that does not.
template <class Buffer, bool Has>
using buffer_if = std::conditional_t<Has, Buffer, AbsentBuffer>;

/// Whether a buffer object is one the build has: false for an AbsentBuffer.
template <class Buffer>
inline constexpr bool is_present = !std::is_same_v<Buffer, AbsentBuffer>;

/// The tensor of spec S at the front of a buffer the build does not have: nothing is waited for, release() does
/// nothing and it must not be used. The unpackers', as the tensor it stands for would be.
template <const Spec& S>
constexpr auto absent_front()
{
    static_assert(S.storage == Storage::L1, "a buffer's pages are in L1");
    return Tensor<owned_by<S, Thread::Unpack, Exu::Unpacker>>(0);
}

/// As absent_front, at the back: the packers'.
template <const Spec& S>
constexpr auto absent_back()
{
    static_assert(S.storage == Storage::L1, "a buffer's pages are in L1");
    return Tensor<owned_by<S, Thread::Pack, Exu::Packer>>(0);
}

// ---- the read side: the unpackers' ---------------------------------------------------------------------------------

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

/// Waits for S's pages at the front of buffer (unpack) and returns the tensor there, the unpackers'. release() pops.
template <const Spec& S, class Buffer>
inline __attribute__((always_inline)) auto wait_front(Buffer& buffer)
{
    static_assert(S.storage == Storage::L1, "a buffer's pages are in L1");
    static_assert(!has_runtime_size(S), "wait_front<S>(buffer): S marks a run-time size: pass the count, wait_front<S>(buffer, count)");
    constexpr std::uint16_t pages = buffer_pages<S>;
    buffer.wait_front(pages);
    return Tensor<owned_by<S, Thread::Unpack, Exu::Unpacker>, BufferPop<Buffer, pages>>(front_address(buffer), BufferPop<Buffer, pages> {&buffer});
}

/// wait_front on a buffer the build does not have, with or without `when`: absent_front<S>().
template <const Spec& S>
constexpr auto wait_front(AbsentBuffer&, bool = true)
{
    return absent_front<S>();
}

/// As BufferPop, for a wait that may not have been taken: pops only if it was.
template <class Buffer, std::uint16_t Pages>
struct BufferPopIf
{
    Buffer* buffer; ///< the buffer the pages are in
    bool taken;     ///< whether the pages were waited for

    /// Unpack: frees the pages once the unpackers have read them, if there are any.
    void operator()() const
    {
        if (taken)
        {
            buffer->pop_front(Pages);
        }
    }
};

/// wait_front, taken only when `when`: otherwise nothing is waited for, release() does nothing and the Tensor must
/// not be used. A bool, and only a bool, is `when`: any other integer is a count and takes the overload below.
template <const Spec& S, class Buffer>
inline __attribute__((always_inline)) auto wait_front(Buffer& buffer, bool when)
{
    static_assert(S.storage == Storage::L1, "a buffer's pages are in L1");
    static_assert(
        !has_runtime_size(S), "wait_front(buffer, when): a spec with a run-time size takes its count, an integer; not ported: a conditional wait for it");
    constexpr std::uint16_t pages = buffer_pages<S>;
    if (when)
    {
        buffer.wait_front(pages);
    }
    return Tensor<owned_by<S, Thread::Unpack, Exu::Unpacker>, BufferPopIf<Buffer, pages>>(front_address(buffer), BufferPopIf<Buffer, pages> {&buffer, when});
}

/// As BufferPop, for a tensor with a run-time size: pops the pages it was made with.
template <class Buffer>
struct BufferPopCount
{
    Buffer* buffer;      ///< the buffer the pages are in
    std::uint16_t pages; ///< the tensor's pages

    /// Unpack: frees the pages once the unpackers have read them.
    void operator()() const
    {
        buffer->pop_front(pages);
    }
};

/// wait_front for a spec with a run-time size (runtime_rows / runtime_cols): `count` of it, 1..the spec's. Waits for
/// that many tile rows' or columns' pages; release() pops them.
template <const Spec& S, class Buffer, class Count>
inline __attribute__((always_inline)) auto wait_front(Buffer& buffer, Count count)
{
    static_assert(std::is_integral_v<Count>, "wait_front(buffer, count): the count is an integer (a bool is `when`)");
    static_assert(S.storage == Storage::L1, "a buffer's pages are in L1");
    static_assert(has_runtime_size(S), "wait_front(buffer, count): the spec marks no run-time size (runtime_rows<S>, runtime_cols<S>); `when` is a bool");
    const std::uint16_t pages = checked_runtime_pages<S>(count);
    buffer.wait_front(pages);
    return Tensor<owned_by<S, Thread::Unpack, Exu::Unpacker>, BufferPopCount<Buffer>>(front_address(buffer), BufferPopCount<Buffer> {&buffer, pages}, count);
}

/// As wait_front, but release() does not pop: the pages stay for later calls until pop_front<S>(buffer). With `when`
/// false nothing is waited for and the Tensor must not be used.
template <const Spec& S, class Buffer>
inline __attribute__((always_inline)) auto peek_front(Buffer& buffer, bool when = true)
{
    static_assert(S.storage == Storage::L1, "a buffer's pages are in L1");
    static_assert(!has_runtime_size(S), "peek_front<S>(buffer): not ported for a spec with a run-time size: wait_front<S>(buffer, count)");
    if (when)
    {
        buffer.wait_front(buffer_pages<S>);
    }
    return Tensor<owned_by<S, Thread::Unpack, Exu::Unpacker>>(front_address(buffer));
}

/// peek_front on a buffer the build does not have: absent_front<S>().
template <const Spec& S>
constexpr auto peek_front(AbsentBuffer&, bool = true)
{
    return absent_front<S>();
}

/// N tile rows of S, from its tile row r0, at the front of buffer (rows<N> of S there): waits (unpack) for S's
/// pages up to the end of those rows only, so a tensor whose rows arrive one after the other can be read as it comes.
/// The one wait whose page count is a run-time value. release() does nothing: pop_front<S>(buffer) once all of S has
/// been read. With `when` false nothing is waited for and the Tensor must not be used.
template <const Spec& S, std::uint32_t N, class Buffer>
inline __attribute__((always_inline)) auto peek_front_rows(Buffer& buffer, std::uint32_t r0, bool when = true)
{
    static_assert(S.storage == Storage::L1, "a buffer's pages are in L1");
    static_assert(!has_runtime_size(S), "peek_front_rows<S, N>(buffer, r0): not ported for a spec with a run-time size");
    if (when)
    {
        buffer.wait_front((r0 + N) * S.shape.sizes[TileX]);
    }
    return rows<N>(Tensor<owned_by<S, Thread::Unpack, Exu::Unpacker>>(front_address(buffer)), r0);
}

/// peek_front_rows on a buffer the build does not have: those rows of absent_front<S>().
template <const Spec& S, std::uint32_t N>
constexpr auto peek_front_rows(AbsentBuffer&, std::uint32_t r0, bool = true)
{
    return rows<N>(absent_front<S>(), r0);
}

/// Pops S's pages from the front of buffer (unpack): the end of a peek_front or of the peek_front_rows of all of S.
template <const Spec& S, class Buffer>
inline __attribute__((always_inline)) void pop_front(Buffer& buffer)
{
    static_assert(
        !has_runtime_size(S), "pop_front<S>: S holds a bound, not the pages a tensor of it has: release() the tensor wait_front(buffer, count) returned");
    buffer.pop_front(buffer_pages<S>);
}

/// pop_front on a buffer the build does not have: nothing.
template <const Spec& S>
constexpr void pop_front(AbsentBuffer&)
{
}

// ---- the write side: the packers' ----------------------------------------------------------------------------------

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

/// Waits for S's pages free at the back of buffer (pack) and returns the tensor there, the packers'. release() pushes.
template <const Spec& S, class Buffer>
inline __attribute__((always_inline)) auto reserve_back(Buffer& buffer)
{
    static_assert(S.storage == Storage::L1, "a buffer's pages are in L1");
    static_assert(!has_runtime_size(S), "reserve_back<S>(buffer): S marks a run-time size: pass the count, reserve_back<S>(buffer, count)");
    constexpr std::uint16_t pages = buffer_pages<S>;
    buffer.reserve_back(pages);
    return Tensor<owned_by<S, Thread::Pack, Exu::Packer>, BufferPush<Buffer, pages>>(back_address(buffer), BufferPush<Buffer, pages> {&buffer});
}

/// reserve_back on a buffer the build does not have, with or without `when`: absent_back<S>().
template <const Spec& S>
constexpr auto reserve_back(AbsentBuffer&, bool = true)
{
    return absent_back<S>();
}

/// As BufferPush, for a reservation that may not have been taken: pushes only if it was.
template <class Buffer, std::uint16_t Pages>
struct BufferPushIf
{
    Buffer* buffer; ///< the buffer the pages are in
    bool taken;     ///< whether the pages were reserved

    /// Pack: hands the pages on once the packer has written them, if there are any.
    void operator()() const
    {
        if (taken)
        {
            buffer->push_back(Pages);
        }
    }
};

/// reserve_back, taken only when `when`: otherwise nothing is waited for, release() does nothing and the Tensor
/// must not be used. A bool, and only a bool, is `when`: any other integer is a count and takes the overload below.
template <const Spec& S, class Buffer>
inline __attribute__((always_inline)) auto reserve_back(Buffer& buffer, bool when)
{
    static_assert(S.storage == Storage::L1, "a buffer's pages are in L1");
    static_assert(
        !has_runtime_size(S),
        "reserve_back(buffer, when): a spec with a run-time size takes its count, an integer; not ported: a conditional reservation for it");
    constexpr std::uint16_t pages = buffer_pages<S>;
    if (when)
    {
        buffer.reserve_back(pages);
    }
    return Tensor<owned_by<S, Thread::Pack, Exu::Packer>, BufferPushIf<Buffer, pages>>(back_address(buffer), BufferPushIf<Buffer, pages> {&buffer, when});
}

/// As BufferPush, for a tensor with a run-time size: pushes the pages it was made with.
template <class Buffer>
struct BufferPushCount
{
    Buffer* buffer;      ///< the buffer the pages are in
    std::uint16_t pages; ///< the tensor's pages

    /// Pack: hands the pages on once the packer has written them.
    void operator()() const
    {
        buffer->push_back(pages);
    }
};

/// reserve_back for a spec with a run-time size (runtime_rows / runtime_cols): `count` of it, 1..the spec's. Waits for
/// that many tile rows' or columns' pages free; release() pushes them.
template <const Spec& S, class Buffer, class Count>
inline __attribute__((always_inline)) auto reserve_back(Buffer& buffer, Count count)
{
    static_assert(std::is_integral_v<Count>, "reserve_back(buffer, count): the count is an integer (a bool is `when`)");
    static_assert(S.storage == Storage::L1, "a buffer's pages are in L1");
    static_assert(has_runtime_size(S), "reserve_back(buffer, count): the spec marks no run-time size (runtime_rows<S>, runtime_cols<S>); `when` is a bool");
    const std::uint16_t pages = checked_runtime_pages<S>(count);
    buffer.reserve_back(pages);
    return Tensor<owned_by<S, Thread::Pack, Exu::Packer>, BufferPushCount<Buffer>>(back_address(buffer), BufferPushCount<Buffer> {&buffer, pages}, count);
}

} // namespace llk2
