// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, dataflow buffers (circular buffers) on Blackhole: how a buffer's pages, pointers and counters are read
// and signalled. The arch-independent API on top is tensor/dataflow_buffer.h.
//
// A dataflow buffer is a ring of pages in L1 shared by a producer and a consumer. Two 16-bit counters count pages:
//   tiles_received   pages the producer has pushed; in stream register STREAM_REMOTE_DEST_BUF_SIZE of the
//                    buffer's stream (stream id = buffer id)
//   tiles_acked      pages the consumer has popped; in STREAM_REMOTE_DEST_BUF_START of the same stream
// Those are the registers today's llk_wait_tiles, llk_pop_tiles, llk_wait_for_free_tiles and llk_push_tiles use
// (tt_metal/hw/ckernels/blackhole/metal/llk_io/, tt_metal/hw/inc/internal/tt-1xx/blackhole/stream_io_map.h).
// Each thread also keeps a local copy of the buffer: the read and write pointers and its own count of the pages it
// has acked and received (dataflow_buffer_interface, the fields of tt-metal's LocalCBInterface).
//
// The calls, in the order they run:
//   dataflow_buffer_params_from(spec)               spec -> dataflow_buffer_params (compile time)
//   dataflow_buffer_setup<P>(base)                  every thread: pointers at base, local counts 0
//   dataflow_buffer_wait_front<On, P, Pages>()      owner (On) only: poll tiles_received until Pages are there.
//                                                   Every thread: returns the read pointer as a byte address
//   dataflow_buffer_pop_front<On, P, Pages>()       every thread: acked += Pages, read pointer on by Pages, wraps.
//                                                   Owner only: wait for its unpackers to finish, then write acked
//   dataflow_buffer_reserve_back<On, P, Pages>()    owner only: poll tiles_acked until Pages are free.
//                                                   Every thread: returns the write pointer as a byte address
//   dataflow_buffer_push_back<On, P, Pages>()       every thread: received += Pages, write pointer on, wraps.
//                                                   Owner only: wait for its packer to finish, then write received
//
// Every thread keeps the same pointers. Today only the unpack thread moves the read pointer and only the pack
// thread the write pointer. Here every thread moves both on every call, and only the owner waits or writes a
// counter register. Each thread starts from the same setup, makes the same calls in the same order (every TRISC
// runs the same kernel body) with the same compile-time page counts, and the pointer update depends only on those,
// never on what the owner read from a register. So after every call the pointers are equal on all three threads,
// and a Tensor from a buffer has the same address on all of them. Only the owner waits, so the other threads may
// run ahead; they must not touch the pages through that address unless the op on their side synchronizes first.
//
// Ordering. A pop must not reach the producer before the unpackers have read the pages, and a push must not reach
// the consumer before the packer has written them. Both write their counter with STOREREG behind a STALLWAIT that
// holds the scalar unit (THCON) until the unpackers or the packer are idle, as llk_pop_tiles and llk_push_to_brisc do.
//
// Not HAL: reading a stream register from the RISC-V (a volatile load, as tt-metal's reg_read) and writing one
// from the RISC-V (only the harness's dataflow_buffer_set_counters does). The HAL covers the Tensix side: SETDMAREG
// (hal::scalar::set), STALLWAIT (hal::sync::stall) and STOREREG (hal::move, GPR to MMIO).

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "hal/move.h"
#include "hal/sync.h"
#include "llk_lib/thread.h"
#include "tensor/tensor.h" // l1_word_bytes

// hal/scalar.h has an LLK_ASSERT inside a constexpr function, which C++17 flags under ENABLE_LLK_ASSERT.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wc++20-extensions"
#include "hal/scalar.h"
#pragma GCC diagnostic pop

namespace llk2::llk_lib
{

// ---- constants -----------------------------------------------------------------------------------------------------
//
// Buffer pointers and page sizes count L1 words (l1_word_bytes, tensor/tensor.h), as in LocalCBInterface.

// Stream registers, from tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_overlay_parameters.h (not on the tt-llk
// include path).
inline constexpr std::uint32_t noc_overlay_start_addr   = 0xFFB40000; ///< NOC_OVERLAY_START_ADDR: stream 0's registers
inline constexpr std::uint32_t stream_reg_space_bytes   = 0x1000;     ///< NOC_STREAM_REG_SPACE_SIZE: one stream's registers
inline constexpr std::uint32_t stream_reg_bytes         = 4;          ///< registers are 32-bit words
inline constexpr std::uint32_t tiles_received_reg_index = 10;         ///< STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX
inline constexpr std::uint32_t tiles_acked_reg_index    = 8;          ///< STREAM_REMOTE_DEST_BUF_START_REG_INDEX
inline constexpr std::uint32_t operand_start_stream     = 0;          ///< OPERAND_START_STREAM: buffer N uses stream N
inline constexpr std::uint32_t max_dataflow_buffers     = 64;         ///< NUM_CIRCULAR_BUFFERS on Blackhole
inline constexpr std::uint32_t counter_mask             = 0xffff;     ///< counters are 16 bits and wrap

/// The GPR each owner stages its counter in before STOREREG. Unpack: a temp the unpack llk-lib does not use (it
/// uses TMP0). Pack: the one llk_push_to_brisc uses.
inline constexpr std::uint32_t pop_counter_gpr  = ckernel::p_gpr_unpack::TMP1;
inline constexpr std::uint32_t push_counter_gpr = ckernel::p_gpr_pack::NUM_MSGS_RECEIVED;

// ---- parameters ----------------------------------------------------------------------------------------------------

/// A buffer as llk-lib needs it. Compile time; the base address is given to setup at run time.
struct dataflow_buffer_params
{
    std::uint32_t id;         ///< buffer id, also its stream id
    std::uint32_t page_words; ///< one page, in 16-byte words
    std::uint32_t pages;      ///< capacity in pages
};

/// Any buffer spec with id, page_bytes and pages (tensor/dataflow_buffer.h's DataflowBufferSpec) as llk-lib params.
template <class Spec>
constexpr dataflow_buffer_params dataflow_buffer_params_from(const Spec& b)
{
    return {b.id, b.page_bytes / l1_word_bytes, b.pages};
}

/// One thread's copy of a buffer's state; the fields of tt-metal's LocalCBInterface that the four calls use.
/// Pointers are in 16-byte words.
struct dataflow_buffer_interface
{
    std::uint32_t fifo_size;      ///< capacity in words
    std::uint32_t fifo_limit;     ///< first word past the buffer
    std::uint32_t fifo_rd_ptr;    ///< read pointer: the oldest page not popped
    std::uint32_t fifo_wr_ptr;    ///< write pointer: the first page not pushed
    std::uint16_t tiles_acked;    ///< pages popped so far, as counted by this thread
    std::uint16_t tiles_received; ///< pages pushed so far, as counted by this thread
};

/// Each TRISC has its own copy (each is its own binary), one per buffer used. tt-metal keeps the same state in
/// cb_interface[id], filled by the firmware from the host's CB config.
template <std::uint32_t Id>
inline dataflow_buffer_interface dataflow_buffer_state {};

/// Byte address of a buffer's tiles_received register.
constexpr std::uint32_t tiles_received_address(std::uint32_t id)
{
    return noc_overlay_start_addr + (operand_start_stream + id) * stream_reg_space_bytes + tiles_received_reg_index * stream_reg_bytes;
}

/// Byte address of a buffer's tiles_acked register.
constexpr std::uint32_t tiles_acked_address(std::uint32_t id)
{
    return noc_overlay_start_addr + (operand_start_stream + id) * stream_reg_space_bytes + tiles_acked_reg_index * stream_reg_bytes;
}

/// Reads a counter register from the RISC-V. Not HAL: a volatile load, as tt-metal's reg_read.
inline __attribute__((always_inline)) std::uint16_t read_stream_counter(std::uint32_t address)
{
    return static_cast<std::uint16_t>(*reinterpret_cast<volatile std::uint32_t*>(address) & counter_mask);
}

/// Writes a counter register from the owner's Tensix stream once Wait's units are idle: SETDMAREG, STALLWAIT, STOREREG.
template <std::uint32_t Address, std::uint32_t Gpr, hal::sync::StallCondition Wait>
inline __attribute__((always_inline)) void write_stream_counter(std::uint16_t value)
{
    namespace move = hal::move;
    constexpr move::Transfer<move::Gpr, move::Mmio> store {
        .source = hal::gpr<Gpr>(), .destination = {.addressing = move::MmioAddressing::Immediate, .byte_address = Address}};
    hal::scalar::set(hal::gpr<Gpr>(), value);
    hal::sync::wait::stall<hal::sync::StallTarget::Thcon, Wait>();
    move::run<store>();
}

/// Moves a pointer on by Words and wraps it at the end of the buffer.
inline __attribute__((always_inline)) void advance_fifo_ptr(std::uint32_t& ptr, std::uint32_t words, const dataflow_buffer_interface& b)
{
    ptr += words;
    if (ptr >= b.fifo_limit)
    {
        ptr -= b.fifo_size;
    }
}

// ---- calls ---------------------------------------------------------------------------------------------------------

/// Every thread: the buffer starts at base_address (bytes, 16-byte aligned), empty as far as this thread has counted.
template <const auto& B>
inline __attribute__((always_inline)) void dataflow_buffer_setup(std::uint32_t base_address)
{
    constexpr dataflow_buffer_params p = dataflow_buffer_params_from(B);
    static_assert(p.id < max_dataflow_buffers, "a buffer id is a stream id: 0 to 63");
    dataflow_buffer_interface& b = dataflow_buffer_state<p.id>;
    b.fifo_size                  = p.page_words * p.pages;
    b.fifo_rd_ptr                = base_address / l1_word_bytes;
    b.fifo_wr_ptr                = b.fifo_rd_ptr;
    b.fifo_limit                 = b.fifo_rd_ptr + b.fifo_size;
    b.tiles_acked                = 0;
    b.tiles_received             = 0;
}

/// On thread On only: writes both counter registers from the RISC-V. Not part of the buffer protocol: in tt-metal
/// the firmware zeroes them and the producer's pushes move them. The harness uses it to start a buffer empty, or
/// full of pages it preloaded.
template <Thread On, const auto& B>
inline __attribute__((always_inline)) void dataflow_buffer_set_counters(std::uint16_t acked, std::uint16_t received)
{
    if constexpr (runs_on<On>)
    {
        constexpr std::uint32_t id                                             = dataflow_buffer_params_from(B).id;
        *reinterpret_cast<volatile std::uint32_t*>(tiles_acked_address(id))    = acked;
        *reinterpret_cast<volatile std::uint32_t*>(tiles_received_address(id)) = received;
    }
}

/// Owner (On) waits until Pages are in the buffer; every thread returns the read pointer as a byte address.
template <Thread On, const auto& B, std::uint32_t Pages>
inline __attribute__((always_inline)) std::uint32_t dataflow_buffer_wait_front()
{
    constexpr dataflow_buffer_params p = dataflow_buffer_params_from(B);
    const dataflow_buffer_interface& b = dataflow_buffer_state<p.id>;
    if constexpr (runs_on<On>)
    {
        // Same 16-bit arithmetic as llk_wait_tiles: pages in the buffer = received (register) - acked (local).
        while (static_cast<std::uint16_t>(read_stream_counter(tiles_received_address(p.id)) - b.tiles_acked) < Pages)
        {
        }
    }
    return b.fifo_rd_ptr * l1_word_bytes;
}

/// Every thread moves the read pointer on by Pages; the owner (On, the unpack thread) also tells the producer,
/// once its unpackers have read the pages.
template <Thread On, const auto& B, std::uint32_t Pages>
inline __attribute__((always_inline)) void dataflow_buffer_pop_front()
{
    constexpr dataflow_buffer_params p = dataflow_buffer_params_from(B);
    dataflow_buffer_interface& b       = dataflow_buffer_state<p.id>;
    b.tiles_acked += Pages;
    advance_fifo_ptr(b.fifo_rd_ptr, Pages * p.page_words, b);
    if constexpr (runs_on<On>)
    {
        using hal::sync::StallCondition;
        write_stream_counter<tiles_acked_address(p.id), pop_counter_gpr, StallCondition::Unpacker0Idle | StallCondition::Unpacker1Idle>(b.tiles_acked);
    }
}

/// Owner (On) waits until Pages are free; every thread returns the write pointer as a byte address.
template <Thread On, const auto& B, std::uint32_t Pages>
inline __attribute__((always_inline)) std::uint32_t dataflow_buffer_reserve_back()
{
    constexpr dataflow_buffer_params p = dataflow_buffer_params_from(B);
    const dataflow_buffer_interface& b = dataflow_buffer_state<p.id>;
    if constexpr (runs_on<On>)
    {
        // Same 16-bit arithmetic as llk_wait_for_free_tiles: free = capacity - (received (local) - acked (register)).
        while (static_cast<std::uint16_t>(p.pages - static_cast<std::uint16_t>(b.tiles_received - read_stream_counter(tiles_acked_address(p.id)))) < Pages)
        {
        }
    }
    return b.fifo_wr_ptr * l1_word_bytes;
}

/// Every thread moves the write pointer on by Pages; the owner (On, the pack thread) also tells the consumer, once
/// its packer has written the pages.
template <Thread On, const auto& B, std::uint32_t Pages>
inline __attribute__((always_inline)) void dataflow_buffer_push_back()
{
    constexpr dataflow_buffer_params p = dataflow_buffer_params_from(B);
    dataflow_buffer_interface& b       = dataflow_buffer_state<p.id>;
    b.tiles_received += Pages;
    advance_fifo_ptr(b.fifo_wr_ptr, Pages * p.page_words, b);
    if constexpr (runs_on<On>)
    {
        write_stream_counter<tiles_received_address(p.id), push_counter_gpr, hal::sync::StallCondition::PackerIdle>(b.tiles_received);
    }
}

} // namespace llk2::llk_lib
