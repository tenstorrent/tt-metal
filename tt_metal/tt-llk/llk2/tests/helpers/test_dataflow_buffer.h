// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TEST ONLY. The tt-llk harness has no tt-metal, so this stands in for tt-metal's DataflowBuffer on the compute
// cores of Blackhole (tt_metal/hw/inc/api/dataflow/dataflow_buffer.h, tt_metal/hw/inc/internal/tt-1xx/
// dataflow_buffer.inl), with the same method names and the same per-thread behavior:
//
//   wait_front(n)     unpack thread: poll tiles_received until n pages are there (llk_wait_tiles). Elsewhere nothing
//   pop_front(n)      unpack thread: read pointer on by n pages; tiles_acked += n, written once the unpackers are idle
//                     (llk_pop_tiles). Elsewhere nothing
//   reserve_back(n)   pack thread: poll tiles_acked until n pages are free (llk_wait_for_free_tiles). Elsewhere nothing
//   push_back(n)      pack thread: write pointer on by n pages; tiles_received += n, written once the packer is idle
//                     (llk_push_tiles). Elsewhere nothing
//   get_read_ptr()    unpack: the read pointer; get_write_ptr() pack: the write pointer. Both in 16-byte words, as on a
//                     TRISC in tt-metal (cb_addr_shift = 4); 0 on math. Other threads' values are not meaningful
//
// The counters are the stream registers tt-metal uses (tiles_received = STREAM_REMOTE_DEST_BUF_SIZE, tiles_acked =
// STREAM_REMOTE_DEST_BUF_START of stream id). The STOREREG that writes them goes through the HAL; polling them is a
// volatile load, as tt-metal's reg_read (the HAL has no RISC-V MMIO read).
//
// What only the harness needs (tt-metal's firmware and data movement kernels do it there):
//   - the constructor takes the buffer's base address, page size and page count, and sets up this thread's side
//     (tt-metal: setup_local_cb_read_write_interfaces from the host's CB config, read side on unpack, write side
//     on pack). DataflowBuffer(id), as a tt-metal kernel constructs one, reads them from buffer_configs[id], which
//     the test fills first;
//   - preload(pages): the consumer (unpack) writes the counters as if a producer had pushed `pages` pages already in
//     L1 (tt-metal: a reader kernel pushes them);
//   - zero_counters(): the producer (pack) writes both counters 0 (tt-metal: the firmware's init_sync_registers
//     before the kernel starts). A buffer both pushed and popped by the kernel also needs every thread to wait
//     until that is done (harness_barrier) before using it; tt-metal's firmware does it before releasing the kernel.

#pragma once

#include <cstdint>

#include "barrier.h"
#include "ckernel.h"
#include "hal/move.h"
#include "hal/sync.h"

// hal/scalar.h has an LLK_ASSERT inside a constexpr function, which C++17 flags under ENABLE_LLK_ASSERT.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wc++20-extensions"
#include "hal/scalar.h"
#pragma GCC diagnostic pop

namespace llk2_test
{

inline constexpr std::uint32_t l1_word_bytes = 16; ///< buffer pointers and page sizes count 16-byte words on a TRISC

// Stream registers, from tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_overlay_parameters.h and
// stream_io_map.h (not on the tt-llk include path).
inline constexpr std::uint32_t noc_overlay_start_addr   = 0xFFB40000; ///< NOC_OVERLAY_START_ADDR: stream 0's registers
inline constexpr std::uint32_t stream_reg_space_bytes   = 0x1000;     ///< NOC_STREAM_REG_SPACE_SIZE: one stream's registers
inline constexpr std::uint32_t stream_reg_bytes         = 4;          ///< registers are 32-bit words
inline constexpr std::uint32_t tiles_received_reg_index = 10;         ///< STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX
inline constexpr std::uint32_t tiles_acked_reg_index    = 8;          ///< STREAM_REMOTE_DEST_BUF_START_REG_INDEX
inline constexpr std::uint32_t counter_mask             = 0xffff;     ///< counters are 16 bits and wrap

/// The GPR each owner stages its counter in before STOREREG: a temp llk2's unpack does not use (it uses TMP0), and
/// the one llk_push_to_brisc uses on pack (llk2's pack uses TMP0).
inline constexpr std::uint32_t pop_counter_gpr  = ckernel::p_gpr_unpack::TMP1;
inline constexpr std::uint32_t push_counter_gpr = ckernel::p_gpr_pack::NUM_MSGS_RECEIVED;

#if defined(LLK_TRISC_UNPACK)
inline constexpr bool on_unpack = true, on_pack = false;
#elif defined(LLK_TRISC_PACK)
inline constexpr bool on_unpack = false, on_pack = true;
#else
inline constexpr bool on_unpack = false, on_pack = false;
#endif

/// TEST ONLY: every thread waits here until all three have arrived, so counters one thread zeroed are zero for all.
inline void harness_barrier()
{
    llk_barrier::rendezvous(llk_barrier::is_action_thread());
}

/// TEST ONLY: one buffer's host config (tt-metal: the CB config the host writes and the firmware reads).
struct BufferConfig
{
    std::uint32_t base_address; ///< bytes
    std::uint32_t page_bytes;   ///< one page
    std::uint32_t pages;        ///< capacity
};

inline constexpr std::uint32_t max_buffers = 32; ///< buffer ids 0..31, as tt-metal's NUM_CIRCULAR_BUFFERS

/// TEST ONLY: every buffer's host config, by id; the test fills it before the kernel constructs its buffers.
inline BufferConfig buffer_configs[max_buffers] {};

/// TEST ONLY: tt-metal's DataflowBuffer on a Blackhole TRISC (see the top of the file).
class DataflowBuffer
{
public:
    /// This thread's side of buffer id, from buffer_configs[id] (tt-metal: the firmware, from the host's CB config).
    explicit DataflowBuffer(std::uint16_t id) :
        DataflowBuffer(id, buffer_configs[id].base_address, buffer_configs[id].page_bytes, buffer_configs[id].pages)
    {
    }

    /// This thread's side of buffer id at base_address (bytes): pages of page_bytes each (tt-metal: the firmware).
    DataflowBuffer(std::uint16_t id, std::uint32_t base_address, std::uint32_t page_bytes, std::uint32_t pages) :
        id_(id), fifo_page_size_(page_bytes / l1_word_bytes), fifo_num_pages_(pages)
    {
        fifo_size_  = fifo_page_size_ * pages;
        fifo_limit_ = base_address / l1_word_bytes + fifo_size_;
        if constexpr (on_unpack)
        {
            fifo_rd_ptr_ = base_address / l1_word_bytes;
        }
        if constexpr (on_pack)
        {
            fifo_wr_ptr_ = base_address / l1_word_bytes;
        }
    }

    /// Unpack: waits until n pages are in the buffer.
    void wait_front(std::uint16_t n) const
    {
        if constexpr (on_unpack)
        {
            // received (register) - acked (local), in 16 bits, as llk_wait_tiles.
            while (static_cast<std::uint16_t>(read_counter(tiles_received_address()) - tiles_acked_) < n)
            {
            }
        }
    }

    /// Unpack: frees the n oldest pages, once the unpackers have read them.
    void pop_front(std::uint16_t n)
    {
        if constexpr (on_unpack)
        {
            tiles_acked_ += n;
            advance(fifo_rd_ptr_, n);
            write_counter<pop_counter_gpr, hal::sync::StallCondition::Unpacker0Idle | hal::sync::StallCondition::Unpacker1Idle>(
                tiles_acked_address(), tiles_acked_);
        }
    }

    /// Pack: waits until n pages are free.
    void reserve_back(std::uint16_t n) const
    {
        if constexpr (on_pack)
        {
            // capacity - (received (local) - acked (register)), in 16 bits, as llk_wait_for_free_tiles.
            while (static_cast<std::uint16_t>(fifo_num_pages_ - static_cast<std::uint16_t>(tiles_received_ - read_counter(tiles_acked_address()))) < n)
            {
            }
        }
    }

    /// Pack: hands n pages to the consumer, once the packer has written them.
    void push_back(std::uint16_t n)
    {
        if constexpr (on_pack)
        {
            tiles_received_ += n;
            advance(fifo_wr_ptr_, n);
            write_counter<push_counter_gpr, hal::sync::StallCondition::PackerIdle>(tiles_received_address(), tiles_received_);
        }
    }

    /// The read pointer in 16-byte words on unpack; 0 on math.
    std::uint32_t get_read_ptr() const
    {
        return on_unpack ? fifo_rd_ptr_ : 0;
    }

    /// The write pointer in 16-byte words on pack; 0 on math.
    std::uint32_t get_write_ptr() const
    {
        return on_pack ? fifo_wr_ptr_ : 0;
    }

    /// TEST ONLY, unpack (the consumer): the first `pages` pages are in L1 and count as pushed.
    void preload(std::uint16_t pages) const
    {
        if constexpr (on_unpack)
        {
            write_register(tiles_acked_address(), 0);
            write_register(tiles_received_address(), pages);
        }
    }

    /// TEST ONLY, pack (the producer): both counters 0, the buffer empty.
    void zero_counters() const
    {
        if constexpr (on_pack)
        {
            write_register(tiles_acked_address(), 0);
            write_register(tiles_received_address(), 0);
        }
    }

private:
    std::uint32_t tiles_received_address() const
    {
        return noc_overlay_start_addr + id_ * stream_reg_space_bytes + tiles_received_reg_index * stream_reg_bytes;
    }

    std::uint32_t tiles_acked_address() const
    {
        return noc_overlay_start_addr + id_ * stream_reg_space_bytes + tiles_acked_reg_index * stream_reg_bytes;
    }

    static std::uint16_t read_counter(std::uint32_t address)
    {
        return static_cast<std::uint16_t>(*reinterpret_cast<volatile std::uint32_t*>(address) & counter_mask);
    }

    static void write_register(std::uint32_t address, std::uint32_t value)
    {
        *reinterpret_cast<volatile std::uint32_t*>(address) = value;
    }

    /// SETDMAREG, STALLWAIT until Wait's units are idle, STOREREG: the counter moves only after the data does.
    template <std::uint32_t Gpr, hal::sync::StallCondition Wait>
    static void write_counter(std::uint32_t address, std::uint16_t value)
    {
        namespace move = hal::move;
        hal::scalar::set(hal::gpr<Gpr>(), value);
        hal::sync::wait::stall<hal::sync::StallTarget::Thcon, Wait>();
        move::run(move::Transfer<move::Gpr, move::Mmio> {
            .source = hal::gpr<Gpr>(), .destination = {.addressing = move::MmioAddressing::Immediate, .byte_address = address}});
    }

    void advance(std::uint32_t& ptr, std::uint16_t n) const
    {
        ptr += n * fifo_page_size_;
        if (ptr >= fifo_limit_)
        {
            ptr -= fifo_size_;
        }
    }

    std::uint16_t id_;                 ///< buffer id, also its stream id
    std::uint32_t fifo_page_size_;     ///< one page, words
    std::uint32_t fifo_num_pages_;     ///< capacity, pages
    std::uint32_t fifo_size_;          ///< capacity, words
    std::uint32_t fifo_limit_;         ///< first word past the buffer
    std::uint32_t fifo_rd_ptr_    = 0; ///< unpack only
    std::uint32_t fifo_wr_ptr_    = 0; ///< pack only
    std::uint16_t tiles_acked_    = 0; ///< unpack only: pages popped
    std::uint16_t tiles_received_ = 0; ///< pack only: pages pushed
};

} // namespace llk2_test
