// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// GlobalCircularBuffer sender and receiver helpers for CCEs, whose receivers are other CCEs.
//
// A CCE has no NoC, so these replace remote_circular_buffer.h's NoC paths with loads and stores: the
// sender writes FIFO data and credits into each receiver's TL1 through cce_sram_address, the receiver
// stores its acks into the sender's TL1 the same way, and each side reads words another agent wrote
// through its own uncached alias. Data stores, then `fence w, w`, then an absolute credit store; FIFO
// reads, then `fence r, w`, then an absolute ack store. Layouts are dram_receiver_config_page.hpp and
// dram_sender_state_block.hpp; counts are kDramReceiverCreditUnitBytes units, and the FIFO geometry
// (page-aligned limit, wrap padding, resize padding) is the Blackhole remote CB's.
//
// Bounded waits take a budget in rdcycle cycles, where 0 never expires, and return WaitResult. They
// leave the interface unchanged when they expire.

#pragma once

#include <stdint.h>

#include "internal/risc_attribs.h"

#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DRISC)

#include "dev_mem_map.h"
#include "experimental/cce_gddr.h"
#include "internal/circular_buffer_interface.h"
#include "tt_metal/impl/buffers/dram_receiver_config_page.hpp"
#include "tt_metal/impl/buffers/dram_sender_state_block.hpp"

namespace experimental::cce_remote_cb {

static_assert(REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE == tt::tt_metal::kDramReceiverCreditUnitBytes);

inline constexpr uint32_t kCreditUnitBytes = tt::tt_metal::kDramReceiverCreditUnitBytes;
inline constexpr uint32_t kCreditStrideWords = tt::tt_metal::kDramReceiverCreditStrideBytes / sizeof(uint32_t);

enum class WaitResult : uint32_t { Ready = 0, Expired = 1, Stopped = 2 };

FORCE_INLINE uint64_t read_cycle_counter() {
    uint64_t cycles;
    asm volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
}

// Counts a budget from construction. A budget of 0 never expires.
class WaitDeadline {
public:
    explicit WaitDeadline(uint64_t budget_cycles) :
        budget_cycles_(budget_cycles), start_cycles_(budget_cycles == 0 ? 0 : read_cycle_counter()) {}

    bool expired() const { return budget_cycles_ != 0 && read_cycle_counter() - start_cycles_ >= budget_cycles_; }

private:
    uint64_t budget_cycles_;
    uint64_t start_cycles_;
};

// A word of this CCE's TL1 that another agent writes or reads.
FORCE_INLINE volatile uint32_t* own_tl1_word(uint32_t tl1_offset) {
    return reinterpret_cast<volatile uint32_t*>(static_cast<uint64_t>(tl1_offset) + MEM_L1_UNCACHED_BASE);
}

FORCE_INLINE volatile uint32_t* peer_tl1_word(uint32_t cce_index, uint32_t tl1_offset) {
    return reinterpret_cast<volatile uint32_t*>(cce_sram_address(cce_index, tl1_offset));
}

// Credit units a transfer of `bytes` at `fifo_ptr` covers: it also covers the gap past the page-aligned
// limit when it reaches that limit.
FORCE_INLINE uint32_t fifo_credit_units(
    uint32_t fifo_ptr, uint32_t bytes, uint32_t fifo_start_addr, uint32_t fifo_size, uint32_t fifo_limit_page_aligned) {
    if (fifo_ptr + bytes >= fifo_limit_page_aligned) {
        bytes += fifo_start_addr + fifo_size - fifo_limit_page_aligned;
    }
    return bytes / kCreditUnitBytes;
}

// The FIFO pointer after a transfer of `bytes` at `fifo_ptr`.
FORCE_INLINE uint32_t
fifo_advance(uint32_t fifo_ptr, uint32_t bytes, uint32_t fifo_start_addr, uint32_t fifo_limit_page_aligned) {
    const uint32_t next_fifo_ptr = fifo_ptr + bytes;
    return next_fifo_ptr >= fifo_limit_page_aligned ? fifo_start_addr + (next_fifo_ptr - fifo_limit_page_aligned)
                                                    : next_fifo_ptr;
}

// The padding, in credit units, that moves `fifo_ptr` to the first `page_size` boundary of the resized
// FIFO, and that boundary.
struct FifoResize {
    uint32_t fifo_ptr;
    uint32_t fifo_limit_page_aligned;
    uint32_t padding_units;
};

FORCE_INLINE FifoResize
fifo_resize(uint32_t fifo_ptr, uint32_t page_size, uint32_t fifo_start_addr, uint32_t fifo_size) {
    FifoResize resize;
    resize.fifo_limit_page_aligned = fifo_start_addr + fifo_size - fifo_size % page_size;
    const uint32_t offset = fifo_ptr - fifo_start_addr;
    const uint32_t next_fifo_ptr = fifo_start_addr + ((offset + page_size - 1) / page_size) * page_size;
    if (next_fifo_ptr >= resize.fifo_limit_page_aligned) {
        resize.fifo_ptr = fifo_start_addr;
        resize.padding_units = (fifo_start_addr + fifo_size - fifo_ptr) / kCreditUnitBytes;
    } else {
        resize.fifo_ptr = next_fifo_ptr;
        resize.padding_units = (next_fifo_ptr - fifo_ptr) / kCreditUnitBytes;
    }
    return resize;
}

struct Receiver {
    uint32_t config_addr;
    uint32_t fifo_start_addr;
    uint32_t fifo_size;
    uint32_t fifo_limit_page_aligned;
    uint32_t fifo_page_size;
    uint32_t fifo_rd_ptr;
    uint32_t pages_acked;
    volatile uint32_t* pages_sent;
    volatile uint32_t* local_pages_acked;
    volatile uint32_t* remote_pages_acked;
};

// Loads the config page at `config_addr` and this receiver's rd_ptr and acked count. Call
// resize_receiver_bounded before the first wait.
FORCE_INLINE Receiver receiver_init(uint32_t config_addr) {
    volatile uint32_t* config_page = own_tl1_word(config_addr);
    Receiver receiver;
    receiver.config_addr = config_addr;
    receiver.fifo_start_addr = config_page[tt::tt_metal::kDramReceiverFifoStartAddrOffset / sizeof(uint32_t)];
    receiver.fifo_size = config_page[tt::tt_metal::kDramReceiverFifoSizeOffset / sizeof(uint32_t)];
    receiver.fifo_limit_page_aligned = receiver.fifo_start_addr + receiver.fifo_size;
    receiver.fifo_page_size = kCreditUnitBytes;
    receiver.fifo_rd_ptr = config_page[tt::tt_metal::kDramReceiverFifoRdPtrOffset / sizeof(uint32_t)];
    receiver.pages_acked = config_page[tt::tt_metal::kDramReceiverPagesAckedOffset / sizeof(uint32_t)];
    receiver.pages_sent = own_tl1_word(config_addr + tt::tt_metal::kDramReceiverCreditLineOffset);
    receiver.local_pages_acked = own_tl1_word(config_addr + tt::tt_metal::kDramReceiverPagesAckedOffset);
    receiver.remote_pages_acked = peer_tl1_word(
        config_page[tt::tt_metal::kDramReceiverSenderIndexOffset / sizeof(uint32_t)],
        config_page[tt::tt_metal::kDramReceiverRemotePagesAckedAddrOffset / sizeof(uint32_t)]);
    return receiver;
}

FORCE_INLINE uint32_t receiver_credit_units(const Receiver& receiver, uint32_t num_pages) {
    return fifo_credit_units(
        receiver.fifo_rd_ptr,
        num_pages * receiver.fifo_page_size,
        receiver.fifo_start_addr,
        receiver.fifo_size,
        receiver.fifo_limit_page_aligned);
}

FORCE_INLINE uint32_t received_units(const Receiver& receiver) { return *receiver.pages_sent - receiver.pages_acked; }

// Whether `num_pages` pages are at the front, without waiting.
FORCE_INLINE bool front_ready(const Receiver& receiver, uint32_t num_pages) {
    return received_units(receiver) >= receiver_credit_units(receiver, num_pages);
}

// The page at the front, read through the uncached alias.
FORCE_INLINE const volatile uint32_t* front_words(const Receiver& receiver) {
    return own_tl1_word(receiver.fifo_rd_ptr);
}

FORCE_INLINE void publish_acked(Receiver& receiver, uint32_t units) {
    receiver.pages_acked += units;
    *receiver.local_pages_acked = receiver.pages_acked;
    asm volatile("fence r, w" ::: "memory");
    *receiver.remote_pages_acked = receiver.pages_acked;
}

FORCE_INLINE void pop_front(Receiver& receiver, uint32_t num_pages) {
    const uint32_t bytes = num_pages * receiver.fifo_page_size;
    const uint32_t units = receiver_credit_units(receiver, num_pages);
    receiver.fifo_rd_ptr =
        fifo_advance(receiver.fifo_rd_ptr, bytes, receiver.fifo_start_addr, receiver.fifo_limit_page_aligned);
    publish_acked(receiver, units);
}

// Moves rd_ptr to the new page size's next boundary. The padding is acked only once the sender has
// credited it, so acked never passes sent; that wait is the bounded part, and it also ends with Stopped
// once `stop_requested()` holds. Either way the interface is unchanged.
template <typename StopRequested>
FORCE_INLINE WaitResult
resize_receiver_bounded(Receiver& receiver, uint32_t page_size, uint64_t budget_cycles, StopRequested stop_requested) {
    const FifoResize resize =
        fifo_resize(receiver.fifo_rd_ptr, page_size, receiver.fifo_start_addr, receiver.fifo_size);
    if (resize.padding_units != 0) {
        const WaitDeadline deadline(budget_cycles);
        while (received_units(receiver) < resize.padding_units) {
            if (stop_requested()) {
                return WaitResult::Stopped;
            }
            if (deadline.expired()) {
                return WaitResult::Expired;
            }
        }
        publish_acked(receiver, resize.padding_units);
    }
    receiver.fifo_rd_ptr = resize.fifo_ptr;
    receiver.fifo_limit_page_aligned = resize.fifo_limit_page_aligned;
    receiver.fifo_page_size = page_size;
    return WaitResult::Ready;
}

// Writes rd_ptr back to the config page, so the next kernel on this GCB resumes there.
FORCE_INLINE void store_receiver_rd_ptr(const Receiver& receiver) {
    *own_tl1_word(receiver.config_addr + tt::tt_metal::kDramReceiverFifoRdPtrOffset) = receiver.fifo_rd_ptr;
}

struct Sender {
    uint32_t state_addr;
    uint32_t fifo_start_addr;
    uint32_t fifo_size;
    uint32_t fifo_limit_page_aligned;
    uint32_t fifo_wr_ptr;
    uint32_t num_receivers;
    // Offset of the credit line, the same on every receiver.
    uint32_t remote_pages_sent_addr;
    // Receiver r's counters are kCreditStrideWords * r words in.
    volatile uint32_t* pages_sent;
    volatile uint32_t* pages_acked;
    // {receiver CCE index, 0} per receiver.
    volatile uint32_t* receiver_table;
    // The receiver with the least free space at the last min_free, so a caller can report whose acks it waited on.
    uint32_t slowest_receiver;
};

// Word `field_offset` of a sender state block. Word accesses, not the packed struct's, which the compiler
// may split into byte accesses.
FORCE_INLINE volatile uint32_t* state_block_word(uint32_t state_addr, uint32_t field_offset) {
    return own_tl1_word(state_addr + field_offset);
}

// Loads the sender state block the host wrote at `state_addr`. Call resize_sender before the first write.
FORCE_INLINE Sender sender_init(uint32_t state_addr) {
    using tt::tt_metal::DramSenderStateBlock;
    Sender sender;
    sender.state_addr = state_addr;
    sender.fifo_start_addr = *state_block_word(state_addr, offsetof(DramSenderStateBlock, fifo_start_addr));
    sender.fifo_size = *state_block_word(state_addr, offsetof(DramSenderStateBlock, fifo_size_per_receiver));
    sender.fifo_limit_page_aligned = sender.fifo_start_addr + sender.fifo_size;
    sender.fifo_wr_ptr = *state_block_word(state_addr, offsetof(DramSenderStateBlock, fifo_wr_ptr));
    sender.num_receivers = *state_block_word(state_addr, offsetof(DramSenderStateBlock, num_receivers));
    sender.remote_pages_sent_addr = remote_cb_remote_pages_sent_ptr(
        *state_block_word(state_addr, offsetof(DramSenderStateBlock, num_receivers_and_remote_pages_sent_ptr)));
    const uint32_t pages_sent_addr =
        *state_block_word(state_addr, offsetof(DramSenderStateBlock, aligned_pages_sent_ptr));
    sender.pages_sent = own_tl1_word(pages_sent_addr);
    sender.pages_acked =
        own_tl1_word(pages_sent_addr + tt::tt_metal::dram_receiver_sender_pages_acked_offset(sender.num_receivers));
    sender.receiver_table =
        own_tl1_word(*state_block_word(state_addr, offsetof(DramSenderStateBlock, receiver_noc_xy_ptr)));
    sender.slowest_receiver = 0;
    return sender;
}

FORCE_INLINE uint32_t receiver_cce_index(const Sender& sender, uint32_t receiver) {
    return sender.receiver_table[2 * receiver];
}

// Receiver `receiver`'s FIFO at wr_ptr, for the data stores.
FORCE_INLINE volatile uint32_t* receiver_back_words(const Sender& sender, uint32_t receiver) {
    return peer_tl1_word(receiver_cce_index(sender, receiver), sender.fifo_wr_ptr);
}

FORCE_INLINE uint32_t sender_credit_units(const Sender& sender, uint32_t bytes) {
    return fifo_credit_units(
        sender.fifo_wr_ptr, bytes, sender.fifo_start_addr, sender.fifo_size, sender.fifo_limit_page_aligned);
}

FORCE_INLINE uint32_t receiver_pages_sent(const Sender& sender, uint32_t receiver) {
    return sender.pages_sent[kCreditStrideWords * receiver];
}

FORCE_INLINE uint32_t receiver_pages_acked(const Sender& sender, uint32_t receiver) {
    return sender.pages_acked[kCreditStrideWords * receiver];
}

// Free credit units at the most backed-up receiver, without waiting, and that receiver in slowest_receiver.
// Clamped, because resize padding can credit past a free-space check. The fence keeps the next FIFO stores
// behind the ack reads.
FORCE_INLINE uint32_t min_free(Sender& sender) {
    const uint32_t ring_units = sender.fifo_size / kCreditUnitBytes;
    uint32_t min_free_units = ring_units;
    sender.slowest_receiver = 0;
    for (uint32_t receiver = 0; receiver < sender.num_receivers; ++receiver) {
        const uint32_t outstanding = receiver_pages_sent(sender, receiver) - receiver_pages_acked(sender, receiver);
        const uint32_t free_units = outstanding >= ring_units ? 0u : ring_units - outstanding;
        if (free_units < min_free_units) {
            min_free_units = free_units;
            sender.slowest_receiver = receiver;
        }
    }
    asm volatile("fence r, w" ::: "memory");
    return min_free_units;
}

// Waits until every receiver has room for `bytes` at wr_ptr, wrap padding included. On expiry,
// slowest_receiver names a receiver that was short.
FORCE_INLINE WaitResult reserve_back_bounded(Sender& sender, uint32_t bytes, uint64_t budget_cycles) {
    const uint32_t units = sender_credit_units(sender, bytes);
    if (min_free(sender) >= units) {
        return WaitResult::Ready;
    }
    const WaitDeadline deadline(budget_cycles);
    while (min_free(sender) < units) {
        if (deadline.expired()) {
            return WaitResult::Expired;
        }
    }
    return WaitResult::Ready;
}

// Credits `units` to every receiver: absolute counts, behind every earlier data store.
FORCE_INLINE void publish_sent(const Sender& sender, uint32_t units) {
    asm volatile("fence w, w" ::: "memory");
    for (uint32_t receiver = 0; receiver < sender.num_receivers; ++receiver) {
        const uint32_t pages_sent = sender.pages_sent[kCreditStrideWords * receiver] + units;
        sender.pages_sent[kCreditStrideWords * receiver] = pages_sent;
        *peer_tl1_word(receiver_cce_index(sender, receiver), sender.remote_pages_sent_addr) = pages_sent;
    }
}

// Credits `bytes_per_receiver` the caller wrote at wr_ptr on every receiver, and advances wr_ptr.
FORCE_INLINE void finalize(Sender& sender, uint32_t bytes_per_receiver) {
    const uint32_t units = sender_credit_units(sender, bytes_per_receiver);
    publish_sent(sender, units);
    sender.fifo_wr_ptr =
        fifo_advance(sender.fifo_wr_ptr, bytes_per_receiver, sender.fifo_start_addr, sender.fifo_limit_page_aligned);
}

// Moves wr_ptr to the new page size's next boundary and credits the padding.
FORCE_INLINE void resize_sender(Sender& sender, uint32_t page_size) {
    const FifoResize resize = fifo_resize(sender.fifo_wr_ptr, page_size, sender.fifo_start_addr, sender.fifo_size);
    if (resize.padding_units != 0) {
        publish_sent(sender, resize.padding_units);
    }
    sender.fifo_wr_ptr = resize.fifo_ptr;
    sender.fifo_limit_page_aligned = resize.fifo_limit_page_aligned;
}

// Waits until every receiver has acked everything this sender credited.
FORCE_INLINE WaitResult barrier_bounded(const Sender& sender, uint64_t budget_cycles) {
    const WaitDeadline deadline(budget_cycles);
    for (uint32_t receiver = 0; receiver < sender.num_receivers; ++receiver) {
        while (sender.pages_acked[kCreditStrideWords * receiver] != sender.pages_sent[kCreditStrideWords * receiver]) {
            if (deadline.expired()) {
                return WaitResult::Expired;
            }
        }
    }
    return WaitResult::Ready;
}

// Writes wr_ptr back to the state block, so the next request to this GCB resumes there.
FORCE_INLINE void store_sender_wr_ptr(const Sender& sender) {
    *state_block_word(sender.state_addr, offsetof(tt::tt_metal::DramSenderStateBlock, fifo_wr_ptr)) =
        sender.fifo_wr_ptr;
}

}  // namespace experimental::cce_remote_cb

#endif
