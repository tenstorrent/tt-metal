// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "tests/tt_metal/tt_fabric/feature_bringup/kernels/fabric_elastic_channels.hpp"

// What do I need to get the next chunk:
// address? (could be an ID instead, in which case it could be packed in a single word)

// What happens if a worker is granted a chunk but it doesn't use it and then it needs to be relinquished to the pool?


// protocol is as follows:

// 1 bit to indicate new value
// value to indicate rest of it

struct SenderChannelView {
    static constexpr uint32_t NEXT_CHUNK_VALID = 1 << 31;
    static constexpr uint32_t NEXT_CHUNK_VALUE_MASK = NEXT_CHUNK_VALID - 1;
    volatile uint32_t *next_chunk_ptr;

    SenderChannelView(volatile uint32_t *next_chunk_ptr) : next_chunk_ptr(next_chunk_ptr) {}

    FORCE_INLINE void wait_for_new_chunk() {
        while (!*next_chunk_ptr) {
        }
    }

    FORCE_INLINE bool has_new_chunk() {
        return *next_chunk_ptr & NEXT_CHUNK_VALID;
    }

    FORCE_INLINE void clear_new_chunk_flag() {
        *next_chunk_ptr = 0;
    }

    FORCE_INLINE uint32_t get_next_chunk() {
        uint32_t value = *next_chunk_ptr;
        return value & NEXT_CHUNK_VALUE_MASK;
    }
};


// Used by the worker to know where to send packets to next
template <size_t N_CHUNKS, size_t CHUNK_N_PKTS>
struct FabricWriterAdapter {
    SenderChannelView sender_channel_view;
    tt::tt_fabric::OnePassIterator<size_t*, CHUNK_N_PKTS> current_chunk;

    FabricWriterAdapter(volatile uint32_t *next_chunk_ptr) : 
        sender_channel_view(next_chunk_ptr), current_chunk() {}

    FORCE_INLINE bool has_valid_destination() {
        return !current_chunk.is_done();
    }

    FORCE_INLINE void advance_to_next_buffer_slot() {
        current_chunk.increment();
    }

    FORCE_INLINE bool new_chunk_is_available() {
        return sender_channel_view.has_new_chunk();
    }

    FORCE_INLINE size_t* get_next_write_address() const {
        return current_chunk.get_current_ptr();
    }
    

    // return true if the new chunk was updated
    FORCE_INLINE void update_to_new_chunk() {
        //...
        ASSERT(sender_channel_view.has_new_chunk());
        ASSERT(current_chunk.is_done());
        auto chunk_base_address = sender_channel_view.get_next_chunk();
        auto new_chunk_base_address = chunk_base_address;
        sender_channel_view.clear_new_chunk_flag();
        current_chunk.reset_to(new_chunk_base_address);
    }
};


void kernel_main() {
    constexpr size_t N_CHUNKS = get_compile_time_arg_val(0);
    constexpr size_t CHUNK_N_PKTS = get_compile_time_arg_val(1);

    size_t arg_idx = 0;
    size_t n_pkts = get_arg_val<size_t>(arg_idx++);
    size_t src_addr = get_arg_val<size_t>(arg_idx++);
    size_t dest_eth_noc_x = get_arg_val<size_t>(arg_idx++);
    size_t dest_eth_noc_y = get_arg_val<size_t>(arg_idx++);
    size_t payload_size = get_arg_val<size_t>(arg_idx++);

    volatile uint32_t *next_chunk_ptr = get_semaphore<CoreType::WORKER>(get_arg_val<uint32_t>(arg_idx++));
    volatile uint32_t *from_eth_flow_control_ptr = get_semaphore<CoreType::ETH>(get_arg_val<uint32_t>(arg_idx++));
    uint32_t to_eth_flow_control_addr = get_semaphore<CoreType::ETH>(get_arg_val<uint32_t>(arg_idx++));

    
    FabricWriterAdapter<N_CHUNKS, CHUNK_N_PKTS> fabric_writer_adapter(next_chunk_ptr);

    const uint64_t dest_sem_noc_addr = get_noc_addr(dest_eth_noc_x, dest_eth_noc_y, to_eth_flow_control_addr);
    size_t pkts_sent = 0;
    while (pkts_sent < n_pkts) {
        if (fabric_writer_adapter.has_valid_destination()) {
            auto dest_bank_addr = fabric_writer_adapter.get_next_write_address();
            auto dest_noc_addr = get_noc_addr(dest_noc_x, dest_noc_y, dest_bank_addr);
            noc_async_write(src_addr, dest_noc_addr, payload_size);
            noc_semaphore_inc(dest_sem_noc_addr, 1);

            fabric_writer_adapter.advance_to_next_buffer_slot();
            pkts_sent++;
        } else if (fabric_writer_adapter.new_chunk_is_available()) {
            fabric_writer_adapter.update_to_new_chunk();
        }
    }
}
