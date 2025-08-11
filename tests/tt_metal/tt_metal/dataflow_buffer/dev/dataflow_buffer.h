// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <vector>
#include <array>

#include "dfb_register.h"
#include "dfb_test_common.hpp"

namespace tt::tt_metal {

    enum DataflowBufferAccessPattern : uint8_t {
        NONE,
        STRIDED,
        BLOCKED,
        GLOBAL,  // no producer, all data is resident
    };
    

namespace dev {

// riscs would have to set this up based on how host config, similar to how we currently set up local cb interface

// each thread should have an instance of this
struct local_dfb_interface_t {
    uint32_t size;
    uintptr_t limit;  // check for wraps
    uint32_t page_size;
    uint32_t num_pages;

    // how to make these multiple rd and wr ptrs?
    uintptr_t rd_ptr;
    uintptr_t wr_ptr;

    DataflowBufferAccessPattern wapt = DataflowBufferAccessPattern::NONE;
    DataflowBufferAccessPattern rapt = DataflowBufferAccessPattern::NONE;
};

extern thread_local local_dfb_interface_t overlay_cluster_dfb_access_pattern_tracker[64];
// Internal mapping table - maps DFB index to register indices. Index from DFB "logial index" and the uint64_t is the
// register assignment mask
extern uint64_t dfb_to_register_allocation[64];

// Internal helper functions for register management
namespace internal {
// Helper function to convert register mask to vector of indices
std::vector<uint8_t> mask_to_indices(uint64_t register_mask);

// Helper function to iterate over set bits in register mask
template <typename Func>
void for_each_register(uint64_t register_mask, Func func) {
    uint64_t mask = register_mask;
    while (mask != 0) {
        uint8_t reg_idx = __builtin_ctzll(mask);  // Count trailing zeros to find first set bit
        func(reg_idx);
        mask &= ~(1ULL << reg_idx);  // Clear the bit we just processed
    }
}
}  // namespace internal

// also templated on whether its intra cluster?
// this is what user interacts with
template <DataflowBufferAccessPattern WAPT, DataflowBufferAccessPattern RAPT>
class DataflowBuffer {
public:
    constexpr DataflowBuffer<WAPT, RAPT>(uint8_t index) : index(index) {
        // Validate index range
        assert(index < 64);

        // Set access patterns in the interface tracker
        assert(
            overlay_cluster_dfb_access_pattern_tracker[index].wapt == WAPT ||
            overlay_cluster_dfb_access_pattern_tracker[index].wapt == NONE);
        assert(
            overlay_cluster_dfb_access_pattern_tracker[index].rapt == RAPT ||
            overlay_cluster_dfb_access_pattern_tracker[index].rapt == NONE);
        overlay_cluster_dfb_access_pattern_tracker[index].wapt = WAPT;
        overlay_cluster_dfb_access_pattern_tracker[index].rapt = RAPT;
    }

    ~DataflowBuffer() {
        // Clean up access patterns
        overlay_cluster_dfb_access_pattern_tracker[index].wapt = NONE;
        overlay_cluster_dfb_access_pattern_tracker[index].rapt = NONE;
    }

    // Get the DataflowBuffer index (user-facing)
    uint8_t get_index() const { return index; }

    void reserve_back(uint32_t num_pages) {
        auto& out = tt::tt_metal::get_thread_output_stream();
        uint8_t register_index = this->get_register_index(tt::tt_metal::current_thread_id);
        out << "TID " << (uint32_t)current_thread_id << " RB reg idx " << std::dec << (uint32_t)register_index << " cap " << std::dec
                  << overlay_cluster_instances[register_index].get_space_avail() << std::endl;
        while (overlay_cluster_instances[register_index].get_space_avail() < num_pages);
    }

    void push_back(uint32_t num_pages) {
        auto& out = tt::tt_metal::get_thread_output_stream();
        uint8_t register_index = this->get_register_index(tt::tt_metal::current_thread_id);
        overlay_cluster_instances[register_index].inc_pages_posted(num_pages);
        overlay_cluster_dfb_access_pattern_tracker[register_index].wr_ptr +=
            (num_pages * overlay_cluster_dfb_access_pattern_tracker[register_index].page_size);
        if (overlay_cluster_dfb_access_pattern_tracker[register_index].wr_ptr ==
            overlay_cluster_dfb_access_pattern_tracker[register_index].limit) {
            overlay_cluster_dfb_access_pattern_tracker[register_index].wr_ptr -=
                overlay_cluster_dfb_access_pattern_tracker[register_index].size;
        }
        out << "TID " << (uint32_t)current_thread_id << " PB pages avail: " << std::dec
                  << overlay_cluster_instances[register_index].get_pages_avail() << std::endl;
    }

    void wait_front(uint32_t num_pages) {
        auto& out = tt::tt_metal::get_thread_output_stream();
        uint8_t register_index = this->get_register_index(tt::tt_metal::current_thread_id);
        out << "TID " << (uint32_t)current_thread_id << " WF pages avail: " << std::dec
                  << overlay_cluster_instances[register_index].get_pages_avail() << std::endl;
        while (overlay_cluster_instances[register_index].get_pages_avail() < num_pages);
    }

    void pop_front(uint32_t num_pages) {
        auto& out = tt::tt_metal::get_thread_output_stream();
        uint8_t register_index = this->get_register_index(tt::tt_metal::current_thread_id);
        overlay_cluster_instances[register_index].inc_pages_acked(num_pages);
        overlay_cluster_dfb_access_pattern_tracker[register_index].rd_ptr +=
            (num_pages * overlay_cluster_dfb_access_pattern_tracker[register_index].page_size);
        if (overlay_cluster_dfb_access_pattern_tracker[register_index].rd_ptr ==
            overlay_cluster_dfb_access_pattern_tracker[register_index].limit) {
            overlay_cluster_dfb_access_pattern_tracker[register_index].rd_ptr -=
                overlay_cluster_dfb_access_pattern_tracker[register_index].size;
        }
        out << "TID " << (uint32_t)current_thread_id << " PF space avail: " << std::dec
        << overlay_cluster_instances[register_index].get_space_avail() << std::endl;
    }

    bool pages_reservable_at_back(DataflowBuffer<WAPT, RAPT>& dfb, uint32_t num_pages) {
        uint8_t register_index = this->get_register_index(tt::tt_metal::current_thread_id);
        return overlay_cluster_instances[register_index].get_space_avail() >= num_pages;
    }

    bool pages_available_at_front(DataflowBuffer<WAPT, RAPT>& dfb, uint32_t num_pages) {
        uint8_t register_index = this->get_register_index(tt::tt_metal::current_thread_id);
        return overlay_cluster_instances[register_index].get_pages_avail() >= num_pages;
    }

private:
    uint8_t get_register_index(uint8_t thread_id) const {
        uint64_t register_mask = dfb_to_register_allocation[index];
        uint8_t first_set_bit = __builtin_ctzll(register_mask);
        // registers allocated sequentially
        return first_set_bit + thread_id;
    }

    uint8_t index;
};

// To be hidden:
// get_write_ptr -> gets address so data can be read over noc and written to addr this returns
// get_read_ptr -> gets address so data can be read from addr this returns and sent over noc

}  // namespace dev

} // namespace tt::tt_metal
