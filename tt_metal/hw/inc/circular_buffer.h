// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/common_runtime_address_map.h"
#include "risc_attribs.h"
#include <array>
#include <cstdint>

// The command queue read interface controls reads from the issue region, host owns the issue region write interface
// Commands and data to send to device are pushed into the issue region
struct CQReadInterface {
    uint32_t issue_fifo_size;
    uint32_t issue_fifo_limit; // range is inclusive of the limit
    uint32_t issue_fifo_rd_ptr;
    uint32_t issue_fifo_rd_toggle;
};

// The command queue write interface controls writes to the completion region, host owns the completion region read interface
// Data requests from device and event states are written to the completion region
struct CQWriteInterface {
    uint32_t completion_fifo_size;
    uint32_t completion_fifo_limit; // range is inclusive of the limit
    uint32_t completion_fifo_wr_ptr;
    uint32_t completion_fifo_wr_toggle;
};

struct CBInterface {
    uint32_t fifo_size;
    uint32_t fifo_limit; // range is inclusive of the limit
    uint32_t fifo_page_size;
    uint32_t fifo_num_pages;

    uint32_t fifo_rd_ptr;
    uint32_t fifo_wr_ptr;

    // Save a cycle during init by writing 0 to the uint32 below
    union {
        uint32_t tiles_acked_received_init;
        struct {
            uint16_t tiles_acked;
            uint16_t tiles_received;
        };
    };

    // used by packer for in-order packing
    uint32_t fifo_wr_tile_ptr;
};

extern CBInterface cb_interface[NUM_CIRCULAR_BUFFERS];

// NCRISC and BRISC setup read and write
// TRISC sets up read or write
inline void setup_cb_read_write_interfaces(uint32_t tt_l1_ptr *cb_l1_base, uint32_t start_cb_index, uint32_t max_cb_index, bool read, bool write, bool init_wr_tile_ptr) {

    constexpr uint32_t WORDS_PER_CIRCULAR_BUFFER_CONFIG = 4;

    volatile tt_l1_ptr uint32_t* circular_buffer_config_addr = cb_l1_base + start_cb_index * WORDS_PER_CIRCULAR_BUFFER_CONFIG;

    // #7493 : this needs to be densely packed
    for (uint32_t cb_id = start_cb_index; cb_id < max_cb_index; cb_id++) {

        // NOTE: fifo_addr, fifo_size and fifo_limit in 16B words!
        uint32_t fifo_addr = circular_buffer_config_addr[0];
        uint32_t fifo_size = circular_buffer_config_addr[1];
        uint32_t fifo_num_pages = circular_buffer_config_addr[2];
        uint32_t fifo_page_size = circular_buffer_config_addr[3];
        uint32_t fifo_limit = fifo_addr + fifo_size;

        cb_interface[cb_id].fifo_limit = fifo_limit;  // to check if we need to wrap
        if (write) {
            cb_interface[cb_id].fifo_wr_ptr = fifo_addr;
        }
        if (read) {
            cb_interface[cb_id].fifo_rd_ptr = fifo_addr;
        }
        cb_interface[cb_id].fifo_size = fifo_size;
        cb_interface[cb_id].tiles_acked_received_init = 0;
        if (write) {
            cb_interface[cb_id].fifo_num_pages = fifo_num_pages;
        }
        cb_interface[cb_id].fifo_page_size = fifo_page_size;

        if (init_wr_tile_ptr) {
            cb_interface[cb_id].fifo_wr_tile_ptr = 0;
        }

        circular_buffer_config_addr += WORDS_PER_CIRCULAR_BUFFER_CONFIG;
    }
}

size_t calculate_max_cb_index(uint32_t mask) {
    if (mask == 0) return 0;
    return 31 - __builtin_clz(mask) + 1;
}

struct array_mask{
  std::array<int, 32> arr;
  size_t size{0};
};

array_mask create_array_mask(uint32_t mask) {
    array_mask result;
    result.arr.fill(-1);

    while (mask != 0) {
        int bit_pos = __builtin_ctz(mask);
        result.arr[result.size] = bit_pos;
        result.size++;
        mask &= (mask - 1);
    }

    return result;
}

class CBSetIterator {
public:
    using value_type = int;
    using difference_type = std::ptrdiff_t;
    using pointer = const int*;
    using reference = const int&;
    using iterator_category = std::input_iterator_tag;

    CBSetIterator(uint32_t mask) : mask(mask), current_bit(find_next()) {}
    CBSetIterator() : mask(0), current_bit(-1) {}

    int operator*() const {
        return current_bit;
    }

    CBSetIterator& operator++() {
        mask &= (mask - 1);
        current_bit = find_next();
        return *this;
    }

    CBSetIterator operator++(int) {
        CBSetIterator temp = *this;
        ++(*this);
        return temp;
    }

    bool operator==(const CBSetIterator& other) const {
        return current_bit == other.current_bit;
    }

    bool operator!=(const CBSetIterator& other) const {
        return !(*this == other);
    }

private:
    uint32_t mask;
    int current_bit;

    int find_next() const {
        if (mask == 0) return -1;
        return __builtin_ctz(mask);
    }
};

class CBSet {
public:
    CBSet(uint32_t mask) : mask(mask) {}

    CBSetIterator begin() const {
        return CBSetIterator(mask);
    }

    CBSetIterator end() const {
        return CBSetIterator();
    }

private:
    uint32_t mask;
};
