// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ckernel::trisc
{

constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS                          = 32;
constexpr static std::uint32_t UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG = 4;
constexpr static std::uint32_t CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT            = 4;

struct LocalCBInterface
{
    uint32_t fifo_size;
    uint32_t fifo_limit; // range is inclusive of the limit
    uint32_t fifo_page_size;
    uint32_t fifo_num_pages;

    uint32_t fifo_rd_ptr;
    uint32_t fifo_wr_ptr;

    // Save a cycle during init by writing 0 to the uint32 below
    union
    {
        uint32_t tiles_acked_received_init;

        struct
        {
            uint16_t tiles_acked;
            uint16_t tiles_received;
        };
    };
};

struct CBInterface
{
    union
    {
        LocalCBInterface local_cb_interface;
    };
};

// Named this way for compatibility with existing code where existing code references local_cb_interface as cb_interface
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS];

inline __attribute__((always_inline)) LocalCBInterface& get_local_cb_interface(uint32_t cb_id)
{
    return cb_interface[cb_id].local_cb_interface;
}

// Use the following API once L1 CBs are aligned with DM

// inline __attribute__((always_inline)) void setup_local_cb_read_write_interfaces(
//     uint32_t* cb_l1_base,
//     uint32_t start_cb_index,
//     uint32_t max_cb_index,
//     bool read,
//     bool write) {
//     volatile uint32_t* circular_buffer_config_addr =
//         cb_l1_base + start_cb_index * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG;

//     for (uint32_t cb_id = start_cb_index; cb_id < max_cb_index; cb_id++) {
//         // NOTE: fifo_addr, fifo_size and fifo_limit in 16B words!
//         uint32_t fifo_addr = circular_buffer_config_addr[0] >> CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT;
//         uint32_t fifo_size = circular_buffer_config_addr[1] >> CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT;
//         uint32_t fifo_num_pages = circular_buffer_config_addr[2];
//         uint32_t fifo_page_size = circular_buffer_config_addr[3] >> CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT;
//         uint32_t fifo_limit = fifo_addr + fifo_size;

//         LocalCBInterface& local_interface = get_local_cb_interface(cb_id);
//         local_interface.fifo_limit = fifo_limit;  // to check if we need to wrap
//         if (write) {
//             local_interface.fifo_wr_ptr = fifo_addr;
//         }
//         if (read) {
//             local_interface.fifo_rd_ptr = fifo_addr;
//         }
//         local_interface.fifo_size = fifo_size;
//         local_interface.tiles_acked_received_init = 0;
//         if (write) {
//             local_interface.fifo_num_pages = fifo_num_pages;
//         }
//         local_interface.fifo_page_size = fifo_page_size;

//         circular_buffer_config_addr += UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG;
//     }
// }

inline __attribute__((always_inline)) void setup_local_cb_read_write_interface(
    uint32_t cb_index, uint32_t setup_fifo_addr, uint32_t setup_fifo_size, uint32_t setup_fifo_num_pages, uint32_t setup_fifo_page_size, bool read, bool write)
{
    // NOTE: fifo_addr, fifo_size and fifo_limit in 16B words!
    uint32_t fifo_addr      = setup_fifo_addr >> CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT;
    uint32_t fifo_size      = setup_fifo_size >> CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT;
    uint32_t fifo_num_pages = setup_fifo_num_pages;
    uint32_t fifo_page_size = setup_fifo_page_size >> CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT;
    uint32_t fifo_limit     = fifo_addr + fifo_size;

    LocalCBInterface& local_interface = get_local_cb_interface(cb_index);
    local_interface.fifo_limit        = fifo_limit; // to check if we need to wrap
    if (write)
    {
        local_interface.fifo_wr_ptr = fifo_addr;
    }
    if (read)
    {
        local_interface.fifo_rd_ptr = fifo_addr;
    }
    local_interface.fifo_size                 = fifo_size;
    local_interface.tiles_acked_received_init = 0;
    if (write)
    {
        local_interface.fifo_num_pages = fifo_num_pages;
    }
    local_interface.fifo_page_size = fifo_page_size;
}

} // namespace ckernel::trisc
