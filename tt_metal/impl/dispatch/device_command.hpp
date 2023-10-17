/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>

#include "dev_mem_map.h"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"

class DeviceCommand {
   public:
    DeviceCommand();

    // Constants
    static constexpr u32 NUM_ENTRIES_IN_COMMAND_HEADER = 16;
    static constexpr u32 NUM_ENTRIES_IN_DEVICE_COMMAND = 5632;
    static constexpr u32 NUM_BYTES_IN_DEVICE_COMMAND = NUM_ENTRIES_IN_DEVICE_COMMAND * sizeof(u32);
    static constexpr u32 DATA_SECTION_ADDRESS = L1_UNRESERVED_BASE + NUM_BYTES_IN_DEVICE_COMMAND;
    static constexpr u32 PROGRAM_PAGE_SIZE = 2048;
    static constexpr u32 PRODUCER_DATA_BUFFER_SIZE =
        (MEM_L1_SIZE - (NUM_ENTRIES_IN_DEVICE_COMMAND * sizeof(u32)) - L1_UNRESERVED_BASE);
    static constexpr u32 CONSUMER_DATA_BUFFER_SIZE = (PRODUCER_DATA_BUFFER_SIZE - NUM_BYTES_IN_DEVICE_COMMAND) / 2;
    static constexpr u32 DEVICE_COMMAND_DATA_ADDR = L1_UNRESERVED_BASE + NUM_BYTES_IN_DEVICE_COMMAND;
    static constexpr u32 NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION = 6;
    static constexpr u32 NUM_POSSIBLE_BUFFER_TRANSFERS = 2;

    // Ensure any changes to this device command have asserts modified/extended
    static_assert(NUM_ENTRIES_IN_COMMAND_HEADER == 16);
    static_assert((NUM_BYTES_IN_DEVICE_COMMAND % 32) == 0);

    // Command header
    static constexpr u32 wrap_idx = 0;
    static constexpr u32 finish_idx = 1;
    static constexpr u32 num_workers_idx = 2;
    static constexpr u32 num_buffer_transfers_idx = 3;
    static constexpr u32 is_program_buffer_idx = 4;
    static constexpr u32 stall_idx = 5;
    static constexpr u32 page_size_idx = 6;
    static constexpr u32 producer_cb_size_idx = 7;
    static constexpr u32 consumer_cb_size_idx = 8;
    static constexpr u32 producer_cb_num_pages_idx = 9;
    static constexpr u32 consumer_cb_num_pages_idx = 10;
    static constexpr u32 num_pages_idx = 11;
    static constexpr u32 data_size_idx = 12;
    static constexpr u32 producer_consumer_transfer_num_pages_idx = 13;

    void wrap();

    void finish();

    void set_num_workers(const u32 num_workers);


    void set_is_program();

    void set_stall();

    void set_page_size(const u32 page_size);

    void set_producer_cb_size(const u32 cb_size);

    void set_consumer_cb_size(const u32 cb_size);

    void set_producer_cb_num_pages(const u32 cb_num_pages);

    void set_consumer_cb_num_pages(const u32 cb_num_pages);

    void set_num_pages(const u32 num_pages);

    void set_data_size(const u32 data_size);

    void set_producer_consumer_transfer_num_pages(const u32 producer_consumer_transfer_num_pages);

    u32 get_data_size() const;

    void add_buffer_transfer_instruction(
        const u32 src,
        const u32 dst,
        const u32 num_pages,
        const u32 padded_page_size,
        const u32 src_buf_type,
        const u32 dst_buf_type);

    void write_program_entry(const u32 val);

    void add_write_page_partial_instruction(
        const u32 num_bytes, const u32 dst, const u32 dst_noc, const u32 num_receivers, const bool advance);

    const std::array<u32, NUM_ENTRIES_IN_DEVICE_COMMAND>& get_desc() const;

   private:
    std::array<u32, DeviceCommand::NUM_ENTRIES_IN_DEVICE_COMMAND> desc;
    u32 buffer_transfer_idx;
    u32 program_transfer_idx;
};
