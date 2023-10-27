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

    enum class TransferType : uint8_t { RUNTIME_ARGS, CB_CONFIGS, PROGRAM_PAGES, GO_SIGNALS, NUM_TRANSFER_TYPES };

    // Constants
    static constexpr uint32_t NUM_ENTRIES_IN_COMMAND_HEADER = 20;
    static constexpr uint32_t NUM_ENTRIES_IN_DEVICE_COMMAND = 5632;
    static constexpr uint32_t NUM_BYTES_IN_DEVICE_COMMAND = NUM_ENTRIES_IN_DEVICE_COMMAND * sizeof(uint32_t);
    static constexpr uint32_t DATA_SECTION_ADDRESS = L1_UNRESERVED_BASE + NUM_BYTES_IN_DEVICE_COMMAND;
    static constexpr uint32_t PROGRAM_PAGE_SIZE = 2048;
    static constexpr uint32_t PRODUCER_DATA_BUFFER_SIZE =
        (MEM_L1_SIZE - (NUM_ENTRIES_IN_DEVICE_COMMAND * sizeof(uint32_t)) - L1_UNRESERVED_BASE);
    static constexpr uint32_t CONSUMER_DATA_BUFFER_SIZE = (PRODUCER_DATA_BUFFER_SIZE - NUM_BYTES_IN_DEVICE_COMMAND) / 2;
    static constexpr uint32_t DEVICE_COMMAND_DATA_ADDR = L1_UNRESERVED_BASE + NUM_BYTES_IN_DEVICE_COMMAND;
    static constexpr uint32_t NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION = 6;
    static constexpr uint32_t NUM_POSSIBLE_BUFFER_TRANSFERS = 2;

    // Ensure any changes to this device command have asserts modified/extended
    static_assert(NUM_ENTRIES_IN_COMMAND_HEADER == 20);
    static_assert((NUM_BYTES_IN_DEVICE_COMMAND % 32) == 0);

    // Command header
    static constexpr uint32_t wrap_idx = 0;
    static constexpr uint32_t finish_idx = 1;
    static constexpr uint32_t num_workers_idx = 2;
    static constexpr uint32_t num_buffer_transfers_idx = 3;
    static constexpr uint32_t is_program_buffer_idx = 4;
    static constexpr uint32_t stall_idx = 5;
    static constexpr uint32_t page_size_idx = 6;
    static constexpr uint32_t producer_cb_size_idx = 7;
    static constexpr uint32_t consumer_cb_size_idx = 8;
    static constexpr uint32_t producer_cb_num_pages_idx = 9;
    static constexpr uint32_t consumer_cb_num_pages_idx = 10;
    static constexpr uint32_t num_pages_idx = 11;
    static constexpr uint32_t num_runtime_arg_pages_idx = 12;
    static constexpr uint32_t num_cb_config_pages_idx = 13;
    static constexpr uint32_t num_program_pages_idx = 14;
    static constexpr uint32_t num_go_signal_pages_idx = 15;
    static constexpr uint32_t data_size_idx = 16;
    static constexpr uint32_t producer_consumer_transfer_num_pages_idx = 17;

    void wrap();

    void finish();

    void set_num_workers(const uint32_t num_workers);

    void set_is_program();

    void set_stall();

    void set_page_size(const uint32_t page_size);

    void set_producer_cb_size(const uint32_t cb_size);

    void set_consumer_cb_size(const uint32_t cb_size);

    void set_producer_cb_num_pages(const uint32_t cb_num_pages);

    void set_consumer_cb_num_pages(const uint32_t cb_num_pages);

    void set_num_pages(const uint32_t num_pages);

    void set_num_pages(const DeviceCommand::TransferType transfer_type, const uint32_t num_pages);

    void set_data_size(const uint32_t data_size);

    void set_producer_consumer_transfer_num_pages(const uint32_t producer_consumer_transfer_num_pages);

    uint32_t get_data_size() const;

    void add_buffer_transfer_instruction(
        const uint32_t src,
        const uint32_t dst,
        const uint32_t num_pages,
        const uint32_t padded_page_size,
        const uint32_t src_buf_type,
        const uint32_t dst_buf_type);

    void write_program_entry(const uint32_t val);

    void add_write_page_partial_instruction(
        const uint32_t num_bytes,
        const uint32_t dst,
        const uint32_t dst_noc,
        const uint32_t num_receivers,
        const bool advance);

    const std::array<uint32_t, NUM_ENTRIES_IN_DEVICE_COMMAND>& get_desc() const;

   private:
    std::array<uint32_t, DeviceCommand::NUM_ENTRIES_IN_DEVICE_COMMAND> desc;
    uint32_t buffer_transfer_idx;
    uint32_t program_transfer_idx;
};
