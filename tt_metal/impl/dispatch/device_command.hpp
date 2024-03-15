// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "dev_mem_map.h"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"

static constexpr uint32_t EVENT_PADDED_SIZE = 16;

struct CommandHeader {
    uint32_t wrap = 0;
    uint32_t finish = 0;
    uint32_t num_workers = 0;
    uint32_t num_buffer_transfers = 0;
    uint32_t is_program_buffer = 0;
    uint32_t stall = 0;
    uint32_t page_size = 0;
    uint32_t pull_and_push_cb_size = 0;
    uint32_t event;
    uint32_t producer_cb_size = 0;
    uint32_t consumer_cb_size = 0;
    uint32_t router_cb_size = 0;
    uint32_t pull_and_push_cb_num_pages = 0;
    uint32_t producer_cb_num_pages = 0;
    uint32_t consumer_cb_num_pages = 0;
    uint32_t router_cb_num_pages = 0;
    uint32_t num_pages = 0;
    uint32_t num_runtime_arg_pages = 0;
    uint32_t num_cb_config_pages = 0;
    uint32_t num_program_multicast_pages = 0;
    uint32_t num_program_unicast_pages = 0;
    uint32_t num_go_signal_multicast_pages = 0;
    uint32_t num_go_signal_unicast_pages = 0;
    uint32_t issue_data_size = 0;
    uint32_t completion_data_size = 0;
    uint32_t program_transfer_num_pages = 0;
    uint32_t router_transfer_num_pages = 0;
    uint32_t producer_consumer_transfer_num_pages = 0;
    uint32_t buffer_type = 0;
    uint32_t sharded_buffer_num_cores = 0;
    uint32_t new_issue_queue_size = 0;
    uint32_t new_completion_queue_size = 0;
    uint16_t is_event_sync = 0;
    uint8_t event_sync_core_x = 0;
    uint8_t event_sync_core_y = 0;
    uint32_t event_sync_event_id = 0;
    uint32_t fwd_path = 1;
};

static_assert((offsetof(CommandHeader, event) % 32) == 0);

class DeviceCommand {
   public:
    DeviceCommand();

    enum class TransferType : uint8_t {
        RUNTIME_ARGS,
        CB_CONFIGS,
        PROGRAM_MULTICAST_PAGES,
        PROGRAM_UNICAST_PAGES,
        GO_SIGNALS_MULTICAST,
        GO_SIGNALS_UNICAST,
        NUM_TRANSFER_TYPES
    };

    // Constants
    //TODO: investigate other num_cores
    static constexpr uint32_t MAX_HUGEPAGE_SIZE = 1 << 30; // 1GB;
    static constexpr uint32_t NUM_MAX_CORES = 108; //12 x 9
    static constexpr uint32_t NUM_ENTRIES_IN_COMMAND_HEADER = sizeof(CommandHeader) / sizeof(uint32_t);
    static constexpr uint32_t NUM_ENTRIES_IN_DEVICE_COMMAND = 5632;
    static constexpr uint32_t NUM_BYTES_IN_DEVICE_COMMAND = NUM_ENTRIES_IN_DEVICE_COMMAND * sizeof(uint32_t);
    static constexpr uint32_t PROGRAM_PAGE_SIZE = 2048;
    static constexpr uint32_t NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION = COMMAND_PTR_SHARD_IDX + NUM_MAX_CORES*NUM_ENTRIES_PER_SHARD;
    static constexpr uint32_t NUM_POSSIBLE_BUFFER_TRANSFERS = 2;
    // Perf measurements showed best results with divisions of 4 pages being transferred from producer -> consumer
    static constexpr uint32_t SYNC_NUM_PAGES = 4;

    // Ensure any changes to this device command have asserts modified/extended
    static_assert((NUM_BYTES_IN_DEVICE_COMMAND % 32) == 0);

    // Denotes which portion of the command queue needs to be wrapped
    enum class WrapRegion : uint8_t {
        NONE = 0,
        ISSUE = 1,
        COMPLETION = 2
    };

    void set_event(uint32_t event);

    void set_issue_queue_size(uint32_t new_issue_queue_size);

    void set_completion_queue_size(uint32_t new_completion_queue_size);

    void set_wrap(WrapRegion wrap_region);

    void set_finish();

    void set_num_workers(const uint32_t num_workers);

    void set_is_program();

    void set_stall();

    void set_page_size(const uint32_t page_size);

    void set_pull_and_push_cb_size(const uint32_t cb_size);

   void set_producer_cb_size(const uint32_t cb_size);

    void set_consumer_cb_size(const uint32_t cb_size);

    void set_router_cb_size(const uint32_t cb_size);

    void set_pull_and_push_cb_num_pages(const uint32_t cb_num_pages);

    void set_producer_consumer_transfer_num_pages(const uint32_t producer_consumer_transfer_num_pages);

    void set_producer_cb_num_pages(const uint32_t cb_num_pages);

    void set_consumer_cb_num_pages(const uint32_t cb_num_pages);

    void set_router_cb_num_pages(const uint32_t cb_num_pages);

    void set_num_pages(const uint32_t num_pages);

    // Denotes the type of buffer
    enum class BufferType : uint8_t {
        INTERLEAVED = 0,
        SHARDED = 1
    };

    void set_buffer_type(BufferType buff_type);

    void set_sharded_buffer_num_cores(uint32_t num_cores);

    void set_num_pages(const DeviceCommand::TransferType transfer_type, const uint32_t num_pages);

    void set_issue_data_size(const uint32_t data_size);

    void set_completion_data_size(const uint32_t data_size);

    void set_program_transfer_num_pages(const uint32_t program_transfer_num_pages);

    void set_router_transfer_num_pages(const uint32_t router_transfer_num_pages);

    void set_is_event_sync(const uint16_t is_event_sync);
    void set_event_sync_core_x(const uint8_t event_sync_core_x);
    void set_event_sync_core_y(const uint8_t event_sync_core_y);
    void set_event_sync_event_id(const uint32_t event_sync_event_id);

    uint32_t get_issue_data_size() const;

    uint32_t get_completion_data_size() const;

    void update_buffer_transfer_src(const uint8_t buffer_transfer_idx, const uint32_t new_src);

    void add_buffer_transfer_interleaved_instruction(
        const uint32_t src,
        const uint32_t dst,
        const uint32_t num_pages,
        const uint32_t padded_page_size,
        const uint32_t src_buf_type,
        const uint32_t dst_buf_type,
        const uint32_t src_page_index,
        const uint32_t dst_page_index
    );

    void add_buffer_transfer_sharded_instruction(
        const uint32_t src,
        const uint32_t dst,
        const uint32_t num_pages,
        const uint32_t padded_page_size,
        const uint32_t src_buf_type,
        const uint32_t dst_buf_type,
        const uint32_t src_page_index,
        const uint32_t dst_page_index,
        const std::vector<uint32_t> num_pages_in_shard,
        const std::vector<uint32_t> core_id_x,
        const std::vector<uint32_t> core_id_y
    );

    void write_program_entry(const uint32_t val);

    void add_write_page_partial_instruction(
        const uint32_t num_bytes,
        const uint32_t dst,
        const uint32_t dst_noc,
        const uint32_t num_receivers,
        const bool advance,
        const bool linked);

    void* data() const;

   private:
    uint32_t buffer_transfer_idx;
    uint32_t program_transfer_idx;
    void add_buffer_transfer_instruction_preamble(
        const uint32_t src,
        const uint32_t dst,
        const uint32_t num_pages,
        const uint32_t padded_page_size,
        const uint32_t src_buf_type,
        const uint32_t dst_buf_type,
        const uint32_t src_page_index,
        const uint32_t dst_page_index
        );
    void add_buffer_transfer_instruction_postamble();

    struct packet_ {
        CommandHeader header;
        std::array<uint32_t, DeviceCommand::NUM_ENTRIES_IN_DEVICE_COMMAND - DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER> data{};
    };

    packet_ packet;
};
