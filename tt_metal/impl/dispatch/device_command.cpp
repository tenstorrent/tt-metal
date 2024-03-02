// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/device_command.hpp"

#include <atomic>

#include "tt_metal/common/assert.hpp"
#include "tt_metal/common/logger.hpp"

DeviceCommand::DeviceCommand() {
    this->buffer_transfer_idx = 0;
    this->program_transfer_idx =
        DeviceCommand::NUM_POSSIBLE_BUFFER_TRANSFERS * DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION;
}

void DeviceCommand::set_event(uint32_t event) { this->packet.header.event = event; }

void DeviceCommand::set_issue_queue_size(uint32_t new_issue_queue_size) {
    this->packet.header.new_issue_queue_size = new_issue_queue_size;
}

void DeviceCommand::set_completion_queue_size(uint32_t new_completion_queue_size) {
    this->packet.header.new_completion_queue_size = new_completion_queue_size;
}

void DeviceCommand::set_wrap(WrapRegion wrap_region) { this->packet.header.wrap = (uint32_t)wrap_region; }

void DeviceCommand::set_finish() { this->packet.header.finish = 1; }

void DeviceCommand::set_num_workers(const uint32_t num_workers) { this->packet.header.num_workers = num_workers; }

void DeviceCommand::set_is_program() { this->packet.header.is_program_buffer = 1; }

void DeviceCommand::set_stall() { this->packet.header.stall = 1; }

void DeviceCommand::set_page_size(const uint32_t page_size) { this->packet.header.page_size = page_size; }

void DeviceCommand::set_pull_and_push_cb_size(const uint32_t cb_size) { this->packet.header.pull_and_push_cb_size = cb_size; }

void DeviceCommand::set_producer_cb_size(const uint32_t cb_size) { this->packet.header.producer_cb_size = cb_size; }

void DeviceCommand::set_consumer_cb_size(const uint32_t cb_size) { this->packet.header.consumer_cb_size = cb_size; }

void DeviceCommand::set_router_cb_size(const uint32_t cb_size) { this->packet.header.router_cb_size = cb_size; }

void DeviceCommand::set_pull_and_push_cb_num_pages(const uint32_t cb_num_pages) {
    this->packet.header.pull_and_push_cb_num_pages = cb_num_pages;
}

void DeviceCommand::set_producer_cb_num_pages(const uint32_t cb_num_pages) {
    this->packet.header.producer_cb_num_pages = cb_num_pages;
}

void DeviceCommand::set_consumer_cb_num_pages(const uint32_t cb_num_pages) { this->packet.header.consumer_cb_num_pages = cb_num_pages; }

void DeviceCommand::set_router_cb_num_pages(const uint32_t cb_num_pages) { this->packet.header.router_cb_num_pages = cb_num_pages; }

void DeviceCommand::set_num_pages(uint32_t num_pages) { this->packet.header.num_pages = num_pages; }

void DeviceCommand::set_sharded_buffer_num_cores(uint32_t num_cores) {
    this->packet.header.sharded_buffer_num_cores = num_cores;
}

void DeviceCommand::set_buffer_type(const DeviceCommand::BufferType buffer_type) {
    this->packet.header.buffer_type = (uint32_t)buffer_type;
}

void DeviceCommand::set_num_pages(const DeviceCommand::TransferType transfer_type, const uint32_t num_pages) {
    switch (transfer_type) {
        case DeviceCommand::TransferType::RUNTIME_ARGS: this->packet.header.num_runtime_arg_pages = num_pages; break;
        case DeviceCommand::TransferType::CB_CONFIGS: this->packet.header.num_cb_config_pages = num_pages; break;
        case DeviceCommand::TransferType::PROGRAM_MULTICAST_PAGES:
            this->packet.header.num_program_multicast_pages = num_pages;
            break;
        case DeviceCommand::TransferType::PROGRAM_UNICAST_PAGES:
            this->packet.header.num_program_unicast_pages = num_pages;
            break;
        case DeviceCommand::TransferType::GO_SIGNALS_MULTICAST:
            this->packet.header.num_go_signal_multicast_pages = num_pages;
            break;
        case DeviceCommand::TransferType::GO_SIGNALS_UNICAST:
            this->packet.header.num_go_signal_unicast_pages = num_pages;
            break;
        default: TT_ASSERT(false, "Invalid transfer type.");
    }
}

void DeviceCommand::set_issue_data_size(const uint32_t data_size) { this->packet.header.issue_data_size = data_size; }

void DeviceCommand::set_completion_data_size(const uint32_t data_size) {
    this->packet.header.completion_data_size = data_size;
}

uint32_t DeviceCommand::get_issue_data_size() const { return this->packet.header.issue_data_size; }

uint32_t DeviceCommand::get_completion_data_size() const { return this->packet.header.completion_data_size; }

void DeviceCommand::set_producer_consumer_transfer_num_pages(const uint32_t producer_consumer_transfer_num_pages) {
    this->packet.header.producer_consumer_transfer_num_pages = producer_consumer_transfer_num_pages;
}

void DeviceCommand::set_program_transfer_num_pages(const uint32_t program_transfer_num_pages) {
    this->packet.header.program_transfer_num_pages = program_transfer_num_pages;
}

void DeviceCommand::set_router_transfer_num_pages(const uint32_t router_transfer_num_pages) {
    this->packet.header.router_transfer_num_pages = router_transfer_num_pages;
}

void DeviceCommand::set_is_event_sync(const uint16_t is_event_sync) { this->packet.header.is_event_sync = is_event_sync; }
void DeviceCommand::set_event_sync_core_x(const uint8_t event_sync_core_x) { this->packet.header.event_sync_core_x = event_sync_core_x; }
void DeviceCommand::set_event_sync_core_y(const uint8_t event_sync_core_y) { this->packet.header.event_sync_core_y = event_sync_core_y; }
void DeviceCommand::set_event_sync_event_id(const uint32_t event_sync_event_id) { this->packet.header.event_sync_event_id = event_sync_event_id; }

void DeviceCommand::update_buffer_transfer_src(const uint8_t buffer_transfer_idx, const uint32_t new_src) {
    this->packet.data
        [DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER +
         buffer_transfer_idx * DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION] = new_src;
}

void DeviceCommand::add_buffer_transfer_instruction_preamble(
    const uint32_t src,
    const uint32_t dst,
    const uint32_t num_pages,
    const uint32_t padded_page_size,
    const uint32_t src_buf_type,
    const uint32_t dst_buf_type,
    const uint32_t src_page_index,
    const uint32_t dst_page_index) {
    this->packet.data[this->buffer_transfer_idx] = src;
    this->packet.data[this->buffer_transfer_idx + 1] = dst;
    this->packet.data[this->buffer_transfer_idx + 2] = num_pages;
    this->packet.data[this->buffer_transfer_idx + 3] = padded_page_size;
    this->packet.data[this->buffer_transfer_idx + 4] = src_buf_type;
    this->packet.data[this->buffer_transfer_idx + 5] = dst_buf_type;
    this->packet.data[this->buffer_transfer_idx + 6] = src_page_index;
    this->packet.data[this->buffer_transfer_idx + 7] = dst_page_index;
}

void DeviceCommand::add_buffer_transfer_instruction_postamble() {
    this->buffer_transfer_idx += DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION;

    this->packet.header.num_buffer_transfers++;
    TT_ASSERT(
        this->packet.header.num_buffer_transfers <= DeviceCommand::NUM_POSSIBLE_BUFFER_TRANSFERS,
        "Surpassing the limit of {} on possible buffer transfers in a single command",
        DeviceCommand::NUM_POSSIBLE_BUFFER_TRANSFERS);
}

void DeviceCommand::add_buffer_transfer_interleaved_instruction(
    const uint32_t src,
    const uint32_t dst,
    const uint32_t num_pages,
    const uint32_t padded_page_size,
    const uint32_t src_buf_type,
    const uint32_t dst_buf_type,
    const uint32_t src_page_index,
    const uint32_t dst_page_index) {
    this->add_buffer_transfer_instruction_preamble(
        src, dst, num_pages, padded_page_size, src_buf_type, dst_buf_type, src_page_index, dst_page_index);
    this->add_buffer_transfer_instruction_postamble();
}

void DeviceCommand::add_buffer_transfer_sharded_instruction(
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
    const std::vector<uint32_t> core_id_y) {
    this->add_buffer_transfer_instruction_preamble(
        src, dst, num_pages, padded_page_size, src_buf_type, dst_buf_type, src_page_index, dst_page_index);

    // Shard specific portion of instruction
    TT_ASSERT(core_id_x.size() == core_id_y.size());
    TT_ASSERT(core_id_x.size() == num_pages_in_shard.size());
    uint32_t num_shards = core_id_x.size();
    uint32_t idx_offset = COMMAND_PTR_SHARD_IDX;
    for (auto shard_id = 0; shard_id < num_shards; shard_id++) {
        this->packet.data[this->buffer_transfer_idx + idx_offset++] = num_pages_in_shard[shard_id];
        this->packet.data[this->buffer_transfer_idx + idx_offset++] = core_id_x[shard_id];
        this->packet.data[this->buffer_transfer_idx + idx_offset++] = core_id_y[shard_id];
    }

    this->add_buffer_transfer_instruction_postamble();
}

void DeviceCommand::write_program_entry(const uint32_t value) {
    this->packet.data.at(this->program_transfer_idx) = value;
    this->program_transfer_idx++;
}

void DeviceCommand::add_write_page_partial_instruction(
    const uint32_t num_bytes,
    const uint32_t dst,
    const uint32_t dst_noc,
    const uint32_t num_receivers,
    const bool advance,
    const bool linked) {
    // This 'at' does size checking
    this->packet.data.at(this->program_transfer_idx + 5) = linked;

    this->packet.data[this->program_transfer_idx] = num_bytes;
    this->packet.data[this->program_transfer_idx + 1] = dst;
    this->packet.data[this->program_transfer_idx + 2] = dst_noc;
    this->packet.data[this->program_transfer_idx + 3] = num_receivers;
    this->packet.data[this->program_transfer_idx + 4] = advance;

    this->program_transfer_idx += 6;
}

void* DeviceCommand::data() const { return (void*)&this->packet; }
