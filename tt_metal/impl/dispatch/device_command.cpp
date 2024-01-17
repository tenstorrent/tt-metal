// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/device_command.hpp"

#include "tt_metal/common/logger.hpp"
#include "tt_metal/common/assert.hpp"

DeviceCommand::DeviceCommand() {
    for (uint32_t idx = 0; idx < DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER; idx++) {
        this->desc[idx] = 0;
    }
    this->buffer_transfer_idx = DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;
    this->program_transfer_idx = this->buffer_transfer_idx + DeviceCommand::NUM_POSSIBLE_BUFFER_TRANSFERS *
                                                                      DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION;

    this->desc[this->sharded_buffer_num_cores_idx] = 1;
}

void DeviceCommand::wrap(WrapRegion wrap_region) { this->desc[this->wrap_idx] = (uint32_t)wrap_region; }

void DeviceCommand::finish() { this->desc[this->finish_idx] = 1; }

void DeviceCommand::set_num_workers(const uint32_t num_workers) { this->desc.at(this->num_workers_idx) = num_workers; }

void DeviceCommand::set_is_program() { this->desc[this->is_program_buffer_idx] = 1; }

void DeviceCommand::set_stall() { this->desc[this->stall_idx] = 1; }

void DeviceCommand::set_page_size(const uint32_t page_size) { this->desc[this->page_size_idx] = page_size; }

void DeviceCommand::set_producer_cb_size(const uint32_t cb_size) { this->desc[this->producer_cb_size_idx] = cb_size; }

void DeviceCommand::set_consumer_cb_size(const uint32_t cb_size) { this->desc[this->consumer_cb_size_idx] = cb_size; }

void DeviceCommand::set_producer_cb_num_pages(const uint32_t cb_num_pages) { this->desc[this->producer_cb_num_pages_idx] = cb_num_pages; }

void DeviceCommand::set_consumer_cb_num_pages(const uint32_t cb_num_pages) { this->desc[this->consumer_cb_num_pages_idx] = cb_num_pages; }

void DeviceCommand::set_num_pages(uint32_t num_pages) { this->desc[this->num_pages_idx] = num_pages; }

void DeviceCommand::set_sharded_buffer_num_cores(uint32_t num_cores) { this->desc[this->sharded_buffer_num_cores_idx] = num_cores; }

void DeviceCommand::set_num_pages(const DeviceCommand::TransferType transfer_type, const uint32_t num_pages) {
    switch (transfer_type) {
        case DeviceCommand::TransferType::RUNTIME_ARGS:
            this->desc[this->num_runtime_arg_pages_idx] = num_pages;
            break;
        case DeviceCommand::TransferType::CB_CONFIGS:
            this->desc[this->num_cb_config_pages_idx] = num_pages;
            break;
        case DeviceCommand::TransferType::PROGRAM_PAGES:
            this->desc[this->num_program_pages_idx] = num_pages;
            break;
        case DeviceCommand::TransferType::GO_SIGNALS:
            this->desc[this->num_go_signal_pages_idx] = num_pages;
            break;
        default:
            TT_ASSERT(false, "Invalid transfer type.");
    }
}

void DeviceCommand::set_data_size(const uint32_t data_size) { this->desc[this->data_size_idx] = data_size; }

uint32_t DeviceCommand::get_data_size() const { return this->desc[this->data_size_idx]; }

void DeviceCommand::set_producer_consumer_transfer_num_pages(const uint32_t producer_consumer_transfer_num_pages) {
    this->desc[this->producer_consumer_transfer_num_pages_idx] = producer_consumer_transfer_num_pages;
}

void DeviceCommand::update_buffer_transfer_src(const uint8_t buffer_transfer_idx, const uint32_t new_src) {
    this->desc[DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER + buffer_transfer_idx * DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION] = new_src;
}


void DeviceCommand::add_buffer_transfer_instruction(
    const uint32_t src,
    const uint32_t dst,
    const uint32_t num_pages,
    const uint32_t padded_page_size,
    const uint32_t src_buf_type,
    const uint32_t dst_buf_type,
    const uint32_t src_page_index,
    const uint32_t dst_page_index) {
    this->desc[this->buffer_transfer_idx] = src;
    this->desc[this->buffer_transfer_idx + 1] = dst;
    this->desc[this->buffer_transfer_idx + 2] = num_pages;
    this->desc[this->buffer_transfer_idx + 3] = padded_page_size;
    this->desc[this->buffer_transfer_idx + 4] = src_buf_type;
    this->desc[this->buffer_transfer_idx + 5] = dst_buf_type;
    this->desc[this->buffer_transfer_idx + 6] = src_page_index;
    this->desc[this->buffer_transfer_idx + 7] = dst_page_index;
    this->buffer_transfer_idx += DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION;

    this->desc[this->num_buffer_transfers_idx]++;
    TT_ASSERT(
        this->desc[this->num_buffer_transfers_idx] <= DeviceCommand::NUM_POSSIBLE_BUFFER_TRANSFERS,
        "Surpassing the limit of {} on possible buffer transfers in a single command",
        DeviceCommand::NUM_POSSIBLE_BUFFER_TRANSFERS);
}


void DeviceCommand::add_buffer_transfer_instruction_sharded(
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
    ) {


    TT_ASSERT(core_id_x.size() == core_id_y.size());
    TT_ASSERT(core_id_x.size() == num_pages_in_shard.size());
    uint32_t num_shards = core_id_x.size();

    uint32_t idx_offset = 0;
    this->desc[this->buffer_transfer_idx + idx_offset++] = src;
    this->desc[this->buffer_transfer_idx + idx_offset++] = dst;
    this->desc[this->buffer_transfer_idx + idx_offset++] = num_pages;
    this->desc[this->buffer_transfer_idx + idx_offset++] = padded_page_size;
    this->desc[this->buffer_transfer_idx + idx_offset++] = src_buf_type;
    this->desc[this->buffer_transfer_idx + idx_offset++] = dst_buf_type;
    this->desc[this->buffer_transfer_idx + idx_offset++] = src_page_index;
    this->desc[this->buffer_transfer_idx + idx_offset++] = dst_page_index;
    for (auto shard_id = 0; shard_id < num_shards; shard_id++) {
        this->desc[this->buffer_transfer_idx + idx_offset++] = num_pages_in_shard[shard_id];
        this->desc[this->buffer_transfer_idx + idx_offset++] = core_id_x[shard_id];
        this->desc[this->buffer_transfer_idx + idx_offset++] = core_id_y[shard_id];
    }
    this->buffer_transfer_idx += DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION;

    this->desc[this->num_buffer_transfers_idx]++;
    TT_ASSERT(
        this->desc[this->num_buffer_transfers_idx] <= DeviceCommand::NUM_POSSIBLE_BUFFER_TRANSFERS,
        "Surpassing the limit of {} on possible buffer transfers in a single command",
        DeviceCommand::NUM_POSSIBLE_BUFFER_TRANSFERS);
}

void DeviceCommand::write_program_entry(const uint32_t value) {
    this->desc.at(this->program_transfer_idx) = value;
    this->program_transfer_idx++;
}

void DeviceCommand::add_write_page_partial_instruction(
    const uint32_t num_bytes, const uint32_t dst, const uint32_t dst_noc, const uint32_t num_receivers, const bool advance, const bool linked) {

    // This 'at' does size checking
    this->desc.at(this->program_transfer_idx + 5) = linked;

    this->desc[this->program_transfer_idx] = num_bytes;
    this->desc[this->program_transfer_idx + 1] = dst;
    this->desc[this->program_transfer_idx + 2] = dst_noc;
    this->desc[this->program_transfer_idx + 3] = num_receivers;
    this->desc[this->program_transfer_idx + 4] = advance;

    this->program_transfer_idx += 6;
}

const std::array<uint32_t, DeviceCommand::NUM_ENTRIES_IN_DEVICE_COMMAND>& DeviceCommand::get_desc() const {
    return this->desc;
}
