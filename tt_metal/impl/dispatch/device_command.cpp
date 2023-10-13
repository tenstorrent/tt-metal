// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/device_command.hpp"

#include "tt_metal/common/logger.hpp"

DeviceCommand::DeviceCommand() {
    for (u32 idx = 0; idx < DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER; idx++) {
        this->desc[idx] = 0;
    }
    this->buffer_transfer_idx = DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;
    this->program_transfer_idx = this->buffer_transfer_idx + DeviceCommand::NUM_POSSIBLE_BUFFER_TRANSFERS *
                                                                      DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION;
}

void DeviceCommand::wrap() { this->desc[this->wrap_idx] = 1; }

void DeviceCommand::finish() { this->desc[this->finish_idx] = 1; }

void DeviceCommand::set_num_workers(const u32 num_workers) { this->desc.at(this->num_workers_idx) = num_workers; }

void DeviceCommand::set_is_program() { this->desc[this->is_program_buffer_idx] = 1; }

void DeviceCommand::set_stall() { this->desc[this->stall_idx] = 1; }

void DeviceCommand::set_page_size(const u32 page_size) { this->desc[this->page_size_idx] = page_size; }

void DeviceCommand::set_producer_cb_size(const u32 cb_size) { this->desc[this->producer_cb_size_idx] = cb_size; }

void DeviceCommand::set_consumer_cb_size(const u32 cb_size) { this->desc[this->consumer_cb_size_idx] = cb_size; }

void DeviceCommand::set_producer_cb_num_pages(const u32 cb_num_pages) { this->desc[this->producer_cb_num_pages_idx] = cb_num_pages; }

void DeviceCommand::set_consumer_cb_num_pages(const u32 cb_num_pages) { this->desc[this->consumer_cb_num_pages_idx] = cb_num_pages; }

void DeviceCommand::set_num_pages(const u32 num_pages) { this->desc[this->num_pages_idx] = num_pages; }

void DeviceCommand::set_data_size(const u32 data_size) { this->desc[this->data_size_idx] = data_size; }

u32 DeviceCommand::get_data_size() const { return this->desc[this->data_size_idx]; }

void DeviceCommand::add_buffer_transfer_instruction(
    const u32 src,
    const u32 dst,
    const u32 num_pages,
    const u32 padded_page_size,
    const u32 src_buf_type,
    const u32 dst_buf_type) {
    this->desc[this->buffer_transfer_idx] = src;
    this->desc[this->buffer_transfer_idx + 1] = dst;
    this->desc[this->buffer_transfer_idx + 2] = num_pages;
    this->desc[this->buffer_transfer_idx + 3] = padded_page_size;
    this->desc[this->buffer_transfer_idx + 4] = src_buf_type;
    this->desc[this->buffer_transfer_idx + 5] = dst_buf_type;
    this->buffer_transfer_idx += DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION;

    this->desc[this->num_buffer_transfers_idx]++;
    tt::log_assert(
        this->desc[this->num_buffer_transfers_idx] <= DeviceCommand::NUM_POSSIBLE_BUFFER_TRANSFERS,
        "Surpassing the limit of {} on possible buffer transfers in a single command",
        DeviceCommand::NUM_POSSIBLE_BUFFER_TRANSFERS);
}

void DeviceCommand::write_program_entry(const u32 value) {
    this->desc.at(this->program_transfer_idx) = value;
    this->program_transfer_idx++;

}

void DeviceCommand::add_write_page_partial_instruction(
    const u32 num_bytes, const u32 dst, const u32 dst_noc, const u32 num_receivers, const bool advance) {

    // This 'at' does size checking
    this->desc.at(this->program_transfer_idx + 4) = advance;

    this->desc[this->program_transfer_idx] = num_bytes;
    this->desc[this->program_transfer_idx + 1] = dst;
    this->desc[this->program_transfer_idx + 2] = dst_noc;
    this->desc[this->program_transfer_idx + 3] = num_receivers;

    // std::cout << "WRITE PAGE PARTIAL AT " << this->program_transfer_idx << ": " << num_bytes << ", " << dst << ", " << dst_noc << std::endl;

    this->program_transfer_idx += 5;
}

const std::array<u32, DeviceCommand::NUM_ENTRIES_IN_DEVICE_COMMAND>& DeviceCommand::get_desc() const {
    return this->desc;
}
