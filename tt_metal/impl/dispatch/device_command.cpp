#include "tt_metal/impl/dispatch/device_command.hpp"

#include "tt_metal/common/logger.hpp"

DeviceCommand::DeviceCommand() {
    this->desc[this->finish_idx] = 0;
    this->desc[this->launch_idx] = 0;
    this->desc[this->data_size_in_bytes_idx] = 0;
    this->desc[this->num_relay_buffer_reads_idx] = 0;
    this->desc[this->num_relay_buffer_writes_idx] = 0;
    this->desc[this->num_relay_program_writes_idx] = 0;
}

void DeviceCommand::finish() { this->desc[this->finish_idx] = 1; }

void DeviceCommand::launch() { this->desc[this->launch_idx] = 1; }

void DeviceCommand::add_buffer_relay(
    u32 addr0,
    u32 addr0_noc,
    u32 addr1,
    u32 addr1_noc_start,
    u32 num_bursts,
    u32 burst_size,
    u32 num_pages_per_burst,
    u32 page_size,
    u32 remainder_burst_size,
    u32 num_pages_per_remainder_burst,
    u32 banking_enum) {
    // tt::log_debug(tt::LogDispatch, "Writing buffer relay to addr {}", this->relay_buffer_entry_idx);
    // tt::log_debug(tt::LogDispatch, "Addr 0 {}", addr0);
    // tt::log_debug(tt::LogDispatch, "addr0_noc {}", addr0_noc);

    this->desc[this->relay_buffer_entry_idx] = addr0;
    this->desc[this->relay_buffer_entry_idx + 1] = addr0_noc;
    this->desc[this->relay_buffer_entry_idx + 2] = addr1;
    this->desc[this->relay_buffer_entry_idx + 3] = addr1_noc_start;
    this->desc[this->relay_buffer_entry_idx + 4] = num_bursts;
    this->desc[this->relay_buffer_entry_idx + 5] = burst_size;
    this->desc[this->relay_buffer_entry_idx + 6] = num_pages_per_burst;
    this->desc[this->relay_buffer_entry_idx + 7] = page_size;
    this->desc[this->relay_buffer_entry_idx + 8] = remainder_burst_size;
    this->desc[this->relay_buffer_entry_idx + 9] = num_pages_per_remainder_burst;
    this->desc[this->relay_buffer_entry_idx + 10] = banking_enum;
    this->relay_buffer_entry_idx += this->num_4B_words_in_relay_buffer_instruction;
}

void DeviceCommand::add_read_buffer_instruction(
    u32 dst,
    u32 dst_noc,
    u32 src,
    u32 src_noc_start,
    u32 num_bursts,
    u32 burst_size,
    u32 num_pages_per_burst,
    u32 page_size,
    u32 remainder_burst_size,
    u32 num_pages_per_remainder_burst,
    u32 banking_enum) {
    this->desc[this->num_relay_buffer_reads_idx]++;
    this->add_buffer_relay(
        dst,
        dst_noc,
        src,
        src_noc_start,
        num_bursts,
        burst_size,
        num_pages_per_burst,
        page_size,
        remainder_burst_size,
        num_pages_per_remainder_burst,
        banking_enum);
}

void DeviceCommand::add_write_buffer_instruction(
    u32 src,
    u32 src_noc,
    u32 dst,
    u32 dst_noc_start,
    u32 num_bursts,
    u32 burst_size,
    u32 num_pages_per_burst,
    u32 page_size,
    u32 remainder_burst_size,
    u32 num_pages_per_remainder_burst,
    u32 banking_enum) {
    this->desc[this->num_relay_buffer_writes_idx]++;
    this->add_buffer_relay(
        src,
        src_noc,
        dst,
        dst_noc_start,
        num_bursts,
        burst_size,
        num_pages_per_burst,
        page_size,
        remainder_burst_size,
        num_pages_per_remainder_burst,
        banking_enum);
}

void DeviceCommand::add_read_multi_write_instruction(
    u32 src, u32 src_noc, u32 transfer_size, vector<TrailingWriteCommand> write_commands) {
    this->desc[this->num_relay_program_writes_idx]++;

    this->desc[this->relay_program_entry_idx] = src;
    this->desc[this->relay_program_entry_idx + 1] = src_noc;
    this->desc[this->relay_program_entry_idx + 2] = transfer_size;
    this->desc[this->relay_program_entry_idx + 3] = write_commands.size();
    this->relay_program_entry_idx += 4;
    for (const TrailingWriteCommand& write_command : write_commands) {
        this->desc[this->relay_program_entry_idx] = write_command.src;
        this->desc[this->relay_program_entry_idx + 1] = write_command.dst;
        this->desc[this->relay_program_entry_idx + 2] = write_command.dst_noc;
        this->desc[this->relay_program_entry_idx + 3] = write_command.transfer_size;
        this->desc[this->relay_program_entry_idx + 4] = write_command.num_receivers;
        this->relay_program_entry_idx += 5;
    }
}

void DeviceCommand::set_data_size_in_bytes(u32 data_size_in_bytes) {
    this->desc[this->data_size_in_bytes_idx] = data_size_in_bytes;
}

u32 DeviceCommand::get_data_size_in_bytes() const { return this->desc[this->data_size_in_bytes_idx]; }

const array<u32, DeviceCommandNumEntries>& DeviceCommand::get_desc() const { return this->desc; }
