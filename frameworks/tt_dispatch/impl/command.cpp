#include "frameworks/tt_dispatch/impl/command.hpp"

DeviceCommand::DeviceCommand() {
    static_assert(DeviceCommandNumEntries * sizeof(u32) % 16 == 0);  // Desc size needs to be 16B-aligned

    this->desc[this->finish_idx] = 0;
    this->desc[this->launch_idx] = 0;
    this->desc[this->data_size_in_bytes_idx] = 0;
    this->desc[this->num_reads_idx] = 0;
    this->desc[this->num_writes_idx] = 0;
}

void DeviceCommand::finish() { this->desc[this->finish_idx] = 1; }

void DeviceCommand::launch() { this->desc[this->launch_idx] = 1; }

void DeviceCommand::add_read_relay(
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
    this->desc[this->num_reads_idx]++;
    this->add_relay(
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

// 'src' must be a single bank
void DeviceCommand::add_write_relay(
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
    this->desc[this->num_writes_idx]++;
    this->add_relay(
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

void DeviceCommand::set_data_size_in_bytes(u32 data_size_in_bytes) {
    this->desc[this->data_size_in_bytes_idx] = data_size_in_bytes;
}

const array<u32, DeviceCommandNumEntries>& DeviceCommand::get_desc() const { return this->desc; }
