#include "tt_metal/impl/buffers/interleaved_buffer.hpp"

#include <algorithm>

namespace tt {

namespace tt_metal {

std::map<int, uint32_t> get_size_per_bank(Device *device, int num_bank_units, int num_entries_per_bank_unit, int num_bytes_per_entry) {
    std::map<int, uint32_t> size_per_bank;

    uint32_t total_size = num_bank_units * num_entries_per_bank_unit * num_bytes_per_entry;
    int num_equally_distributed_units = num_bank_units / device->num_dram_banks();
    int remaining_units_after_equally_distributing = num_bank_units % device->num_dram_banks();

    uint32_t total_allocated = 0;
    for (int dram_bank = 0; dram_bank < device->num_dram_banks(); dram_bank++) {
        int num_units_in_bank = num_equally_distributed_units;
        if (remaining_units_after_equally_distributing > 0) {
            num_units_in_bank += 1;
            remaining_units_after_equally_distributing -= 1;
        }
        uint32_t buffer_size = num_units_in_bank * (num_entries_per_bank_unit * num_bytes_per_entry);
        size_per_bank.insert({dram_bank, buffer_size});
        total_allocated += buffer_size;
        if (total_allocated == total_size) {
            break;
        }
    }

    return size_per_bank;
}

InterleavedDramBuffer::InterleavedDramBuffer(Device *device, int num_bank_units, int num_entries_per_bank_unit, int num_bytes_per_entry)
    : num_bank_units_(num_bank_units),
      num_entries_per_bank_unit_(num_entries_per_bank_unit),
      num_bytes_per_entry_(num_bytes_per_entry_),
      Buffer(device, num_bank_units * num_entries_per_bank_unit * num_bytes_per_entry, 0, true) {

    auto size_per_bank = get_size_per_bank(device, num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry);
    this->set_address(size_per_bank);

    for (auto &[dram_bank, required_size_bytes] : size_per_bank) {
        auto dram_buffer = new DramBuffer(device, dram_bank, required_size_bytes, this->address_);
        this->bank_to_buffer_.insert({dram_bank, dram_buffer});
    }
}

InterleavedDramBuffer::InterleavedDramBuffer(
    Device *device, int num_bank_units, int num_entries_per_bank_unit, int num_bytes_per_entry, const std::map<int, DramBuffer *> &bank_to_buffer)
    : num_bank_units_(num_bank_units),
      num_entries_per_bank_unit_(num_entries_per_bank_unit),
      num_bytes_per_entry_(num_bytes_per_entry),
      Buffer(device, num_bank_units * num_entries_per_bank_unit * num_bytes_per_entry, 0, true) {

    auto size_per_bank = get_size_per_bank(device, num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry);
    this->set_address(size_per_bank);

    for (auto &[dram_bank, dram_buffer] : bank_to_buffer) {
        auto new_dram_buffer = new DramBuffer(device, dram_buffer->dram_channel(), dram_buffer->size(), this->address_);
        this->bank_to_buffer_.insert({dram_bank, new_dram_buffer});
    }
}

void InterleavedDramBuffer::set_address(const std::map<int, uint32_t> &size_per_bank) {
    this->address_ = 0;
    std::vector<std::pair<uint32_t, uint32_t>> candidate_addr_ranges;
    for (auto &[dram_bank, required_size_bytes] : size_per_bank) {
        auto potential_addr_ranges = this->device_->banked_dram_manager_.at(dram_bank)->available_addresses(required_size_bytes);
        if (candidate_addr_ranges.empty()) {
            candidate_addr_ranges = potential_addr_ranges;
            continue;
        }
        int i = 0;
        int j = 0;
        std::vector<std::pair<uint32_t, uint32_t>> intersecting_addr_ranges;
        while (i < candidate_addr_ranges.size() and j < potential_addr_ranges.size()) {
            uint32_t lower_addr = std::max(candidate_addr_ranges[i].first, potential_addr_ranges[j].first);
            uint32_t upper_addr = std::min(candidate_addr_ranges[i].second, potential_addr_ranges[j].second);
            if (lower_addr <= upper_addr) {
                intersecting_addr_ranges.push_back({lower_addr, upper_addr});
            }
            if (candidate_addr_ranges[i].second < potential_addr_ranges[j].second) {
                i++;
            } else {
                j++;
            }
        }
        candidate_addr_ranges = intersecting_addr_ranges;
    }

    if (candidate_addr_ranges.empty()) {
        TT_THROW("Not enough space to hold interleave " + std::to_string(this->size_in_bytes_) + " bytes across DRAM channels");
    }

    uint32_t smallest_size = candidate_addr_ranges[0].second - candidate_addr_ranges[0].first;
    uint32_t address = candidate_addr_ranges[0].first;
    for (auto candidate_addr_range : candidate_addr_ranges) {
        uint32_t range_size = candidate_addr_range.second - candidate_addr_range.first;
        if (range_size < smallest_size) {
            smallest_size = range_size;
            address = candidate_addr_range.first;
        }
    }
    this->address_ = address;
}

Buffer *InterleavedDramBuffer::clone() {
    return new InterleavedDramBuffer(this->device_, this->num_bank_units_, this->num_entries_per_bank_unit_, this->num_bytes_per_entry_, this->bank_to_buffer_);
}

uint32_t InterleavedDramBuffer::buffer_size(int dram_channel) const {
    if (this->bank_to_buffer_.find(dram_channel) == this->bank_to_buffer_.end()) {
        TT_THROW("No buffer has been allocated at DRAM channel " + std::to_string(dram_channel));
    }
    return this->bank_to_buffer_.at(dram_channel)->size();
}

tt_xy_pair InterleavedDramBuffer::noc_coordinates() const {
    return this->noc_coordinates(0);
}

tt_xy_pair InterleavedDramBuffer::noc_coordinates(int dram_channel) const {
    if (this->bank_to_buffer_.find(dram_channel) == this->bank_to_buffer_.end()) {
        TT_THROW("No buffer has been allocated at DRAM channel " + std::to_string(dram_channel));
    }
    return this->bank_to_buffer_.at(dram_channel)->noc_coordinates();
}

std::vector<tt_xy_pair> InterleavedDramBuffer::interleaved_noc_coordinates() const {
    std::vector<tt_xy_pair> dram_noc_coordinates;
    for (int dram_bank = 0; dram_bank < this->device_->num_dram_banks(); dram_bank++) {
        if (this->bank_to_buffer_.find(dram_bank) != this->bank_to_buffer_.end()) {
            dram_noc_coordinates.push_back(this->bank_to_buffer_.at(dram_bank)->noc_coordinates());
        }
    }
    TT_ASSERT(not dram_noc_coordinates.empty());
    return dram_noc_coordinates;
}

void InterleavedDramBuffer::free() {
    for (auto &[dram_bank, dram_buffer] : bank_to_buffer_) {
        dram_buffer->free();
    }
    this->allocated_on_device_ = false;
    bank_to_buffer_.clear();
}

InterleavedDramBuffer::~InterleavedDramBuffer() {
    if (this->allocated_on_device_) {
        this->free();
    }
}

}  // namespace tt_metal

}  // namespace tt
