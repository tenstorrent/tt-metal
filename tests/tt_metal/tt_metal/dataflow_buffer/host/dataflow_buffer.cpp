// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/assert.hpp>

#include <stdexcept>
#include <cstdint>

#include "dev/dataflow_buffer.h"
#include "host/dataflow_buffer.hpp"

namespace dev {

dfb_register_t overlay_cluster_instances[64] = {};
local_dfb_interface_t overlay_cluster_dfb_access_pattern_tracker[64] = {};
uint64_t dfb_to_register_allocation[64] = {};

}  // namespace dev

namespace tt::tt_metal {

DataflowBufferConfig::DataflowBufferConfig(uint32_t total_size, const tt::DataFormat& data_format) :
    total_size_(total_size), data_format_(data_format), page_size_(0), max_size_(0), access_patterns_() {}

DataflowBufferConfig::DataflowBufferConfig(uint32_t total_size) :
    total_size_(total_size), data_format_(tt::DataFormat::Float16_b), page_size_(0), max_size_(0), access_patterns_() {}

DataflowBufferConfig& DataflowBufferConfig::set_total_size(uint32_t total_size) {
    total_size_ = total_size;
    return *this;
}

DataflowBufferConfig& DataflowBufferConfig::set_page_size(uint32_t page_size) {
    page_size_ = page_size;
    return *this;
}

DataflowBufferConfig& DataflowBufferConfig::set_access_pattern(const AccessPattern& pattern) {
    access_patterns_ = pattern;
    return *this;
}

uint32_t DataflowBufferConfig::total_size() const { return total_size_; }

tt::DataFormat DataflowBufferConfig::data_format() const { return data_format_; }

uint32_t DataflowBufferConfig::page_size() const { return page_size_; }

const DataflowBufferConfig::AccessPattern& DataflowBufferConfig::access_pattern() const { return access_patterns_; }

DataflowBufferConfig::Builder DataflowBufferConfig::Builder::LocalBuilder(DataflowBufferConfig& parent) {
    auto builder = Builder(parent);
    return builder;
}

DataflowBufferConfig::Builder::Builder(DataflowBufferConfig& parent) : parent_(parent) {}

const DataflowBufferConfig::Builder& DataflowBufferConfig::Builder::set_data_format(tt::DataFormat data_format) const {
    parent_.data_format_ = data_format;
    return *this;
}

const DataflowBufferConfig::Builder& DataflowBufferConfig::Builder::set_total_size(uint32_t total_size) const {
    parent_.set_total_size(total_size);
    return *this;
}

const DataflowBufferConfig::Builder& DataflowBufferConfig::Builder::set_page_size(uint32_t page_size) const {
    parent_.set_page_size(page_size);
    return *this;
}

const DataflowBufferConfig::Builder& DataflowBufferConfig::Builder::set_access_pattern(
    const AccessPattern& pattern) const {
    parent_.set_access_pattern(pattern);
    return *this;
}

DataflowBufferConfig::Builder DataflowBufferConfig::builder() { return Builder::LocalBuilder(*this); }

bool operator==(const DataflowBufferConfig::AccessPattern& lhs, const DataflowBufferConfig::AccessPattern& rhs) {
    return lhs.write_pattern == rhs.write_pattern && lhs.read_pattern == rhs.read_pattern &&
           lhs.num_reader_threads == rhs.num_reader_threads && lhs.num_writer_threads == rhs.num_writer_threads;
}

bool operator!=(const DataflowBufferConfig::AccessPattern& lhs, const DataflowBufferConfig::AccessPattern& rhs) {
    return !(lhs == rhs);
}

bool operator==(const DataflowBufferConfig& lhs, const DataflowBufferConfig& rhs) {
    return lhs.total_size() == rhs.total_size() && lhs.data_format() == rhs.data_format() &&
           lhs.page_size() == rhs.page_size() && lhs.access_pattern() == rhs.access_pattern();
}

bool operator!=(const DataflowBufferConfig& lhs, const DataflowBufferConfig& rhs) { return !(lhs == rhs); }

// this needs to return different logical space for the GlobalDFBs
uint8_t CreateDataflowBuffer(
    const DataflowBufferConfig& config, const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec) {
    // This needs to be per core but for now (local DFB testing)leave this
    static uint8_t logical_dfb_index = 0;
    if (logical_dfb_index == 64) {
        TT_THROW("Exceeded max number of Dataflow Buffers");
    }
    uint8_t current_index = logical_dfb_index++;

    // Do register allocation
    static uint64_t register_allocator_mask = 0;
    TT_FATAL(register_allocator_mask != UINT64_MAX, "No available registers for dataflow buffer");

    // Calculate number of registers to allocate based on thread count + access pattern
    // hardcoded for now...
    uint32_t num_registers_allocated = 0;
    auto access_pattern = config.access_pattern();
    if (access_pattern.write_pattern == DataflowBufferAccessPattern::STRIDED &&
        access_pattern.read_pattern == DataflowBufferAccessPattern::STRIDED) {
        num_registers_allocated = std::max(access_pattern.num_reader_threads, access_pattern.num_writer_threads);
    } else {
        TT_THROW("Unsupported case for register allocation");
    }

    // Ensure we don't exceed the 64-bit mask capacity
    TT_FATAL(num_registers_allocated <= 64, "Cannot allocate more than 64 registers");

    // Find the first non-set bit
    uint32_t first_non_set_bit = 0;
    first_non_set_bit = __builtin_ctzll(~register_allocator_mask);

    TT_FATAL(first_non_set_bit + num_registers_allocated <= 64, "Not enough registers available to allocate");

    // Set the required number of consecutive bits starting from first_non_set_bit
    uint64_t mask_to_set = ((1ULL << num_registers_allocated) - 1) << first_non_set_bit;
    register_allocator_mask |= mask_to_set;

    // Set the logical to physical reg mapping
    dev::dfb_to_register_allocation[current_index] = mask_to_set;
    log_info(
        tt::LogMetal,
        "Logical DFB {}'s register allocation: {:#x} (set bits are the hw register indices)",
        (uint32_t)current_index,
        mask_to_set);

    // Iterate over all set bits in mask_to_set
    std::cout << "Set bit positions (HW register indices): ";
    uint64_t temp_mask = mask_to_set;
    uint16_t capacity = config.total_size() / config.page_size();
    std::cout << "capacity " << capacity << std::endl;
    while (temp_mask) {
        int bit_position = __builtin_ctzll(temp_mask);  // Find lowest set bit
        std::cout << bit_position << " ";
        dev::overlay_cluster_instances[current_index].set_capacity(capacity);

        temp_mask &= (temp_mask - 1);  // Clear the lowest set bit
    }
    std::cout << std::endl;

    return current_index;
}

}  // namespace tt::tt_metal
