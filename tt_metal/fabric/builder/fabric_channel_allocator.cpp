// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_channel_allocator.hpp"
#include <tt_stl/assert.hpp>
#include <algorithm>
#include <numeric>

namespace tt::tt_fabric {

// Base FabricChannelAllocator implementation
FabricChannelAllocator::FabricChannelAllocator(const std::vector<MemoryRegion>& memory_regions) :
    memory_regions_(memory_regions) {
    TT_FATAL(!memory_regions_.empty(), "At least one memory region must be provided");

    // Validate that regions don't overlap
    for (size_t i = 0; i < memory_regions_.size(); ++i) {
        for (size_t j = i + 1; j < memory_regions_.size(); ++j) {
            const auto& region1 = memory_regions_[i];
            const auto& region2 = memory_regions_[j];

            // Check for overlap: regions overlap if one starts before the other ends
            bool overlap =
                (region1.start_address < region2.end_address) && (region2.start_address < region1.end_address);
            TT_FATAL(!overlap, "Memory regions {} and {} overlap", i, j);
        }
    }
}

size_t FabricChannelAllocator::get_total_available_memory() const {
    return std::accumulate(
        memory_regions_.begin(), memory_regions_.end(), size_t{0}, [](size_t sum, const MemoryRegion& region) {
            return sum + region.get_size();
        });
}

// StaticChannelsAllocator implementation
StaticChannelsAllocator::StaticChannelsAllocator(
    const std::vector<MemoryRegion>& memory_regions,
    size_t num_sender_channels,
    size_t num_receiver_channels,
    size_t buffer_size_bytes,
    size_t buffers_per_sender_channel,
    size_t buffers_per_receiver_channel) :
    FabricChannelAllocator(memory_regions),
    num_sender_channels_(num_sender_channels),
    num_receiver_channels_(num_receiver_channels),
    buffer_size_bytes_(buffer_size_bytes),
    buffers_per_sender_channel_(buffers_per_sender_channel),
    buffers_per_receiver_channel_(buffers_per_receiver_channel) {
    TT_FATAL(num_sender_channels_ > 0, "Number of sender channels must be > 0");
    TT_FATAL(num_receiver_channels_ > 0, "Number of receiver channels must be > 0");
    TT_FATAL(buffer_size_bytes_ > 0, "Buffer size must be > 0");
    TT_FATAL(buffers_per_sender_channel_ > 0, "Buffers per sender channel must be > 0");
    TT_FATAL(buffers_per_receiver_channel_ > 0, "Buffers per receiver channel must be > 0");

    allocate_channels();
}

void StaticChannelsAllocator::allocate_channels() {
    // Calculate total memory needed
    size_t total_sender_memory = num_sender_channels_ * buffers_per_sender_channel_ * buffer_size_bytes_;
    size_t total_receiver_memory = num_receiver_channels_ * buffers_per_receiver_channel_ * buffer_size_bytes_;
    size_t total_memory_needed = total_sender_memory + total_receiver_memory;

    TT_FATAL(
        total_memory_needed <= get_total_available_memory(),
        "Insufficient memory: need {} bytes, have {} bytes",
        total_memory_needed,
        get_total_available_memory());

    // Initialize address vectors
    sender_channel_base_addresses_.resize(num_sender_channels_);
    receiver_channel_base_addresses_.resize(num_receiver_channels_);

    // Allocate from the first region (static allocation uses first available region)
    const auto& region = memory_regions_[0];
    size_t current_address = region.start_address;

    // Allocate sender channels
    for (size_t i = 0; i < num_sender_channels_; ++i) {
        sender_channel_base_addresses_[i] = current_address;
        current_address += buffers_per_sender_channel_ * buffer_size_bytes_;
    }

    // Allocate receiver channels
    for (size_t i = 0; i < num_receiver_channels_; ++i) {
        receiver_channel_base_addresses_[i] = current_address;
        current_address += buffers_per_receiver_channel_ * buffer_size_bytes_;
    }

    TT_FATAL(current_address <= region.end_address, "Channel allocation exceeded available memory in region");
}

void StaticChannelsAllocator::emit_ct_args(std::vector<uint32_t>& ct_args) const {
    // Add allocation type identifier
    ct_args.push_back(0);  // 0 = StaticChannelsAllocator

    // Add basic configuration
    ct_args.push_back(static_cast<uint32_t>(num_sender_channels_));
    ct_args.push_back(static_cast<uint32_t>(num_receiver_channels_));
    ct_args.push_back(static_cast<uint32_t>(buffer_size_bytes_));
    ct_args.push_back(static_cast<uint32_t>(buffers_per_sender_channel_));
    ct_args.push_back(static_cast<uint32_t>(buffers_per_receiver_channel_));

    // Add sender channel base addresses
    for (size_t i = 0; i < num_sender_channels_; ++i) {
        ct_args.push_back(static_cast<uint32_t>(sender_channel_base_addresses_[i]));
    }

    // Add receiver channel base addresses
    for (size_t i = 0; i < num_receiver_channels_; ++i) {
        ct_args.push_back(static_cast<uint32_t>(receiver_channel_base_addresses_[i]));
    }
}

size_t StaticChannelsAllocator::get_sender_channel_base_address(size_t channel_id) const {
    TT_FATAL(channel_id < num_sender_channels_, "Sender channel ID {} out of bounds", channel_id);
    return sender_channel_base_addresses_[channel_id];
}

size_t StaticChannelsAllocator::get_receiver_channel_base_address(size_t channel_id) const {
    TT_FATAL(channel_id < num_receiver_channels_, "Receiver channel ID {} out of bounds", channel_id);
    return receiver_channel_base_addresses_[channel_id];
}

// ElasticChannelsAllocator implementation
ElasticChannelsAllocator::ElasticChannelsAllocator(
    const std::vector<MemoryRegion>& memory_regions,
    size_t buffer_size_bytes,
    size_t min_buffers_per_channel,
    size_t max_buffers_per_channel) :
    FabricChannelAllocator(memory_regions),
    buffer_size_bytes_(buffer_size_bytes),
    min_buffers_per_channel_(min_buffers_per_channel),
    max_buffers_per_channel_(max_buffers_per_channel),
    num_sender_channels_(0),
    num_receiver_channels_(0),
    allocation_successful_(false) {
    TT_FATAL(buffer_size_bytes_ > 0, "Buffer size must be > 0");
    TT_FATAL(min_buffers_per_channel_ > 0, "Min buffers per channel must be > 0");
    TT_FATAL(
        max_buffers_per_channel_ >= min_buffers_per_channel_,
        "Max buffers per channel must be >= min buffers per channel");
}

bool ElasticChannelsAllocator::allocate_channels(size_t num_sender_channels, size_t num_receiver_channels) {
    num_sender_channels_ = num_sender_channels;
    num_receiver_channels_ = num_receiver_channels;

    TT_FATAL(num_sender_channels_ > 0, "Number of sender channels must be > 0");
    TT_FATAL(num_receiver_channels_ > 0, "Number of receiver channels must be > 0");

    size_t total_channels = num_sender_channels_ + num_receiver_channels_;
    size_t available_memory = get_total_available_memory();

    // Calculate optimal buffers per channel
    size_t optimal_buffers = calculate_optimal_buffers_per_channel(total_channels, available_memory);

    // Clamp to valid range
    optimal_buffers = std::max(min_buffers_per_channel_, std::min(max_buffers_per_channel_, optimal_buffers));

    // Check if allocation is possible
    size_t total_memory_needed = total_channels * optimal_buffers * buffer_size_bytes_;
    if (total_memory_needed > available_memory) {
        allocation_successful_ = false;
        return false;
    }

    // Initialize buffer count vectors
    sender_channel_buffer_counts_.resize(num_sender_channels_, optimal_buffers);
    receiver_channel_buffer_counts_.resize(num_receiver_channels_, optimal_buffers);

    // Initialize address vectors
    sender_channel_base_addresses_.resize(num_sender_channels_);
    receiver_channel_base_addresses_.resize(num_receiver_channels_);

    // Allocate from the first region
    const auto& region = memory_regions_[0];
    size_t current_address = region.start_address;

    // Allocate sender channels
    for (size_t i = 0; i < num_sender_channels_; ++i) {
        sender_channel_base_addresses_[i] = current_address;
        current_address += sender_channel_buffer_counts_[i] * buffer_size_bytes_;
    }

    // Allocate receiver channels
    for (size_t i = 0; i < num_receiver_channels_; ++i) {
        receiver_channel_base_addresses_[i] = current_address;
        current_address += receiver_channel_buffer_counts_[i] * buffer_size_bytes_;
    }

    allocation_successful_ = true;
    return true;
}

size_t ElasticChannelsAllocator::calculate_optimal_buffers_per_channel(
    size_t total_channels, size_t available_memory) const {
    if (total_channels == 0) {
        return min_buffers_per_channel_;
    }

    // Calculate how many buffers we can fit per channel
    size_t max_possible_buffers = available_memory / (total_channels * buffer_size_bytes_);

    // Return the maximum possible, clamped to our valid range
    return std::max(min_buffers_per_channel_, std::min(max_buffers_per_channel_, max_possible_buffers));
}

void ElasticChannelsAllocator::emit_ct_args(std::vector<uint32_t>& ct_args) const {
    TT_FATAL(allocation_successful_, "Channels must be allocated before emitting CT args");

    // Add allocation type identifier
    ct_args.push_back(1);  // 1 = ElasticChannelsAllocator

    // Add basic configuration
    ct_args.push_back(static_cast<uint32_t>(num_sender_channels_));
    ct_args.push_back(static_cast<uint32_t>(num_receiver_channels_));
    ct_args.push_back(static_cast<uint32_t>(buffer_size_bytes_));
    ct_args.push_back(static_cast<uint32_t>(min_buffers_per_channel_));
    ct_args.push_back(static_cast<uint32_t>(max_buffers_per_channel_));

    // Add sender channel buffer counts
    for (size_t i = 0; i < num_sender_channels_; ++i) {
        ct_args.push_back(static_cast<uint32_t>(sender_channel_buffer_counts_[i]));
    }

    // Add receiver channel buffer counts
    for (size_t i = 0; i < num_receiver_channels_; ++i) {
        ct_args.push_back(static_cast<uint32_t>(receiver_channel_buffer_counts_[i]));
    }

    // Add sender channel base addresses
    for (size_t i = 0; i < num_sender_channels_; ++i) {
        ct_args.push_back(static_cast<uint32_t>(sender_channel_base_addresses_[i]));
    }

    // Add receiver channel base addresses
    for (size_t i = 0; i < num_receiver_channels_; ++i) {
        ct_args.push_back(static_cast<uint32_t>(receiver_channel_base_addresses_[i]));
    }
}

size_t ElasticChannelsAllocator::get_sender_channel_buffer_count(size_t channel_id) const {
    TT_FATAL(allocation_successful_, "Channels must be allocated first");
    TT_FATAL(channel_id < num_sender_channels_, "Sender channel ID {} out of bounds", channel_id);
    return sender_channel_buffer_counts_[channel_id];
}

size_t ElasticChannelsAllocator::get_receiver_channel_buffer_count(size_t channel_id) const {
    TT_FATAL(allocation_successful_, "Channels must be allocated first");
    TT_FATAL(channel_id < num_receiver_channels_, "Receiver channel ID {} out of bounds", channel_id);
    return receiver_channel_buffer_counts_[channel_id];
}

size_t ElasticChannelsAllocator::get_sender_channel_base_address(size_t channel_id) const {
    TT_FATAL(allocation_successful_, "Channels must be allocated first");
    TT_FATAL(channel_id < num_sender_channels_, "Sender channel ID {} out of bounds", channel_id);
    return sender_channel_base_addresses_[channel_id];
}

size_t ElasticChannelsAllocator::get_receiver_channel_base_address(size_t channel_id) const {
    TT_FATAL(allocation_successful_, "Channels must be allocated first");
    TT_FATAL(channel_id < num_receiver_channels_, "Receiver channel ID {} out of bounds", channel_id);
    return receiver_channel_base_addresses_[channel_id];
}

}  // namespace tt::tt_fabric
