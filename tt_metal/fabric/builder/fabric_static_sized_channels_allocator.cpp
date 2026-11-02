// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-logger/tt-logger.hpp>
#include <enchantum/enchantum.hpp>
#include <tt_stl/indestructible.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/fmt.hpp>
#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <string_view>

namespace tt::tt_fabric {

size_t FabricStaticSizedChannelsAllocator::get_sender_channel_base_address(size_t vc_id, size_t channel_id) const {
    TT_FATAL(
        vc_id < builder_config::MAX_NUM_VCS, "VC ID {} out of bounds (max {})", vc_id, builder_config::MAX_NUM_VCS);
    TT_FATAL(
        channel_id < sender_channels_base_address[vc_id].size(),
        "Sender channel ID {} out of bounds for VC{}",
        channel_id,
        vc_id);
    return sender_channels_base_address[vc_id][channel_id];
}

size_t FabricStaticSizedChannelsAllocator::get_sender_channel_number_of_slots(size_t vc_id, size_t channel_id) const {
    TT_FATAL(
        vc_id < builder_config::MAX_NUM_VCS, "VC ID {} out of bounds (max {})", vc_id, builder_config::MAX_NUM_VCS);
    TT_FATAL(
        channel_id < sender_channels_num_buffers[vc_id].size(),
        "Sender channel ID {} out of bounds for VC{}",
        channel_id,
        vc_id);
    return sender_channels_num_buffers[vc_id][channel_id];
}

size_t FabricStaticSizedChannelsAllocator::get_receiver_channel_number_of_slots(size_t vc_id, size_t channel_id) const {
    TT_FATAL(
        vc_id < builder_config::MAX_NUM_VCS, "VC ID {} out of bounds (max {})", vc_id, builder_config::MAX_NUM_VCS);
    TT_FATAL(
        channel_id < receiver_channels_num_buffers[vc_id].size(),
        "Receiver channel ID {} out of bounds for VC{}",
        channel_id,
        vc_id);
    return receiver_channels_num_buffers[vc_id][channel_id];
}

size_t FabricStaticSizedChannelsAllocator::get_remote_receiver_channel_number_of_slots(
    size_t vc_id, size_t channel_id) const {
    TT_FATAL(
        vc_id < builder_config::MAX_NUM_VCS, "VC ID {} out of bounds (max {})", vc_id, builder_config::MAX_NUM_VCS);
    TT_FATAL(
        channel_id < remote_receiver_channels_num_buffers[vc_id].size(),
        "Remote receiver channel ID {} out of bounds for VC{}",
        channel_id,
        vc_id);
    return remote_receiver_channels_num_buffers[vc_id][channel_id];
}

size_t FabricStaticSizedChannelsAllocator::get_receiver_channel_base_address(size_t vc_id, size_t channel_id) const {
    TT_FATAL(
        vc_id < builder_config::MAX_NUM_VCS, "VC ID {} out of bounds (max {})", vc_id, builder_config::MAX_NUM_VCS);
    TT_FATAL(
        channel_id < receiver_channels_base_address[vc_id].size(),
        "Receiver channel ID {} out of bounds for VC{}",
        channel_id,
        vc_id);
    return receiver_channels_base_address[vc_id][channel_id];
}

namespace {
// Experimental caps on VC0 channel depth for latency/credit-loop sweeps (0 = no cap).
size_t fabric_experiment_slot_cap(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::strtoul(value, nullptr, 0) : 0;
}
}  // namespace

FabricStaticSizedChannelsAllocator::FabricStaticSizedChannelsAllocator(
    tt::tt_fabric::Topology topology,
    const FabricEriscDatamoverOptions& options,
    const std::array<size_t, builder_config::MAX_NUM_VCS>& num_used_sender_channels_per_vc,
    const std::array<size_t, builder_config::MAX_NUM_VCS>& num_used_receiver_channels_per_vc,
    size_t channel_buffer_size_bytes,
    size_t available_channel_buffering_space,
    const std::vector<MemoryRegion>& memory_regions) :
    FabricChannelAllocator(topology, options, memory_regions),
    num_used_sender_channels_per_vc(num_used_sender_channels_per_vc),
    num_used_receiver_channels_per_vc(num_used_receiver_channels_per_vc),
    channel_buffer_size_bytes(channel_buffer_size_bytes),
    available_channel_buffering_space(available_channel_buffering_space) {
    // Compute buffer region start from memory regions
    TT_FATAL(!memory_regions.empty(), "Memory regions must not be empty");
    this->buffer_region_start = memory_regions[0].start_address;

    // Set max_l1_loading_size from memory regions
    this->max_l1_loading_size = memory_regions[0].get_end_address();
    for (const auto& region : memory_regions) {
        this->max_l1_loading_size = std::max(this->max_l1_loading_size, region.get_end_address());
    }

    // Per-VC buffer slot arrays
    std::array<std::array<size_t, builder_config::num_max_sender_channels>, builder_config::MAX_NUM_VCS>
        num_sender_buffer_slots_per_vc = {};
    std::array<std::array<size_t, builder_config::num_max_sender_channels>, builder_config::MAX_NUM_VCS>
        num_remote_sender_buffer_slots_per_vc = {};
    std::array<std::array<size_t, builder_config::num_max_receiver_channels>, builder_config::MAX_NUM_VCS>
        num_receiver_buffer_slots_per_vc = {};
    std::array<std::array<size_t, builder_config::num_max_receiver_channels>, builder_config::MAX_NUM_VCS>
        num_remote_receiver_buffer_slots_per_vc = {};

    bool has_tensix_extension = options.fabric_tensix_config != tt::tt_fabric::FabricTensixConfig::DISABLED;

    configure_buffer_slots_helper(
        topology,
        options,
        num_sender_buffer_slots_per_vc,
        num_remote_sender_buffer_slots_per_vc,
        num_receiver_buffer_slots_per_vc,
        num_remote_receiver_buffer_slots_per_vc);

    // Calculate total slots across all VCs
    size_t total_slot_count = 0;
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        size_t vc_sender_slots = std::accumulate(
            num_sender_buffer_slots_per_vc[vc].begin(),
            num_sender_buffer_slots_per_vc[vc].begin() + num_used_sender_channels_per_vc[vc],
            size_t{0});
        size_t vc_receiver_slots = std::accumulate(
            num_receiver_buffer_slots_per_vc[vc].begin(),
            num_receiver_buffer_slots_per_vc[vc].begin() + num_used_receiver_channels_per_vc[vc],
            size_t{0});
        total_slot_count += vc_sender_slots + vc_receiver_slots;

        log_trace(tt::LogFabric, "VC{} num_sender_buffer_slots: {}", vc, num_sender_buffer_slots_per_vc[vc]);
        if (std::getenv("TT_FABRIC_EXP_LOG_SLOTS") != nullptr && num_used_sender_channels_per_vc[vc] > 0) {
            log_info(
                tt::LogFabric,
                "VC{} sender slots {} receiver slots {} (channel buffer {} B; buffering space {} B = {} slots)",
                vc,
                std::vector<size_t>(
                    num_sender_buffer_slots_per_vc[vc].begin(),
                    num_sender_buffer_slots_per_vc[vc].begin() + num_used_sender_channels_per_vc[vc]),
                std::vector<size_t>(
                    num_receiver_buffer_slots_per_vc[vc].begin(),
                    num_receiver_buffer_slots_per_vc[vc].begin() + num_used_receiver_channels_per_vc[vc]),
                channel_buffer_size_bytes,
                available_channel_buffering_space,
                available_channel_buffering_space / channel_buffer_size_bytes);
        }
        log_trace(
            tt::LogFabric, "VC{} num_remote_sender_buffer_slots: {}", vc, num_remote_sender_buffer_slots_per_vc[vc]);
        log_trace(tt::LogFabric, "VC{} num_receiver_buffer_slots: {}", vc, num_receiver_buffer_slots_per_vc[vc]);
        log_trace(
            tt::LogFabric,
            "VC{} num_remote_receiver_buffer_slots: {}",
            vc,
            num_remote_receiver_buffer_slots_per_vc[vc]);
    }
    TT_FATAL(
        total_slot_count * channel_buffer_size_bytes <= available_channel_buffering_space,
        "Total channel size of {} B exceeds available space of {} B",
        total_slot_count * channel_buffer_size_bytes,
        available_channel_buffering_space);

    log_trace(tt::LogFabric, "Available channel buffering space: {}", this->available_channel_buffering_space);

    // Set channel sizes and num buffers per VC
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        // set the sender channel sizes and num buffers
        for (uint32_t i = 0; i < num_used_sender_channels_per_vc[vc]; i++) {
            this->sender_channels_size_bytes[vc][i] = channel_buffer_size_bytes * num_sender_buffer_slots_per_vc[vc][i];
            this->sender_channels_num_buffers[vc][i] = num_sender_buffer_slots_per_vc[vc][i];
        }
        // set the remote sender channel sizes and num buffers
        for (uint32_t i = 0; i < num_used_sender_channels_per_vc[vc]; i++) {
            this->remote_sender_channels_size_bytes[vc][i] =
                channel_buffer_size_bytes * num_remote_sender_buffer_slots_per_vc[vc][i];
            this->remote_sender_channels_num_buffers[vc][i] = num_remote_sender_buffer_slots_per_vc[vc][i];
        }
        // set the local receiver channel sizes and num buffers
        for (uint32_t i = 0; i < num_used_receiver_channels_per_vc[vc]; i++) {
            this->receiver_channels_size_bytes[vc][i] =
                channel_buffer_size_bytes * num_receiver_buffer_slots_per_vc[vc][i];
            this->receiver_channels_num_buffers[vc][i] = num_receiver_buffer_slots_per_vc[vc][i];
        }
        // set the remote receiver channel sizes and num buffers
        for (uint32_t i = 0; i < num_used_receiver_channels_per_vc[vc]; i++) {
            this->remote_receiver_channels_size_bytes[vc][i] =
                channel_buffer_size_bytes * num_remote_receiver_buffer_slots_per_vc[vc][i];
            this->remote_receiver_channels_num_buffers[vc][i] = num_remote_receiver_buffer_slots_per_vc[vc][i];
        }
    }

    // set the base addresses for the local channels (allocate sequentially: VC0 channels, then VC1 channels)
    uint32_t sender_buffer_addr = buffer_region_start;
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (uint32_t i = 0; i < num_used_sender_channels_per_vc[vc]; i++) {
            this->sender_channels_base_address[vc][i] = sender_buffer_addr;
            sender_buffer_addr += this->sender_channels_size_bytes[vc][i];
            log_trace(
                tt::LogFabric, "VC{} Sender {} channel_start: {}", vc, i, this->sender_channels_base_address[vc][i]);
        }
    }

    uint32_t receiver_buffer_addr = sender_buffer_addr;
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (uint32_t i = 0; i < num_used_receiver_channels_per_vc[vc]; i++) {
            this->receiver_channels_base_address[vc][i] = receiver_buffer_addr;
            receiver_buffer_addr += this->receiver_channels_size_bytes[vc][i];
            log_trace(
                tt::LogFabric,
                "VC{} Receiver {} channel_start: {}",
                vc,
                i,
                this->receiver_channels_base_address[vc][i]);
        }
    }
    uint32_t buffer_addr_end = receiver_buffer_addr;

    // set the base addresses for the remote channels
    uint32_t remote_sender_buffer_addr = buffer_region_start;
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (uint32_t i = 0; i < num_used_sender_channels_per_vc[vc]; i++) {
            this->remote_sender_channels_base_address[vc][i] = remote_sender_buffer_addr;
            remote_sender_buffer_addr += this->remote_sender_channels_size_bytes[vc][i];
            log_trace(
                tt::LogFabric,
                "VC{} Remote Sender {} channel_start: {}",
                vc,
                i,
                this->remote_sender_channels_base_address[vc][i]);
        }
    }

    uint32_t remote_receiver_buffer_addr = remote_sender_buffer_addr;
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (uint32_t i = 0; i < num_used_receiver_channels_per_vc[vc]; i++) {
            this->remote_receiver_channels_base_address[vc][i] = remote_receiver_buffer_addr;
            remote_receiver_buffer_addr += this->remote_receiver_channels_size_bytes[vc][i];
            log_trace(
                tt::LogFabric,
                "VC{} Remote Receiver {} channel_start: {}",
                vc,
                i,
                this->remote_receiver_channels_base_address[vc][i]);
        }
    }

    log_trace(tt::LogFabric, "Available channel buffering space: {}", this->available_channel_buffering_space);

    auto skip_current_sender_channel = [&](uint32_t vc_id, uint32_t idx) -> bool {
        // for fabric with tensix extension, only check the worker channel (VC0 channel 0), other channels are skipped
        uint32_t target_channel = get_worker_connected_sender_channel();
        return !((vc_id == 0 && idx == target_channel) || !has_tensix_extension);
    };

    // Validate sender channels per VC
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (uint32_t i = 0; i < num_used_sender_channels_per_vc[vc]; i++) {
            if (!skip_current_sender_channel(vc, i)) {
                TT_FATAL(
                    this->sender_channels_size_bytes[vc][i] > 0,
                    "Internal error when computing `sender_channels_size_bytes[VC{}][{}]` which was computed to be "
                    "size 0",
                    vc,
                    i);
            }
        }
    }

    // Validate receiver channels per VC
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (uint32_t i = 0; i < num_used_receiver_channels_per_vc[vc]; i++) {
            TT_FATAL(
                this->receiver_channels_size_bytes[vc][i] > 0,
                "Internal error when computing `receiver_channels_size_bytes[VC{}][{}]` which was computed to be size "
                "0",
                vc,
                i);
        }
    }

    // Validate total size across all VCs
    size_t total_size = 0;
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        total_size += std::accumulate(
            this->sender_channels_size_bytes[vc].begin(),
            this->sender_channels_size_bytes[vc].begin() + num_used_sender_channels_per_vc[vc],
            0ul);
        total_size += std::accumulate(
            this->receiver_channels_size_bytes[vc].begin(),
            this->receiver_channels_size_bytes[vc].begin() + num_used_receiver_channels_per_vc[vc],
            0ul);
    }

    TT_FATAL(
        total_size <= this->available_channel_buffering_space,
        "Internal error when computing channel sizes. Total channel size {} exceeds available space {}",
        total_size,
        this->available_channel_buffering_space);
    TT_FATAL(
        buffer_addr_end <= this->max_l1_loading_size,
        "Internal error - channel buffers spilled past the end of usable L1 region.");
}

void FabricStaticSizedChannelsAllocator::configure_buffer_slots_helper(
    Topology topology,
    const FabricEriscDatamoverOptions& options,
    std::array<std::array<size_t, builder_config::num_max_sender_channels>, builder_config::MAX_NUM_VCS>&
        num_sender_buffer_slots_per_vc,
    std::array<std::array<size_t, builder_config::num_max_sender_channels>, builder_config::MAX_NUM_VCS>&
        num_remote_sender_buffer_slots_per_vc,
    std::array<std::array<size_t, builder_config::num_max_receiver_channels>, builder_config::MAX_NUM_VCS>&
        num_receiver_buffer_slots_per_vc,
    std::array<std::array<size_t, builder_config::num_max_receiver_channels>, builder_config::MAX_NUM_VCS>&
        num_remote_receiver_buffer_slots_per_vc) {
    // Per-VC buffer slot configuration: {vc0_sender, vc0_receiver, vc1_sender, vc1_receiver, vc2_sender, vc2_receiver}
    struct PerVcBufferSlots {
        size_t vc0_sender_slots;
        size_t vc0_receiver_slots;
        size_t vc1_sender_slots;
        size_t vc1_receiver_slots;
        size_t vc2_sender_slots;
        size_t vc2_receiver_slots;
    };

    // fabric with tensix extension uses different buffer slots options, since only one or two sender channels are
    // used by fabric router, while other sender channels are skipped and have 0 buffer slots.
    // Format: {vc0_sender, vc0_receiver, vc1_sender, vc1_receiver}
    static const std::vector<std::vector<PerVcBufferSlots>> default_with_tensix_buffer_slot_options = {
        // WORMHOLE_B0
        {
            {16, 16, 0, 0, 0, 0},  // Option 1
            {8, 16, 0, 0, 0, 0},   // Option 2
            {8, 8, 0, 0, 0, 0},    // Option 3
            {4, 8, 0, 0, 0, 0},    // Option 4
            {4, 4, 0, 0, 0, 0},    // Option 5: VC0 only, smaller
            {2, 4, 0, 0, 0, 0},    // Option 6: VC0 only, smaller
            {2, 2, 0, 0, 0, 0},    // Option 7: VC0 only, smaller
            {1, 2, 0, 0, 0, 0},    // Option 8: VC0 only, smallest
            {1, 1, 0, 0, 0, 0},    // Option 9: VC0 only, smallest
            {4, 8, 4, 4, 0, 0},    // Option 10: supports VC0+VC1
            {4, 8, 2, 2, 0, 0},    // Option 11: supports VC0+VC1
            {4, 4, 2, 2, 0, 0},    // Option 12: supports VC0+VC1
            {2, 2, 2, 2, 0, 0}     // Option 13: supports VC0+VC1
        },
        // BLACKHOLE
        {
            {32, 32, 0, 0, 0, 0},  // Option 1
            {16, 32, 0, 0, 0, 0},  // Option 2
            {16, 16, 0, 0, 0, 0},  // Option 3
            {8, 16, 0, 0, 0, 0},   // Option 4
            {8, 8, 0, 0, 0, 0},    // Option 5
            {4, 8, 0, 0, 0, 0},    // Option 6
            {4, 4, 0, 0, 0, 0},    // Option 7
            {2, 4, 0, 0, 0, 0},    // Option 8
            {2, 2, 0, 0, 0, 0},    // Option 9
            {1, 2, 0, 0, 0, 0},    // Option 10
            {1, 1, 0, 0, 0, 0},    // Option 11
            {4, 8, 2, 4, 0, 0},    // Option 12: supports VC0+VC1
            {4, 8, 2, 2, 0, 0},    // Option 13: supports VC0+VC1, smaller VC1 receiver
            {2, 4, 2, 2, 0, 0},    // Option 14: supports VC0+VC1, smaller overall
            {2, 4, 1, 1, 0, 0},    // Option 15: supports VC0+VC1, smaller overall
            {2, 2, 1, 1, 0, 0},    // Option 16: supports VC0+VC1, smaller overall
            {1, 1, 1, 1, 0, 0}     // Option 17: supports VC0+VC1, smaller overall

        }};

    auto get_num_buffer_slots = [](Topology topology,
                                   size_t arch_index,
                                   bool vc1_present,
                                   bool vc2_present) -> const std::vector<PerVcBufferSlots>& {
        // Architecture-specific buffer slot configurations per VC, split by active VC count
        // Format: {vc0_sender, vc0_receiver, vc1_sender, vc1_receiver, vc2_sender, vc2_receiver}

        // VC0-only mesh options: vc1 slots = 0, vc2 slots = 0
        static const std::vector<std::vector<PerVcBufferSlots>> vc0_only_mesh_buffer_slot_options = {
            // WORMHOLE_B0
            {
                {7, 11, 0, 0, 0, 0},  // Option 1: VC0 only
                {4, 8, 0, 0, 0, 0},   // Option 2: VC0 only, smaller
                {4, 4, 0, 0, 0, 0},   // Option 3: VC0 only, smaller
                {2, 4, 0, 0, 0, 0},   // Option 4: VC0 only, smaller
                {2, 2, 0, 0, 0, 0},   // Option 5: VC0 only, smaller
                {1, 2, 0, 0, 0, 0},   // Option 6: VC0 only, smallest
                {1, 1, 0, 0, 0, 0}    // Option 7: VC0 only, smallest
            },
            // BLACKHOLE
            {
                {32, 32, 0, 0, 0, 0},  // Option 1: VC0 only
                {16, 32, 0, 0, 0, 0},  // Option 2: VC0 only
                {16, 16, 0, 0, 0, 0},  // Option 3: VC0 only
                {8, 16, 0, 0, 0, 0},   // Option 4: VC0 only
                {8, 8, 0, 0, 0, 0},    // Option 5: VC0 only
                {4, 8, 0, 0, 0, 0},    // Option 6: VC0 only
                {4, 4, 0, 0, 0, 0},    // Option 7: VC0 only, smaller
                {2, 4, 0, 0, 0, 0},    // Option 8: VC0 only, smaller
                {2, 2, 0, 0, 0, 0},    // Option 9: VC0 only, smaller
                {1, 2, 0, 0, 0, 0},    // Option 10: VC0 only, smallest
                {1, 1, 0, 0, 0, 0}     // Option 11: VC0 only, smallest
            }};

        // VC0+VC1 mesh options: vc1 slots > 0, vc2 slots = 0
        static const std::vector<std::vector<PerVcBufferSlots>> vc0_vc1_mesh_buffer_slot_options = {
            // WORMHOLE_B0
            {
                {4, 8, 2, 4, 0, 0},  // Option 1: supports VC0+VC1
                {4, 8, 2, 2, 0, 0},  // Option 2: supports VC0+VC1, smaller VC1 receiver
                {2, 4, 2, 2, 0, 0},  // Option 3: supports VC0+VC1, smaller overall
                {2, 4, 1, 1, 0, 0},  // Option 4: supports VC0+VC1, smaller overall
                {2, 2, 1, 1, 0, 0},  // Option 5: supports VC0+VC1, smaller overall
                {1, 1, 1, 1, 0, 0}   // Option 6: supports VC0+VC1, smallest
            },
            // BLACKHOLE
            {
                {4, 8, 2, 4, 0, 0},  // Option 1: supports VC0+VC1
                {4, 8, 2, 2, 0, 0},  // Option 2: supports VC0+VC1, smaller VC1 receiver
                {2, 4, 2, 2, 0, 0},  // Option 3: supports VC0+VC1, smaller overall
                {2, 4, 1, 1, 0, 0},  // Option 4: supports VC0+VC1, smaller overall
                {2, 2, 1, 1, 0, 0},  // Option 5: supports VC0+VC1, smaller overall
                {1, 1, 1, 1, 0, 0}   // Option 6: supports VC0+VC1, smallest
            }};

        // VC0+VC1+VC2 mesh options: vc2 mirrors vc1 values in each row
        static const std::vector<std::vector<PerVcBufferSlots>> vc0_vc1_vc2_mesh_buffer_slot_options = {
            // WORMHOLE_B0
            {
                {4, 8, 2, 4, 2, 4},  // Option 1: supports VC0+VC1+VC2
                {4, 8, 2, 2, 2, 2},  // Option 2: supports VC0+VC1+VC2, smaller VC1/VC2 receiver
                {2, 4, 2, 2, 2, 2},  // Option 3: supports VC0+VC1+VC2, smaller overall
                {2, 4, 1, 1, 1, 1},  // Option 4: supports VC0+VC1+VC2, smaller overall
                {2, 2, 1, 1, 1, 1},  // Option 5: supports VC0+VC1+VC2, smaller overall
                {1, 1, 1, 1, 1, 1}   // Option 6: supports VC0+VC1+VC2, smallest
            },
            // BLACKHOLE
            {
                {4, 8, 2, 4, 2, 4},  // Option 1: supports VC0+VC1+VC2
                {4, 8, 2, 2, 2, 2},  // Option 2: supports VC0+VC1+VC2, smaller VC1/VC2 receiver
                {2, 4, 2, 2, 2, 2},  // Option 3: supports VC0+VC1+VC2, smaller overall
                {2, 4, 1, 1, 1, 1},  // Option 4: supports VC0+VC1+VC2, smaller overall
                {2, 2, 1, 1, 1, 1},  // Option 5: supports VC0+VC1+VC2, smaller overall
                {1, 1, 1, 1, 1, 1}   // Option 6: supports VC0+VC1+VC2, smallest
            }};

        // VC0+VC2 mesh options (no VC1): vc1 slots = 0, vc2 mirrors vc0-only reduced values
        static const std::vector<std::vector<PerVcBufferSlots>> vc0_vc2_mesh_buffer_slot_options = {
            // WORMHOLE_B0
            {
                {7, 11, 0, 0, 2, 4},  // Option 1: supports VC0+VC2
                {4, 8, 0, 0, 2, 4},   // Option 2: supports VC0+VC2, smaller VC0
                {4, 8, 0, 0, 2, 2},   // Option 3: supports VC0+VC2, smaller VC2 receiver
                {2, 4, 0, 0, 2, 2},   // Option 4: supports VC0+VC2, smaller overall
                {2, 4, 0, 0, 1, 1},   // Option 5: supports VC0+VC2, smaller overall
                {2, 2, 0, 0, 1, 1},   // Option 6: supports VC0+VC2, smaller overall
                {1, 1, 0, 0, 1, 1}    // Option 7: supports VC0+VC2, smallest
            },
            // BLACKHOLE
            {
                {7, 11, 0, 0, 2, 4},  // Option 1: supports VC0+VC2
                {4, 8, 0, 0, 2, 4},   // Option 2: supports VC0+VC2, smaller VC0
                {4, 8, 0, 0, 2, 2},   // Option 3: supports VC0+VC2, smaller VC2 receiver
                {2, 4, 0, 0, 2, 2},   // Option 4: supports VC0+VC2, smaller overall
                {2, 4, 0, 0, 1, 1},   // Option 5: supports VC0+VC2, smaller overall
                {2, 2, 0, 0, 1, 1},   // Option 6: supports VC0+VC2, smaller overall
                {1, 1, 0, 0, 1, 1}    // Option 7: supports VC0+VC2, smallest
            }};

        static const std::vector<std::vector<PerVcBufferSlots>> other_buffer_slot_options = {
            // WORMHOLE_B0
            {{16, 16, 0, 0, 0, 0},  // Only VC0 for non-mesh topologies.
             {8, 16, 0, 0, 0, 0},
             {8, 8, 0, 0, 0, 0},
             {4, 8, 0, 0, 0, 0},
             {4, 4, 0, 0, 0, 0},
             {2, 4, 0, 0, 0, 0},
             {2, 2, 0, 0, 0, 0},
             {1, 2, 0, 0, 0, 0},
             {1, 1, 0, 0, 0, 0}},
            // BLACKHOLE
            {{32, 32, 0, 0, 0, 0},  // Only VC0 for non-mesh topologies.
             {16, 32, 0, 0, 0, 0},
             {16, 16, 0, 0, 0, 0},
             {8, 16, 0, 0, 0, 0},
             {8, 8, 0, 0, 0, 0},
             {4, 8, 0, 0, 0, 0},
             {4, 4, 0, 0, 0, 0},
             {2, 4, 0, 0, 0, 0},
             {2, 2, 0, 0, 0, 0},
             {1, 2, 0, 0, 0, 0},
             {1, 1, 0, 0, 0, 0}}};

        static ttsl::Indestructible<std::vector<std::vector<PerVcBufferSlots>>> vc0_only_mesh_slots(
            vc0_only_mesh_buffer_slot_options);
        static ttsl::Indestructible<std::vector<std::vector<PerVcBufferSlots>>> vc0_vc1_mesh_slots(
            vc0_vc1_mesh_buffer_slot_options);
        static ttsl::Indestructible<std::vector<std::vector<PerVcBufferSlots>>> vc0_vc2_mesh_slots(
            vc0_vc2_mesh_buffer_slot_options);
        static ttsl::Indestructible<std::vector<std::vector<PerVcBufferSlots>>> vc0_vc1_vc2_mesh_slots(
            vc0_vc1_vc2_mesh_buffer_slot_options);
        static ttsl::Indestructible<std::vector<std::vector<PerVcBufferSlots>>> other_slots(other_buffer_slot_options);

        if (topology == Topology::Mesh || topology == Topology::Torus) {
            // Select table based on which VCs are actually active
            if (vc1_present && vc2_present) {
                return vc0_vc1_vc2_mesh_slots.get()[arch_index];
            }
            if (vc2_present) {
                return vc0_vc2_mesh_slots.get()[arch_index];
            }
            if (vc1_present) {
                return vc0_vc1_mesh_slots.get()[arch_index];
            }
            return vc0_only_mesh_slots.get()[arch_index];
        }
        return other_slots.get()[arch_index];
    };

    auto get_optimal_num_slots_per_vc = [this](
                                            auto& buffer_slot_options,
                                            size_t num_vc0_sender_channels,
                                            size_t num_vc0_receiver_channels,
                                            size_t num_vc1_sender_channels,
                                            size_t num_vc1_receiver_channels,
                                            size_t num_vc2_sender_channels,
                                            size_t num_vc2_receiver_channels,
                                            size_t& vc0_sender_buffer_slots,
                                            size_t& vc0_receiver_buffer_slots,
                                            size_t& vc1_sender_buffer_slots,
                                            size_t& vc1_receiver_buffer_slots,
                                            size_t& vc2_sender_buffer_slots,
                                            size_t& vc2_receiver_buffer_slots) {
        bool vc1_needed = (num_vc1_sender_channels > 0) || (num_vc1_receiver_channels > 0);
        bool vc2_needed = (num_vc2_sender_channels > 0) || (num_vc2_receiver_channels > 0);
        bool found_valid_option = false;
        const size_t sender_slot_cap = fabric_experiment_slot_cap("TT_FABRIC_EXP_MAX_SENDER_SLOTS");
        const size_t receiver_slot_cap = fabric_experiment_slot_cap("TT_FABRIC_EXP_MAX_RECEIVER_SLOTS");
        for (auto& option : buffer_slot_options) {
            if ((sender_slot_cap != 0 && option.vc0_sender_slots > sender_slot_cap) ||
                (receiver_slot_cap != 0 && option.vc0_receiver_slots > receiver_slot_cap)) {
                continue;
            }
            vc0_sender_buffer_slots = option.vc0_sender_slots;
            vc0_receiver_buffer_slots = option.vc0_receiver_slots;
            vc1_sender_buffer_slots = option.vc1_sender_slots;
            vc1_receiver_buffer_slots = option.vc1_receiver_slots;
            vc2_sender_buffer_slots = option.vc2_sender_slots;
            vc2_receiver_buffer_slots = option.vc2_receiver_slots;
            // skip the VC0 only options if VC1 is needed (either sender or receiver channels)
            if (vc1_needed) {
                // Check if we need VC1 sender channels but this option doesn't provide them
                bool skip_due_to_vc1_sender = (num_vc1_sender_channels > 0) && (vc1_sender_buffer_slots == 0);
                // Check if we need VC1 receiver channels but this option doesn't provide them
                bool skip_due_to_vc1_receiver = (num_vc1_receiver_channels > 0) && (vc1_receiver_buffer_slots == 0);

                if (skip_due_to_vc1_sender || skip_due_to_vc1_receiver) {
                    continue;  // Skip this option - VC1 is needed but this option doesn't support it
                }
            }
            // skip options without VC2 slots if VC2 is needed
            if (vc2_needed) {
                bool skip_due_to_vc2_sender = (num_vc2_sender_channels > 0) && (vc2_sender_buffer_slots == 0);
                bool skip_due_to_vc2_receiver = (num_vc2_receiver_channels > 0) && (vc2_receiver_buffer_slots == 0);

                if (skip_due_to_vc2_sender || skip_due_to_vc2_receiver) {
                    continue;  // Skip this option - VC2 is needed but this option doesn't support it
                }
            }

            // Calculate total slots across all VCs
            auto vc0_total_sender_slots = num_vc0_sender_channels * vc0_sender_buffer_slots;
            auto vc0_total_receiver_slots = num_vc0_receiver_channels * vc0_receiver_buffer_slots;
            auto vc1_total_sender_slots = num_vc1_sender_channels * vc1_sender_buffer_slots;
            auto vc1_total_receiver_slots = num_vc1_receiver_channels * vc1_receiver_buffer_slots;
            auto vc2_total_sender_slots = num_vc2_sender_channels * vc2_sender_buffer_slots;
            auto vc2_total_receiver_slots = num_vc2_receiver_channels * vc2_receiver_buffer_slots;

            auto total_num_bytes = (vc0_total_sender_slots + vc0_total_receiver_slots + vc1_total_sender_slots +
                                    vc1_total_receiver_slots + vc2_total_sender_slots + vc2_total_receiver_slots) *
                                   this->channel_buffer_size_bytes;

            if (total_num_bytes <= this->available_channel_buffering_space) {
                found_valid_option = true;
                break;  // Found a configuration that fits
            }
        }

        // Validate that we found a valid option
        if (!found_valid_option) {
            // Debug: print topology and table info
            log_warning(tt::LogFabric, "  buffer_slot_options.size(): {}", buffer_slot_options.size());
            // Debug: print all options tried
            for (size_t idx = 0; idx < buffer_slot_options.size(); ++idx) {
                auto& opt = buffer_slot_options[idx];
                auto total = (num_vc0_sender_channels * opt.vc0_sender_slots +
                              num_vc0_receiver_channels * opt.vc0_receiver_slots +
                              num_vc1_sender_channels * opt.vc1_sender_slots +
                              num_vc1_receiver_channels * opt.vc1_receiver_slots +
                              num_vc2_sender_channels * opt.vc2_sender_slots +
                              num_vc2_receiver_channels * opt.vc2_receiver_slots) *
                             this->channel_buffer_size_bytes;
                log_warning(
                    tt::LogFabric,
                    "  Option {}: vc0({},{}) vc1({},{}) vc2({},{}) = {} bytes (avail: {}, buf_size: {})",
                    idx,
                    opt.vc0_sender_slots,
                    opt.vc0_receiver_slots,
                    opt.vc1_sender_slots,
                    opt.vc1_receiver_slots,
                    opt.vc2_sender_slots,
                    opt.vc2_receiver_slots,
                    total,
                    this->available_channel_buffering_space,
                    this->channel_buffer_size_bytes);
            }
            TT_THROW(
                "Failed to find suitable buffer slot configuration. VC1 needed: {}, VC2 needed: {}, VC0 channels: {} "
                "senders/{} "
                "receivers, VC1 channels: {} senders/{} receivers, VC2 channels: {} senders/{} receivers, Available "
                "space: {} bytes",
                vc1_needed,
                vc2_needed,
                num_vc0_sender_channels,
                num_vc0_receiver_channels,
                num_vc1_sender_channels,
                num_vc1_receiver_channels,
                num_vc2_sender_channels,
                num_vc2_receiver_channels,
                this->available_channel_buffering_space);
        }

        // Additional validation: ensure VC1/VC2 buffer slots are non-zero if those channels are needed
    };

    // auto axis_index = static_cast<std::size_t>(options.edm_axis);
    auto arch = options.arch;
    size_t arch_index;
    if (arch == tt::ARCH::WORMHOLE_B0) {
        arch_index = 0;
    } else if (arch == tt::ARCH::BLACKHOLE) {
        arch_index = 1;
    } else {
        TT_THROW("Unsupported architecture: {}", enchantum::to_string(arch));
    }

    switch (options.fabric_tensix_config) {
        case tt::tt_fabric::FabricTensixConfig::MUX: {
            // MUX mode: Only VC0 channel 0 is used for worker
            uint32_t target_channel = get_worker_connected_sender_channel();
            size_t vc0_sender_buffer_slots, vc0_receiver_buffer_slots;
            size_t vc1_sender_buffer_slots, vc1_receiver_buffer_slots;
            size_t vc2_sender_buffer_slots, vc2_receiver_buffer_slots;

            // get the optimal buffer slots for MUX mode (per-VC)
            get_optimal_num_slots_per_vc(
                default_with_tensix_buffer_slot_options[arch_index],
                num_used_sender_channels_per_vc[0],
                num_used_receiver_channels_per_vc[0],
                num_used_sender_channels_per_vc[1],
                num_used_receiver_channels_per_vc[1],
                num_used_sender_channels_per_vc[2],
                num_used_receiver_channels_per_vc[2],
                vc0_sender_buffer_slots,
                vc0_receiver_buffer_slots,
                vc1_sender_buffer_slots,
                vc1_receiver_buffer_slots,
                vc2_sender_buffer_slots,
                vc2_receiver_buffer_slots);

            // set buffer slots for VC0 worker channel only
            num_sender_buffer_slots_per_vc[0][target_channel] = vc0_sender_buffer_slots;
            num_remote_sender_buffer_slots_per_vc[0][target_channel] = vc0_sender_buffer_slots;

            // Fill receiver buffer slots for all VCs
            num_receiver_buffer_slots_per_vc[0].fill(vc0_receiver_buffer_slots);
            num_remote_receiver_buffer_slots_per_vc[0].fill(vc0_receiver_buffer_slots);
            num_receiver_buffer_slots_per_vc[1].fill(vc1_receiver_buffer_slots);
            num_remote_receiver_buffer_slots_per_vc[1].fill(vc1_receiver_buffer_slots);
            num_receiver_buffer_slots_per_vc[2].fill(vc2_receiver_buffer_slots);
            num_remote_receiver_buffer_slots_per_vc[2].fill(vc2_receiver_buffer_slots);
            return;
        }
        default: break;
    }

    // A 1D fabric with several line VCs (TT_METAL_FABRIC_LINE_VCS; fabric_exp/consolidation/LINE_VCS_HOST_DESIGN.md):
    // every line VC has a worker channel, a forwarding channel and a receiver, at one row of depths so a receiver's
    // depth matches at both ends of a link. Each VC gets an equal share of the slots: the receiver the largest power of
    // 2 within half of it (the receiver ring's index arithmetic is cheapest so), the forwarding channel half of the
    // rest, the worker the remainder; at 82 slots, 2 VCs get 13/12/16 each and 3 VCs 10/9/8.
    // TT_FABRIC_EXP_SLOTS_1D_VCS=worker,forwarding,receiver overrides the row.
    if ((topology == Topology::Linear || topology == Topology::Ring) &&
        (num_used_sender_channels_per_vc[1] > 0 || num_used_sender_channels_per_vc[2] > 0)) {
        const size_t slot_capacity = available_channel_buffering_space / channel_buffer_size_bytes;
        size_t line_vcs = 0;
        for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
            TT_FATAL(
                num_used_sender_channels_per_vc[vc] == 0 ||
                    (num_used_sender_channels_per_vc[vc] == 2 && num_used_receiver_channels_per_vc[vc] == 1),
                "1D line VC{}: expected a worker channel, a forwarding channel and a receiver channel",
                vc);
            line_vcs += num_used_sender_channels_per_vc[vc] > 0 ? 1 : 0;
        }
        const size_t per_vc = slot_capacity / line_vcs;
        size_t receiver = std::bit_floor(per_vc / 2);
        size_t forwarding = (per_vc - receiver) / 2;
        size_t worker = per_vc - receiver - forwarding;
        if (const char* row = std::getenv("TT_FABRIC_EXP_SLOTS_1D_VCS"); row != nullptr) {
            TT_FATAL(
                std::sscanf(row, "%zu,%zu,%zu", &worker, &forwarding, &receiver) == 3,
                "TT_FABRIC_EXP_SLOTS_1D_VCS={}: expected worker,forwarding,receiver",
                row);
        }
        TT_FATAL(
            worker > 0 && forwarding > 0 && receiver > 0 && worker <= MAX_CHANNEL_BUFFER_SLOTS &&
                forwarding <= MAX_CHANNEL_BUFFER_SLOTS && receiver <= MAX_CHANNEL_BUFFER_SLOTS &&
                line_vcs * (worker + forwarding + receiver) <= slot_capacity,
            "1D line VCs: {} VCs of {} worker, {} forwarding and {} receiver slots do not fit {} slots (at most {} per "
            "channel)",
            line_vcs,
            worker,
            forwarding,
            receiver,
            slot_capacity,
            MAX_CHANNEL_BUFFER_SLOTS);
        const uint32_t worker_channel = get_worker_connected_sender_channel();
        for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
            if (num_used_sender_channels_per_vc[vc] == 0) {
                continue;
            }
            num_sender_buffer_slots_per_vc[vc].fill(forwarding);
            num_remote_sender_buffer_slots_per_vc[vc].fill(forwarding);
            num_sender_buffer_slots_per_vc[vc][worker_channel] = worker;
            num_remote_sender_buffer_slots_per_vc[vc][worker_channel] = worker;
            num_receiver_buffer_slots_per_vc[vc].fill(receiver);
            num_remote_receiver_buffer_slots_per_vc[vc].fill(receiver);
        }
        return;
    }

    // Default case: Configure buffer slots with per-VC options
    size_t vc0_sender_buffer_slots, vc0_receiver_buffer_slots;
    size_t vc1_sender_buffer_slots, vc1_receiver_buffer_slots;
    size_t vc2_sender_buffer_slots, vc2_receiver_buffer_slots;

    // Determine which VCs are active based on channel usage
    bool vc1_present = (num_used_sender_channels_per_vc[1] > 0 || num_used_receiver_channels_per_vc[1] > 0);
    bool vc2_present = (num_used_sender_channels_per_vc[2] > 0 || num_used_receiver_channels_per_vc[2] > 0);

    // Get optimal buffer slots considering active VCs
    get_optimal_num_slots_per_vc(
        get_num_buffer_slots(topology, arch_index, vc1_present, vc2_present),
        num_used_sender_channels_per_vc[0],
        num_used_receiver_channels_per_vc[0],
        num_used_sender_channels_per_vc[1],
        num_used_receiver_channels_per_vc[1],
        num_used_sender_channels_per_vc[2],
        num_used_receiver_channels_per_vc[2],
        vc0_sender_buffer_slots,
        vc0_receiver_buffer_slots,
        vc1_sender_buffer_slots,
        vc1_receiver_buffer_slots,
        vc2_sender_buffer_slots,
        vc2_receiver_buffer_slots);

    // Apply the buffer slot configuration to each VC
    num_sender_buffer_slots_per_vc[0].fill(vc0_sender_buffer_slots);
    num_remote_sender_buffer_slots_per_vc[0].fill(vc0_sender_buffer_slots);
    num_receiver_buffer_slots_per_vc[0].fill(vc0_receiver_buffer_slots);
    num_remote_receiver_buffer_slots_per_vc[0].fill(vc0_receiver_buffer_slots);

    num_sender_buffer_slots_per_vc[1].fill(vc1_sender_buffer_slots);
    num_remote_sender_buffer_slots_per_vc[1].fill(vc1_sender_buffer_slots);
    num_receiver_buffer_slots_per_vc[1].fill(vc1_receiver_buffer_slots);
    num_remote_receiver_buffer_slots_per_vc[1].fill(vc1_receiver_buffer_slots);

    num_sender_buffer_slots_per_vc[2].fill(vc2_sender_buffer_slots);
    num_remote_sender_buffer_slots_per_vc[2].fill(vc2_sender_buffer_slots);
    num_receiver_buffer_slots_per_vc[2].fill(vc2_receiver_buffer_slots);
    num_remote_receiver_buffer_slots_per_vc[2].fill(vc2_receiver_buffer_slots);

    // Mesh allocation uses one conservative depth for every channel in a VC,
    // which can leave whole packet slots unused. Give those slots to the live
    // local-worker injection channel; forwarding and receive depths stay
    // unchanged.
    // Study switch (fabric_exp, 1D analysis): TT_FABRIC_EXP_LINEAR_SPARE_SLOTS=1 gives Linear's spare slots to its
    // worker channel too.
    const bool linear_spare_slots =
        topology == Topology::Linear && std::getenv("TT_FABRIC_EXP_LINEAR_SPARE_SLOTS") != nullptr;
    if ((topology == Topology::Mesh || topology == Topology::Torus || linear_spare_slots) &&
        options.fabric_tensix_config == tt::tt_fabric::FabricTensixConfig::DISABLED) {
        size_t allocated_slots = 0;
        for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
            allocated_slots += std::accumulate(
                num_sender_buffer_slots_per_vc[vc].begin(),
                num_sender_buffer_slots_per_vc[vc].begin() + num_used_sender_channels_per_vc[vc],
                size_t{0});
            allocated_slots += std::accumulate(
                num_receiver_buffer_slots_per_vc[vc].begin(),
                num_receiver_buffer_slots_per_vc[vc].begin() + num_used_receiver_channels_per_vc[vc],
                size_t{0});
        }
        const size_t slot_capacity = available_channel_buffering_space / channel_buffer_size_bytes;
        TT_FATAL(allocated_slots <= slot_capacity, "Fabric channel allocation exceeds its slot capacity");
        const size_t spare_slots = slot_capacity - allocated_slots;
        const uint32_t worker_channel = get_worker_connected_sender_channel();

        const size_t requested_worker_slots = num_sender_buffer_slots_per_vc[0][worker_channel] + spare_slots;
        size_t worker_slots = std::min(requested_worker_slots, MAX_CHANNEL_BUFFER_SLOTS);
        if (const size_t cap = fabric_experiment_slot_cap("TT_FABRIC_EXP_MAX_SENDER_SLOTS"); cap != 0) {
            worker_slots = std::min(worker_slots, std::max(cap, num_sender_buffer_slots_per_vc[0][worker_channel]));
        }
        if (const size_t exp_worker_slots = fabric_experiment_slot_cap("TT_FABRIC_EXP_WORKER_SLOTS");
            exp_worker_slots != 0) {
            worker_slots = std::min(exp_worker_slots, MAX_CHANNEL_BUFFER_SLOTS);
            const size_t non_vc0_sender_slots =
                allocated_slots - num_used_sender_channels_per_vc[0] * vc0_sender_buffer_slots;
            TT_FATAL(
                worker_slots + non_vc0_sender_slots <= slot_capacity,
                "TT_FABRIC_EXP_WORKER_SLOTS={} exceeds slot capacity {} (other channels use {})",
                worker_slots,
                slot_capacity,
                non_vc0_sender_slots);
            const size_t fwd_channels =
                num_used_sender_channels_per_vc[0] > 1 ? num_used_sender_channels_per_vc[0] - 1 : 0;
            if (fwd_channels > 0) {
                const size_t max_fwd_slots = (slot_capacity - non_vc0_sender_slots - worker_slots) / fwd_channels;
                const size_t fwd_slots = std::min(vc0_sender_buffer_slots, max_fwd_slots);
                TT_FATAL(
                    fwd_slots > 0,
                    "TT_FABRIC_EXP_WORKER_SLOTS={} leaves 0 slots for {} forwarding channels",
                    worker_slots,
                    fwd_channels);
                num_sender_buffer_slots_per_vc[0].fill(fwd_slots);
                num_remote_sender_buffer_slots_per_vc[0].fill(fwd_slots);
            }
        }

        num_sender_buffer_slots_per_vc[0][worker_channel] = worker_slots;
        num_remote_sender_buffer_slots_per_vc[0][worker_channel] = worker_slots;
    }

    // 1D (Linear, Ring) on Blackhole: the tier table's first fit (16 + 16 sender, 32 receiver slots) leaves ~19 of 83
    // slots unused. The receiver keeps its tier depth, a power of 2 (the receiver ring's index arithmetic is cheapest
    // so); the sender channels share every other slot evenly, the remainder to the worker channel. Measured (fabric_exp
    // RESULTS §98, slot allocation sweep): 16 worker slots bind 1-hop and 16 forwarding slots bind 3-hop on Linear;
    // rings are not slot-bound. Only for the study's router in the v2 slot (TT_FABRIC_KERNEL_VERSION=2, v4): main's
    // router has power-of-2 paths in its channel index arithmetic, and a sender channel whose depth is not a power of
    // 2 costs the flows through it 2-6 cycles/packet there (§98, 1D slot split), so main keeps the tier depths.
    // TT_FABRIC_EXP_TIER_SLOTS=1 keeps the tier depths, as do the older study switches.
    const char* kernel_version = std::getenv("TT_FABRIC_KERNEL_VERSION");
    const bool split_1d_slots =
        kernel_version != nullptr && std::string_view(kernel_version) == "2" &&
        (topology == Topology::Linear || topology == Topology::Ring) && arch == tt::ARCH::BLACKHOLE &&
        options.fabric_tensix_config == tt::tt_fabric::FabricTensixConfig::DISABLED && !vc1_present && !vc2_present &&
        !linear_spare_slots && std::getenv("TT_FABRIC_EXP_TIER_SLOTS") == nullptr &&
        fabric_experiment_slot_cap("TT_FABRIC_EXP_MAX_SENDER_SLOTS") == 0 &&
        fabric_experiment_slot_cap("TT_FABRIC_EXP_MAX_RECEIVER_SLOTS") == 0 &&
        num_used_sender_channels_per_vc[0] >= 2;  // a router with only the worker channel (neighbor exchange) keeps its tier
    if (split_1d_slots) {
        const size_t slot_capacity = available_channel_buffering_space / channel_buffer_size_bytes;
        const size_t senders = num_used_sender_channels_per_vc[0];
        const size_t receiver_slots = num_used_receiver_channels_per_vc[0] * vc0_receiver_buffer_slots;
        TT_FATAL(
            receiver_slots + senders <= slot_capacity,
            "1D slot split: {} sender channels and {} receiver slots do not fit {} slots",
            senders,
            receiver_slots,
            slot_capacity);
        const size_t sender_slots = slot_capacity - receiver_slots;
        const size_t forwarding = std::min(sender_slots / senders, MAX_CHANNEL_BUFFER_SLOTS);
        const size_t worker = std::min(sender_slots - (senders - 1) * forwarding, MAX_CHANNEL_BUFFER_SLOTS);
        const uint32_t worker_channel = get_worker_connected_sender_channel();
        num_sender_buffer_slots_per_vc[0].fill(forwarding);
        num_remote_sender_buffer_slots_per_vc[0].fill(forwarding);
        num_sender_buffer_slots_per_vc[0][worker_channel] = worker;
        num_remote_sender_buffer_slots_per_vc[0][worker_channel] = worker;
    }

    // Study switch (fabric_exp, slot allocation): TT_FABRIC_EXP_SLOTS=worker,forwarding,receiver sets VC0's depths on
    // VC0-only fabrics: the worker channel, every forwarding channel, the receiver channel. A worker of 0 takes every
    // slot the others leave, up to MAX_CHANNEL_BUFFER_SLOTS. Every router gets the same depths, so the receiver depth
    // matches at both ends of a link. TT_FABRIC_EXP_SLOTS_1D (Linear, Ring) and TT_FABRIC_EXP_SLOTS_2D (Mesh, Torus)
    // set one topology family and take precedence, so a suite with both (the sanity yamls) can give each its own.
    const bool exp_slots_2d = topology == Topology::Mesh || topology == Topology::Torus;
    const char* exp_slots = std::getenv(exp_slots_2d ? "TT_FABRIC_EXP_SLOTS_2D" : "TT_FABRIC_EXP_SLOTS_1D");
    if (exp_slots == nullptr) {
        exp_slots = std::getenv("TT_FABRIC_EXP_SLOTS");
    }
    if (exp_slots != nullptr && options.fabric_tensix_config == tt::tt_fabric::FabricTensixConfig::DISABLED) {
        size_t worker = 0, forwarding = 0, receiver = 0;
        TT_FATAL(
            std::sscanf(exp_slots, "%zu,%zu,%zu", &worker, &forwarding, &receiver) == 3,
            "TT_FABRIC_EXP_SLOTS={}: expected worker,forwarding,receiver",
            exp_slots);
        TT_FATAL(!vc1_present && !vc2_present, "TT_FABRIC_EXP_SLOTS is for VC0-only fabrics");
        const size_t slot_capacity = available_channel_buffering_space / channel_buffer_size_bytes;
        const size_t others = (num_used_sender_channels_per_vc[0] - 1) * forwarding +
                              num_used_receiver_channels_per_vc[0] * receiver;
        TT_FATAL(others < slot_capacity, "TT_FABRIC_EXP_SLOTS={}: no slots left for the worker", exp_slots);
        if (worker == 0) {
            worker = std::min(slot_capacity - others, MAX_CHANNEL_BUFFER_SLOTS);
        }
        TT_FATAL(
            worker > 0 && forwarding > 0 && receiver > 0 && worker <= MAX_CHANNEL_BUFFER_SLOTS &&
                forwarding <= MAX_CHANNEL_BUFFER_SLOTS && receiver <= MAX_CHANNEL_BUFFER_SLOTS &&
                worker + others <= slot_capacity,
            "TT_FABRIC_EXP_SLOTS={}: {} worker + {} other slots, capacity {}, at most {} per channel",
            exp_slots,
            worker,
            others,
            slot_capacity,
            MAX_CHANNEL_BUFFER_SLOTS);
        const uint32_t worker_channel = get_worker_connected_sender_channel();
        num_sender_buffer_slots_per_vc[0].fill(forwarding);
        num_remote_sender_buffer_slots_per_vc[0].fill(forwarding);
        num_sender_buffer_slots_per_vc[0][worker_channel] = worker;
        num_remote_sender_buffer_slots_per_vc[0][worker_channel] = worker;
        num_receiver_buffer_slots_per_vc[0].fill(receiver);
        num_remote_receiver_buffer_slots_per_vc[0].fill(receiver);
    }

    // Study rows (fabric_exp, slot allocation) for fabrics with VC1 and/or VC2. A row is the depths
    //   {VC0 worker, every VC0 forwarding channel, VC0 receiver, every VC1 sender channel (all forwarding),
    //    VC1 receiver, VC2's worker channel, VC2 receiver, VC2's forwarding channel (line mode)},
    // the entries of an absent VC 0. Every router gets the same row, so a receiver's depth matches at both ends of a link.
    // Switches: TT_FABRIC_EXP_SLOTS_VC1=worker,forwarding0,receiver0,forwarding1,receiver1 on VC0+VC1 fabrics, with
    // ",worker2,receiver2" added on VC0+VC1+VC2 fabrics; TT_FABRIC_EXP_SLOTS_VC2=worker,forwarding0,receiver0,worker2,
    // receiver2 on VC0+VC2 fabrics; in VC2 line mode either switch takes ",forwarding2" at the end. A worker of 0 takes
    // every slot the others leave, up to MAX_CHANNEL_BUFFER_SLOTS.
    // Without a switch, the study's router in the v2 slot (TT_FABRIC_KERNEL_VERSION=2, v4) on Blackhole gets the
    // study's row for its shape (4 VC0 sender channels, 3 VC1 and 1 VC2 where present, one receiver channel per VC),
    // its worker the rest of the slots, when the row's design worker fits (the rows are set at the default 4352 B
    // packet: 81 slots; a larger packet keeps the tier table):
    //   VC0+VC1      19, 9, 16, 5, 4: the tier table's 51,4,8,2,4 binds the rows the study measures (VC0 receiver 8:
    //                1-hop +26 to +43; VC0 forwarding 4: 2-hop 482 against 264; VC1 forwarding 2: inter-mesh flows
    //                with VC1 hops 780-970 against 320-405; fabric_exp RESULTS §98, jobs 109-112), from the offline
    //                model (consolidation/alloc_model.py, between its VC0:VC1 weights 3:2 and 2:1);
    //   VC0+VC2      17, 14, 16 + VC2 2, 4: VC0 as on VC0-only fabrics (17, 16, 16) but 2 forwarding slots fewer per
    //                channel for VC2's tier depths (the tier row's 43, 7, 11 makes VC0's receiver ring 11 slots);
    //   VC0+VC1+VC2  19, 8, 16, 4, 4 + VC2 2, 4: the VC0+VC1 row, one forwarding slot fewer per VC0 and VC1 channel
    //                for VC2 (not yet measured);
    //   VC0+VC2 line 17, 12, 16 + VC2 4, 4, 4 (worker, receiver, forwarding): VC2's forwarding channel takes 2 slots
    //                from each VC0 forwarding channel, its worker 2 more (a line's source, not a neighbour's); the
    //                receiver at 4 keeps the 2 slots bubble flow control needs on VC2 rings (VC2_LINE_BRINGUP.md;
    //                correctness first, not tuned);
    //   VC0+VC1+VC2 line  16, 7, 16, 4, 4 + VC2 4, 4, 4: the VC0+VC1+VC2 row, one VC0 forwarding slot fewer per channel
    //                and VC2's worker 2 more for VC2's forwarding channel (worker 16 at 4352 B; not tuned).
    // Main's router keeps the tier table; TT_FABRIC_EXP_TIER_SLOTS=1 keeps it for the study's router too.
    using StudyRow = std::array<size_t, 8>;
    // VC2 in line mode: VC2's channels past its worker are forwarding channels (one, fed by the router facing the
    // opposite way).
    const size_t vc2_forwarding_channels =
        num_used_sender_channels_per_vc[2] > 1 ? num_used_sender_channels_per_vc[2] - 1 : 0;
    const auto parse_row = [](const char* name, const char* text, size_t fields, bool vc1) {
        size_t v[8] = {};
        TT_FATAL(
            std::sscanf(
                text, "%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]) ==
                static_cast<int>(fields),
            "{}={}: expected {} depths",
            name,
            text,
            fields);
        // TT_FABRIC_EXP_SLOTS_VC2 has no VC1 fields: worker, forwarding0, receiver0, worker2, receiver2[, forwarding2].
        return vc1 ? StudyRow{v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]}
                   : StudyRow{v[0], v[1], v[2], 0, 0, v[3], v[4], v[5]};
    };
    const char* const row_switch = vc1_present ? "TT_FABRIC_EXP_SLOTS_VC1" : "TT_FABRIC_EXP_SLOTS_VC2";
    const char* const row_text = std::getenv(row_switch);
    std::optional<StudyRow> row;
    if ((vc1_present || vc2_present) && options.fabric_tensix_config == tt::tt_fabric::FabricTensixConfig::DISABLED) {
        if (row_text != nullptr) {
            row = parse_row(
                row_switch,
                row_text,
                (vc1_present && vc2_present ? 7 : 5) + (vc2_forwarding_channels > 0 ? 1 : 0),
                vc1_present);
        } else if (
            kernel_version != nullptr && std::string_view(kernel_version) == "2" && arch == tt::ARCH::BLACKHOLE &&
            std::getenv("TT_FABRIC_EXP_TIER_SLOTS") == nullptr &&
            fabric_experiment_slot_cap("TT_FABRIC_EXP_WORKER_SLOTS") == 0 &&
            fabric_experiment_slot_cap("TT_FABRIC_EXP_MAX_SENDER_SLOTS") == 0 &&
            fabric_experiment_slot_cap("TT_FABRIC_EXP_MAX_RECEIVER_SLOTS") == 0 &&
            num_used_sender_channels_per_vc[0] == 4 && num_used_receiver_channels_per_vc[0] == 1 &&
            num_used_sender_channels_per_vc[1] == (vc1_present ? 3 : 0) &&
            num_used_receiver_channels_per_vc[1] == (vc1_present ? 1 : 0) &&
            (num_used_sender_channels_per_vc[2] == (vc2_present ? 1 : 0) ||
             (vc2_present && num_used_sender_channels_per_vc[2] == 2)) &&
            num_used_receiver_channels_per_vc[2] == (vc2_present ? 1 : 0)) {
            const StudyRow design = !vc2_present ? StudyRow{19, 9, 16, 5, 4, 0, 0, 0}
                                    : vc2_forwarding_channels > 0
                                        ? (vc1_present ? StudyRow{16, 7, 16, 4, 4, 4, 4, 4} : StudyRow{17, 12, 16, 0, 0, 4, 4, 4})
                                    : !vc1_present ? StudyRow{17, 14, 16, 0, 0, 2, 4, 0}
                                                   : StudyRow{19, 8, 16, 4, 4, 2, 4, 0};
            size_t design_total = design[0] + vc2_forwarding_channels * design[7];
            for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
                // Per VC: its forwarding channels' depth (VC2: its worker's; its forwarding channels are added above).
                const size_t forwarding = vc == 0 ? design[1] : vc == 1 ? design[3] : design[5];
                const size_t forwarding_channels = vc == 0   ? num_used_sender_channels_per_vc[0] - 1
                                                   : vc == 1 ? num_used_sender_channels_per_vc[1]
                                                             : num_used_sender_channels_per_vc[2] - vc2_forwarding_channels;
                const size_t receiver = vc == 0 ? design[2] : vc == 1 ? design[4] : design[6];
                design_total += forwarding_channels * forwarding + num_used_receiver_channels_per_vc[vc] * receiver;
            }
            if (design_total <= available_channel_buffering_space / channel_buffer_size_bytes) {
                row = design;
                (*row)[0] = 0;  // the worker takes the rest
            }
        }
    }
    if (row.has_value()) {
        const auto [worker_in, forwarding0, receiver0, forwarding1, receiver1, worker2, receiver2, forwarding2] = *row;
        const char* const name = row_text != nullptr ? row_switch : "study slot row";
        const size_t slot_capacity = available_channel_buffering_space / channel_buffer_size_bytes;
        const size_t others = (num_used_sender_channels_per_vc[0] - 1) * forwarding0 +
                              num_used_receiver_channels_per_vc[0] * receiver0 +
                              num_used_sender_channels_per_vc[1] * forwarding1 +
                              num_used_receiver_channels_per_vc[1] * receiver1 +
                              (num_used_sender_channels_per_vc[2] - vc2_forwarding_channels) * worker2 +
                              vc2_forwarding_channels * forwarding2 +
                              num_used_receiver_channels_per_vc[2] * receiver2;
        TT_FATAL(others < slot_capacity, "{}: no slots left for the worker", name);
        const size_t worker = worker_in != 0 ? worker_in : std::min(slot_capacity - others, MAX_CHANNEL_BUFFER_SLOTS);
        const size_t present[8] = {
            1,
            1,
            1,
            vc1_present ? 1u : 0u,
            vc1_present ? 1u : 0u,
            vc2_present ? 1u : 0u,
            vc2_present ? 1u : 0u,
            vc2_forwarding_channels > 0 ? 1u : 0u};
        const size_t depths[8] = {worker, forwarding0, receiver0, forwarding1, receiver1, worker2, receiver2, forwarding2};
        for (size_t i = 0; i < 8; ++i) {
            TT_FATAL(
                present[i] == 0 || (depths[i] > 0 && depths[i] <= MAX_CHANNEL_BUFFER_SLOTS),
                "{}: every depth of a present VC must be 1 to {}",
                name,
                MAX_CHANNEL_BUFFER_SLOTS);
        }
        TT_FATAL(
            worker + others <= slot_capacity,
            "{}: {} worker + {} other slots, capacity {}",
            name,
            worker,
            others,
            slot_capacity);
        const uint32_t worker_channel = get_worker_connected_sender_channel();
        num_sender_buffer_slots_per_vc[0].fill(forwarding0);
        num_remote_sender_buffer_slots_per_vc[0].fill(forwarding0);
        num_sender_buffer_slots_per_vc[0][worker_channel] = worker;
        num_remote_sender_buffer_slots_per_vc[0][worker_channel] = worker;
        num_receiver_buffer_slots_per_vc[0].fill(receiver0);
        num_remote_receiver_buffer_slots_per_vc[0].fill(receiver0);
        if (vc1_present) {
            num_sender_buffer_slots_per_vc[1].fill(forwarding1);
            num_remote_sender_buffer_slots_per_vc[1].fill(forwarding1);
            num_receiver_buffer_slots_per_vc[1].fill(receiver1);
            num_remote_receiver_buffer_slots_per_vc[1].fill(receiver1);
        }
        if (vc2_present) {
            // VC2's worker channel (0), then its forwarding channels (line mode).
            num_sender_buffer_slots_per_vc[2].fill(vc2_forwarding_channels > 0 ? forwarding2 : worker2);
            num_remote_sender_buffer_slots_per_vc[2].fill(vc2_forwarding_channels > 0 ? forwarding2 : worker2);
            num_sender_buffer_slots_per_vc[2][0] = worker2;
            num_remote_sender_buffer_slots_per_vc[2][0] = worker2;
            num_receiver_buffer_slots_per_vc[2].fill(receiver2);
            num_remote_receiver_buffer_slots_per_vc[2].fill(receiver2);
        }
    }
}

void FabricStaticSizedChannelsAllocator::emit_ct_args(std::vector<uint32_t>& ct_args) const {
    // Emit sender channel args for all VCs
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (size_t i = 0; i < this->num_used_sender_channels_per_vc[vc]; ++i) {
            ct_args.push_back(static_cast<uint32_t>(this->sender_channels_base_address[vc][i]));
            ct_args.push_back(this->sender_channels_num_buffers[vc][i]);
            ct_args.push_back(static_cast<uint32_t>(this->remote_sender_channels_base_address[vc][i]));
            ct_args.push_back(this->remote_sender_channels_num_buffers[vc][i]);
        }
    }

    // Emit receiver channel args for all VCs
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (size_t i = 0; i < this->num_used_receiver_channels_per_vc[vc]; ++i) {
            ct_args.push_back(static_cast<uint32_t>(this->receiver_channels_base_address[vc][i]));
            ct_args.push_back(this->receiver_channels_num_buffers[vc][i]);
            ct_args.push_back(static_cast<uint32_t>(this->remote_receiver_channels_base_address[vc][i]));
            ct_args.push_back(this->remote_receiver_channels_num_buffers[vc][i]);
        }
    }
}

void FabricStaticSizedChannelsAllocator::emit_channel_allocations_ct_args(
    std::vector<uint32_t>& ct_args,
    size_t num_used_vc0_sender_channels,
    size_t num_used_vc1_sender_channels,
    size_t num_used_vc2_sender_channels,
    size_t num_used_receiver_channels) const {
    // Tag
    ct_args.push_back(0xabcd1234);

    // num_entries = total sender + receiver channels in the allocator
    size_t total_sender_channels = get_num_sender_channels();
    size_t total_receiver_channels = get_num_receiver_channels();
    size_t num_entries = total_sender_channels + total_receiver_channels;
    ct_args.push_back(static_cast<uint32_t>(num_entries));

    // Per-entry data (reuse existing emit_ct_args which emits per-channel data)
    emit_ct_args(ct_args);

    // Channel-to-entry index mappings
    //
    // The allocator emits entries in order: VC0 senders [0 .. total_vc0), VC1 senders
    // [total_vc0 .. total_vc0+total_vc1), VC2 senders [total_vc0+total_vc1 .. total_sender),
    // then receiver entries. The router may use fewer sender channels per VC than were
    // allocated (e.g. interior chips have no Z upstream, so the last VC0/VC1 slot is
    // allocated but unused). When emitting the kernel-side SENDER_TO_ENTRY_IDX mapping we
    // must skip past the *per-VC* unused slots, not the total unused slots — the kernel
    // compacts its sender channels in (VC0, VC1, VC2) order using the ACTUAL_* counts
    // (see VC1_LOCAL_CHANNEL_START / VC2_LOCAL_CHANNEL_START in
    // fabric_erisc_router_ct_args.hpp), so each VC's used channels must point into its
    // own allocator-entry region.
    //
    // VC1 sch_idx i (in [used_vc0 .. used_vc0+used_vc1)) maps to entry
    //   total_vc0 + (i - used_vc0) = i + unused_vc0
    // VC2 sch_idx i (in [used_vc0+used_vc1 .. used_vc0+used_vc1+used_vc2)) maps to
    //   total_vc0 + total_vc1 + (i - used_vc0 - used_vc1) = i + unused_vc0 + unused_vc1
    //
    // The historical bug was using `num_unused_channels` (the total across VCs) as the
    // skip count for VC1, which silently routed VC1 sch_idx[k] to the *next* VC1 entry —
    // e.g. sch=6 (the third used VC1 channel = "from-N upstream" on a SOUTH router) ended
    // up reading the unused VC1 "from-Z" slot. The upstream router was still configured
    // (correctly) to write into the original base address, so forwarded data was dropped
    // on the floor while credits kept flowing, producing the classic "downstream transmits
    // zero-content packets" symptom on multi-hop cross-mesh VC1 paths through interior
    // chips. VC2 had the same shape of bug: VC2 used channels fell into the padding
    // region because num_used_sender_channels excluded VC2.
    const size_t total_vc0_sender_channels = get_num_sender_channels(0);
    const size_t total_vc1_sender_channels = get_num_sender_channels(1);
    const size_t total_vc2_sender_channels = get_num_sender_channels(2);
    const size_t num_unused_vc0_sender_channels = total_vc0_sender_channels - num_used_vc0_sender_channels;
    const size_t num_unused_vc1_sender_channels = total_vc1_sender_channels - num_used_vc1_sender_channels;
    const size_t num_unused_vc2_sender_channels = total_vc2_sender_channels - num_used_vc2_sender_channels;
    const size_t num_used_sender_channels =
        num_used_vc0_sender_channels + num_used_vc1_sender_channels + num_used_vc2_sender_channels;
    const size_t num_unused_channels =
        num_unused_vc0_sender_channels + num_unused_vc1_sender_channels + num_unused_vc2_sender_channels;
    const bool has_unused_channels = (num_unused_channels > 0) && num_used_sender_channels > 0;

    // Sender channel-to-entry index
    if (has_unused_channels) {
        const size_t used_vc0_end = num_used_vc0_sender_channels;
        const size_t used_vc1_end = used_vc0_end + num_used_vc1_sender_channels;
        for (size_t i = 0; i < num_used_sender_channels; ++i) {
            if (i < used_vc0_end) {
                ct_args.push_back(static_cast<uint32_t>(i));
            } else if (i < used_vc1_end) {
                // VC1 channels skip the unused VC0 slots only.
                ct_args.push_back(static_cast<uint32_t>(i + num_unused_vc0_sender_channels));
            } else {
                // VC2 channels skip the unused VC0 and VC1 slots.
                ct_args.push_back(
                    static_cast<uint32_t>(i + num_unused_vc0_sender_channels + num_unused_vc1_sender_channels));
            }
        }
        // Padding for unused channels — map to their actual (unserviced) entry indices.
        // The unused entries are the per-VC tails appended in VC order.
        for (size_t i = 0; i < num_unused_vc0_sender_channels; ++i) {
            ct_args.push_back(static_cast<uint32_t>(num_used_vc0_sender_channels + i));
        }
        for (size_t i = 0; i < num_unused_vc1_sender_channels; ++i) {
            ct_args.push_back(static_cast<uint32_t>(total_vc0_sender_channels + num_used_vc1_sender_channels + i));
        }
        for (size_t i = 0; i < num_unused_vc2_sender_channels; ++i) {
            ct_args.push_back(static_cast<uint32_t>(
                total_vc0_sender_channels + total_vc1_sender_channels + num_used_vc2_sender_channels + i));
        }
    } else {
        for (size_t i = 0; i < num_used_sender_channels; ++i) {
            ct_args.push_back(static_cast<uint32_t>(i));
        }
    }

    // Receiver channel-to-entry index (receivers start after all sender entries)
    size_t receiver_entry_base = total_sender_channels;
    for (size_t i = 0; i < num_used_receiver_channels; ++i) {
        ct_args.push_back(static_cast<uint32_t>(i + receiver_entry_base));
    }
}

};  // namespace tt::tt_fabric
