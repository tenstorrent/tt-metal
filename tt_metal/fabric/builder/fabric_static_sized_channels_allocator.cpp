// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-logger/tt-logger.hpp>
#include <enchantum/enchantum.hpp>
#include <tt_stl/indestructible.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/fmt.hpp>
#include <algorithm>
#include <numeric>

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

size_t FabricStaticSizedChannelsAllocator::get_receiver_channel_number_of_slots(size_t vc_id) const {
    TT_FATAL(
        vc_id < builder_config::MAX_NUM_VCS, "VC ID {} out of bounds (max {})", vc_id, builder_config::MAX_NUM_VCS);
    TT_FATAL(is_receiver_channel_active_per_vc[vc_id], "VC{} has no active receiver channel", vc_id);
    return receiver_channel_num_buffers[vc_id];
}

size_t FabricStaticSizedChannelsAllocator::get_receiver_channel_base_address(size_t vc_id) const {
    TT_FATAL(
        vc_id < builder_config::MAX_NUM_VCS, "VC ID {} out of bounds (max {})", vc_id, builder_config::MAX_NUM_VCS);
    TT_FATAL(is_receiver_channel_active_per_vc[vc_id], "VC{} has no active receiver channel", vc_id);
    return receiver_channel_base_address[vc_id];
}

FabricStaticSizedChannelsAllocator::FabricStaticSizedChannelsAllocator(
    tt::tt_fabric::Topology topology,
    const FabricEriscDatamoverOptions& options,
    const std::array<size_t, builder_config::MAX_NUM_VCS>& num_used_sender_channels_per_vc,
    const std::array<bool, builder_config::MAX_NUM_VCS>& is_receiver_channel_active_per_vc,
    size_t channel_buffer_size_bytes,
    size_t available_channel_buffering_space,
    const std::vector<MemoryRegion>& memory_regions) :
    FabricChannelAllocator(topology, options, memory_regions),
    num_used_sender_channels_per_vc(num_used_sender_channels_per_vc),
    is_receiver_channel_active_per_vc(is_receiver_channel_active_per_vc),
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
        size_t vc_receiver_slots = is_receiver_channel_active_per_vc[vc] ? num_receiver_buffer_slots_per_vc[vc][0] : 0;
        total_slot_count += vc_sender_slots + vc_receiver_slots;

        log_trace(tt::LogFabric, "VC{} num_sender_buffer_slots: {}", vc, num_sender_buffer_slots_per_vc[vc]);
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
        // set the local receiver channel size and num buffers (one per VC)
        if (is_receiver_channel_active_per_vc[vc]) {
            this->receiver_channel_size_bytes[vc] = channel_buffer_size_bytes * num_receiver_buffer_slots_per_vc[vc][0];
            this->receiver_channel_num_buffers[vc] = num_receiver_buffer_slots_per_vc[vc][0];
        }
        // set the remote receiver channel size and num buffers (one per VC)
        if (is_receiver_channel_active_per_vc[vc]) {
            this->remote_receiver_channel_size_bytes[vc] =
                channel_buffer_size_bytes * num_remote_receiver_buffer_slots_per_vc[vc][0];
            this->remote_receiver_channel_num_buffers[vc] = num_remote_receiver_buffer_slots_per_vc[vc][0];
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
        if (is_receiver_channel_active_per_vc[vc]) {
            this->receiver_channel_base_address[vc] = receiver_buffer_addr;
            receiver_buffer_addr += this->receiver_channel_size_bytes[vc];
            log_trace(tt::LogFabric, "VC{} Receiver channel_start: {}", vc, this->receiver_channel_base_address[vc]);
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
        if (is_receiver_channel_active_per_vc[vc]) {
            this->remote_receiver_channel_base_address[vc] = remote_receiver_buffer_addr;
            remote_receiver_buffer_addr += this->remote_receiver_channel_size_bytes[vc];
            log_trace(
                tt::LogFabric,
                "VC{} Remote Receiver channel_start: {}",
                vc,
                this->remote_receiver_channel_base_address[vc]);
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
        if (is_receiver_channel_active_per_vc[vc]) {
            TT_FATAL(
                this->receiver_channel_size_bytes[vc] > 0,
                "Internal error when computing `receiver_channel_size_bytes[VC{}]` which was computed to be size 0",
                vc);
        }
    }

    // Validate total size across all VCs
    size_t total_size = 0;
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        total_size += std::accumulate(
            this->sender_channels_size_bytes[vc].begin(),
            this->sender_channels_size_bytes[vc].begin() + num_used_sender_channels_per_vc[vc],
            0ul);
        if (is_receiver_channel_active_per_vc[vc]) {
            total_size += this->receiver_channel_size_bytes[vc];
        }
    }

    TT_FATAL(
        total_size <= this->available_channel_buffering_space,
        "Internal error when computing channel sizes. Total channel size {} exceeds available space {}",
        total_size,
        this->available_channel_buffering_space);
    TT_FATAL(
        buffer_addr_end < this->max_l1_loading_size,
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
    // Per-VC buffer slot configuration using array-of-struct indexed by VC.
    // Each entry is std::array<VcSlotConfig, MAX_NUM_VCS> where VcSlotConfig has {sender_slots, receiver_slots}.

    // fabric with tensix extension uses different buffer slots options, since only one or two sender channels are
    // used by fabric router, while other sender channels are skipped and have 0 buffer slots.
    static const std::vector<std::vector<std::array<VcSlotConfig, builder_config::MAX_NUM_VCS>>>
        default_with_tensix_buffer_slot_options = {
            // WORMHOLE_B0
            {
                {VcSlotConfig{16, 16}, VcSlotConfig{0, 0}},  // Option 1
                {VcSlotConfig{8, 16}, VcSlotConfig{0, 0}},   // Option 2
                {VcSlotConfig{8, 8}, VcSlotConfig{0, 0}},    // Option 3
                {VcSlotConfig{4, 8}, VcSlotConfig{0, 0}},    // Option 4
                {VcSlotConfig{4, 4}, VcSlotConfig{0, 0}},    // Option 5: VC0 only, smaller
                {VcSlotConfig{2, 4}, VcSlotConfig{0, 0}},    // Option 6: VC0 only, smaller
                {VcSlotConfig{2, 2}, VcSlotConfig{0, 0}},    // Option 7: VC0 only, smaller
                {VcSlotConfig{1, 2}, VcSlotConfig{0, 0}},    // Option 8: VC0 only, smallest
                {VcSlotConfig{1, 1}, VcSlotConfig{0, 0}},    // Option 9: VC0 only, smallest
                {VcSlotConfig{4, 8}, VcSlotConfig{4, 4}},    // Option 10: supports both VCs
                {VcSlotConfig{4, 8}, VcSlotConfig{2, 2}},    // Option 11: supports both VCs
                {VcSlotConfig{4, 4}, VcSlotConfig{2, 2}},    // Option 12: supports both VCs
                {VcSlotConfig{2, 2}, VcSlotConfig{2, 2}}     // Option 13: supports both VCs
            },
            // BLACKHOLE
            {
                {VcSlotConfig{32, 32}, VcSlotConfig{0, 0}},  // Option 1
                {VcSlotConfig{16, 32}, VcSlotConfig{0, 0}},  // Option 2
                {VcSlotConfig{16, 16}, VcSlotConfig{0, 0}},  // Option 3
                {VcSlotConfig{8, 16}, VcSlotConfig{0, 0}},   // Option 4
                {VcSlotConfig{8, 8}, VcSlotConfig{0, 0}},    // Option 5
                {VcSlotConfig{4, 8}, VcSlotConfig{0, 0}},    // Option 6
                {VcSlotConfig{4, 4}, VcSlotConfig{0, 0}},    // Option 7
                {VcSlotConfig{2, 4}, VcSlotConfig{0, 0}},    // Option 8
                {VcSlotConfig{2, 2}, VcSlotConfig{0, 0}},    // Option 9
                {VcSlotConfig{1, 2}, VcSlotConfig{0, 0}},    // Option 10
                {VcSlotConfig{1, 1}, VcSlotConfig{0, 0}},    // Option 11
                {VcSlotConfig{4, 8}, VcSlotConfig{2, 4}},    // Option 12: supports both VCs
                {VcSlotConfig{4, 8}, VcSlotConfig{2, 2}},    // Option 13: supports both VCs, smaller VC1 receiver
                {VcSlotConfig{2, 4}, VcSlotConfig{2, 2}},    // Option 14: supports both VCs, smaller overall
                {VcSlotConfig{2, 4}, VcSlotConfig{1, 1}},    // Option 15: supports both VCs, smaller overall
                {VcSlotConfig{2, 2}, VcSlotConfig{1, 1}},    // Option 16: supports both VCs, smaller overall
                {VcSlotConfig{1, 1}, VcSlotConfig{1, 1}}     // Option 17: supports both VCs, smaller overall
            }};

    using VcSlotConfigArray = std::array<VcSlotConfig, builder_config::MAX_NUM_VCS>;
    auto get_num_buffer_slots = [](Topology topology, size_t arch_index) -> const std::vector<VcSlotConfigArray>& {
        // Architecture-specific buffer slot configurations per VC
        static const std::vector<std::vector<VcSlotConfigArray>> mesh_buffer_slot_options = {
            // WORMHOLE_B0
            {
                {VcSlotConfig{7, 11}, VcSlotConfig{0, 0}},  // Option 1: VC0 only
                {VcSlotConfig{4, 8}, VcSlotConfig{0, 0}},   // Option 2: VC0 only, smaller
                {VcSlotConfig{4, 4}, VcSlotConfig{0, 0}},   // Option 3: VC0 only, smaller
                {VcSlotConfig{2, 4}, VcSlotConfig{0, 0}},   // Option 4: VC0 only, smaller
                {VcSlotConfig{2, 2}, VcSlotConfig{0, 0}},   // Option 5: VC0 only, smaller
                {VcSlotConfig{1, 2}, VcSlotConfig{0, 0}},   // Option 6: VC0 only, smallest
                {VcSlotConfig{1, 1}, VcSlotConfig{0, 0}},   // Option 7: VC0 only, smallest
                {VcSlotConfig{4, 8}, VcSlotConfig{2, 4}},   // Option 8: supports both VCs
                {VcSlotConfig{4, 8}, VcSlotConfig{2, 2}},   // Option 9: supports both VCs, smaller VC1 receiver
                {VcSlotConfig{2, 4}, VcSlotConfig{2, 2}},   // Option 10: supports both VCs, smaller overall
                {VcSlotConfig{2, 4}, VcSlotConfig{1, 1}},   // Option 11: supports both VCs, smaller overall
                {VcSlotConfig{2, 2}, VcSlotConfig{1, 1}},   // Option 12: supports both VCs, smaller overall
                {VcSlotConfig{1, 1}, VcSlotConfig{1, 1}}    // Option 13: supports both VCs, smaller overall
            },
            // BLACKHOLE
            {
                {VcSlotConfig{8, 16}, VcSlotConfig{0, 0}},  // Option 1: VC0 only
                {VcSlotConfig{8, 8}, VcSlotConfig{0, 0}},   // Option 2: VC0 only
                {VcSlotConfig{4, 8}, VcSlotConfig{0, 0}},   // Option 3: VC0 only
                {VcSlotConfig{4, 4}, VcSlotConfig{0, 0}},   // Option 4: VC0 only, smaller
                {VcSlotConfig{2, 4}, VcSlotConfig{0, 0}},   // Option 5: VC0 only, smaller
                {VcSlotConfig{2, 2}, VcSlotConfig{0, 0}},   // Option 6: VC0 only, smaller
                {VcSlotConfig{1, 2}, VcSlotConfig{0, 0}},   // Option 7: VC0 only, smallest
                {VcSlotConfig{1, 1}, VcSlotConfig{0, 0}},   // Option 8: VC0 only, smallest
                {VcSlotConfig{4, 8}, VcSlotConfig{2, 4}},   // Option 9: supports both VCs
                {VcSlotConfig{4, 8}, VcSlotConfig{2, 2}},   // Option 10: supports both VCs, smaller VC1 receiver
                {VcSlotConfig{2, 4}, VcSlotConfig{2, 2}},   // Option 11: supports both VCs, smaller overall
                {VcSlotConfig{2, 4}, VcSlotConfig{1, 1}},   // Option 12: supports both VCs, smaller overall
                {VcSlotConfig{2, 2}, VcSlotConfig{1, 1}},   // Option 13: supports both VCs, smaller overall
                {VcSlotConfig{1, 1}, VcSlotConfig{1, 1}}    // Option 14: supports both VCs, smaller overall
            }};
        static const std::vector<std::vector<VcSlotConfigArray>> other_buffer_slot_options = {
            // WORMHOLE_B0
            {{VcSlotConfig{16, 16}, VcSlotConfig{0, 0}},  // Only VC0 for non-mesh topologies.
             {VcSlotConfig{8, 16}, VcSlotConfig{0, 0}},
             {VcSlotConfig{8, 8}, VcSlotConfig{0, 0}},
             {VcSlotConfig{4, 8}, VcSlotConfig{0, 0}},
             {VcSlotConfig{4, 4}, VcSlotConfig{0, 0}},
             {VcSlotConfig{2, 4}, VcSlotConfig{0, 0}},
             {VcSlotConfig{2, 2}, VcSlotConfig{0, 0}},
             {VcSlotConfig{1, 2}, VcSlotConfig{0, 0}},
             {VcSlotConfig{1, 1}, VcSlotConfig{0, 0}}},
            // BLACKHOLE
            {{VcSlotConfig{32, 32}, VcSlotConfig{0, 0}},  // Only VC0 for non-mesh topologies.
             {VcSlotConfig{16, 32}, VcSlotConfig{0, 0}},
             {VcSlotConfig{16, 16}, VcSlotConfig{0, 0}},
             {VcSlotConfig{8, 16}, VcSlotConfig{0, 0}},
             {VcSlotConfig{8, 8}, VcSlotConfig{0, 0}},
             {VcSlotConfig{4, 8}, VcSlotConfig{0, 0}},
             {VcSlotConfig{4, 4}, VcSlotConfig{0, 0}},
             {VcSlotConfig{2, 4}, VcSlotConfig{0, 0}},
             {VcSlotConfig{2, 2}, VcSlotConfig{0, 0}},
             {VcSlotConfig{1, 2}, VcSlotConfig{0, 0}},
             {VcSlotConfig{1, 1}, VcSlotConfig{0, 0}}}};

        static tt::stl::Indestructible<std::vector<std::vector<VcSlotConfigArray>>> mesh_slots(
            mesh_buffer_slot_options);
        static tt::stl::Indestructible<std::vector<std::vector<VcSlotConfigArray>>> other_slots(
            other_buffer_slot_options);

        if (topology == Topology::Mesh || topology == Topology::Torus) {
            return mesh_slots.get()[arch_index];
        }
        return other_slots.get()[arch_index];
    };

    auto get_optimal_num_slots_per_vc =
        [this](
            const auto& buffer_slot_options,
            const std::array<size_t, builder_config::MAX_NUM_VCS>& num_sender_channels_per_vc,
            const std::array<size_t, builder_config::MAX_NUM_VCS>& num_receiver_channels_per_vc)
        -> std::array<VcSlotConfig, builder_config::MAX_NUM_VCS> {
        bool vc1_needed = (num_sender_channels_per_vc[1] > 0) || (num_receiver_channels_per_vc[1] > 0);
        for (const auto& option : buffer_slot_options) {
            // skip the VC0 only options if VC1 is needed (either sender or receiver channels)
            if (vc1_needed) {
                // Check if we need VC1 sender channels but this option doesn't provide them
                bool skip_due_to_vc1_sender = (num_sender_channels_per_vc[1] > 0) && (option[1].sender_slots == 0);
                // Check if we need VC1 receiver channels but this option doesn't provide them
                bool skip_due_to_vc1_receiver =
                    (num_receiver_channels_per_vc[1] > 0) && (option[1].receiver_slots == 0);

                if (skip_due_to_vc1_sender || skip_due_to_vc1_receiver) {
                    continue;  // Skip this option - VC1 is needed but this option doesn't support it
                }
            }

            // Calculate total slots across all VCs
            size_t total_slots = 0;
            for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
                total_slots += num_sender_channels_per_vc[vc] * option[vc].sender_slots;
                total_slots += num_receiver_channels_per_vc[vc] * option[vc].receiver_slots;
            }

            auto total_num_bytes = total_slots * this->channel_buffer_size_bytes;

            if (total_num_bytes <= this->available_channel_buffering_space) {
                return option;  // Found a configuration that fits
            }
        }

        // No valid option found
        TT_THROW(
            "Failed to find suitable buffer slot configuration. VC1 needed: {}, VC0 channels: {} senders/{} "
            "receivers, VC1 channels: {} senders/{} receivers, Available space: {} bytes",
            vc1_needed,
            num_sender_channels_per_vc[0],
            num_receiver_channels_per_vc[0],
            num_sender_channels_per_vc[1],
            num_receiver_channels_per_vc[1],
            this->available_channel_buffering_space);
    };

    // auto axis_index = static_cast<std::size_t>(options.edm_axis);
    auto arch = tt::tt_metal::MetalContext::instance().hal().get_arch();
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

            std::array<size_t, builder_config::MAX_NUM_VCS> num_sender_chs = {
                num_used_sender_channels_per_vc[0], num_used_sender_channels_per_vc[1]};
            std::array<size_t, builder_config::MAX_NUM_VCS> num_receiver_chs = {
                static_cast<size_t>(is_receiver_channel_active_per_vc[0]),
                static_cast<size_t>(is_receiver_channel_active_per_vc[1])};

            // get the optimal buffer slots for MUX mode (per-VC)
            auto slot_config = get_optimal_num_slots_per_vc(
                default_with_tensix_buffer_slot_options[arch_index], num_sender_chs, num_receiver_chs);

            // set buffer slots for VC0 worker channel only
            num_sender_buffer_slots_per_vc[0][target_channel] = slot_config[0].sender_slots;
            num_remote_sender_buffer_slots_per_vc[0][target_channel] = slot_config[0].sender_slots;

            // Fill receiver buffer slots for both VCs
            num_receiver_buffer_slots_per_vc[0].fill(slot_config[0].receiver_slots);
            num_remote_receiver_buffer_slots_per_vc[0].fill(slot_config[0].receiver_slots);
            num_receiver_buffer_slots_per_vc[1].fill(slot_config[1].receiver_slots);
            num_remote_receiver_buffer_slots_per_vc[1].fill(slot_config[1].receiver_slots);
            return;
        }
        default: break;
    }

    // Default case: Configure buffer slots with per-VC options
    std::array<size_t, builder_config::MAX_NUM_VCS> num_sender_chs = {
        num_used_sender_channels_per_vc[0], num_used_sender_channels_per_vc[1]};
    std::array<size_t, builder_config::MAX_NUM_VCS> num_receiver_chs = {
        static_cast<size_t>(is_receiver_channel_active_per_vc[0]),
        static_cast<size_t>(is_receiver_channel_active_per_vc[1])};

    // Get optimal buffer slots considering both VCs
    auto slot_config =
        get_optimal_num_slots_per_vc(get_num_buffer_slots(topology, arch_index), num_sender_chs, num_receiver_chs);

    // Apply the buffer slot configuration to each VC
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        num_sender_buffer_slots_per_vc[vc].fill(slot_config[vc].sender_slots);
        num_remote_sender_buffer_slots_per_vc[vc].fill(slot_config[vc].sender_slots);
        num_receiver_buffer_slots_per_vc[vc].fill(slot_config[vc].receiver_slots);
        num_remote_receiver_buffer_slots_per_vc[vc].fill(slot_config[vc].receiver_slots);
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

    // Emit receiver channel args for all VCs (one channel per VC)
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        if (this->is_receiver_channel_active_per_vc[vc]) {
            ct_args.push_back(static_cast<uint32_t>(this->receiver_channel_base_address[vc]));
            ct_args.push_back(this->receiver_channel_num_buffers[vc]);
            ct_args.push_back(static_cast<uint32_t>(this->remote_receiver_channel_base_address[vc]));
            ct_args.push_back(this->remote_receiver_channel_num_buffers[vc]);
        }
    }
}

void FabricStaticSizedChannelsAllocator::emit_channel_allocations_ct_args(
    std::vector<uint32_t>& ct_args,
    const std::array<size_t, builder_config::MAX_NUM_VCS>& num_used_sender_channels_per_vc,
    const std::array<bool, builder_config::MAX_NUM_VCS>& is_receiver_channel_active_per_vc) const {
    size_t num_used_vc0_sender_channels = num_used_sender_channels_per_vc[0];
    size_t num_used_vc1_sender_channels = num_used_sender_channels_per_vc[1];
    size_t num_used_receiver_channels = static_cast<size_t>(is_receiver_channel_active_per_vc[0]) +
                                        static_cast<size_t>(is_receiver_channel_active_per_vc[1]);
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
    // The allocator emits entries in order: VC0 senders, VC1 senders, VC0 receivers, VC1 receivers.
    // The router may use fewer channels than allocated. When there are unused channels,
    // VC1 sender entries need to skip over the unused VC0 entry slots.
    size_t num_used_sender_channels = num_used_vc0_sender_channels + num_used_vc1_sender_channels;
    size_t num_unused_channels = total_sender_channels - num_used_sender_channels;
    bool has_unused_channels = (num_unused_channels > 0) && num_used_sender_channels > 0;

    // Sender channel-to-entry index
    if (has_unused_channels) {
        for (size_t i = 0; i < num_used_sender_channels; ++i) {
            if (i < num_used_vc0_sender_channels) {
                ct_args.push_back(static_cast<uint32_t>(i));
            } else {
                // VC1 channels skip the unused VC0 channel entries
                ct_args.push_back(static_cast<uint32_t>(i + num_unused_channels));
            }
        }
        // Padding for unused channels — map to their actual (unserviced) entry indices
        for (size_t i = 0; i < num_unused_channels; ++i) {
            ct_args.push_back(static_cast<uint32_t>(num_used_vc0_sender_channels + i));
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
