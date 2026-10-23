// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_v2_kernel_ct_args.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_utils.h"
#include <hostdevcommon/fabric_mux_v2_common.h>

namespace tt::tt_fabric::mux_v2::kernel {

namespace ct_args = tt::tt_fabric::mux_v2::ct_args;

constexpr uint32_t kMaxRuntimeChannels = 64;

static_assert(
    ct_args::num_channels <= kMaxRuntimeChannels, "FabricMuxV2 num_channels exceeds forwarder channel storage");

struct ForwarderChannel {
    uint32_t pending_packets_stream_id = 0;
    uint32_t channel_base_address = 0;
    uint32_t next_send_slot = 0;

    bool has_pending_packets() const {
        return get_ptr_val(pending_packets_stream_id) != static_cast<int32_t>(ct_args::num_buffers_per_channel);
    }

    uint32_t get_slot_address(uint32_t slot_index) const {
        return channel_base_address + (slot_index * ct_args::channel_buffer_size_bytes);
    }

    uint32_t get_current_slot_address() const { return get_slot_address(next_send_slot); }

    void advance_slot() {
        if constexpr (ct_args::num_buffers_per_channel_is_pow2) {
            next_send_slot = (next_send_slot + 1) & ct_args::num_buffers_per_channel_mask;
        } else {
            next_send_slot = next_send_slot + 1 == ct_args::num_buffers_per_channel ? 0 : next_send_slot + 1;
        }
    }
};

struct ForwarderContext {
    std::array<ForwarderChannel, kMaxRuntimeChannels> channels{};
};

constexpr ForwarderContext make_forwarder_context() {
    ForwarderContext context{};
    const uint32_t channel_stride_bytes = ct_args::num_buffers_per_channel * ct_args::channel_buffer_size_bytes;

    for (uint32_t channel_idx = 0; channel_idx < ct_args::num_channels; ++channel_idx) {
        auto& channel = context.channels[channel_idx];
        channel.pending_packets_stream_id = channel_idx;
        channel.channel_base_address = ct_args::channel_region_base_address + (channel_idx * channel_stride_bytes);
    }

    return context;
}

inline volatile tt_l1_ptr uint32_t* get_status_ptr() {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct_args::mux_status_address);
}

inline volatile tt_l1_ptr uint32_t* get_forwarder_ready_sem_ptr() {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore<ProgrammableCoreType::TENSIX>(ct_args::forwarder_ready_sem_id));
}

inline volatile tt_l1_ptr uint32_t* get_manager_init_done_sem_ptr() {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore<ProgrammableCoreType::TENSIX>(ct_args::manager_init_done_sem_id));
}

inline volatile tt_l1_ptr tt::tt_fabric::FabricMuxV2SharedControlBlock* get_shared_control_ptr() {
    return reinterpret_cast<volatile tt_l1_ptr tt::tt_fabric::FabricMuxV2SharedControlBlock*>(
        ct_args::shared_control_region_base_address);
}

constexpr uint32_t get_per_channel_scalar_address(uint32_t base_address, uint32_t logical_channel_id) {
    return base_address + (logical_channel_id * ct_args::per_channel_scalar_region_stride_bytes);
}

inline volatile tt_l1_ptr uint32_t* get_connection_handshake_ptr(uint32_t logical_channel_id) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_per_channel_scalar_address(ct_args::connection_handshake_region_base_address, logical_channel_id));
}

inline uint32_t get_credit_notify_scratch_address(uint32_t logical_channel_id) {
    return get_per_channel_scalar_address(ct_args::credit_notify_scratch_region_base_address, logical_channel_id);
}

inline volatile tt_l1_ptr tt::tt_fabric::FabricMuxV2SharedTridRingHeader* get_shared_ring_header_ptr() {
    return reinterpret_cast<volatile tt_l1_ptr tt::tt_fabric::FabricMuxV2SharedTridRingHeader*>(
        ct_args::shared_ring_region_base_address);
}

inline volatile tt_l1_ptr tt::tt_fabric::FabricMuxV2SharedTridRingEntry* get_shared_ring_entries_ptr() {
    return reinterpret_cast<volatile tt_l1_ptr tt::tt_fabric::FabricMuxV2SharedTridRingEntry*>(
        ct_args::shared_ring_region_base_address + sizeof(tt::tt_fabric::FabricMuxV2SharedTridRingHeader));
}

inline volatile tt_l1_ptr tt::tt_fabric::EDMChannelWorkerLocationInfo* get_connection_info_ptr(
    uint32_t logical_channel_id) {
    return reinterpret_cast<volatile tt_l1_ptr tt::tt_fabric::EDMChannelWorkerLocationInfo*>(
        ct_args::connection_info_region_base_address +
        (logical_channel_id * sizeof(tt::tt_fabric::EDMChannelWorkerLocationInfo)));
}

inline void zero_per_channel_scalar_region(uint32_t base_address, uint32_t stride_bytes, uint32_t count) {
    for (uint32_t channel_id = 0; channel_id < count; ++channel_id) {
        auto word_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base_address + (channel_id * stride_bytes));
        word_ptr[0] = 0;
    }
}

inline void initialize_shared_control_block(
    volatile tt_l1_ptr tt::tt_fabric::FabricMuxV2SharedControlBlock* shared_control_ptr) {
    shared_control_ptr->drain_initiated = 0;
    shared_control_ptr->forwarder_stop_tracking = 0;
    shared_control_ptr->forwarder_done = 0;
}

inline void initialize_shared_ring_header() {
    auto shared_ring_header_ptr = get_shared_ring_header_ptr();
    shared_ring_header_ptr->write_count = 0;
    shared_ring_header_ptr->read_count = 0;
}

inline void initialize_pending_stream_state(const ForwarderContext& context) {
    for (uint32_t channel_idx = 0; channel_idx < ct_args::num_channels; ++channel_idx) {
        const auto& channel = context.channels[channel_idx];
        init_ptr_val(channel.pending_packets_stream_id, ct_args::num_buffers_per_channel);
    }
}

inline tt::tt_fabric::WorkerToFabricEdmSender build_downstream_sender() {
    auto arg_idx = std::size_t{0};
    return tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
}

}  // namespace tt::tt_fabric::mux_v2::kernel
