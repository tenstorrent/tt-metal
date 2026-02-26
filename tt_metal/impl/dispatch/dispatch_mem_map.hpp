// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <umd/device/types/core_coordinates.hpp>
#include "command_queue_common.hpp"
#include "dispatch_settings.hpp"

namespace tt::tt_metal {
class Hal;
enum class CommandQueueDeviceAddrType : uint8_t;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

//
// Dispatch Memory Map
// Assigns each CommandQueueDeviceAddrType in a linear
// order. The size of each address type and L1 base is
// set by DispatchSettings.
//
class DispatchMemMap {
public:
    DispatchMemMap& operator=(const DispatchMemMap&) = delete;
    DispatchMemMap& operator=(DispatchMemMap&& other) noexcept = delete;
    DispatchMemMap(const DispatchMemMap&) = delete;
    DispatchMemMap(DispatchMemMap&& other) noexcept = delete;
    // Create a DispatchMemMap
    DispatchMemMap(const CoreType& core_type, uint32_t num_hw_cqs, const Hal& hal, bool is_galaxy_cluster);

    uint32_t prefetch_q_entries() const;

    uint32_t prefetch_q_size() const;

    uint32_t max_prefetch_command_size() const;

    uint32_t cmddat_q_base() const;

    uint32_t cmddat_q_size() const;

    uint32_t scratch_db_base() const;

    uint32_t scratch_db_size() const;

    uint32_t ringbuffer_size() const;

    uint32_t dispatch_buffer_block_size_pages() const;

    uint32_t dispatch_buffer_base() const;

    uint32_t dispatch_buffer_pages() const;

    uint32_t prefetch_d_buffer_size() const;

    uint32_t prefetch_d_buffer_pages() const;

    uint32_t dispatch_s_buffer_size() const;

    uint32_t dispatch_s_buffer_pages() const;

    uint32_t get_device_command_queue_addr(const CommandQueueDeviceAddrType& device_addr_type) const;

    uint32_t get_host_command_queue_addr(const CommandQueueHostAddrType& host_addr) const;

    uint32_t get_sync_offset(uint32_t index) const;

    uint32_t get_dispatch_message_addr_start() const;

    uint32_t get_dispatch_stream_index(uint32_t index) const;

    // Offset to be passed in the go message.
    uint8_t get_dispatch_message_update_offset(uint32_t index) const;

    uint32_t get_prefetcher_l1_size() const;

private:
    uint32_t cmddat_q_base_ = 0;
    uint32_t scratch_db_base_ = 0;
    uint32_t dispatch_buffer_base_ = 0;

    uint32_t dispatch_buffer_block_size_pages_ = 0;
    std::vector<uint32_t> device_cq_addrs_;

    DispatchSettings settings;

    uint32_t host_alignment_ = 0;
    uint32_t l1_alignment_ = 0;
    uint32_t l1_size_ = 0;
    uint32_t noc_overlay_start_addr_ = 0;
    uint32_t noc_stream_reg_space_size_ = 0;
    uint32_t noc_stream_remote_dest_buf_space_available_update_reg_index_ = 0;
    uint32_t dispatch_stream_base_ = 0;
};

}  // namespace tt::tt_metal
