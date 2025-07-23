// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <utility>
#include <vector>

#include <umd/device/tt_core_coordinates.h>
#include "command_queue_common.hpp"
#include "dispatch_settings.hpp"

namespace tt {
namespace tt_metal {
enum class CommandQueueDeviceAddrType : uint8_t;
}  // namespace tt_metal
}  // namespace tt

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
    DispatchMemMap(const CoreType& core_type, uint32_t num_hw_cqs);

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
    // Reset the instance using the settings for the core_type and num_hw_cqs.
    void reset(const CoreType& core_type, uint32_t num_hw_cqs);

    std::pair<uint32_t, uint32_t> get_device_l1_info(const CoreType& core_type) const;

    uint32_t cmddat_q_base_ = 0;
    uint32_t scratch_db_base_ = 0;
    uint32_t dispatch_buffer_base_ = 0;

    uint32_t dispatch_buffer_block_size_pages_ = 0;
    std::vector<uint32_t> device_cq_addrs_;

    DispatchSettings settings;

    uint32_t hw_cqs{0};  // 0 means uninitialized
    CoreType last_core_type{CoreType::WORKER};
};

}  // namespace tt::tt_metal
