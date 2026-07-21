// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <umd/device/types/core_coordinates.hpp>
#include "command_queue_common.hpp"
#include "dispatch_core_common.hpp"
#include "dispatch_settings.hpp"

namespace tt::llrt {
class RunTimeOptions;
}  // namespace tt::llrt

namespace tt::tt_metal {
class Hal;
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
    DispatchMemMap(
        const CoreType& core_type,
        uint32_t num_hw_cqs,
        const Hal& hal,
        bool is_galaxy_cluster,
        const CommandQueueDispatchLayout& cq_layout,
        const tt::llrt::RunTimeOptions& rtoptions);

    uint32_t prefetch_q_entries() const;

    uint32_t prefetch_q_entry_size_bytes() const;

    uint32_t prefetch_q_size() const;

    uint32_t max_prefetch_command_size() const;

    uint32_t cmddat_q_base(uint8_t cq_id) const;

    uint32_t cmddat_q_size() const;

    uint32_t scratch_db_base(uint8_t cq_id) const;

    uint32_t scratch_db_size() const;

    uint32_t ringbuffer_size() const;

    uint32_t dispatch_buffer_block_size_pages() const;

    uint32_t dispatch_buffer_base(uint8_t cq_id) const;

    uint32_t dispatch_buffer_pages() const;

    uint32_t prefetch_d_buffer_size() const;

    uint32_t prefetch_d_buffer_pages() const;

    uint32_t dispatch_s_buffer_size() const;

    uint32_t dispatch_s_buffer_pages() const;

    // L1 layout for the DEVICE_PRINT dispatch region (placed immediately after the dispatch_s CB):
    // noc_locations and l1_cache OVERLAP at the same L1 address. The kernel reads noc_locations[]
    // once during init() and caches everything into LDM (rw_noc_addresses, cache_buffer_offsets,
    // cache_buffer_sizes), so after init the L1 region can be reused as the l1_cache buffer.
    // This is safe because:
    //   * sizeof(NocLocationInputInfo) (12B) < rw_pointers_entry_size (>= 16B), so the
    //     noc_locations bytes occupy strictly less L1 than the kernel's first writes will use.
    //   * the kernel's first writes to that L1 region happen AFTER the noc_locations cache loop.
    // Everything is precomputed in the constructor from rtoptions + hal; callers (DispatchSKernel)
    // just read the addresses and pass the runtime per-device print-core count to the kernel
    // via static_config_.device_print_noc_locations_count.
    uint32_t dispatch_s_device_print_l1_cache_size() const;
    uint32_t device_print_dispatch_noc_locations_addr(uint8_t cq_id) const;
    uint32_t device_print_dispatch_l1_cache_addr(uint8_t cq_id) const;

    uint32_t get_device_command_queue_addr(const CommandQueueDeviceAddrType& device_addr_type, uint8_t cq_id) const;

    uint32_t get_host_command_queue_addr(const CommandQueueHostAddrType& host_addr) const;

    uint32_t get_sync_offset(uint32_t index) const;

    uint32_t get_dispatch_message_addr_start() const;

    uint32_t get_dispatch_stream_index(uint32_t index) const;

    // Offset to be passed in the go message.
    uint8_t get_dispatch_message_update_offset(uint32_t index) const;

    // Index of the first worker-completion counter reserved for cq_id's workers, in the shared
    // WORKER_COMPLETION_SEMAPHORES L1 region.
    uint32_t get_completion_counter_offset(uint8_t cq_id) const;

private:
    uint32_t cmddat_q_base_ = 0;
    uint32_t scratch_db_base_ = 0;
    uint32_t dispatch_buffer_base_ = 0;
    uint32_t dispatch_s_device_print_l1_cache_size_ = 0;
    uint32_t dispatch_s_buffer_end_ = 0;

    uint32_t dispatch_buffer_block_size_pages_ = 0;
    std::vector<uint32_t> device_cq_addrs_;

    DispatchSettings settings;

    uint32_t num_cqs_per_core_ = 0;
    uint32_t cq_zone_stride_ = 0;

    uint32_t host_alignment_ = 0;
    uint32_t l1_alignment_ = 0;
    uint32_t noc_overlay_start_addr_ = 0;
    uint32_t noc_stream_reg_space_size_ = 0;
    uint32_t noc_stream_remote_dest_buf_space_available_update_reg_index_ = 0;
    uint32_t dispatch_stream_base_ = 0;
    bool has_stream_registers_ = false;
};

}  // namespace tt::tt_metal
