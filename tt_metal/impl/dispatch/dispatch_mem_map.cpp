// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <enchantum/enchantum.hpp>
#include <tt-metalium/tt_align.hpp>

#include "dispatch_mem_map.hpp"
#include <tt_stl/assert.hpp>
#include "command_queue_common.hpp"
#include "dispatch_settings.hpp"
#include "hal_types.hpp"
#include "llrt/hal.hpp"
#include "llrt/rtoptions.hpp"
#include <hostdevcommon/dispatch_telemetry_types.hpp>
#include <tt_stl/enum.hpp>

namespace tt::tt_metal {

DispatchMemMap::DispatchMemMap(
    const CoreType& core_type,
    uint32_t num_hw_cqs,
    const Hal& hal,
    bool is_galaxy_cluster,
    const CommandQueueDispatchLayout& cq_layout,
    const tt::llrt::RunTimeOptions& rtoptions) :
    settings(DispatchSettings(
        num_hw_cqs,

        core_type,

        is_galaxy_cluster,

        rtoptions.get_dram_backed_cq(),

        hal.get_alignment(HalMemType::L1),

        // Prefetch queue entry width: each entry encodes the prefetch command size with the MSB reserved as a
        // stall flag. Both 2-byte (15 size bits) and 4-byte (31 size bits) widths cover today's
        // prefetch_max_cmd_size on every arch. We prefer 4 bytes because sub-32-bit reads/writes have been
        // observed to fail on Quasar, and using one width everywhere keeps host/kernel code simple. WH ETH
        // stays at 2 bytes because of tighter memory constraints.
        (hal.get_arch() == tt::ARCH::WORMHOLE_B0 && core_type == CoreType::ETH) ? 2u : 4u)),
    num_cqs_per_core_(cq_layout.num_cqs_per_core),
    host_alignment_(hal.get_alignment(HalMemType::HOST)),
    l1_alignment_(hal.get_alignment(HalMemType::L1)),
    noc_overlay_start_addr_(hal.get_noc_overlay_start_addr()),
    noc_stream_reg_space_size_(hal.get_noc_stream_reg_space_size()),
    noc_stream_remote_dest_buf_space_available_update_reg_index_(
        hal.get_noc_stream_remote_dest_buf_space_available_update_reg_index()),
    has_stream_registers_(hal.has_stream_registers()) {
    uint32_t l1_base;
    uint32_t l1_size;
    if (core_type == CoreType::WORKER) {
        l1_base = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
        l1_size = hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
        dispatch_stream_base_ = 48u;  // 64 streams; CBs use entries 8-39
    } else if (core_type == CoreType::ETH) {
        l1_base = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
        l1_size = hal.get_dev_size(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::BASE);
        dispatch_stream_base_ = 16u;  // 32 streams
    } else {
        TT_THROW("DispatchMemMap not implemented for core type");
    }

    const auto dispatch_buffer_block_size = settings.dispatch_size_;
    const auto pcie_alignment = host_alignment_;
    const auto l1_alignment = l1_alignment_;

    TT_ASSERT(settings.prefetch_cmddat_q_size_ >= 2 * settings.prefetch_max_cmd_size_);
    TT_ASSERT(settings.prefetch_scratch_db_size_ % 2 == 0);
    TT_ASSERT((dispatch_buffer_block_size & (dispatch_buffer_block_size - 1)) == 0);
    TT_ASSERT(
        DispatchSettings::DISPATCH_MESSAGE_ENTRIES <= DispatchSettings::DISPATCH_MESSAGES_MAX_OFFSET / l1_alignment + 1,
        "Number of dispatch message entries exceeds max representable offset");

    constexpr uint8_t num_dev_cq_addrs = enchantum::count<CommandQueueDeviceAddrType>;
    std::vector<uint32_t> device_cq_addr_sizes_(num_dev_cq_addrs, 0);
    for (auto dev_addr_idx = 0; dev_addr_idx < num_dev_cq_addrs; dev_addr_idx++) {
        CommandQueueDeviceAddrType dev_addr_type = enchantum::cast<CommandQueueDeviceAddrType>(dev_addr_idx).value();
        if (dev_addr_type == CommandQueueDeviceAddrType::PREFETCH_Q_RD) {
            device_cq_addr_sizes_[dev_addr_idx] = settings.prefetch_q_rd_ptr_size_;
        } else if (dev_addr_type == CommandQueueDeviceAddrType::PREFETCH_Q_PCIE_RD) {
            device_cq_addr_sizes_[dev_addr_idx] = settings.prefetch_q_pcie_rd_ptr_size_;
        } else if (dev_addr_type == CommandQueueDeviceAddrType::DISPATCH_S_SYNC_SEM) {
            device_cq_addr_sizes_[dev_addr_idx] = settings.dispatch_s_sync_sem_;
        } else if (dev_addr_type == CommandQueueDeviceAddrType::FABRIC_HEADER_RB) {
            // At this point fabric context is not initialized yet
            // Hardcode to 128B (more than enough space) for now
            device_cq_addr_sizes_[dev_addr_idx] = tt::tt_metal::DispatchSettings::FABRIC_HEADER_RB_ENTRIES * 128;
        } else if (dev_addr_type == CommandQueueDeviceAddrType::REALTIME_PROFILER_MSG) {
            // Real-time profiler mailbox: dispatch-core-local L1 region shared between the
            // dispatch cores and the reserved RT-profiler tensix core.
            device_cq_addr_sizes_[dev_addr_idx] =
                hal.get_realtime_profiler_msgs_factory(HalProgrammableCoreType::TENSIX)
                    .size_of<realtime_profiler_msgs::realtime_profiler_msg_t>();
        } else if (dev_addr_type == CommandQueueDeviceAddrType::DISPATCH_TELEMETRY) {
            device_cq_addr_sizes_[dev_addr_idx] = dispatch_telemetry_types::DISPATCH_TELEMETRY_SIZE;
        } else if (dev_addr_type == CommandQueueDeviceAddrType::DISPATCH_TELEMETRY_CONTROL) {
            device_cq_addr_sizes_[dev_addr_idx] = sizeof(dispatch_telemetry_types::DispatchTelemetryControl);
        } else if (dev_addr_type == CommandQueueDeviceAddrType::WORKER_COMPLETION_SEMAPHORES) {
            device_cq_addr_sizes_[dev_addr_idx] =
                has_stream_registers_
                    ? 0
                    : cq_layout.num_cqs_per_core * DispatchSettings::DISPATCH_MESSAGE_ENTRIES * l1_alignment;
        } else if (
            dev_addr_type == CommandQueueDeviceAddrType::FABRIC_SYNC_STATUS ||
            dev_addr_type == CommandQueueDeviceAddrType::DISPATCH_PROGRESS) {
            device_cq_addr_sizes_[dev_addr_idx] = sizeof(uint32_t);
        } else {
            device_cq_addr_sizes_[dev_addr_idx] = settings.other_ptrs_size;
        }
    }

    device_cq_addrs_.resize(num_dev_cq_addrs);
    device_cq_addrs_[0] = l1_base;
    for (auto dev_addr_idx = 1; dev_addr_idx < num_dev_cq_addrs; dev_addr_idx++) {
        device_cq_addrs_[dev_addr_idx] = device_cq_addrs_[dev_addr_idx - 1] + device_cq_addr_sizes_[dev_addr_idx - 1];
        auto dev_addr_type = *enchantum::index_to_enum<CommandQueueDeviceAddrType>(dev_addr_idx);
        if (dev_addr_type == CommandQueueDeviceAddrType::UNRESERVED) {
            device_cq_addrs_[dev_addr_idx] = align(device_cq_addrs_[dev_addr_idx], pcie_alignment);
        } else if (
            dev_addr_type == CommandQueueDeviceAddrType::DISPATCH_PROGRESS ||
            dev_addr_type == CommandQueueDeviceAddrType::FABRIC_HEADER_RB ||
            dev_addr_type == CommandQueueDeviceAddrType::FABRIC_SYNC_STATUS ||
            dev_addr_type == CommandQueueDeviceAddrType::REALTIME_PROFILER_MSG ||
            dev_addr_type == CommandQueueDeviceAddrType::DISPATCH_TELEMETRY ||
            dev_addr_type == CommandQueueDeviceAddrType::DISPATCH_TELEMETRY_CONTROL ||
            dev_addr_type == CommandQueueDeviceAddrType::WORKER_COMPLETION_SEMAPHORES) {
            device_cq_addrs_[dev_addr_idx] = align(device_cq_addrs_[dev_addr_idx], l1_alignment);
        }
    }

    uint32_t prefetch_dispatch_unreserved_base =
        device_cq_addrs_[ttsl::as_underlying_type<CommandQueueDeviceAddrType>(CommandQueueDeviceAddrType::UNRESERVED)];
    cmddat_q_base_ = align(prefetch_dispatch_unreserved_base + settings.prefetch_q_size_, pcie_alignment);
    scratch_db_base_ = align(cmddat_q_base_ + settings.prefetch_cmddat_q_size_, pcie_alignment);
    if (cq_layout.fd_kernels_on_same_core) {
        // All FD kernels share one core (Quasar), so dispatch_buffer must not alias
        // any of prefetch_q / cmddat_q / scratch_db / ringbuffer. scratch_db and
        // ringbuffer are two views of the same region rooted at scratch_db_base_, so
        // the prefetch footprint ends at scratch_db_base_ + max(scratch_db, ringbuffer).
        uint32_t prefetch_top =
            scratch_db_base_ + std::max(settings.prefetch_scratch_db_size_, settings.prefetch_ringbuffer_size_);
        dispatch_buffer_base_ = align(prefetch_top, 1u << DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE);
    } else {
        dispatch_buffer_base_ =
            align(prefetch_dispatch_unreserved_base, 1 << DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE);
    }
    dispatch_buffer_block_size_pages_ = settings.dispatch_pages_ / DispatchSettings::DISPATCH_BUFFER_SIZE_BLOCKS;
    const uint32_t dispatch_cb_end = dispatch_buffer_base_ + settings.dispatch_size_;

    TT_FATAL(
        scratch_db_base_ + settings.prefetch_ringbuffer_size_ <= l1_size,
        "Ringbuffer (start: {}, end: {}) extends past L1 end (size: {})",
        scratch_db_base_,
        scratch_db_base_ + settings.prefetch_ringbuffer_size_,
        l1_size);

    TT_ASSERT(
        dispatch_cb_end + (cq_layout.fd_kernels_on_same_core ? settings.dispatch_s_buffer_size_ : 0) <= l1_size,
        "Dispatch layout overflows L1 (dispatch_cb_end=0x{:X}, l1_size=0x{:X})",
        dispatch_cb_end,
        l1_size);
    TT_ASSERT(dispatch_cb_end < l1_size);

    const uint32_t dispatch_s_buffer_base = (core_type == CoreType::WORKER) ? dispatch_cb_end : dispatch_buffer_base_;
    dispatch_s_buffer_end_ = dispatch_s_buffer_base + settings.dispatch_s_buffer_size_;
    TT_FATAL(
        dispatch_s_buffer_end_ <= l1_size,
        "dispatch_s buffer end ({}) extends past L1 end (size {})",
        dispatch_s_buffer_end_,
        l1_size);

    // Per-arch defaults for the dispatch_s DEVICE_PRINT L1 cache buffer. Lives here (rather than
    // on Hal) because this is a dispatch-side memory layout decision, and callers should reach
    // it through DispatchMemMap. A non-zero rtoptions override replaces the per-arch default.
    const uint32_t device_print_override = rtoptions.get_device_print_dispatch_l1_cache_bytes();
    if (device_print_override != 0) {
        dispatch_s_device_print_l1_cache_size_ = device_print_override;
    } else {
        switch (hal.get_arch()) {
            case tt::ARCH::WORMHOLE_B0: dispatch_s_device_print_l1_cache_size_ = 8 * 1024; break;
            case tt::ARCH::BLACKHOLE: dispatch_s_device_print_l1_cache_size_ = 64 * 1024; break;
            // With multiple CQs sharing one dispatch core's L1, there isn't enough space left over for the full device
            // print L1 cache.
            case tt::ARCH::QUASAR:
                dispatch_s_device_print_l1_cache_size_ = cq_layout.num_cqs_per_core > 1 ? 32 * 1024 : 128 * 1024;
                break;
            default: dispatch_s_device_print_l1_cache_size_ = 0; break;
        }
    }

    TT_FATAL(
        dispatch_s_buffer_end_ + dispatch_s_device_print_l1_cache_size_ <= l1_size,
        "DPRINT dispatch L1 region (l1_cache {} bytes after dispatch_s end {}) exceeds L1 size {}",
        dispatch_s_device_print_l1_cache_size_,
        dispatch_s_buffer_end_,
        l1_size);

    // Each CQ co-located on this core gets its own zone: a fixed-size shift of the base (CQ0) address
    // space computed above. cq_zone_stride_ is 0 when this core hosts a single CQ, so per-CQ accessors
    // are a no-op in that case.
    cq_zone_stride_ = (num_cqs_per_core_ > 1)
                          ? align(
                                dispatch_s_buffer_end_ + dispatch_s_device_print_l1_cache_size_ - l1_base,
                                1u << DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE)
                          : 0;

    const uint32_t last_cq_zone_end =
        dispatch_s_buffer_end_ + dispatch_s_device_print_l1_cache_size_ + (num_cqs_per_core_ - 1) * cq_zone_stride_;
    TT_FATAL(
        last_cq_zone_end <= l1_size, "Last CQ zone end ({}) extends past L1 end (size {})", last_cq_zone_end, l1_size);
}

uint32_t DispatchMemMap::prefetch_q_entries() const { return settings.prefetch_q_entries_; }

uint32_t DispatchMemMap::prefetch_q_entry_size_bytes() const { return settings.prefetch_q_entry_size_bytes_; }

uint32_t DispatchMemMap::prefetch_q_size() const { return settings.prefetch_q_size_; }

uint32_t DispatchMemMap::max_prefetch_command_size() const { return settings.prefetch_max_cmd_size_; }

uint32_t DispatchMemMap::cmddat_q_base(uint8_t cq_id) const { return cmddat_q_base_ + cq_id * cq_zone_stride_; }

uint32_t DispatchMemMap::cmddat_q_size() const { return settings.prefetch_cmddat_q_size_; }

uint32_t DispatchMemMap::scratch_db_base(uint8_t cq_id) const { return scratch_db_base_ + cq_id * cq_zone_stride_; }

uint32_t DispatchMemMap::scratch_db_size() const { return settings.prefetch_scratch_db_size_; }

uint32_t DispatchMemMap::ringbuffer_size() const { return settings.prefetch_ringbuffer_size_; }

uint32_t DispatchMemMap::dispatch_buffer_block_size_pages() const { return dispatch_buffer_block_size_pages_; }

uint32_t DispatchMemMap::dispatch_buffer_base(uint8_t cq_id) const {
    return dispatch_buffer_base_ + cq_id * cq_zone_stride_;
}

uint32_t DispatchMemMap::dispatch_buffer_pages() const { return settings.dispatch_pages_; }

uint32_t DispatchMemMap::prefetch_d_buffer_size() const { return settings.prefetch_d_buffer_size_; }

uint32_t DispatchMemMap::prefetch_d_buffer_pages() const { return settings.prefetch_d_pages_; }

uint32_t DispatchMemMap::dispatch_s_buffer_size() const { return settings.dispatch_s_buffer_size_; }

uint32_t DispatchMemMap::dispatch_s_buffer_pages() const {
    return settings.dispatch_s_buffer_size_ / (1 << tt::tt_metal::DispatchSettings::DISPATCH_S_BUFFER_LOG_PAGE_SIZE);
}

uint32_t DispatchMemMap::dispatch_s_device_print_l1_cache_size() const {
    return dispatch_s_device_print_l1_cache_size_;
}

uint32_t DispatchMemMap::device_print_dispatch_noc_locations_addr(uint8_t cq_id) const {
    return dispatch_s_buffer_end_ + cq_id * cq_zone_stride_;
}

uint32_t DispatchMemMap::device_print_dispatch_l1_cache_addr(uint8_t cq_id) const {
    return dispatch_s_buffer_end_ + cq_id * cq_zone_stride_;
}

uint32_t DispatchMemMap::get_device_command_queue_addr(
    const CommandQueueDeviceAddrType& device_addr_type, uint8_t cq_id) const {
    const uint32_t index = ttsl::as_underlying_type<CommandQueueDeviceAddrType>(device_addr_type);
    TT_ASSERT(index < this->device_cq_addrs_.size());
    if (device_addr_type == CommandQueueDeviceAddrType::WORKER_COMPLETION_SEMAPHORES) {
        TT_FATAL(!this->has_stream_registers_, "Attempting to read address of unallocated memory region");
    }
    uint32_t addr = device_cq_addrs_[index];
    if (!is_cq_shared(device_addr_type)) {
        addr += cq_id * cq_zone_stride_;
    }
    return addr;
}

uint32_t DispatchMemMap::get_host_command_queue_addr(const CommandQueueHostAddrType& host_addr) const {
    return ttsl::as_underlying_type<CommandQueueHostAddrType>(host_addr) * host_alignment_;
}

uint32_t DispatchMemMap::get_sync_offset(uint32_t index) const {
    TT_ASSERT(index < tt::tt_metal::DispatchSettings::DISPATCH_MESSAGE_ENTRIES);
    return index * l1_alignment_;
}

uint32_t DispatchMemMap::get_dispatch_message_addr_start() const {
    if (!has_stream_registers_) {
        // On arches without stream registers (Quasar), use the dedicated L1 worker completion counters region.
        // WORKER_COMPLETION_SEMAPHORES is shared across CQs, so cq_id here does not affect the address.
        return get_device_command_queue_addr(CommandQueueDeviceAddrType::WORKER_COMPLETION_SEMAPHORES, 0);
    }
    return noc_overlay_start_addr_ + (noc_stream_reg_space_size_ * get_dispatch_stream_index(0)) +
           (noc_stream_remote_dest_buf_space_available_update_reg_index_ * sizeof(uint32_t));
}

uint32_t DispatchMemMap::get_dispatch_stream_index(uint32_t index) const { return dispatch_stream_base_ + index; }

uint8_t DispatchMemMap::get_dispatch_message_update_offset(uint32_t index) const {
    TT_ASSERT(index < tt::tt_metal::DispatchSettings::DISPATCH_MESSAGES_MAX_OFFSET);
    return index;
}

uint32_t DispatchMemMap::get_completion_counter_offset(uint8_t cq_id) const {
    const uint8_t slot = (num_cqs_per_core_ > 1) ? cq_id : 0;
    return slot * DispatchSettings::DISPATCH_MESSAGE_ENTRIES;
}

}  // namespace tt::tt_metal
