// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
#include <tt_stl/enum.hpp>

namespace tt::tt_metal {

DispatchMemMap::DispatchMemMap(const CoreType& core_type, uint32_t num_hw_cqs, const Hal& hal, bool is_galaxy_cluster) :
    settings(DispatchSettings(num_hw_cqs, core_type, is_galaxy_cluster, hal.get_alignment(HalMemType::L1))),
    host_alignment_(hal.get_alignment(HalMemType::HOST)),
    l1_alignment_(hal.get_alignment(HalMemType::L1)),
    noc_overlay_start_addr_(hal.get_noc_overlay_start_addr()),
    noc_stream_reg_space_size_(hal.get_noc_stream_reg_space_size()),
    noc_stream_remote_dest_buf_space_available_update_reg_index_(
        hal.get_noc_stream_remote_dest_buf_space_available_update_reg_index()) {
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
    l1_size_ = l1_size;

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
            dev_addr_type == CommandQueueDeviceAddrType::FABRIC_SYNC_STATUS) {
            device_cq_addrs_[dev_addr_idx] = align(device_cq_addrs_[dev_addr_idx], l1_alignment);
        }
    }

    uint32_t prefetch_dispatch_unreserved_base =
        device_cq_addrs_[ttsl::as_underlying_type<CommandQueueDeviceAddrType>(CommandQueueDeviceAddrType::UNRESERVED)];
    cmddat_q_base_ = align(prefetch_dispatch_unreserved_base + settings.prefetch_q_size_, pcie_alignment);
    scratch_db_base_ = align(cmddat_q_base_ + settings.prefetch_cmddat_q_size_, pcie_alignment);
    dispatch_buffer_base_ =
        align(prefetch_dispatch_unreserved_base, 1 << DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE);
    dispatch_buffer_block_size_pages_ = settings.dispatch_pages_ / DispatchSettings::DISPATCH_BUFFER_SIZE_BLOCKS;
    const uint32_t dispatch_cb_end = dispatch_buffer_base_ + settings.dispatch_size_;

    TT_ASSERT(scratch_db_base_ + settings.prefetch_scratch_db_size_ < l1_size);
    TT_FATAL(
        scratch_db_base_ + settings.prefetch_ringbuffer_size_ <= l1_size,
        "Ringbuffer (start: {}, end: {}) extends past L1 end (size: {})",
        scratch_db_base_,
        scratch_db_base_ + settings.prefetch_ringbuffer_size_,
        l1_size);

    TT_ASSERT(dispatch_cb_end < l1_size);
}

uint32_t DispatchMemMap::prefetch_q_entries() const { return settings.prefetch_q_entries_; }

uint32_t DispatchMemMap::prefetch_q_size() const { return settings.prefetch_q_size_; }

uint32_t DispatchMemMap::max_prefetch_command_size() const { return settings.prefetch_max_cmd_size_; }

uint32_t DispatchMemMap::cmddat_q_base() const { return cmddat_q_base_; }

uint32_t DispatchMemMap::cmddat_q_size() const { return settings.prefetch_cmddat_q_size_; }

uint32_t DispatchMemMap::scratch_db_base() const { return scratch_db_base_; }

uint32_t DispatchMemMap::scratch_db_size() const { return settings.prefetch_scratch_db_size_; }

uint32_t DispatchMemMap::ringbuffer_size() const { return settings.prefetch_ringbuffer_size_; }

uint32_t DispatchMemMap::dispatch_buffer_block_size_pages() const { return dispatch_buffer_block_size_pages_; }

uint32_t DispatchMemMap::dispatch_buffer_base() const { return dispatch_buffer_base_; }

uint32_t DispatchMemMap::dispatch_buffer_pages() const { return settings.dispatch_pages_; }

uint32_t DispatchMemMap::prefetch_d_buffer_size() const { return settings.prefetch_d_buffer_size_; }

uint32_t DispatchMemMap::prefetch_d_buffer_pages() const { return settings.prefetch_d_pages_; }

uint32_t DispatchMemMap::dispatch_s_buffer_size() const { return settings.dispatch_s_buffer_size_; }

uint32_t DispatchMemMap::dispatch_s_buffer_pages() const {
    return settings.dispatch_s_buffer_size_ / (1 << tt::tt_metal::DispatchSettings::DISPATCH_S_BUFFER_LOG_PAGE_SIZE);
}

uint32_t DispatchMemMap::get_device_command_queue_addr(const CommandQueueDeviceAddrType& device_addr_type) const {
    uint32_t index = ttsl::as_underlying_type<CommandQueueDeviceAddrType>(device_addr_type);
    TT_ASSERT(index < this->device_cq_addrs_.size());
    return device_cq_addrs_[index];
}

uint32_t DispatchMemMap::get_host_command_queue_addr(const CommandQueueHostAddrType& host_addr) const {
    return ttsl::as_underlying_type<CommandQueueHostAddrType>(host_addr) * host_alignment_;
}

uint32_t DispatchMemMap::get_sync_offset(uint32_t index) const {
    TT_ASSERT(index < tt::tt_metal::DispatchSettings::DISPATCH_MESSAGE_ENTRIES);
    return index * l1_alignment_;
}

uint32_t DispatchMemMap::get_dispatch_message_addr_start() const {
    // Address of the first dispatch message entry. Remaining entries are each offset by
    // noc_stream_reg_space_size bytes.
    return noc_overlay_start_addr_ + (noc_stream_reg_space_size_ * get_dispatch_stream_index(0)) +
           (noc_stream_remote_dest_buf_space_available_update_reg_index_ * sizeof(uint32_t));
}

uint32_t DispatchMemMap::get_dispatch_stream_index(uint32_t index) const { return dispatch_stream_base_ + index; }

uint8_t DispatchMemMap::get_dispatch_message_update_offset(uint32_t index) const {
    TT_ASSERT(index < tt::tt_metal::DispatchSettings::DISPATCH_MESSAGES_MAX_OFFSET);
    return index;
}

uint32_t DispatchMemMap::get_prefetcher_l1_size() const { return l1_size_; }

}  // namespace tt::tt_metal
