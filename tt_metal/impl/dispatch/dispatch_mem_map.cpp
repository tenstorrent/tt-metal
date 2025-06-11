// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <magic_enum/magic_enum.hpp>
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include <tt-metalium/tt_align.hpp>

#include "dispatch_mem_map.hpp"
#include "assert.hpp"
#include "command_queue_common.hpp"
#include "control_plane.hpp"
#include "dispatch_settings.hpp"
#include "fabric/fabric_context.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "utils.hpp"

namespace tt::tt_metal {

DispatchMemMap::DispatchMemMap(const CoreType& core_type, const uint32_t num_hw_cqs) {
    this->reset(core_type, num_hw_cqs);
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

uint32_t DispatchMemMap::mux_buffer_size(uint8_t num_hw_cqs) const {
    return settings.tunneling_buffer_size_ / num_hw_cqs;
}

uint32_t DispatchMemMap::mux_buffer_pages(uint8_t num_hw_cqs) const {
    return settings.tunneling_buffer_pages_ / num_hw_cqs;
}

uint32_t DispatchMemMap::dispatch_s_buffer_size() const { return settings.dispatch_s_buffer_size_; }

uint32_t DispatchMemMap::dispatch_s_buffer_pages() const {
    return settings.dispatch_s_buffer_size_ / (1 << tt::tt_metal::DispatchSettings::DISPATCH_S_BUFFER_LOG_PAGE_SIZE);
}

uint32_t DispatchMemMap::get_device_command_queue_addr(const CommandQueueDeviceAddrType& device_addr_type) const {
    uint32_t index = tt::utils::underlying_type<CommandQueueDeviceAddrType>(device_addr_type);
    TT_ASSERT(index < this->device_cq_addrs_.size());
    return device_cq_addrs_[index];
}

uint32_t DispatchMemMap::get_host_command_queue_addr(const CommandQueueHostAddrType& host_addr) const {
    return tt::utils::underlying_type<CommandQueueHostAddrType>(host_addr) *
           tt::tt_metal::MetalContext::instance().hal().get_alignment(tt::tt_metal::HalMemType::HOST);
}

uint32_t DispatchMemMap::get_sync_offset(uint32_t index) const {
    TT_ASSERT(index < tt::tt_metal::DispatchSettings::DISPATCH_MESSAGE_ENTRIES);
    uint32_t offset = index * MetalContext::instance().hal().get_alignment(HalMemType::L1);
    return offset;
}

uint32_t DispatchMemMap::get_dispatch_message_addr_start() const {
    // Address of the first dispatch message entry. Remaining entries are each offset by
    // get_noc_stream_reg_space_size() bytes.
    return tt::tt_metal::MetalContext::instance().hal().get_noc_overlay_start_addr() +
           tt::tt_metal::MetalContext::instance().hal().get_noc_stream_reg_space_size() * get_dispatch_stream_index(0) +
           tt::tt_metal::MetalContext::instance()
                   .hal()
                   .get_noc_stream_remote_dest_buf_space_available_update_reg_index() *
               sizeof(uint32_t);
}

uint32_t DispatchMemMap::get_dispatch_stream_index(uint32_t index) const {
    if (last_core_type == CoreType::WORKER) {
        // There are 64 streams. CBs use entries 8-39.
        return 48u + index;
    } else if (last_core_type == CoreType::ETH) {
        // There are 32 streams.
        return 16u + index;
    } else {
        TT_THROW("get_dispatch_starting_stream_index not implemented for core type");
    }
}

uint8_t DispatchMemMap::get_dispatch_message_update_offset(uint32_t index) const {
    TT_ASSERT(index < tt::tt_metal::DispatchSettings::DISPATCH_MESSAGES_MAX_OFFSET);
    return index;
}

// Reset the instance using the settings for the core_type and num_hw_cqs.
void DispatchMemMap::reset(const CoreType& core_type, const uint32_t num_hw_cqs) {
    const auto dispatch_settings = DispatchSettings::get(core_type, num_hw_cqs);
    this->settings = dispatch_settings;
    last_core_type = settings.core_type_;
    hw_cqs = settings.num_hw_cqs_;

    const auto dispatch_buffer_block_size = settings.dispatch_size_;
    const auto [l1_base, l1_size] = get_device_l1_info(settings.core_type_);
    const auto pcie_alignment =
        tt::tt_metal::MetalContext::instance().hal().get_alignment(tt::tt_metal::HalMemType::HOST);
    const auto l1_alignment = tt::tt_metal::MetalContext::instance().hal().get_alignment(tt::tt_metal::HalMemType::L1);

    TT_ASSERT(settings.prefetch_cmddat_q_size_ >= 2 * settings.prefetch_max_cmd_size_);
    TT_ASSERT(settings.prefetch_scratch_db_size_ % 2 == 0);
    TT_ASSERT((dispatch_buffer_block_size & (dispatch_buffer_block_size - 1)) == 0);
    TT_ASSERT(
        DispatchSettings::DISPATCH_MESSAGE_ENTRIES <= DispatchSettings::DISPATCH_MESSAGES_MAX_OFFSET / l1_alignment + 1,
        "Number of dispatch message entries exceeds max representable offset");

    constexpr uint8_t num_dev_cq_addrs = magic_enum::enum_count<CommandQueueDeviceAddrType>();
    std::vector<uint32_t> device_cq_addr_sizes_(num_dev_cq_addrs, 0);
    for (auto dev_addr_idx = 0; dev_addr_idx < num_dev_cq_addrs; dev_addr_idx++) {
        CommandQueueDeviceAddrType dev_addr_type =
            magic_enum::enum_cast<CommandQueueDeviceAddrType>(dev_addr_idx).value();
        if (dev_addr_type == CommandQueueDeviceAddrType::PREFETCH_Q_RD) {
            device_cq_addr_sizes_[dev_addr_idx] = settings.prefetch_q_rd_ptr_size_;
        } else if (dev_addr_type == CommandQueueDeviceAddrType::PREFETCH_Q_PCIE_RD) {
            device_cq_addr_sizes_[dev_addr_idx] = settings.prefetch_q_pcie_rd_ptr_size_;
        } else if (dev_addr_type == CommandQueueDeviceAddrType::DISPATCH_S_SYNC_SEM) {
            device_cq_addr_sizes_[dev_addr_idx] = settings.dispatch_s_sync_sem_;
        } else if (dev_addr_type == CommandQueueDeviceAddrType::FABRIC_HEADER_RB) {
            // At this point fabric context is not initialized yet
            // Hardcode to 64B (more than enough space) for now
            // const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
            // const auto& fabric_context = control_plane.get_fabric_context();
            device_cq_addr_sizes_[dev_addr_idx] = tt::tt_metal::DispatchSettings::FABRIC_HEADER_RB_ENTRIES * 64;
        } else if (dev_addr_type == CommandQueueDeviceAddrType::FABRIC_SYNC_STATUS) {
            device_cq_addr_sizes_[dev_addr_idx] = sizeof(uint32_t);
        } else {
            device_cq_addr_sizes_[dev_addr_idx] = settings.other_ptrs_size;
        }
    }

    device_cq_addrs_.resize(num_dev_cq_addrs);
    device_cq_addrs_[0] = l1_base;
    for (auto dev_addr_idx = 1; dev_addr_idx < num_dev_cq_addrs; dev_addr_idx++) {
        device_cq_addrs_[dev_addr_idx] = device_cq_addrs_[dev_addr_idx - 1] + device_cq_addr_sizes_[dev_addr_idx - 1];
        CommandQueueDeviceAddrType dev_addr_type = magic_enum::enum_value<CommandQueueDeviceAddrType>(dev_addr_idx);
        if (dev_addr_type == CommandQueueDeviceAddrType::UNRESERVED) {
            device_cq_addrs_[dev_addr_idx] = align(device_cq_addrs_[dev_addr_idx], pcie_alignment);
        } else if (
            dev_addr_type == CommandQueueDeviceAddrType::FABRIC_HEADER_RB ||
            dev_addr_type == CommandQueueDeviceAddrType::FABRIC_SYNC_STATUS) {
            device_cq_addrs_[dev_addr_idx] = align(device_cq_addrs_[dev_addr_idx], l1_alignment);
        }
    }

    uint32_t prefetch_dispatch_unreserved_base =
    device_cq_addrs_[tt::utils::underlying_type<CommandQueueDeviceAddrType>(
        CommandQueueDeviceAddrType::UNRESERVED)];
    cmddat_q_base_ = align(prefetch_dispatch_unreserved_base + settings.prefetch_q_size_, pcie_alignment);
    scratch_db_base_ = align(cmddat_q_base_ + settings.prefetch_cmddat_q_size_, pcie_alignment);
    dispatch_buffer_base_ = align(prefetch_dispatch_unreserved_base, 1 << DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE);
    dispatch_buffer_block_size_pages_ = settings.dispatch_pages_ / DispatchSettings::DISPATCH_BUFFER_SIZE_BLOCKS;
    const uint32_t dispatch_cb_end = dispatch_buffer_base_ + settings.dispatch_size_;

    TT_ASSERT(scratch_db_base_ + settings.prefetch_scratch_db_size_ < l1_size);
    TT_FATAL(
        scratch_db_base_ + settings.prefetch_ringbuffer_size_ <= l1_size,
        "Ringbuffer (start: {}, end: {}) extends past L1 end (size: {})",
        scratch_db_base_,
        scratch_db_base_ + settings.prefetch_scratch_db_size_,
        l1_size);

    TT_ASSERT(dispatch_cb_end < l1_size);
}

std::pair<uint32_t, uint32_t> DispatchMemMap::get_device_l1_info(const CoreType& core_type) const {
    uint32_t l1_base;
    uint32_t l1_size;
    if (core_type == CoreType::WORKER) {
        l1_base = MetalContext::instance().hal().get_dev_addr(
            tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::DEFAULT_UNRESERVED);
        l1_size = MetalContext::instance().hal().get_dev_size(
            tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::BASE);
    } else if (core_type == CoreType::ETH) {
        l1_base = MetalContext::instance().hal().get_dev_addr(
            tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
        l1_size = MetalContext::instance().hal().get_dev_size(
            tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::BASE);
    } else {
        TT_THROW("get_base_device_command_queue_addr not implemented for core type");
    }

    return {l1_base, l1_size};
}

}  // namespace tt::tt_metal
