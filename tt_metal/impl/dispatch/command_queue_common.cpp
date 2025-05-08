// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "command_queue_common.hpp"
#include "dispatch_settings.hpp"

#include "impl/context/metal_context.hpp"

enum class CoreType;

namespace tt::tt_metal {

uint32_t get_relative_cq_offset(uint8_t cq_id, uint32_t cq_size) { return cq_id * cq_size; }

uint16_t get_umd_channel(uint16_t channel) { return channel & 0x3; }

uint32_t get_absolute_cq_offset(uint16_t channel, uint8_t cq_id, uint32_t cq_size) {
    return (DispatchSettings::MAX_HUGEPAGE_SIZE * get_umd_channel(channel)) +
           ((channel >> 2) * DispatchSettings::MAX_DEV_CHANNEL_SIZE) + get_relative_cq_offset(cq_id, cq_size);
}

template <bool addr_16B>
uint32_t get_cq_issue_rd_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(chip_id);
    uint32_t channel_offset = (channel >> 2) * tt::tt_metal::DispatchSettings::MAX_DEV_CHANNEL_SIZE;
    uint32_t issue_q_rd_ptr =
        MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::ISSUE_Q_RD);
    tt::tt_metal::MetalContext::instance().get_cluster().read_sysmem(
        &recv,
        sizeof(uint32_t),
        issue_q_rd_ptr + channel_offset + get_relative_cq_offset(cq_id, cq_size),
        mmio_device_id,
        channel);
    if constexpr (!addr_16B) {
        return recv << 4;
    }
    return recv;
}

template uint32_t get_cq_issue_rd_ptr<true>(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size);
template uint32_t get_cq_issue_rd_ptr<false>(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size);

template <bool addr_16B>
uint32_t get_cq_issue_wr_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(chip_id);
    uint32_t issue_q_wr_ptr =
        MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::ISSUE_Q_WR);
    tt::tt_metal::MetalContext::instance().get_cluster().read_sysmem(
        &recv, sizeof(uint32_t), issue_q_wr_ptr + get_relative_cq_offset(cq_id, cq_size), mmio_device_id, channel);
    if constexpr (!addr_16B) {
        return recv << 4;
    }
    return recv;
}

template uint32_t get_cq_issue_wr_ptr<true>(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size);
template uint32_t get_cq_issue_wr_ptr<false>(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size);

template <bool addr_16B>
uint32_t get_cq_completion_wr_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(chip_id);
    uint32_t channel_offset = (channel >> 2) * tt::tt_metal::DispatchSettings::MAX_DEV_CHANNEL_SIZE;
    uint32_t completion_q_wr_ptr = MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(
        CommandQueueHostAddrType::COMPLETION_Q_WR);
    tt::tt_metal::MetalContext::instance().get_cluster().read_sysmem(
        &recv,
        sizeof(uint32_t),
        completion_q_wr_ptr + channel_offset + get_relative_cq_offset(cq_id, cq_size),
        mmio_device_id,
        channel);
    if constexpr (!addr_16B) {
        return recv << 4;
    }
    return recv;
}

template uint32_t get_cq_completion_wr_ptr<true>(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size);
template uint32_t get_cq_completion_wr_ptr<false>(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size);

template <bool addr_16B>
inline uint32_t get_cq_completion_rd_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(chip_id);
    uint32_t completion_q_rd_ptr = MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(
        CommandQueueHostAddrType::COMPLETION_Q_RD);
    tt::tt_metal::MetalContext::instance().get_cluster().read_sysmem(
        &recv, sizeof(uint32_t), completion_q_rd_ptr + get_relative_cq_offset(cq_id, cq_size), mmio_device_id, channel);
    if constexpr (!addr_16B) {
        return recv << 4;
    }
    return recv;
}

template uint32_t get_cq_completion_rd_ptr<true>(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size);
template uint32_t get_cq_completion_rd_ptr<false>(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size);

}  // namespace tt::tt_metal
