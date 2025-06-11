// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include <umd/device/types/cluster_descriptor_types.h>

namespace tt::tt_metal {

enum class CommandQueueDeviceAddrType : uint8_t {
    PREFETCH_Q_RD = 0,
    // Used to notify host of how far device has gotten, doesn't need L1 alignment because it's only written locally by
    // prefetch kernel.
    PREFETCH_Q_PCIE_RD = 1,
    COMPLETION_Q_WR = 2,
    COMPLETION_Q_RD = 3,
    // Max of 2 CQs. COMPLETION_Q*_LAST_EVENT_PTR track the last completed event in the respective CQs
    COMPLETION_Q0_LAST_EVENT = 4,
    COMPLETION_Q1_LAST_EVENT = 5,
    DISPATCH_S_SYNC_SEM = 6,
    FABRIC_HEADER_RB = 7,
    FABRIC_SYNC_STATUS = 8,
    UNRESERVED = 9,
};

// likely only used in impl
enum class CommandQueueHostAddrType : uint8_t {
    ISSUE_Q_RD = 0,
    ISSUE_Q_WR = 1,
    COMPLETION_Q_WR = 2,
    COMPLETION_Q_RD = 3,
    UNRESERVED = 4
};

// used in system_memory_manager.cpp and command_queue_interface.cpp

/// @brief Get offset of the command queue relative to its channel
/// @param cq_id uint8_t ID the command queue
/// @param cq_size uint32_t size of the command queue
/// @return uint32_t relative offset
uint32_t get_relative_cq_offset(uint8_t cq_id, uint32_t cq_size);

// used in system_memory_manager and device
uint16_t get_umd_channel(uint16_t channel);

// only used in impl

/// @brief Get absolute offset of the command queue
/// @param channel uint16_t channel ID (hugepage)
/// @param cq_id uint8_t ID the command queue
/// @param cq_size uint32_t size of the command queue
/// @return uint32_t absolute offset
uint32_t get_absolute_cq_offset(uint16_t channel, uint8_t cq_id, uint32_t cq_size);

// mostly used in debug_tools
template <bool addr_16B>
uint32_t get_cq_issue_rd_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size);

template <bool addr_16B>
uint32_t get_cq_issue_wr_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size);

// has usage in system_memory_manager.cpp
template <bool addr_16B>
uint32_t get_cq_completion_wr_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size);

template <bool addr_16B>
uint32_t get_cq_completion_rd_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size);

}  // namespace tt::tt_metal
