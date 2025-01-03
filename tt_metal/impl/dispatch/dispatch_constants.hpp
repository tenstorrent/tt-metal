// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <climits>
#include <cstdint>
#include <limits>
#include "tt_metal/impl/dispatch/cq_commands.hpp"
#include "tt_metal/hw/inc/dev_msgs.h"

namespace tt::tt_metal::dispatch {

// FD Constants
struct DispatchConstants {
    // Prefetch Queue entry type // Same as the one in cq_prefetch.cpp
    using prefetch_q_entry_type = uint16_t;

    // Prefetch Queue pointer type // Same as the one in cq_prefetch.cpp
    using prefetch_q_ptr_type = uint32_t;

    static constexpr uint32_t MAX_NUM_HW_CQS = 2;

    static constexpr uint32_t DISPATCH_MESSAGE_ENTRIES = 16;

    static constexpr uint32_t DISPATCH_MESSAGES_MAX_OFFSET =
        std::numeric_limits<decltype(go_msg_t::dispatch_message_offset)>::max();

    static constexpr uint32_t DISPATCH_BUFFER_LOG_PAGE_SIZE = 12;

    static constexpr uint32_t DISPATCH_GO_SIGNAL_NOC_DATA_ENTRIES = 64;

    // dispatch_s CB page size is 128 bytes. This should currently be enough to accomodate all commands that
    // are sent to it. Change as needed, once this endpoint is required to handle more than go signal mcasts.
    static constexpr uint32_t DISPATCH_S_BUFFER_LOG_PAGE_SIZE = 7;

    static constexpr uint32_t GO_SIGNAL_BITS_PER_TXN_TYPE = 4;

    static constexpr uint32_t PREFETCH_Q_LOG_MINSIZE = 4;

    static constexpr uint32_t LOG_TRANSFER_PAGE_SIZE = 12;

    static constexpr uint32_t TRANSFER_PAGE_SIZE = 1 << LOG_TRANSFER_PAGE_SIZE;

    static constexpr uint32_t PREFETCH_D_BUFFER_LOG_PAGE_SIZE = 12;

    static constexpr uint32_t EVENT_PADDED_SIZE = 16;

    // When page size of buffer to write/read exceeds MAX_PREFETCH_COMMAND_SIZE, the PCIe aligned page size is broken
    // down into equal sized partial pages BASE_PARTIAL_PAGE_SIZE denotes the initial partial page size to use, it is
    // incremented by PCIe alignment until page size can be evenly split
    static constexpr uint32_t BASE_PARTIAL_PAGE_SIZE = 4096;

    static_assert(
        DISPATCH_MESSAGE_ENTRIES <=
        sizeof(decltype(CQDispatchCmd::notify_dispatch_s_go_signal.index_bitmask)) * CHAR_BIT);

    static constexpr uint32_t MAX_HUGEPAGE_SIZE = 1 << 30;                                        // 1GB
    static constexpr uint32_t MAX_DEV_CHANNEL_SIZE = 1 << 28;                                     // 256 MB;
    static constexpr uint32_t DEVICES_PER_UMD_CHANNEL = MAX_HUGEPAGE_SIZE / MAX_DEV_CHANNEL_SIZE; // 256 MB;
};

}  // namespace tt::tt_metal::dispatch
