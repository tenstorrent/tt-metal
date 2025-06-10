// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <magic_enum/magic_enum.hpp>
#include <umd/device/tt_core_coordinates.h>
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

enum class CoreType;

namespace tt {
class Cluster;
}

namespace tt::tt_metal {

//
// Dispatch Kernel Settings
//
class DispatchSettings {
public:
    // Returns the default settings for WORKER cores
    static DispatchSettings worker_defaults(const tt::Cluster& cluster, const uint32_t num_hw_cqs);

    // Returns the default settings for ETH cores
    static DispatchSettings eth_defaults(const tt::Cluster& cluster, const uint32_t num_hw_cqs);

    // Returns the default settings
    static DispatchSettings defaults(const CoreType& core_type, const tt::Cluster& cluster, const uint32_t num_hw_cqs);

    // Returns the settings for a core type and number hw cqs. The values can be modified, but customization must occur
    // before command queue kernels are created.
    static DispatchSettings& get(const CoreType& core_type, const uint32_t num_hw_cqs);

    // Reset the settings
    static void initialize(const tt::Cluster& cluster);

    // Reset the settings for a core type and number hw cqs to the provided settings
    static void initialize(const DispatchSettings& other);

    bool operator==(const DispatchSettings& other) const;

    bool operator!=(const DispatchSettings& other) const;

    DispatchSettings& core_type(const CoreType& val);

    // Trivial setter for num_hw_cqs
    DispatchSettings& num_hw_cqs(uint32_t val);

    // Trivial setter for prefetch_max_cmd_size
    DispatchSettings& prefetch_max_cmd_size(uint32_t val);

    // Trivial setter for prefetch_cmddat_q_size
    DispatchSettings& prefetch_cmddat_q_size(uint32_t val);

    // Trivial setter for prefetch_scratch_db_size
    DispatchSettings& prefetch_scratch_db_size(uint32_t val);

    // Trivial setter for prefetch_ringbuffer_size
    DispatchSettings& prefetch_ringbuffer_size(uint32_t val);

    // Setter for prefetch_q_entries and update prefetch_q_size
    DispatchSettings& prefetch_q_entries(uint32_t val);

    // Setter for prefetch_d_buffer_size and update prefetch_d_pages
    DispatchSettings& prefetch_d_buffer_size(uint32_t val);

    // Setter for dispatch_block_size and update dispatch_pages
    DispatchSettings& dispatch_size(uint32_t val);

    // Setter for dispatch_s_buffer_size and update dispatch_s_buffer_pages
    DispatchSettings& dispatch_s_buffer_size(uint32_t val);

    // Setter for tunneling_buffer_size and update tunneling_buffer_pages
    DispatchSettings& tunneling_buffer_size(uint32_t val);

    // Sets pointer values based on L1 alignment
    DispatchSettings& with_alignment(uint32_t l1_alignment);

    // Returns a list of errors
    std::vector<std::string> get_errors() const;

    // Throws if any settings are not set properly and not empty. Does not confirm if the settings will work.
    DispatchSettings& build();

    //
    // Non Configurable Settings
    //

    // Prefetch Queue entry type
    // Same as the one in cq_prefetch.cpp
    using prefetch_q_entry_type = uint16_t;

    // Prefetch Queue pointer type
    // Same as the one in cq_prefetch.cpp
    using prefetch_q_ptr_type = uint32_t;

    static constexpr uint32_t MAX_NUM_HW_CQS = 2;

    static constexpr uint32_t DISPATCH_MESSAGE_ENTRIES = 8;

    // correctness asserted in .cpp
    static constexpr uint32_t DISPATCH_MESSAGES_MAX_OFFSET = 255;

    static constexpr uint32_t DISPATCH_BUFFER_LOG_PAGE_SIZE = 12;

    static constexpr uint32_t DISPATCH_BUFFER_SIZE_BLOCKS = 4;

    static constexpr uint32_t DISPATCH_GO_SIGNAL_NOC_DATA_ENTRIES = 64;

    // dispatch_s CB page size is 256 bytes. This should currently be enough to accomodate all commands that
    // are sent to it. Change as needed.
    static constexpr uint32_t DISPATCH_S_BUFFER_LOG_PAGE_SIZE = 8;

    static constexpr uint32_t GO_SIGNAL_BITS_PER_TXN_TYPE = 4;

    static constexpr uint32_t PREFETCH_Q_LOG_MINSIZE = 4;

    static constexpr uint32_t LOG_TRANSFER_PAGE_SIZE = 12;

    static constexpr uint32_t TRANSFER_PAGE_SIZE = 1 << LOG_TRANSFER_PAGE_SIZE;

    static constexpr uint32_t PREFETCH_D_BUFFER_LOG_PAGE_SIZE = 12;

    static constexpr uint32_t PREFETCH_D_BUFFER_BLOCKS = 4;

    static constexpr uint32_t EVENT_PADDED_SIZE = 16;

    // When page size of buffer to write/read exceeds the max prefetch command size, the PCIe-aligned page size is
    // broken down into equal sized partial pages. The base partial page size is incremented until it
    // is PCIE-aligned. If the resulting partial page size doesn't evenly divide the full page size, the last partial
    // page size is padded appropriately.
    static constexpr uint32_t BASE_PARTIAL_PAGE_SIZE_DISPATCH = 4096;

    static constexpr uint32_t MAX_HUGEPAGE_SIZE = 1 << 30;                                         // 1GB
    static constexpr uint32_t MAX_DEV_CHANNEL_SIZE = 1 << 28;                                      // 256 MB;
    static constexpr uint32_t DEVICES_PER_UMD_CHANNEL = MAX_HUGEPAGE_SIZE / MAX_DEV_CHANNEL_SIZE;  // 256 MB;

    // Number of entries in the fabric header ring buffer
    static constexpr uint32_t FABRIC_HEADER_RB_ENTRIES = 1;

    //
    // Configurable Settings
    //

    // common
    uint32_t num_hw_cqs_{0};

    // Rd/Wr/Msg pointer sizes
    uint32_t prefetch_q_rd_ptr_size_{0};    // configured with alignment
    uint32_t prefetch_q_pcie_rd_ptr_size_;  // configured with alignment
    uint32_t dispatch_s_sync_sem_;          // configured with alignment
    uint32_t other_ptrs_size;               // configured with alignment

    // cq_prefetch
    uint32_t prefetch_q_entries_{0};
    uint32_t prefetch_q_size_;
    uint32_t prefetch_max_cmd_size_;
    uint32_t prefetch_cmddat_q_size_;
    uint32_t prefetch_scratch_db_size_;
    uint32_t prefetch_ringbuffer_size_;
    uint32_t prefetch_d_buffer_size_;
    uint32_t prefetch_d_pages_;  // prefetch_d_buffer_size_ / PREFETCH_D_BUFFER_LOG_PAGE_SIZE

    // cq_dispatch
    uint32_t dispatch_size_;   // total buffer size
    uint32_t dispatch_pages_;  // total buffer size / page size
    uint32_t dispatch_s_buffer_size_;
    uint32_t dispatch_s_buffer_pages_;  // dispatch_s_buffer_size_ / DISPATCH_S_BUFFER_LOG_PAGE_SIZE

    // packet_mux, packet_demux, vc_eth_tunneler, vc_packet_router
    uint32_t tunneling_buffer_size_;
    uint32_t tunneling_buffer_pages_;  // tunneling_buffer_size_ / PREFETCH_D_BUFFER_LOG_PAGE_SIZE

    CoreType core_type_;  // Which core this settings is for
};

// Convenience type alias for arrays of `DISPATCH_MESSAGE_ENTRIES` size.
template <typename T>
using DispatchArray = std::array<T, DispatchSettings::DISPATCH_MESSAGE_ENTRIES>;

}  // namespace tt::tt_metal
