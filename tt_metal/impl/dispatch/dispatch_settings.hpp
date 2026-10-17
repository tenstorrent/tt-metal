// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <umd/device/types/core_coordinates.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "llrt/hal.hpp"
#include <hostdevcommon/common_values.hpp>

namespace tt {
class Cluster;
}

namespace tt::tt_metal {
class Hal;

//
// Dispatch Kernel Settings
//
class DispatchSettings {
public:
    explicit DispatchSettings(
        uint32_t num_hw_cqs,
        uint32_t num_cqs_per_core,
        const CoreType& core_type,
        bool is_galaxy_cluster,
        bool are_cqs_dram_backed,
        uint32_t l1_alignment,
        uint32_t prefetch_q_entry_size_bytes);

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

    // Sets pointer values based on L1 alignment
    DispatchSettings& with_alignment(uint32_t l1_alignment);

    // Returns a list of errors
    std::vector<std::string> get_errors() const;

    // Throws if any settings are not set properly and not empty. Does not confirm if the settings will work.
    DispatchSettings& build();

    //
    // Non Configurable Settings
    //

    // Prefetch Queue pointer type
    // Same as the one in cq_prefetch.cpp
    using prefetch_q_ptr_type = uint32_t;

    static constexpr uint32_t MAX_NUM_HW_CQS = ::MAX_NUM_HW_CQS;

    static constexpr uint32_t DISPATCH_MESSAGE_ENTRIES = ::DISPATCH_MAX_MESSAGE_ENTRIES;

    // Worker completion counters per CQ; each sub-device's programs report done into one of them.
    static constexpr uint32_t DISPATCH_COMPLETION_COUNTERS = ::DISPATCH_MAX_COMPLETION_COUNTERS;

    // FDS group (Quasar) that carries the gos of, and the dones into, a completion counter. Group zero is the idle
    // value on the wire.
    static constexpr uint32_t fds_group_for_completion_counter(uint32_t counter_index) { return counter_index + 1; }

    // Most completion slots a sub-device can own: one per launch message ring entry, since no more programs than that
    // can be in flight on a core.
    static constexpr uint32_t MAX_COMPLETION_SLOTS_PER_SUB_DEVICE = 8;

    // Completion slots (K) each sub-device owns when num_sub_devices share a CQ's completion counters. Sub-device i
    // owns counters [i * K, (i + 1) * K); counter i * K is its base slot.
    static constexpr uint32_t num_completion_slots(uint32_t num_sub_devices) {
        return std::min(MAX_COMPLETION_SLOTS_PER_SUB_DEVICE, DISPATCH_COMPLETION_COUNTERS / num_sub_devices);
    }

    // Base completion counter of a sub-device, within a CQ's counter block, when each sub-device owns
    // num_completion_slots counters. Traced programs report done into it; enqueued programs rotate through the
    // sub-device's counters (ExpectedWorkerCompletions).
    static constexpr uint32_t completion_counter_index(uint32_t sub_device_index, uint32_t num_completion_slots) {
        return sub_device_index * num_completion_slots;
    }

    // correctness asserted in .cpp
    static constexpr uint32_t DISPATCH_MESSAGES_MAX_OFFSET = 255;

    static constexpr uint32_t DISPATCH_BUFFER_LOG_PAGE_SIZE = 12;

    static constexpr uint32_t DISPATCH_BUFFER_SIZE_BLOCKS = 4;

    static constexpr uint32_t DISPATCH_GO_SIGNAL_NOC_DATA_ENTRIES = 64;

    // dispatch_s CB page size is 256 bytes. This should currently be enough to accommodate all commands that
    // are sent to it. Change as needed.
    static constexpr uint32_t DISPATCH_S_BUFFER_LOG_PAGE_SIZE = 8;

    static constexpr uint32_t GO_SIGNAL_BITS_PER_TXN_TYPE = 4;

    static constexpr uint32_t PREFETCH_Q_LOG_MINSIZE = 4;

    static constexpr uint32_t LOG_TRANSFER_PAGE_SIZE = 12;

    static constexpr uint32_t TRANSFER_PAGE_SIZE = 1 << LOG_TRANSFER_PAGE_SIZE;

    static constexpr uint32_t PREFETCH_D_BUFFER_LOG_PAGE_SIZE = 12;

    static constexpr uint32_t EVENT_PADDED_SIZE = 16;

    // When page size of buffer to write/read exceeds the max prefetch command size, the PCIe-aligned page size is
    // broken down into equal sized partial pages. The base partial page size is incremented until it
    // is PCIE-aligned. If the resulting partial page size doesn't evenly divide the full page size, the last partial
    // page size is padded appropriately.
    static constexpr uint32_t BASE_PARTIAL_PAGE_SIZE_DISPATCH = 4096;

    static constexpr uint32_t MAX_HUGEPAGE_SIZE = 1 << 30;                                         // 1GB
    static constexpr uint32_t MAX_DEV_CHANNEL_SIZE = 1 << 28;                                      // 256 MB;
    static constexpr uint32_t DEVICES_PER_UMD_CHANNEL = MAX_HUGEPAGE_SIZE / MAX_DEV_CHANNEL_SIZE;  // 256 MB;

    static constexpr uint32_t HUGEPAGE_D2H_FALLBACK_RESERVE_BYTES = 2 * 1024 * 1024;  // 2 MiB

    // Number of entries in the fabric header ring buffer
    static constexpr uint32_t FABRIC_HEADER_RB_ENTRIES = 1;

    //
    // Configurable Settings
    //

    // common
    uint32_t num_hw_cqs_{0};

    // Rd/Wr/Msg pointer sizes
    uint32_t prefetch_q_rd_ptr_size_{0};      // configured with alignment
    uint32_t dispatch_s_sync_sem_{};          // configured with alignment
    uint32_t other_ptrs_size{};               // configured with alignment

    // cq_prefetch
    uint32_t prefetch_q_entry_size_bytes_{};  // 2 for WH ETH, 4 otherwise
    uint32_t prefetch_q_entries_{0};
    uint32_t prefetch_q_size_{};
    uint32_t prefetch_max_cmd_size_{};
    uint32_t prefetch_cmddat_q_size_{};
    uint32_t prefetch_scratch_db_size_{};
    uint32_t prefetch_ringbuffer_size_{};
    uint32_t prefetch_d_buffer_size_{};
    uint32_t prefetch_d_pages_{};  // prefetch_d_buffer_size_ / PREFETCH_D_BUFFER_LOG_PAGE_SIZE

    // cq_dispatch
    uint32_t dispatch_size_{};   // total buffer size
    uint32_t dispatch_pages_{};  // total buffer size / page size
    uint32_t dispatch_s_buffer_size_{};
    uint32_t dispatch_s_buffer_pages_{};  // dispatch_s_buffer_size_ / DISPATCH_S_BUFFER_LOG_PAGE_SIZE

    CoreType core_type_{0};  // Which core this settings is for

private:
    void init_worker_defaults(
        uint32_t num_hw_cqs, bool is_galaxy_cluster, bool are_cqs_dram_backed, uint32_t l1_alignment);
    void init_dispatch_defaults(
        uint32_t num_hw_cqs, uint32_t num_cqs_per_core, bool are_cqs_dram_backed, uint32_t l1_alignment);
    void init_eth_defaults(uint32_t num_hw_cqs, uint32_t l1_alignment);
    uint32_t get_prefetch_q_entries(
        CoreType core_type, uint32_t num_hw_cqs, bool is_galaxy_cluster, bool are_cqs_dram_backed);
};

// Convenience type alias for arrays of `DISPATCH_MESSAGE_ENTRIES` size.
template <typename T>
using DispatchArray = std::array<T, DispatchSettings::DISPATCH_MESSAGE_ENTRIES>;

// Per-CQ arrays indexed by completion counter (DispatchSettings::completion_counter_index), not by sub-device.
template <typename T>
using CompletionCounterArray = std::array<T, DispatchSettings::DISPATCH_COMPLETION_COUNTERS>;

}  // namespace tt::tt_metal
