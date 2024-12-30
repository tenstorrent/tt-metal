// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <unordered_map>
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"
#include "umd/device/tt_core_coordinates.h"
#include "tt_metal/impl/dispatch/dispatch_constants.hpp"

namespace tt::tt_metal::dispatch {

// Settings for a dispatch core
struct DispatchSettings {
    // common
    uint32_t num_hw_cqs_{0};

    // Rd/Wr/Msg pointer sizes
    uint32_t prefetch_q_rd_ptr_size_{0};    // configured with alignment
    uint32_t prefetch_q_pcie_rd_ptr_size_;  // configured with alignment
    uint32_t dispatch_s_sync_sem_;          // configured with alignment
    uint32_t dispatch_message_;             // configured with alignment
    uint32_t other_ptrs_size;               // configured with alignment

    // cq_prefetch
    uint32_t prefetch_q_entries_{0};
    uint32_t prefetch_q_size_;
    uint32_t prefetch_max_cmd_size_;
    uint32_t prefetch_cmddat_q_size_;
    uint32_t prefetch_scratch_db_size_;
    uint32_t prefetch_d_buffer_size_;
    uint32_t prefetch_d_pages_;  // prefetch_d_buffer_size_ / PREFETCH_D_BUFFER_LOG_PAGE_SIZE
    uint32_t prefetch_d_blocks_;

    // cq_dispatch
    uint32_t dispatch_pages_per_block_;  // number of pages per block
    uint32_t dispatch_size_;             // total buffer size
    uint32_t dispatch_pages_;            // total buffer size / page size
    uint32_t dispatch_s_buffer_size_;
    uint32_t dispatch_s_buffer_pages_;  // dispatch_s_buffer_size_ / DISPATCH_S_BUFFER_LOG_PAGE_SIZE

    // packet_mux, packet_demux, vc_eth_tunneler, vc_packet_router
    uint32_t tunneling_buffer_size_;
    uint32_t tunneling_buffer_pages_;  // tunneling_buffer_size_ / PREFETCH_D_BUFFER_LOG_PAGE_SIZE

    CoreType core_type_;  // Which core this settings is for

    bool operator==(const DispatchSettings& other) const {
        return num_hw_cqs_ == other.num_hw_cqs_ && prefetch_q_rd_ptr_size_ == other.prefetch_q_rd_ptr_size_ &&
               prefetch_q_pcie_rd_ptr_size_ == other.prefetch_q_pcie_rd_ptr_size_ &&
               dispatch_s_sync_sem_ == other.dispatch_s_sync_sem_ && dispatch_message_ == other.dispatch_message_ &&
               other_ptrs_size == other.other_ptrs_size && prefetch_q_entries_ == other.prefetch_q_entries_ &&
               prefetch_q_size_ == other.prefetch_q_size_ && prefetch_max_cmd_size_ == other.prefetch_max_cmd_size_ &&
               prefetch_cmddat_q_size_ == other.prefetch_cmddat_q_size_ &&
               prefetch_scratch_db_size_ == other.prefetch_scratch_db_size_ &&
               prefetch_d_buffer_size_ == other.prefetch_d_buffer_size_ &&
               prefetch_d_pages_ == other.prefetch_d_pages_ && prefetch_d_blocks_ == other.prefetch_d_blocks_ &&
               dispatch_pages_per_block_ == other.dispatch_pages_per_block_ && dispatch_size_ == other.dispatch_size_ &&
               dispatch_pages_ == other.dispatch_pages_ && dispatch_s_buffer_size_ == other.dispatch_s_buffer_size_ &&
               dispatch_s_buffer_pages_ == other.dispatch_s_buffer_pages_ &&
               tunneling_buffer_size_ == other.tunneling_buffer_size_ &&
               tunneling_buffer_pages_ == other.tunneling_buffer_pages_ && core_type_ == other.core_type_;
    }

    bool operator!=(const DispatchSettings& other) const {
        return !(*this == other);
    }

    // Default settings for WORKER cores
    static DispatchSettings worker_defaults(const tt::Cluster& cluster, const uint32_t num_hw_cqs);

    // Default settings for ETH cores
    static DispatchSettings eth_defaults(const tt::Cluster& cluster, const uint32_t num_hw_cqs);

    // Returns default dispatch settings for a core
    static DispatchSettings defaults(const CoreType& core_type, const tt::Cluster& cluster, const uint32_t num_hw_cqs);

    DispatchSettings& core_type(const CoreType& val) {
        this->core_type_ = val;
        return *this;
    }

    // Trivial setter for num_hw_cqs
    DispatchSettings& num_hw_cqs(uint32_t val) {
        this->num_hw_cqs_ = val;
        return *this;
    }

    // Trivial setter for prefetch_max_cmd_size
    DispatchSettings& prefetch_max_cmd_size(uint32_t val) {
        this->prefetch_max_cmd_size_ = val;
        return *this;
    }

    // Trivial setter for prefetch_cmddat_q_size
    DispatchSettings& prefetch_cmddat_q_size(uint32_t val) {
        this->prefetch_cmddat_q_size_ = val;
        return *this;
    }

    // Trivial setter for prefetch_scratch_db_size
    DispatchSettings& prefetch_scratch_db_size(uint32_t val) {
        this->prefetch_scratch_db_size_ = val;
        return *this;
    }

    // Setter for prefetch_q_entries and update prefetch_q_size
    DispatchSettings& prefetch_q_entries(uint32_t val) {
        this->prefetch_q_entries_ = val;
        this->prefetch_q_size_ = val * sizeof(DispatchConstants::prefetch_q_entry_type);
        return *this;
    }

    // Setter for prefetch_d_buffer_size and update prefetch_d_pages
    DispatchSettings& prefetch_d_buffer_size(uint32_t val) {
        this->prefetch_d_buffer_size_ = val;
        this->prefetch_d_pages_ =
            this->prefetch_d_buffer_size_ / (1 << DispatchConstants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE);

        return *this;
    }

    // Trivial setter for dispatch_pages_per_block
    DispatchSettings& dispatch_pages_per_block(uint32_t val) {
        this->dispatch_pages_per_block_ = val;
        return *this;
    }

    // Trivial setter for prefetch_d_blocks
    DispatchSettings& prefetch_d_blocks(uint32_t val) {
        this->prefetch_d_blocks_ = val;
        return *this;
    }

    // Setter for dispatch_block_size and update dispatch_pages
    DispatchSettings& dispatch_size(uint32_t val) {
        this->dispatch_size_ = val;
        this->dispatch_pages_ = this->dispatch_size_ / (1 << DispatchConstants::DISPATCH_BUFFER_LOG_PAGE_SIZE);
        return *this;
    }

    // Setter for dispatch_s_buffer_size and update dispatch_s_buffer_pages
    DispatchSettings& dispatch_s_buffer_size(uint32_t val) {
        this->dispatch_s_buffer_size_ = val;
        this->dispatch_s_buffer_pages_ =
            this->dispatch_s_buffer_size_ / (1 << DispatchConstants::DISPATCH_S_BUFFER_LOG_PAGE_SIZE);
        return *this;
    }

    // Setter for tunneling_buffer_size and update tunneling_buffer_pages
    DispatchSettings& tunneling_buffer_size(uint32_t val) {
        this->tunneling_buffer_size_ = val;
        this->tunneling_buffer_pages_ =
            this->tunneling_buffer_size_ / (1 << DispatchConstants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE); // match legacy dispatch_constants
        return *this;
    }

    // Sets pointer values based on L1 alignment
    DispatchSettings& with_alignment(uint32_t l1_alignment) {
        this->prefetch_q_rd_ptr_size_ = sizeof(DispatchConstants::prefetch_q_ptr_type);
        this->prefetch_q_pcie_rd_ptr_size_ = l1_alignment - sizeof(DispatchConstants::prefetch_q_ptr_type);
        this->dispatch_s_sync_sem_ = DispatchConstants::DISPATCH_MESSAGE_ENTRIES * l1_alignment;
        this->dispatch_message_ = DispatchConstants::DISPATCH_MESSAGE_ENTRIES * l1_alignment;
        this->other_ptrs_size = l1_alignment;

        return *this;
    }

    // Returns a list of errors
    std::vector<std::string> get_errors() const;

    // Throws if any settings are not set properly and not empty. Does not confirm if the settings will work.
    DispatchSettings& build();
};

struct DispatchSettingsContainerKey {
    CoreType core_type;
    uint32_t num_hw_cqs;

    bool operator==(const DispatchSettingsContainerKey& other) const {
        return core_type == other.core_type && num_hw_cqs == other.num_hw_cqs;
    }
};

using DispatchSettingsContainer = std::unordered_map<DispatchSettingsContainerKey, DispatchSettings>;

}  // namespace tt::tt_metal::dispatch

namespace std {
template <>
struct hash<tt::tt_metal::dispatch::DispatchSettingsContainerKey> {
    size_t operator()(const tt::tt_metal::dispatch::DispatchSettingsContainerKey& k) const {
        const auto h1 = std::hash<uint32_t>{}(static_cast<int>(k.core_type));
        const auto h2 = std::hash<uint32_t>{}(k.num_hw_cqs);
        return h1 ^ (h2 << 1);
    }
};
}  // namespace std
