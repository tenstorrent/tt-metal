// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <limits.h>
#include "dev_msgs.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string_view>
#include <unordered_map>

#include "assert.hpp"
#include "fmt/base.h"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "magic_enum/magic_enum.hpp"
#include "size_literals.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_commands.hpp"
#include <umd/device/tt_core_coordinates.h>

namespace tt::tt_metal {

static_assert(
    DispatchSettings::DISPATCH_MESSAGE_ENTRIES <=
    sizeof(decltype(CQDispatchCmd::notify_dispatch_s_go_signal.index_bitmask)) * CHAR_BIT);

static_assert(
    DispatchSettings::DISPATCH_MESSAGES_MAX_OFFSET ==
        std::numeric_limits<decltype(go_msg_t::dispatch_message_offset)>::max(),
    "DISPATCH_MESSAGES_MAX_OFFSET does not match the maximum value of go_msg_t::dispatch_message_offset. "
    "Fix the value in dispatch_settings.hpp");

namespace {

struct DispatchSettingsContainerKey {
    CoreType core_type;
    uint32_t num_hw_cqs;

    bool operator==(const DispatchSettingsContainerKey& other) const {
        return core_type == other.core_type && num_hw_cqs == other.num_hw_cqs;
    }
};

struct DispatchSettingsContainerKeyHasher {
    size_t operator()(const DispatchSettingsContainerKey& k) const {
        const auto h1 = std::hash<uint32_t>{}(static_cast<int>(k.core_type));
        const auto h2 = std::hash<uint32_t>{}(k.num_hw_cqs);
        return h1 ^ (h2 << 1);
    }
};

using DispatchSettingsContainer =
    std::unordered_map<DispatchSettingsContainerKey, DispatchSettings, DispatchSettingsContainerKeyHasher>;

DispatchSettingsContainer& get_store() {
    static DispatchSettingsContainer store;
    return store;
}
}  // namespace

DispatchSettings DispatchSettings::worker_defaults(const tt::Cluster& cluster, const uint32_t num_hw_cqs) {
    uint32_t prefetch_q_entries;
    if (cluster.is_galaxy_cluster()) {
        prefetch_q_entries = 1532 / num_hw_cqs;
    } else {
        prefetch_q_entries = 1534;
    }

    return DispatchSettings()
        .num_hw_cqs(num_hw_cqs)
        .core_type(CoreType::WORKER)
        .prefetch_q_entries(prefetch_q_entries)
        .prefetch_max_cmd_size(128_KB)
        .prefetch_cmddat_q_size(256_KB)
        .prefetch_scratch_db_size(128_KB)
        .prefetch_ringbuffer_size(1024_KB)
        .prefetch_d_buffer_size(256_KB)

        .dispatch_size(512_KB)
        .dispatch_s_buffer_size(32_KB)

        .with_alignment(MetalContext::instance().hal().get_alignment(HalMemType::L1))

        .build();
}

DispatchSettings DispatchSettings::eth_defaults(const tt::Cluster& /*cluster*/, const uint32_t num_hw_cqs) {
    return DispatchSettings()
        .num_hw_cqs(num_hw_cqs)
        .core_type(CoreType::ETH)
        .prefetch_q_entries(128)
        .prefetch_max_cmd_size(32_KB)
        .prefetch_cmddat_q_size(64_KB)
        .prefetch_scratch_db_size(19_KB)
        .prefetch_ringbuffer_size(70_KB)
        .prefetch_d_buffer_size(128_KB)

        .dispatch_size(128_KB)
        .dispatch_s_buffer_size(32_KB)

        .with_alignment(MetalContext::instance().hal().get_alignment(HalMemType::L1))

        .build();
}

DispatchSettings DispatchSettings::defaults(
    const CoreType& core_type, const tt::Cluster& cluster, const uint32_t num_hw_cqs) {
    if (!num_hw_cqs) {
        TT_THROW("0 CQs is invalid");
    }

    if (core_type == CoreType::WORKER) {
        return worker_defaults(cluster, num_hw_cqs);
    } else if (core_type == CoreType::ETH) {
        return eth_defaults(cluster, num_hw_cqs);
    }

    TT_THROW("Default settings for core_type {} is not implemented", magic_enum::enum_name(core_type));
}

std::vector<std::string> DispatchSettings::get_errors() const {
    std::vector<std::string> msgs;

    if (!prefetch_q_rd_ptr_size_ || !prefetch_q_pcie_rd_ptr_size_ || !dispatch_s_sync_sem_ || !other_ptrs_size) {
        msgs.push_back(fmt::format("configuration with_alignment() is a required\n"));
    }

    if (!num_hw_cqs_) {
        msgs.push_back(fmt::format("num_hw_cqs must be set to non zero\n"));
    } else if (num_hw_cqs_ > DispatchSettings::MAX_NUM_HW_CQS) {
        msgs.push_back(fmt::format(
            "{} CQs specified. the maximum number for num_hw_cqs is {}\n",
            num_hw_cqs_,
            DispatchSettings::MAX_NUM_HW_CQS));
    }

    if (prefetch_cmddat_q_size_ < 2 * prefetch_max_cmd_size_) {
        msgs.push_back(fmt::format(
            "prefetch_cmddat_q_size_ {} is too small. It must be >= 2 * prefetch_max_cmd_size_\n",
            prefetch_cmddat_q_size_));
    }
    if (prefetch_scratch_db_size_ % 2) {
        msgs.push_back(fmt::format("prefetch_scratch_db_size_ {} must be even\n", prefetch_scratch_db_size_));
    }
    if ((dispatch_size_ & (dispatch_size_ - 1)) != 0) {
        msgs.push_back(fmt::format("dispatch_block_size_ {} must be power of 2\n", dispatch_size_));
    }

    return msgs;
}

DispatchSettings& DispatchSettings::build() {
    const auto msgs = get_errors();
    if (msgs.empty()) {
        return *this;
    }
    TT_THROW("Validation errors in dispatch_settings. Call validate() for a list of errors");
}

// Returns the settings for a core type and number hw cqs. The values can be modified, but customization must occur
// before command queue kernels are created.
DispatchSettings& DispatchSettings::get(const CoreType& core_type, const uint32_t num_hw_cqs) {
    DispatchSettingsContainerKey k{core_type, num_hw_cqs};
    auto& store = get_store();
    if (!store.contains(k)) {
        TT_THROW(
            "DispatchSettings is not initialized for CoreType {}, {} CQs",
            magic_enum::enum_name(core_type),
            num_hw_cqs);
    }
    return store[k];
}

// Reset the settings
void DispatchSettings::initialize(const tt::Cluster& cluster) {
    static constexpr std::array<CoreType, 2> k_SupportedCoreTypes{CoreType::ETH, CoreType::WORKER};
    auto& store = get_store();
    for (const auto& core_type : k_SupportedCoreTypes) {
        for (uint32_t hw_cqs = 1; hw_cqs <= MAX_NUM_HW_CQS; ++hw_cqs) {
            DispatchSettingsContainerKey k{core_type, hw_cqs};
            store[k] = DispatchSettings::defaults(core_type, cluster, hw_cqs);
        }
    }
}

// Reset the settings for a core type and number hw cqs to the provided settings
void DispatchSettings::initialize(const DispatchSettings& other) {
    auto& store = get_store();
    DispatchSettingsContainerKey k{other.core_type_, other.num_hw_cqs_};
    store[k] = other;
}

bool DispatchSettings::operator==(const DispatchSettings& other) const {
    return num_hw_cqs_ == other.num_hw_cqs_ && prefetch_q_rd_ptr_size_ == other.prefetch_q_rd_ptr_size_ &&
           prefetch_q_pcie_rd_ptr_size_ == other.prefetch_q_pcie_rd_ptr_size_ &&
           dispatch_s_sync_sem_ == other.dispatch_s_sync_sem_ && other_ptrs_size == other.other_ptrs_size &&
           prefetch_q_entries_ == other.prefetch_q_entries_ && prefetch_q_size_ == other.prefetch_q_size_ &&
           prefetch_max_cmd_size_ == other.prefetch_max_cmd_size_ &&
           prefetch_cmddat_q_size_ == other.prefetch_cmddat_q_size_ &&
           prefetch_scratch_db_size_ == other.prefetch_scratch_db_size_ &&
           prefetch_d_buffer_size_ == other.prefetch_d_buffer_size_ && prefetch_d_pages_ == other.prefetch_d_pages_ &&
           dispatch_size_ == other.dispatch_size_ && dispatch_pages_ == other.dispatch_pages_ &&
           dispatch_s_buffer_size_ == other.dispatch_s_buffer_size_ &&
           dispatch_s_buffer_pages_ == other.dispatch_s_buffer_pages_;
}

bool DispatchSettings::operator!=(const DispatchSettings& other) const { return !(*this == other); }

DispatchSettings& DispatchSettings::core_type(const CoreType& val) {
    this->core_type_ = val;
    return *this;
}

// Trivial setter for num_hw_cqs
DispatchSettings& DispatchSettings::num_hw_cqs(uint32_t val) {
    this->num_hw_cqs_ = val;
    return *this;
}

// Trivial setter for prefetch_max_cmd_size
DispatchSettings& DispatchSettings::prefetch_max_cmd_size(uint32_t val) {
    this->prefetch_max_cmd_size_ = val;
    return *this;
}

// Trivial setter for prefetch_cmddat_q_size
DispatchSettings& DispatchSettings::prefetch_cmddat_q_size(uint32_t val) {
    this->prefetch_cmddat_q_size_ = val;
    return *this;
}

// Trivial setter for prefetch_scratch_db_size
DispatchSettings& DispatchSettings::prefetch_scratch_db_size(uint32_t val) {
    this->prefetch_scratch_db_size_ = val;
    return *this;
}

// Trivial setter for prefetch_ringbuffer_size
DispatchSettings& DispatchSettings::prefetch_ringbuffer_size(uint32_t val) {
    this->prefetch_ringbuffer_size_ = val;
    return *this;
}

// Setter for prefetch_q_entries and update prefetch_q_size
DispatchSettings& DispatchSettings::prefetch_q_entries(uint32_t val) {
    this->prefetch_q_entries_ = val;
    this->prefetch_q_size_ = val * sizeof(prefetch_q_entry_type);
    return *this;
}

// Setter for prefetch_d_buffer_size and update prefetch_d_pages
DispatchSettings& DispatchSettings::prefetch_d_buffer_size(uint32_t val) {
    this->prefetch_d_buffer_size_ = val;
    this->prefetch_d_pages_ = this->prefetch_d_buffer_size_ / (1 << PREFETCH_D_BUFFER_LOG_PAGE_SIZE);

    return *this;
}

// Setter for dispatch_block_size and update dispatch_pages
DispatchSettings& DispatchSettings::dispatch_size(uint32_t val) {
    this->dispatch_size_ = val;
    this->dispatch_pages_ = this->dispatch_size_ / (1 << DISPATCH_BUFFER_LOG_PAGE_SIZE);
    return *this;
}

// Setter for dispatch_s_buffer_size and update dispatch_s_buffer_pages
DispatchSettings& DispatchSettings::dispatch_s_buffer_size(uint32_t val) {
    this->dispatch_s_buffer_size_ = val;
    this->dispatch_s_buffer_pages_ = this->dispatch_s_buffer_size_ / (1 << DISPATCH_S_BUFFER_LOG_PAGE_SIZE);
    return *this;
}

// Sets pointer values based on L1 alignment
DispatchSettings& DispatchSettings::with_alignment(uint32_t l1_alignment) {
    this->prefetch_q_rd_ptr_size_ = sizeof(prefetch_q_ptr_type);
    this->prefetch_q_pcie_rd_ptr_size_ = l1_alignment - sizeof(prefetch_q_ptr_type);
    this->dispatch_s_sync_sem_ = DISPATCH_MESSAGE_ENTRIES * l1_alignment;
    this->other_ptrs_size = l1_alignment;

    return *this;
}

}  // namespace tt::tt_metal
