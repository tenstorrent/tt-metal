// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <mutex>
#include <vector>

#include "core_coord.hpp"
#include "data_types.hpp"
#include "dispatch_core_common.hpp"
#include "dispatch_core_manager.hpp"
#include <umd/device/tt_xy_pair.h>

namespace tt::tt_metal {

// Cluster level interface through which device-dispatch characteristics can be
// queried. This layer builds on top of the dispatch core manager (responsible for
// assigning cores to specific dispatch tasks) and the Cluster (tracks multi-chip topology)
// to provide users with higher level queries about the dispatch topology.

// Any new functions querying dispatch state should be placed in this interface (along
// with any existing functions that are in the device class but are exposing lower
// level dispatch details to the user)
class DispatchQueryManager {
public:
    DispatchQueryManager& operator=(const DispatchQueryManager&) = delete;
    DispatchQueryManager& operator=(DispatchQueryManager&& other) noexcept = delete;
    DispatchQueryManager(const DispatchQueryManager&) = delete;
    DispatchQueryManager(DispatchQueryManager&& other) noexcept = delete;
    DispatchQueryManager(uint8_t num_hw_cqs);

    // dispatch_s related queries
    bool dispatch_s_enabled() const;
    bool distributed_dispatcher() const;
    NOC go_signal_noc() const;
    // General Dispatch related queries - configs and core placement
    const std::vector<CoreCoord>& get_logical_storage_cores(uint32_t device_id) const;
    const std::vector<CoreCoord>& get_logical_dispatch_cores(uint32_t device_id) const;
    const std::vector<CoreCoord>& get_logical_dispatch_cores_on_user_chips() const;
    const std::vector<CoreCoord>& get_logical_storage_cores_on_user_chips() const;
    tt_cxy_pair get_dispatch_core(uint8_t cq_id) const;

private:
    void reset(uint8_t num_hw_cqs);

    bool dispatch_s_enabled_ = false;
    bool distributed_dispatcher_ = false;
    NOC go_signal_noc_ = NOC::NOC_0;
    uint8_t num_hw_cqs_ = 0;
    DispatchCoreConfig dispatch_core_config_;  // The config this object was initialized with, need to store it so we
                                               // know when to reset if it changes.
    // Store the list of reserved storage and dispatch cores on
    // user exposed chips. Expected to be identical across chips.
    std::vector<CoreCoord> logical_dispatch_cores_on_user_chips_;
    std::vector<CoreCoord> logical_storage_cores_on_user_chips_;
    // Make this mutable, since this is JIT populated
    // through a const instance when queried
    mutable std::vector<tt_cxy_pair> dispatch_cores_;
    mutable std::mutex modifier_mutex;
};

}  // namespace tt::tt_metal
