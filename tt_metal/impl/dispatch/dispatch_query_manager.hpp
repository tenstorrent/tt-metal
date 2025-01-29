// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <dispatch_core_manager.hpp>
#include <tt-metalium/tt_cluster.hpp>

namespace tt::tt_metal {

// Interface through which device-dispatch characteristics can be queried.
// This layer builds on top of the dispatch core manager (responsible for assigning
// cores to specific dispatch tasks) and the Cluster (tracks multi-chip topology)
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

    static void initialize(uint8_t num_hw_cqs);
    static const DispatchQueryManager& instance();
    // dispatch_s related queries
    bool dispatch_s_enabled() const;
    bool distributed_dispatcher() const;
    NOC go_signal_noc() const;

private:
    void reset(uint8_t num_hw_cqs);
    DispatchQueryManager(uint8_t num_hw_cqs);

    bool dispatch_s_enabled_ = false;
    bool distributed_dispatcher_ = false;
    NOC go_signal_noc_ = NOC::NOC_0;
    uint8_t num_hw_cqs_ = 0;
    CoreType dispatch_core_type_ = CoreType::WORKER;
};

}  // namespace tt::tt_metal
