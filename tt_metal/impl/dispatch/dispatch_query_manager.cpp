// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_query_manager.hpp"

#include <initializer_list>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "assert.hpp"
#include "core_descriptor.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/types/cluster_descriptor_types.h>
#include <umd/device/types/xy_pair.h>

namespace {

tt::tt_metal::DispatchCoreConfig dispatch_core_config() {
    return tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
}

tt_cxy_pair dispatch_core(uint8_t cq_id) {
    tt_cxy_pair dispatch_core = tt_cxy_pair(0, 0, 0);
    std::optional<tt_cxy_pair> first_dispatch_core = std::nullopt;
    for (chip_id_t device_id : tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids()) {
        uint16_t channel =
            tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device_id);
        if (tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id) == device_id) {
            // Dispatch core is not allocated on this MMIO device or this is a TG system, skip it
            // On TG, local dispatch cores are allocated on MMIO devices, but are not used
            // since programs are not run on these devices. The placement of these cores is
            // irrelevant for the runtime layer, since these are not used. Hence, these are
            // skipped.
            if (not tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().is_dispatcher_core_allocated(
                    device_id, channel, cq_id) or
                tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster()) {
                continue;
            }
            dispatch_core = tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().dispatcher_core(
                device_id, channel, cq_id);
        } else {
            // Dispatch core is not allocated on this Non-MMIO device, skip it
            if (not tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().is_dispatcher_d_core_allocated(
                    device_id, channel, cq_id)) {
                continue;
            }
            dispatch_core = tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().dispatcher_d_core(
                device_id, channel, cq_id);
        }
        if (not first_dispatch_core.has_value()) {
            first_dispatch_core = dispatch_core;
        } else {
            TT_FATAL(
                dispatch_core.x == first_dispatch_core.value().x and dispatch_core.y == first_dispatch_core.value().y,
                "Expected the Dispatch Cores to be consistent across physical devices");
        }
    }
    TT_FATAL(first_dispatch_core.has_value(), "Could not find the dispatch core for {}", cq_id);
    return dispatch_core;
}

template <typename F>
std::vector<CoreCoord> get_consistent_logical_cores(
    uint8_t num_hw_cqs, const tt::tt_metal::DispatchCoreConfig& dispatch_core_config, F&& func) {
    auto user_chips = tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids();
    std::vector<CoreCoord> first_core_set;
    std::vector<CoreCoord> current_cores;

    for (auto chip : user_chips) {
        current_cores = std::forward<F>(func)(chip, num_hw_cqs, dispatch_core_config);
        if (!first_core_set.empty()) {
            TT_FATAL(first_core_set == current_cores, "Expected logical cores to match across user exposed devices");
        } else {
            first_core_set = current_cores;
        }
    }
    return current_cores;
}

std::vector<CoreCoord> populate_all_logical_storage_cores(
    uint8_t num_hw_cqs, const tt::tt_metal::DispatchCoreConfig& dispatch_core_config) {
    return get_consistent_logical_cores(num_hw_cqs, dispatch_core_config, tt::get_logical_storage_cores);
}

std::vector<CoreCoord> populate_all_logical_dispatch_cores(
    uint8_t num_hw_cqs, const tt::tt_metal::DispatchCoreConfig& dispatch_core_config) {
    return get_consistent_logical_cores(num_hw_cqs, dispatch_core_config, tt::get_logical_dispatch_cores);
}

}  // namespace

namespace tt::tt_metal {

bool DispatchQueryManager::dispatch_s_enabled() const { return dispatch_s_enabled_; }

bool DispatchQueryManager::distributed_dispatcher() const { return distributed_dispatcher_; }

NOC DispatchQueryManager::go_signal_noc() const { return go_signal_noc_; }

void DispatchQueryManager::reset(uint8_t num_hw_cqs) {
    num_hw_cqs_ = num_hw_cqs;
    dispatch_core_config_ = dispatch_core_config();
    dispatch_s_enabled_ =
        (num_hw_cqs == 1 or dispatch_core_config().get_dispatch_core_type() == DispatchCoreType::WORKER);
    distributed_dispatcher_ =
        (num_hw_cqs == 1 and dispatch_core_config().get_dispatch_core_type() == DispatchCoreType::ETH);
    go_signal_noc_ = dispatch_s_enabled_ ? NOC::NOC_1 : NOC::NOC_0;
    // Reset the dispatch cores reported by the manager. Will be re-populated when the associated query is made
    dispatch_cores_ = {};
    // Populate dispatch and storage
    logical_dispatch_cores_on_user_chips_ = populate_all_logical_dispatch_cores(num_hw_cqs_, dispatch_core_config());
    logical_storage_cores_on_user_chips_ = populate_all_logical_storage_cores(num_hw_cqs_, dispatch_core_config());
}

const std::vector<CoreCoord>& DispatchQueryManager::get_logical_storage_cores(uint32_t device_id) const {
    return tt::get_logical_storage_cores(device_id, num_hw_cqs_, dispatch_core_config());
}

const std::vector<CoreCoord>& DispatchQueryManager::get_logical_dispatch_cores(uint32_t device_id) const {
    return tt::get_logical_dispatch_cores(device_id, num_hw_cqs_, dispatch_core_config());
}

const std::vector<CoreCoord>& DispatchQueryManager::get_logical_storage_cores_on_user_chips() const {
    return logical_storage_cores_on_user_chips_;
}

const std::vector<CoreCoord>& DispatchQueryManager::get_logical_dispatch_cores_on_user_chips() const {
    return logical_dispatch_cores_on_user_chips_;
}

tt_cxy_pair DispatchQueryManager::get_dispatch_core(uint8_t cq_id) const {
    std::scoped_lock<std::mutex> lock(modifier_mutex);
    if (dispatch_cores_.empty()) {
        for (auto cq = 0; cq < num_hw_cqs_; cq++) {
            // Populate when queried. Statically allocating at
            // the start of the process causes the dispatch core
            // order to change, which leads to lower performance
            // with ethernet dispatch.
            dispatch_cores_.push_back(dispatch_core(cq));
        }
    }
    return dispatch_cores_[cq_id];
}

DispatchQueryManager::DispatchQueryManager(uint8_t num_hw_cqs) { this->reset(num_hw_cqs); }

}  // namespace tt::tt_metal
