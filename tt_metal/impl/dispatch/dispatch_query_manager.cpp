// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_query_manager.hpp"

#include <initializer_list>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <tt_stl/assert.hpp>
#include "context/metal_env_accessor.hpp"
#include "core_descriptor.hpp"
#include "dispatch/dispatch_core_manager.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <llrt/tt_cluster.hpp>

namespace {

tt_cxy_pair dispatch_core(
    tt::tt_metal::MetalEnv& env, tt::tt_metal::dispatch_core_manager& core_manager, uint8_t cq_id) {
    tt_cxy_pair dispatch_core = tt_cxy_pair(0, 0, 0);
    std::optional<tt_cxy_pair> first_dispatch_core = std::nullopt;
    auto& cluster = tt::tt_metal::MetalEnvAccessor(env).impl().get_cluster();
    for (tt::ChipId device_id : cluster.all_chip_ids()) {
        uint16_t channel = cluster.get_assigned_channel_for_device(device_id);
        if (cluster.get_associated_mmio_device(device_id) == device_id) {
            // Dispatch core is not allocated on this MMIO device or this is a TG system, skip it
            // On TG, local dispatch cores are allocated on MMIO devices, but are not used
            // since programs are not run on these devices. The placement of these cores is
            // irrelevant for the runtime layer, since these are not used. Hence, these are
            // skipped.
            if (not core_manager.is_dispatcher_core_allocated(device_id, channel, cq_id) or
                cluster.is_galaxy_cluster()) {
                continue;
            }
            dispatch_core = core_manager.dispatcher_core(device_id, channel, cq_id);
        } else {
            // Dispatch core is not allocated on this Non-MMIO device, skip it
            if (not core_manager.is_dispatcher_d_core_allocated(device_id, channel, cq_id)) {
                continue;
            }
            dispatch_core = core_manager.dispatcher_d_core(device_id, channel, cq_id);
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

std::vector<tt::tt_metal::CoreCoord> get_consistent_logical_cores(
    tt::tt_metal::MetalEnv& env, uint8_t num_hw_cqs, const tt::tt_metal::DispatchCoreConfig& dispatch_core_config) {
    auto user_chips = tt::tt_metal::MetalEnvAccessor(env).impl().get_cluster().user_exposed_chip_ids();
    std::vector<tt::tt_metal::CoreCoord> first_core_set;
    std::vector<tt::tt_metal::CoreCoord> current_cores;

    for (auto chip : user_chips) {
        current_cores = tt::get_logical_dispatch_cores(
            tt::tt_metal::MetalEnvAccessor(env).impl(), chip, num_hw_cqs, dispatch_core_config);
        if (!first_core_set.empty()) {
            TT_FATAL(first_core_set == current_cores, "Expected logical cores to match across user exposed devices");
        } else {
            first_core_set = current_cores;
        }
    }
    return current_cores;
}

std::vector<tt::tt_metal::CoreCoord> populate_all_logical_dispatch_cores(
    tt::tt_metal::MetalEnv& env, uint8_t num_hw_cqs, const tt::tt_metal::DispatchCoreConfig& dispatch_core_config) {
    return get_consistent_logical_cores(env, num_hw_cqs, dispatch_core_config);
}

tt::tt_metal::CommandQueueDispatchLayout generate_cq_dispatch_layout(
    tt::ARCH arch, tt::CoreType core_type, tt::CoreType dispatch_core_type, uint8_t num_hw_cqs) {
    if (core_type != dispatch_core_type || arch != tt::ARCH::QUASAR) {
        return {.fd_kernels_on_same_core = false, .num_cqs_per_core = 1};
    }
    return {.fd_kernels_on_same_core = true, .num_cqs_per_core = num_hw_cqs};
}

}  // namespace

namespace tt::tt_metal {

bool DispatchQueryManager::dispatch_s_enabled() const { return dispatch_s_enabled_; }

bool DispatchQueryManager::distributed_dispatcher() const { return distributed_dispatcher_; }

NOC DispatchQueryManager::go_signal_noc() const { return go_signal_noc_; }

void DispatchQueryManager::reset(DispatchCoreConfig& dispatch_core_config, uint8_t num_hw_cqs) {
    num_hw_cqs_ = num_hw_cqs;
    dispatch_core_config_ = dispatch_core_config;
    const tt::ARCH arch = MetalEnvAccessor(env_).impl().get_cluster().arch();
    dispatch_s_enabled_ =
        (num_hw_cqs == 1 or dispatch_core_config_.get_dispatch_core_type() == DispatchCoreType::WORKER);
    distributed_dispatcher_ =
        (num_hw_cqs == 1 and dispatch_core_config_.get_dispatch_core_type() == DispatchCoreType::ETH);
    go_signal_noc_ = (dispatch_s_enabled_ and arch != tt::ARCH::QUASAR) ? NOC::NOC_1 : NOC::NOC_0;
    const CoreType dispatch_core_type = get_core_type_from_config(dispatch_core_config);
    worker_cq_dispatch_layout_ = generate_cq_dispatch_layout(arch, CoreType::WORKER, dispatch_core_type, num_hw_cqs);
    eth_cq_dispatch_layout_ = generate_cq_dispatch_layout(arch, CoreType::ETH, dispatch_core_type, num_hw_cqs);
    // Reset the dispatch cores reported by the manager. Will be re-populated when the associated query is made
    dispatch_cores_ = {};
    // Populate dispatch
    logical_dispatch_cores_on_user_chips_ =
        populate_all_logical_dispatch_cores(env_, num_hw_cqs_, dispatch_core_config_);
}

const std::vector<tt::tt_metal::CoreCoord>& DispatchQueryManager::get_logical_dispatch_cores(uint32_t device_id) const {
    return tt::get_logical_dispatch_cores(MetalEnvAccessor(env_).impl(), device_id, num_hw_cqs_, dispatch_core_config_);
}

const std::vector<tt::tt_metal::CoreCoord>& DispatchQueryManager::get_logical_dispatch_cores_on_user_chips() const {
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
            dispatch_cores_.push_back(dispatch_core(env_, core_manager_, cq));
        }
        const CommandQueueDispatchLayout& layout = cq_dispatch_layout(get_core_type_from_config(dispatch_core_config_));
        if (layout.fd_kernels_on_same_core) {
            // The shared, non-offset L1 regions and the per-CQ zoning in DispatchMemMap are only valid if these CQs
            // really do land on one physical core.
            for (uint8_t cq = 1; cq < layout.num_cqs_per_core; cq++) {
                TT_FATAL(
                    dispatch_cores_[cq] == dispatch_cores_[0],
                    "CQs sharing a dispatch core diverged: CQ 0 resolved to chip {} ({}, {}), CQ {} resolved to "
                    "chip {} ({}, {})",
                    dispatch_cores_[0].chip,
                    dispatch_cores_[0].x,
                    dispatch_cores_[0].y,
                    cq,
                    dispatch_cores_[cq].chip,
                    dispatch_cores_[cq].x,
                    dispatch_cores_[cq].y);
            }
        }
    }
    return dispatch_cores_[cq_id];
}

const CommandQueueDispatchLayout& DispatchQueryManager::cq_dispatch_layout(CoreType core_type) const {
    return core_type == CoreType::WORKER ? worker_cq_dispatch_layout_ : eth_cq_dispatch_layout_;
}

DispatchQueryManager::DispatchQueryManager(
    MetalEnv& env, dispatch_core_manager& core_manager, DispatchCoreConfig& dispatch_core_config, uint8_t num_hw_cqs) :
    env_(env), core_manager_(core_manager) {
    this->reset(dispatch_core_config, num_hw_cqs);
}

}  // namespace tt::tt_metal
