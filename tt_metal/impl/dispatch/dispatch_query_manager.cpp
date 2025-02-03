// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/dispatch_query_manager.hpp"

namespace {

tt::tt_metal::DispatchCoreConfig dispatch_core_config() {
    tt::tt_metal::DispatchCoreConfig dispatch_core_config;
    tt::tt_metal::DispatchCoreConfig first_dispatch_core_config;

    for (chip_id_t device_id = 0; device_id < tt::Cluster::instance().number_of_devices(); device_id++) {
        dispatch_core_config = tt::tt_metal::dispatch_core_manager::instance().get_dispatch_core_config(device_id);
        if (device_id == 0) {
            first_dispatch_core_config = dispatch_core_config;
        } else {
            TT_FATAL(
                dispatch_core_config == first_dispatch_core_config,
                "Expected the Dispatch Core Config to be consistent across physical devices");
        }
    }

    return dispatch_core_config;
}

tt::tt_metal::DispatchQueryManager* inst = nullptr;

}  // namespace
namespace tt::tt_metal {

void DispatchQueryManager::initialize(uint8_t num_hw_cqs) {
    if (inst == nullptr) {
        static DispatchQueryManager DispatchQueryManager(num_hw_cqs);
        inst = &DispatchQueryManager;
    } else if (num_hw_cqs != inst->num_hw_cqs_ or dispatch_core_config() != inst->dispatch_core_config_) {
        inst->reset(num_hw_cqs);
        inst->num_hw_cqs_ = num_hw_cqs;
    }
}

const DispatchQueryManager& DispatchQueryManager::instance() {
    TT_FATAL(inst != nullptr, "Trying to acess the dispatch query layer without initializing it.");
    return *inst;
}

bool DispatchQueryManager::dispatch_s_enabled() const { return dispatch_s_enabled_; }

bool DispatchQueryManager::distributed_dispatcher() const { return distributed_dispatcher_; }

NOC DispatchQueryManager::go_signal_noc() const { return go_signal_noc_; }

void DispatchQueryManager::reset(uint8_t num_hw_cqs) {
    num_hw_cqs_ = num_hw_cqs;
    dispatch_core_config_ = dispatch_core_config();
    dispatch_s_enabled_ =
        (num_hw_cqs == 1 or dispatch_core_config_.get_dispatch_core_type() == DispatchCoreType::WORKER);
    distributed_dispatcher_ =
        (num_hw_cqs == 1 and dispatch_core_config_.get_dispatch_core_type() == DispatchCoreType::ETH);
    go_signal_noc_ = dispatch_s_enabled_ ? NOC::NOC_1 : NOC::NOC_0;
}

const DispatchCoreConfig& DispatchQueryManager::get_dispatch_core_config() const { return dispatch_core_config_; }

const std::vector<CoreCoord>& DispatchQueryManager::get_logical_storage_cores(uint32_t device_id) const {
    return tt::get_logical_storage_cores(device_id, num_hw_cqs_, dispatch_core_config_);
}

const std::vector<CoreCoord>& DispatchQueryManager::get_logical_dispatch_cores(uint32_t device_id) const {
    return tt::get_logical_dispatch_cores(device_id, num_hw_cqs_, dispatch_core_config_);
}

DispatchQueryManager::DispatchQueryManager(uint8_t num_hw_cqs) { this->reset(num_hw_cqs); }

}  // namespace tt::tt_metal
