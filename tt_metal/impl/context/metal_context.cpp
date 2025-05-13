// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "metal_context.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "tt_metal/impl/allocator/l1_banking_allocator.hpp"
#include "tt_metal/impl/dispatch/topology.hpp"
#include "tt_metal/impl/debug/dprint_server.hpp"
#include "tt_metal/impl/debug/noc_logging.hpp"
#include "tt_metal/impl/debug/watcher_server.hpp"
#include "tt_metal/impl/debug/debug_helpers.hpp"
#include "tt_metal/jit_build/build_env_manager.hpp"
#include "tt_metal/llrt/llrt.hpp"
#include "tt_metal/llrt/get_platform_architecture.hpp"

namespace tt::tt_metal {

void MetalContext::initialize(
    const DispatchCoreConfig& dispatch_core_config, uint8_t num_hw_cqs, const BankMapping& l1_bank_remap) {
    if (initialized_) {
        if (this->dispatch_core_config_ != dispatch_core_config or num_hw_cqs != this->num_hw_cqs_ or
            l1_bank_remap != this->l1_bank_remap_) {
            log_warning("Closing and re-initializing MetalContext with new parameters.");
        } else {
            // Re-init request with the same parameters, do nothing
            return;
        }
    }

    initialized_ = true;
    dispatch_core_config_ = dispatch_core_config;
    num_hw_cqs_ = num_hw_cqs;
    l1_bank_remap_ = l1_bank_remap;

    // Initialize dispatch state
    dispatch_core_manager_ = std::make_unique<dispatch_core_manager>(dispatch_core_config, num_hw_cqs);
    dispatch_query_manager_ = std::make_unique<DispatchQueryManager>(num_hw_cqs);
    // Need DispatchMemMap for both dispatch core types
    tt_metal::DispatchSettings::initialize(*cluster_);
    dispatch_mem_map_[magic_enum::enum_integer(CoreType::WORKER)] =
        std::make_unique<DispatchMemMap>(CoreType::WORKER, num_hw_cqs);
    dispatch_mem_map_[magic_enum::enum_integer(CoreType::ETH)] =
        std::make_unique<DispatchMemMap>(CoreType::ETH, num_hw_cqs);

    // TODO: Move FW, fabric, dispatch init here
}

MetalContext& MetalContext::instance() {
    static tt::stl::Indestructible<MetalContext> inst;
    return inst.get();
}

MetalContext::MetalContext() {
    bool is_base_routing_fw_enabled =
        Cluster::is_base_routing_fw_enabled(Cluster::get_cluster_type_from_cluster_desc(rtoptions_));
    hal_ = std::make_unique<Hal>(get_platform_architecture(rtoptions_), is_base_routing_fw_enabled);
    cluster_ = std::make_unique<Cluster>(rtoptions_, *hal_);
}

llrt::RunTimeOptions& MetalContext::rtoptions() { return rtoptions_; }

Cluster& MetalContext::get_cluster() {
    TT_FATAL(cluster_, "Trying to get cluster before intializing it.");
    return *cluster_;
}

const llrt::RunTimeOptions& MetalContext::rtoptions() const { return rtoptions_; }

const Cluster& MetalContext::get_cluster() const {
    TT_FATAL(cluster_, "Trying to get cluster before intializing it.");
    return *cluster_;
}

const Hal& MetalContext::hal() const {
    TT_FATAL(hal_, "Trying to get hal before intializing it.");
    return *hal_;
}

dispatch_core_manager& MetalContext::get_dispatch_core_manager() {
    TT_FATAL(dispatch_core_manager_, "Trying to get dispatch_core_manager before intializing it.");
    return *dispatch_core_manager_;
}

DispatchQueryManager& MetalContext::get_dispatch_query_manager() {
    TT_FATAL(dispatch_query_manager_, "Trying to get dispatch_query_manager before intializing it.");
    return *dispatch_query_manager_;
}

const DispatchMemMap& MetalContext::dispatch_mem_map() const {
    return dispatch_mem_map(dispatch_core_config_.get_core_type());
}

const DispatchMemMap& MetalContext::dispatch_mem_map(const CoreType& core_type) const {
    auto& mem_map = dispatch_mem_map_[magic_enum::enum_integer(core_type)];
    TT_FATAL(mem_map, "Tried to get dispatch_mem_map for {} before intializing it.", core_type);
    return *mem_map;
}

}  // namespace tt::tt_metal
