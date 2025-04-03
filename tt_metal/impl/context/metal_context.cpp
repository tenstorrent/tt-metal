// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "metal_context.hpp"
#include <tt-metalium/dispatch_settings.hpp>
#include "tt_metal/impl/allocator/l1_banking_allocator.hpp"
#include "tt_metal/impl/dispatch/topology.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/impl/debug/dprint_server.hpp"
#include "tt_metal/impl/debug/noc_logging.hpp"
#include "tt_metal/impl/debug/watcher_server.hpp"
#include "tt_metal/impl/debug/debug_helpers.hpp"
#include "tt_metal/jit_build/build_env_manager.hpp"
#include "tt_metal/llrt/llrt.hpp"

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
    tt_metal::DispatchSettings::initialize(cluster_);

    // TODO: Move FW, fabric, dispatch init here
}

MetalContext& MetalContext::instance() {
    static tt::stl::Indestructible<MetalContext> inst;
    return inst.get();
}

MetalContext::MetalContext() {}

Cluster& MetalContext::get_cluster() { return cluster_; }

dispatch_core_manager& MetalContext::get_dispatch_core_manager() {
    TT_FATAL(dispatch_core_manager_, "Trying to get dispatch_core_manager before intializing it.");
    return *dispatch_core_manager_;
}

DispatchQueryManager& MetalContext::get_dispatch_query_manager() {
    TT_FATAL(dispatch_query_manager_, "Trying to get dispatch_query_manager before intializing it.");
    return *dispatch_query_manager_;
}

}  // namespace tt::tt_metal
