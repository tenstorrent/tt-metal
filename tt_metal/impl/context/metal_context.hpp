// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/indestructible.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/core_descriptor.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/dev_msgs.h>
#include <tt-metalium/allocator_types.hpp>
#include <llrt/tt_cluster.hpp>
#include <impl/dispatch/dispatch_core_manager.hpp>
#include <impl/dispatch/dispatch_query_manager.hpp>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tt::tt_metal {

// A class to manage one-time initialization and teardown (FW, dispatch, fabric, cluster) and access to related state.
// Dispatch-independent state (Cluster) is initialized with the creation of MetalContext and accessible right after.
// Dispatch-dependent state (FW, dispatch, fabric) is initialized explicitly with a MetalContext::initialize() call, and
// only accessible after that.
class MetalContext {
public:
    MetalContext& operator=(const MetalContext&) = delete;
    MetalContext& operator=(MetalContext&& other) noexcept = delete;
    MetalContext(const MetalContext&) = delete;
    MetalContext(MetalContext&& other) noexcept = delete;
    static MetalContext& instance();

    Cluster& get_cluster();
    dispatch_core_manager& get_dispatch_core_manager();
    DispatchQueryManager& get_dispatch_query_manager();

    void initialize(
        const DispatchCoreConfig& dispatch_core_config, uint8_t num_hw_cqs, const BankMapping& l1_bank_remap);

private:
    friend class tt::stl::Indestructible<MetalContext>;
    MetalContext();
    ~MetalContext();

    bool initialized_ = false;

    uint8_t num_hw_cqs_ = 0;
    BankMapping l1_bank_remap_;
    DispatchCoreConfig dispatch_core_config_;

    Cluster cluster_;
    std::unique_ptr<dispatch_core_manager> dispatch_core_manager_;
    std::unique_ptr<DispatchQueryManager> dispatch_query_manager_;
};

}  // namespace tt::tt_metal
