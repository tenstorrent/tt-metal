// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/dispatch/dispatch_core_common.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/types/arch.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <impl/dispatch/dispatch_core_manager.hpp>
#include <llrt/tt_cluster.hpp>

namespace tt::tt_metal {

DispatchCoreAxis DispatchCoreConfig::get_default_axis() {
    if (MetalContext::instance().get_cluster().arch() == tt::ARCH::BLACKHOLE) {
        if (MetalContext::instance().get_fabric_tensix_config() == tt_fabric::FabricTensixConfig::DISABLED) {
            return DispatchCoreAxis::COL;
        }
    }
    return DispatchCoreAxis::ROW;
}

CoreType get_core_type_from_config(const DispatchCoreConfig& config) {
    switch (config.get_dispatch_core_type()) {
        case DispatchCoreType::WORKER: return CoreType::WORKER;
        case DispatchCoreType::ETH: return CoreType::ETH;
        default: TT_THROW("invalid dispatch core type");
    }
}

DispatchCoreConfig get_dispatch_core_config() {
    return MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
}

}  // namespace tt::tt_metal
