// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_core_common.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/types/arch.hpp>
#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_metal {

DispatchCoreAxis DispatchCoreConfig::get_default_axis() {
    if (MetalContext::instance().get_cluster().arch() == tt::ARCH::BLACKHOLE) {
        if (MetalContext::instance().get_fabric_tensix_config() == tt_fabric::FabricTensixConfig::DISABLED) {
            return DispatchCoreAxis::COL;
        }
    }
    return DispatchCoreAxis::ROW;
}

DispatchCoreConfig get_dispatch_core_config() {
    return MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
}

CoreType get_dispatch_core_type() {
    return MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_type();
};

}  // namespace tt::tt_metal
