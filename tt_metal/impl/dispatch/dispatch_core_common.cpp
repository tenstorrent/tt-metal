// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/dispatch/dispatch_core_common.hpp"
#include <tt_stl/reflection.hpp>
#include "impl/context/metal_context.hpp"
#include <umd/device/types/arch.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <impl/dispatch/dispatch_core_manager.hpp>
#include <llrt/tt_cluster.hpp>

namespace tt::tt_metal {

DispatchCoreAxis DispatchCoreConfig::get_default_axis() {
    // Deprecated fallback: only checks the DEFAULT context, so it fails for non-default
    // contexts (e.g. mock devices).  All internal callers should use
    // resolve_dispatch_core_axis(arch, fabric_tensix_config) instead.
    // This remains for legacy external callers that go through get_dispatch_core_axis().
    // TODO: remove this once the header API can be updated.
    // https://github.com/tenstorrent/tt-metal/issues/39974
    // We first check if the instance exists to avoid implicit creation of the default context with MetalEnv
    if (MetalContext::instance_exists(DEFAULT_CONTEXT_ID)) {
        if (MetalContext::instance().get_cluster().arch() == tt::ARCH::BLACKHOLE) {
            if (MetalContext::instance().get_fabric_tensix_config() == tt_fabric::FabricTensixConfig::DISABLED) {
                return DispatchCoreAxis::COL;
            }
        }
    }
    return DispatchCoreAxis::ROW;
}

DispatchCoreAxis resolve_dispatch_core_axis(tt::ARCH arch, tt_fabric::FabricTensixConfig fabric_tensix_config) {
    if (arch == tt::ARCH::BLACKHOLE && fabric_tensix_config == tt_fabric::FabricTensixConfig::DISABLED) {
        return DispatchCoreAxis::COL;
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
    if (MetalContext::instance_exists(DEFAULT_CONTEXT_ID)) {
        return MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
    }
    return DispatchCoreConfig();
}

}  // namespace tt::tt_metal

std::size_t std::hash<tt::tt_metal::DispatchCoreConfig>::operator()(
    const tt::tt_metal::DispatchCoreConfig& dispatch_core_config) const {
    return ttsl::hash::hash_objects_with_default_seed(dispatch_core_config.attribute_values());
}
