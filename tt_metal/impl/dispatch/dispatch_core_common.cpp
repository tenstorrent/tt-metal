// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/dispatch/dispatch_core_common.hpp"
#include <tt_stl/assert.hpp>
#include <tt_stl/reflection.hpp>
#include "dispatch_core_common.hpp"
#include <umd/device/types/arch.hpp>
#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_metal::detail {
CoreType resolve_dispatch_core_type(
    tt::tt_metal::MetalEnvImpl& env, ChipId device_id, const DispatchCoreConfig& dispatch_core_config);
}  // namespace tt::tt_metal::detail

namespace tt::tt_metal {

DispatchCoreAxis resolve_dispatch_core_axis(
    const DispatchCoreConfig& config, tt::ARCH arch, tt_fabric::FabricTensixConfig fabric_tensix_config) {
    if (config.has_dispatch_core_axis()) {
        return config.get_dispatch_core_axis();
    }
    if (arch == tt::ARCH::BLACKHOLE && fabric_tensix_config == tt_fabric::FabricTensixConfig::DISABLED) {
        return DispatchCoreAxis::COL;
    }
    return DispatchCoreAxis::ROW;
}

DispatchCoreConfig resolve_dispatch_core_config(
    tt::ARCH arch,
    tt_fabric::FabricTensixConfig fabric_tensix_config,
    std::optional<DispatchCoreType> type,
    std::optional<DispatchCoreAxis> axis) {
    const auto resolved_type = type.value_or(DispatchCoreType::WORKER);
    const auto resolved_axis = axis.value_or(
        arch == tt::ARCH::BLACKHOLE && fabric_tensix_config == tt_fabric::FabricTensixConfig::DISABLED
            ? DispatchCoreAxis::COL
            : DispatchCoreAxis::ROW);
    return DispatchCoreConfig{resolved_type, resolved_axis};
}

CoreType get_core_type_from_config(const DispatchCoreConfig& config) {
    switch (config.get_dispatch_core_type()) {
        case DispatchCoreType::WORKER: return CoreType::WORKER;
        case DispatchCoreType::ETH: return CoreType::ETH;
        default: TT_THROW("invalid dispatch core type");
    }
}

CoreType resolve_dispatch_core_type(
    tt::tt_metal::MetalEnvImpl& env, ChipId device_id, const DispatchCoreConfig& dispatch_core_config) {
    return ::tt::tt_metal::detail::resolve_dispatch_core_type(env, device_id, dispatch_core_config);
}

CoreType resolve_dispatch_core_type(tt::ARCH arch, DispatchCoreType dispatch_core_type) {
    TT_FATAL(
        arch != tt::ARCH::QUASAR,
        "Offline dispatch-core resolution is not implemented for Quasar; DISPATCH vs WORKER "
        "depends on the SoC descriptor and TT_METAL_TENSIX_DISPATCH_CORES");
    return get_core_type_from_config(DispatchCoreConfig{dispatch_core_type});
}

}  // namespace tt::tt_metal

std::size_t std::hash<tt::tt_metal::DispatchCoreConfig>::operator()(
    const tt::tt_metal::DispatchCoreConfig& dispatch_core_config) const {
    return ttsl::hash::hash_objects_with_default_seed(dispatch_core_config.attribute_values());
}
