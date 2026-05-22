// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/dispatch/dispatch_core_common.hpp"
#include <tt_stl/reflection.hpp>
#include "dispatch_core_common.hpp"
#include <tt-metalium/cluster.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/context/metal_context.hpp"
#include <umd/device/types/arch.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <impl/dispatch/dispatch_core_manager.hpp>
#include <llrt/tt_cluster.hpp>

namespace tt::tt_metal {

namespace {

bool is_blackhole_arch() {
    return tt::tt_metal::detail::get_platform_architecture_name().find("blackhole") != std::string::npos;
}

DispatchCoreAxis get_default_dispatch_core_axis(std::optional<tt::tt_fabric::FabricTensixConfig> fabric_tensix_config) {
    if (is_blackhole_arch()) {
        if (fabric_tensix_config.value_or(tt::tt_fabric::FabricTensixConfig::DISABLED) ==
            tt::tt_fabric::FabricTensixConfig::MUX) {
            return DispatchCoreAxis::ROW;
        }
        return DispatchCoreAxis::COL;
    }
    return DispatchCoreAxis::ROW;
}

}  // namespace

DispatchCoreType DispatchCoreConfig::get_default_type() {
    const auto cluster_type = tt::tt_metal::GetClusterType();
    if (cluster_type == tt::tt_metal::ClusterType::N300 || cluster_type == tt::tt_metal::ClusterType::T3K ||
        cluster_type == tt::tt_metal::ClusterType::N300_2x2) {
        return DispatchCoreType::ETH;
    }
    return DispatchCoreType::WORKER;
}

DispatchCoreConfig DispatchCoreConfig::create_dispatch_core_config(
    std::optional<DispatchCoreType> dispatch_core_type,
    std::optional<DispatchCoreAxis> dispatch_core_axis,
    std::optional<tt::tt_fabric::FabricTensixConfig> fabric_tensix_config) {
    if (dispatch_core_type.has_value() && dispatch_core_axis.has_value() &&
        dispatch_core_type.value() == DispatchCoreType::ETH && dispatch_core_axis.value() == DispatchCoreAxis::COL) {
        TT_THROW("COL axis is not supported for ETH dispatch core type");
    }

    if (dispatch_core_axis.has_value() && dispatch_core_axis.value() == DispatchCoreAxis::ROW && is_blackhole_arch() &&
        fabric_tensix_config.value_or(tt::tt_fabric::FabricTensixConfig::DISABLED) !=
            tt::tt_fabric::FabricTensixConfig::MUX) {
        TT_THROW("ROW dispatch core axis is not supported for blackhole arch unless fabric tensix MUX is enabled");
    }

    if (dispatch_core_type.has_value() && dispatch_core_axis.has_value()) {
        return DispatchCoreConfig(dispatch_core_type.value(), dispatch_core_axis.value());
    }
    if (dispatch_core_type.has_value()) {
        return DispatchCoreConfig(dispatch_core_type.value(), get_default_dispatch_core_axis(fabric_tensix_config));
    }
    if (dispatch_core_axis.has_value()) {
        if (dispatch_core_axis.value() == DispatchCoreAxis::COL) {
            return DispatchCoreConfig(DispatchCoreType::WORKER, dispatch_core_axis.value());
        }
        return DispatchCoreConfig(get_default_type(), dispatch_core_axis.value());
    }

    return DispatchCoreConfig(get_default_type(), get_default_dispatch_core_axis(fabric_tensix_config));
}

DispatchCoreAxis DispatchCoreConfig::get_default_axis() {
    // All internal callers should use resolve_dispatch_core_axis(arch, fabric_tensix_config) instead.

    // Check if the instance exists to prevent implicit init of a second cluster
    // if we already have one in MetalEnv
    // TOOD: https://github.com/tenstorrent/tt-metal/issues/39974
    if (MetalContext::instance_exists(DEFAULT_CONTEXT_ID)) {
        if (MetalContext::instance().get_cluster().arch() == tt::ARCH::BLACKHOLE) {
            if (MetalContext::instance().get_fabric_tensix_config() == tt_fabric::FabricTensixConfig::DISABLED) {
                return DispatchCoreAxis::COL;
            }
        }
    }
    return DispatchCoreAxis::ROW;
}

DispatchCoreAxis resolve_dispatch_core_axis(
    const DispatchCoreConfig& config, tt::ARCH arch, tt_fabric::FabricTensixConfig fabric_tensix_config) {
    const auto& axis = std::get<1>(config.attribute_values());
    if (axis.has_value()) {
        return axis.value();
    }
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
    // Check if the instance exists to prevent implicit init of a second cluster
    // if we already have one in MetalEnv
    // TODO: https://github.com/tenstorrent/tt-metal/issues/39974
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
