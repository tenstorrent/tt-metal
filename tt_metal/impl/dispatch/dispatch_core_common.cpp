// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/dispatch/dispatch_core_common.hpp"
#include <tt_stl/reflection.hpp>
#include "dispatch_core_common.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/types/arch.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <impl/dispatch/dispatch_core_manager.hpp>
#include <llrt/tt_cluster.hpp>
#include <tt-metalium/cluster.hpp>

namespace tt::tt_metal {

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

// Cluster types that use ETH as default dispatch core type
static bool is_eth_default_cluster_type(ClusterType cluster_type) {
    return cluster_type == ClusterType::N300 || cluster_type == ClusterType::T3K ||
           cluster_type == ClusterType::N300_2x2;
}

DispatchCoreConfig get_default_dispatch_core_config() {
    // Determine default dispatch core type based on cluster type
    DispatchCoreType default_type = DispatchCoreType::WORKER;
    DispatchCoreAxis default_axis = DispatchCoreAxis::ROW;

    // Check if MetalContext is available (preferred path)
    if (MetalContext::instance_exists(DEFAULT_CONTEXT_ID)) {
        const auto& metal_context = MetalContext::instance();
        const auto cluster_type = metal_context.get_cluster().get_cluster_type();

        // ETH dispatch for N300, T3K, N300_2x2 clusters
        if (is_eth_default_cluster_type(cluster_type)) {
            default_type = DispatchCoreType::ETH;
        }

        // Determine axis: Blackhole + no fabric tensix -> COL, otherwise ROW
        if (metal_context.get_cluster().arch() == tt::ARCH::BLACKHOLE) {
            if (metal_context.get_fabric_tensix_config() == tt_fabric::FabricTensixConfig::DISABLED) {
                default_axis = DispatchCoreAxis::COL;
            } else {
                default_axis = DispatchCoreAxis::ROW;
            }
        } else {
            default_axis = DispatchCoreAxis::ROW;
        }
    } else {
        // MetalContext not available, detect cluster type from descriptor
        // This mirrors the logic in Cluster::get_cluster_type_from_cluster_desc
        auto cluster_desc = tt::umd::Cluster::create_cluster_descriptor();
        if (cluster_desc) {
            // Check for ETH-default clusters (N300, T3K, N300_2x2)
            const auto num_chips = cluster_desc->get_all_chips().size();
            if (num_chips > 0) {
                const auto board_type = cluster_desc->get_board_type(*cluster_desc->get_all_chips().begin());

                // Check if all chips have the same board type
                bool all_same_board = true;
                for (const auto& chip_id : cluster_desc->get_all_chips()) {
                    if (cluster_desc->get_board_type(chip_id) != board_type) {
                        all_same_board = false;
                        break;
                    }
                }

                if (all_same_board && board_type == BoardType::N300) {
                    if (num_chips == 8) {
                        // Could be T3K - verify by checking connections
                        bool is_t3k = true;
                        for (const auto& [chip_id, connections] : cluster_desc->get_ethernet_connections()) {
                            std::unordered_set<ChipId> remote_chips;
                            for (const auto& [channel, remote_chip_and_channel] : connections) {
                                remote_chips.insert(std::get<0>(remote_chip_and_channel));
                            }
                            if (cluster_desc->is_chip_mmio_capable(chip_id)) {
                                if (remote_chips.size() != 3) {
                                    is_t3k = false;
                                    break;
                                }
                            } else {
                                if (remote_chips.size() != 2) {
                                    is_t3k = false;
                                    break;
                                }
                            }
                        }
                        if (is_t3k) {
                            default_type = DispatchCoreType::ETH;
                        }
                    } else if (num_chips == 4) {
                        // Could be N300_2x2 - verify all chips have exactly 2 connections
                        bool is_n300_2x2 = true;
                        for (const auto& [chip_id, connections] : cluster_desc->get_ethernet_connections()) {
                            std::unordered_set<ChipId> remote_chips;
                            for (const auto& [channel, remote_chip_and_channel] : connections) {
                                remote_chips.insert(std::get<0>(remote_chip_and_channel));
                            }
                            if (remote_chips.size() != 2) {
                                is_n300_2x2 = false;
                                break;
                            }
                        }
                        if (is_n300_2x2) {
                            default_type = DispatchCoreType::ETH;
                        }
                    } else if (num_chips == 2) {
                        // N300 single card
                        default_type = DispatchCoreType::ETH;
                    }
                }
            }

            // Determine axis from architecture
            // Get arch from any chip's soc descriptor
            for (const auto& chip_id : cluster_desc->get_all_chips()) {
                auto arch = cluster_desc->get_arch(chip_id);
                if (arch == tt::ARCH::BLACKHOLE) {
                    // For Blackhole without MetalContext, we can't determine fabric tensix config
                    // Default to COL (safer default without MUX)
                    default_axis = DispatchCoreAxis::COL;
                } else {
                    default_axis = DispatchCoreAxis::ROW;
                }
                break;  // Only need to check first chip
            }
        }
    }

    return DispatchCoreConfig(default_type, default_axis);
}

}  // namespace tt::tt_metal

std::size_t std::hash<tt::tt_metal::DispatchCoreConfig>::operator()(
    const tt::tt_metal::DispatchCoreConfig& dispatch_core_config) const {
    return ttsl::hash::hash_objects_with_default_seed(dispatch_core_config.attribute_values());
}
