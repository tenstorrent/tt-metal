// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <filesystem>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include "tools/scaleout/validation/utils/ethernet_link_metrics.hpp"
#include "board/board.hpp"
#include <factory_system_descriptor/utils.hpp>

namespace tt::scaleout_tools {

using tt::ChipId;
using tt::CoordSystem;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::PhysicalSystemDescriptor;


enum class ResetSequence {
    PORT_DOWN,
    LINK_DOWN,
};

enum class ResetAction {
    PORT_ACTION_UP,
    PORT_ACTION_DOWN,
    PORT_RETRAIN,
    PORT_RETRAIN_FORCE,
    PORT_REINIT,
    PORT_REINIT_FORCE,
};

// ============================================================================
// Utility Functions
// ============================================================================

template <typename T1, typename T2>
constexpr std::common_type_t<T1, T2> align_down(T1 value, T2 alignment) {
    static_assert(std::is_integral<T1>::value, "align_down() requires integral types");
    static_assert(std::is_integral<T2>::value, "align_down() requires integral types");
    using T = std::common_type_t<T1, T2>;
    return static_cast<T>(value) & ~(static_cast<T>(alignment) - 1);
}

void log_output_rank0(const std::string& message);

// ============================================================================
// Logging Functions (Metrics and Connectivity)
// ============================================================================

void print_ethernet_connectivity(
    bool print_connectivity, const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor);

// ============================================================================
// Link Metrics Generation
// ============================================================================

bool generate_link_metrics(
    PhysicalSystemDescriptor& physical_system_descriptor,
    uint32_t num_iterations,
    bool log_ethernet_metrics,
    bool send_traffic,
    bool sweep_traffic_configs,
    uint32_t packet_size_bytes,
    uint32_t data_size,
    const std::filesystem::path& output_path);

void reset_ethernet_links(
    const PhysicalSystemDescriptor& physical_system_descriptor, const tt_metal::AsicTopology& asic_topology);

void reset_ethernet_links_bh(const tt_metal::AsicTopology& asic_topology, uint32_t reset_step);

void issue_reset_step(const EthChannelIdentifier& link, const ResetAction& reset_action);

tt_metal::AsicTopology generate_asic_topology_from_connections(
    const std::set<PhysicalChannelConnection>& physical_connections,
    PhysicalSystemDescriptor& physical_system_descriptor);

}  // namespace tt::scaleout_tools
