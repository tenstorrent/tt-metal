// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <filesystem>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include "tt_metal/impl/context/metal_context.hpp"
#include "tools/scaleout/validation/utils/ethernet_link_metrics.hpp"
#include <board/board.hpp>
#include <factory_system_descriptor/query.hpp>
#include <factory_system_descriptor/utils.hpp>

// Forward declarations for in-memory validation
namespace YAML {
class Node;
}

namespace tt::scaleout_tools::fsd::proto {
class FactorySystemDescriptor;
}

namespace tt::scaleout_tools {

using tt::ChipId;
using tt::CoordSystem;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::PhysicalSystemDescriptor;

struct ConnectivityValidationConfig {
    std::filesystem::path output_path;
    std::optional<std::string> cabling_descriptor_path = std::nullopt;
    std::optional<std::string> deployment_descriptor_path = std::nullopt;
    std::optional<std::string> fsd_path = std::nullopt;
    bool fail_on_warning = false;
};

// ============================================================================
// Utility Functions
// ============================================================================

template <typename T1, typename T2>
constexpr std::common_type_t<T1, T2> align_down(T1 value, T2 alignment) {
    static_assert(std::is_integral_v<T1>, "align_down() requires integral types");
    static_assert(std::is_integral_v<T2>, "align_down() requires integral types");
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
    const ConnectivityValidationConfig& validation_config);

void reset_ethernet_links(
    const PhysicalSystemDescriptor& physical_system_descriptor, const tt_metal::AsicTopology& asic_topology);

std::vector<EthChannelIdentifier> collect_retrained_link_identifiers(
    const tt_metal::AsicTopology& missing_topology, const PhysicalSystemDescriptor& physical_system_descriptor);

void log_link_retrain_summary(
    const std::unordered_map<EthChannelIdentifier, uint32_t>& link_retrain_counts,
    uint32_t total_retrain_iterations,
    const std::filesystem::path& output_path);

void log_unretrainable_channels(
    const tt_metal::AsicTopology& missing_topology,
    const PhysicalSystemDescriptor& physical_system_descriptor,
    uint32_t total_retrain_iterations,
    const std::filesystem::path& output_path);

tt_metal::AsicTopology build_reset_topology(
    const std::string& reset_host,
    uint32_t reset_tray_id,
    uint32_t reset_asic_location,
    uint32_t reset_channel,
    PhysicalSystemDescriptor& physical_system_descriptor);

void perform_link_reset(
    const std::string& reset_host,
    uint32_t reset_tray_id,
    uint32_t reset_asic_location,
    uint32_t reset_channel,
    PhysicalSystemDescriptor& physical_system_descriptor);

tt_metal::AsicTopology generate_asic_topology_from_connections(
    const std::set<PhysicalChannelConnection>& physical_connections,
    PhysicalSystemDescriptor& physical_system_descriptor);

fsd::proto::FactorySystemDescriptor get_factory_system_descriptor(
    const std::optional<std::string>& cabling_descriptor_path,
    const std::optional<std::string>& deployment_descriptor_path,
    const std::optional<std::string>& fsd_path,
    const std::vector<std::string>& hostnames);

tt_metal::AsicTopology validate_connectivity(
    const fsd::proto::FactorySystemDescriptor& fsd_proto,
    const YAML::Node& gsd_yaml_node,
    bool fail_on_warning,
    PhysicalSystemDescriptor& physical_system_descriptor,
    std::optional<uint32_t> min_connections = std::nullopt);

// Filter a missing-connections topology down to the links whose two endpoint hosts sit at the given
// hierarchy tier (instance_path common-prefix length == depth), per fsd_query. Used to bring the cluster
// up tier by tier, deepest (closest) first.
tt_metal::AsicTopology filter_topology_by_tier(
    const tt_metal::AsicTopology& topology,
    const FsdQuery& fsd_query,
    uint32_t depth,
    const PhysicalSystemDescriptor& physical_system_descriptor);

// Phased (per-hierarchy-node) discovery for one tier: split the world into subgroups by depth-`depth`
// instance_path prefix (via FsdQuery::hierarchy_partition), discover each subgroup independently on its own
// sub-context, and merge the result into `physical_system_descriptor`. One collective `split` forms every
// subgroup at once; each rank then discovers only within its subgroup.
//
// NOTE: this is the per-subgroup building block. Each rank ends with ITS subgroup's connectivity only —
// assembling a single global PSD across subgroups needs a cross-subgroup gather, and no public PSD
// serialization exists today (see PHASED_DISCOVERY_DESIGN.md). Until that lands, a caller using this must
// scope validation per subgroup rather than validate a global PSD.
void rediscover_by_hierarchy_subgroups(
    PhysicalSystemDescriptor& physical_system_descriptor, const FsdQuery& fsd_query, uint32_t depth);

// Per-subgroup phased bring-up for one tier (option-(b) validation): each iteration collectively splits into
// depth-`depth` hierarchy-node subgroups, discovers each subgroup, and validates/retrains each subgroup's OWN
// tier links (instance_path LCP == depth) — the both-endpoints-discovered guard in validate_fsd_against_gsd
// scopes the result to intra-subgroup connections, so no global PSD is needed. Ranks stay globally lockstepped
// (all_gather of a per-rank converged flag) so the collective split/reset can't deadlock when subgroups
// converge at different rates. Retrained link endpoints accumulate into `link_retrain_counts`. Runs at most
// `max_retrains` reset rounds for this tier; returns the number of reset rounds performed. Requires ethernet
// link retraining support.
uint32_t phased_bring_up_tier(
    const fsd::proto::FactorySystemDescriptor& fsd_proto,
    const FsdQuery& fsd_query,
    uint32_t depth,
    PhysicalSystemDescriptor& physical_system_descriptor,
    uint32_t max_retrains,
    std::optional<uint32_t> min_connections,
    std::unordered_map<EthChannelIdentifier, uint32_t>& link_retrain_counts);

}  // namespace tt::scaleout_tools
