// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>

#include <memory>

namespace tt::umd {
class Cluster;
}

namespace tt::tt_metal {
class Hal;
}

namespace tt::tt_metal::distributed::multihost {
class DistributedContext;
}

namespace tt {
enum class TargetDevice : std::uint8_t;
}

namespace tt::tt_metal {

// Main discovery function - runs physical system discovery and returns a populated PSD
PhysicalSystemDescriptor run_physical_system_discovery(
    tt::umd::Cluster& cluster,
    const std::shared_ptr<distributed::multihost::DistributedContext>& distributed_context,
    tt::TargetDevice target_device_type,
    bool run_global_discovery = true,
    bool run_live_discovery = true);

// Free function to query local ethernet metrics
LocalEthernetMetrics query_local_ethernet_metrics(
    const PhysicalSystemDescriptor& psd, tt::umd::Cluster& cluster, const Hal* hal);

namespace discovery_impl {
// Internal discovery function - runs local discovery only
PhysicalSystemDescriptor run_local_discovery(
    tt::umd::Cluster& cluster,
    const std::shared_ptr<distributed::multihost::DistributedContext>& distributed_context,
    tt::TargetDevice target_device_type,
    bool run_live_discovery);
}  // namespace discovery_impl

}  // namespace tt::tt_metal
