// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/fabric/physical_system_descriptor.hpp"

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
// Note: Default arguments are declared in physical_system_descriptor.hpp
PhysicalSystemDescriptor run_physical_system_discovery(
    tt::umd::Cluster& cluster,
    std::shared_ptr<distributed::multihost::DistributedContext> distributed_context,
    const Hal* hal,
    tt::TargetDevice target_device_type,
    bool run_global_discovery,
    bool run_live_discovery);

// Free function to query local ethernet metrics
LocalEthernetMetrics query_local_ethernet_metrics(
    const PhysicalSystemDescriptor& psd, tt::umd::Cluster& cluster, const Hal* hal);

namespace discovery_impl {
// Internal discovery function - runs local discovery only
PhysicalSystemDescriptor run_local_discovery(
    tt::umd::Cluster& cluster,
    std::shared_ptr<distributed::multihost::DistributedContext> distributed_context,
    const Hal* hal,
    tt::TargetDevice target_device_type,
    bool run_live_discovery);
}  // namespace discovery_impl

}  // namespace tt::tt_metal
