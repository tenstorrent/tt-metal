// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <umd/device/cluster_descriptor.hpp>

#include <memory>

namespace tt::tt_metal {
class Hal;
}

namespace tt::tt_metal::distributed::multihost {
class DistributedContext;
}

namespace tt {
enum class TargetDevice : std::uint8_t;
}

namespace tt::tt_fabric::coordination {
class SystemCoordinator;
}

namespace tt::tt_metal {

// Main discovery function - runs physical system discovery and returns a populated PSD
//
// Option B2-i: when `coordinator` is non-null, the cross-host exchanges (hostname-uniqueness
// resolution and the gather -> controller-process -> scatter of the PhysicalSystemDescriptor)
// are routed through the domain-level SystemCoordinator instead of the raw DistributedContext
// send/recv collectives. This is the injection seam a no-MPI ServiceCoordinator plugs into.
// When null (default), the legacy DistributedContext path is used verbatim.
PhysicalSystemDescriptor run_physical_system_discovery(
    tt::umd::ClusterDescriptor& cluster_desc,
    const std::shared_ptr<distributed::multihost::DistributedContext>& distributed_context,
    tt::TargetDevice target_device_type,
    bool run_global_discovery = true,
    bool run_live_discovery = true,
    const std::shared_ptr<tt::tt_fabric::coordination::SystemCoordinator>& coordinator = nullptr);

// Free function to query local ethernet metrics
LocalEthernetMetrics query_local_ethernet_metrics(
    const PhysicalSystemDescriptor& psd, tt::umd::Cluster& cluster, const Hal* hal);

namespace discovery_impl {
// Internal discovery function - runs local discovery only.
// all_hostnames_unique must be from resolve_hostname_uniqueness() called before this.
PhysicalSystemDescriptor run_local_discovery(
    tt::umd::ClusterDescriptor& cluster_desc,
    const std::shared_ptr<distributed::multihost::DistributedContext>& distributed_context,
    tt::TargetDevice target_device_type,
    bool all_hostnames_unique);

}  // namespace discovery_impl

}  // namespace tt::tt_metal
