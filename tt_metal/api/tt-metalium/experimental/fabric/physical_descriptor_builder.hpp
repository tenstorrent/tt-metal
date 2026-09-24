// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// physical_descriptor_builder.hpp - Offline FactorySystemDescriptor (FSD) -> PhysicalSystemDescriptor (PSD) conversion.
//
// These free functions build a `tt::fabric::proto::PhysicalSystemDescriptor` purely from a
// `tt::scaleout_tools::fsd::proto::FactorySystemDescriptor` (the desired/as-built topology). They perform
// NO hardware/UMD discovery and do not depend on the tt-metal runtime — the intent (see
// https://github.com/tenstorrent/tt-metal/issues/52859) is that tooling such as tt-run / generate_rank_bindings
// can map against a known-good FSD instead of a live-discovered PSD.
//
// Ported from tenstorrent/tt-fabric-manager `controller/physical_system_descriptor_builder.cpp` (the FSD
// overloads only; the HostPhysicalTopologies / hybrid overloads stay in Fabric Manager as they depend on
// FM's runtime-discovery types). The produced PSD proto is the exact type consumed by tt-metal's
// `tt::tt_metal::PhysicalSystemDescriptor(const tt::fabric::proto::PhysicalSystemDescriptor&)` bridge and the
// topology mapper.

#pragma once

#include <string>
#include <vector>

#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>  // tt::tt_metal::PhysicalSystemDescriptor

// The generated proto headers are private (not part of the installed public API), so they are not included here.
// The types below are only named by reference / returned by value, so forward declarations suffice; callers that
// consume the returned proto must include the generated headers themselves (the .cpp does).
namespace tt::scaleout_tools::fsd::proto {
class FactorySystemDescriptor;
}  // namespace tt::scaleout_tools::fsd::proto
namespace tt::fabric::proto {
class PhysicalSystemDescriptor;
}  // namespace tt::fabric::proto

// Experimental API — under tt-metalium/experimental, so it lives in the experimental namespace.
namespace tt::tt_metal::experimental::tt_fabric {

// Convenience, proto-free entry point: parse an FSD textproto file and return the ready-to-use C++
// PhysicalSystemDescriptor. Consumers with only a file path (e.g. tt-run / generate_rank_bindings) can use this
// and need no protobuf headers on their include path.
//
// host_filter (optional): restrict the descriptor to these hostnames before building. In the wild an FSD often
// covers a whole superpod (e.g. SC36) or an aggregated datacenter (exabox), so a consumer wanting a single pod
// passes just that pod's hostnames. Empty = no filter (use the whole FSD). Throws if a requested hostname is
// absent from the FSD.
::tt::tt_metal::PhysicalSystemDescriptor build_physical_descriptor_from_file(
    const std::string& fsd_path, const std::vector<std::string>& host_filter = {});

// Parse a Factory System Descriptor textproto file from disk.
// Throws std::runtime_error if the file cannot be read or parsed.
::tt::scaleout_tools::fsd::proto::FactorySystemDescriptor load_factory_descriptor(const std::string& path);

// Restrict an FSD to a subset of hosts by hostname, densely renumbering host_ids and filtering board_types /
// eth_connections to match (connections touching a filtered-out host are dropped). Useful for carving a single
// pod out of a superpod/datacenter FSD before building. Empty hostnames returns the FSD unchanged; throws if a
// requested hostname is not present.
::tt::scaleout_tools::fsd::proto::FactorySystemDescriptor filter_factory_descriptor(
    const ::tt::scaleout_tools::fsd::proto::FactorySystemDescriptor& fsd, const std::vector<std::string>& hostnames);

// Build a PhysicalSystemDescriptor proto from a FactorySystemDescriptor (desired-state), fully offline.
//
// Only the fields the topology mapper needs are populated:
//   - asic_descriptors: tray_id, asic_location, board_type, host_name, and a synthesized unique_id stable
//     across runs (keyed by host_id/tray_id/asic_location).
//   - system_graph.asic_connectivity_graph: bidirectional edges from eth_connections, with is_local derived
//     from whether the two endpoints share a host_id.
//   - system_graph.host_connectivity_graph + exit_node_connection_table: inter-host connections.
//   - host_to_mobo_name: from FSD Host.motherboard.
//   - host_to_rank: assigned as the FSD host index (hosts[i] -> rank i).
// Runtime-only fields (ethernet_firmware_version, pcie_devices_per_tray, pcie_id_to_asic_location,
// target_device_type, umd_unique_id) are left unset.
//
// The ASIC set is the union of endpoints in eth_connections; ASICs declared in board_types but with zero
// eth_connections do not appear (the mapper has no edges to place on them).
//
// Throws std::runtime_error if an eth_connection endpoint references a host_id outside the hosts[] array, or
// if a referenced (host_id, tray_id) has no matching entry in board_types.
::tt::fabric::proto::PhysicalSystemDescriptor build_physical_descriptor(
    const ::tt::scaleout_tools::fsd::proto::FactorySystemDescriptor& fsd);

// Build one PhysicalSystemDescriptor per connected component of FSD hosts.
//
// Hosts are partitioned into disjoint connected components (two hosts are connected when an eth_connection
// has its endpoints on different hosts, directly or transitively). Each returned PSD covers exactly one
// component: a sub-FSD restricted to that component (with host_ids densely renumbered) is fed through
// build_physical_descriptor above, so each descriptor is self-consistent and its host_to_rank is 0-based within the
// group. Components are ordered deterministically by their group's lowest member host name.
std::vector<::tt::fabric::proto::PhysicalSystemDescriptor> build_physical_descriptors(
    const ::tt::scaleout_tools::fsd::proto::FactorySystemDescriptor& fsd);

}  // namespace tt::tt_metal::experimental::tt_fabric
