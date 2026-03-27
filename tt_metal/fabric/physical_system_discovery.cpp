// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include "tt_metal/fabric/physical_system_discovery.hpp"
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>

#include <unistd.h>
#include <climits>
#include <fstream>
#include <algorithm>
#include <set>
#include <unordered_map>

#include <umd/device/cluster.hpp>
#include <umd/device/soc_descriptor.hpp>
#include <tt-metalium/distributed_context.hpp>
#include "tt_metal/llrt/tunnels_from_mmio_device.hpp"
#include "tt_metal/llrt/hal.hpp"
#include "tt_metal/fabric/serialization/physical_system_descriptor_serialization.hpp"
#include "tt_metal/fabric/fabric_host_utils.hpp"

namespace tt::tt_metal {

namespace {

std::string get_mobo_name() {
    std::ifstream file("/sys/class/dmi/id/board_name");
    std::string motherboard;

    if (file.is_open()) {
        std::getline(file, motherboard);
        file.close();
    }

    return motherboard;
}

TrayID get_tray_id_for_chip(
    tt::umd::Cluster& cluster, ChipId chip_id, const std::string& mobo_name, bool using_mock_cluster_desc) {
    static const std::unordered_map<std::string, std::vector<uint16_t>> mobo_to_bus_ids = {
        {"SIENAD8-2L2T", {0xc1, 0x01, 0x41, 0x42}},
        {"X12DPG-QT6", {0xb1, 0xca, 0x31, 0x4b}},
        {"H13DSG-O-CPU", {0x01, 0x21, 0x41, 0x61, 0x81, 0xa1, 0xc1, 0xe1}},
    };

    if (using_mock_cluster_desc || !mobo_to_bus_ids.contains(mobo_name)) {
        return TrayID{0};
    }
    const auto& ordered_bus_ids = mobo_to_bus_ids.at(mobo_name);
    auto bus_id = tt::tt_fabric::get_bus_id(cluster, chip_id);
    auto bus_id_it = std::find(ordered_bus_ids.begin(), ordered_bus_ids.end(), bus_id);
    TT_FATAL(bus_id_it != ordered_bus_ids.end(), "Bus ID {} not found.", bus_id);
    auto tray_id = std::distance(ordered_bus_ids.begin(), bus_id_it) + 1;
    return TrayID{static_cast<unsigned int>(tray_id)};
}

std::pair<TrayID, ASICLocation> get_asic_position(
    tt::umd::Cluster& cluster,
    ChipId chip_id,
    bool using_mock_cluster_desc,
    std::unordered_map<uint32_t, std::unordered_set<uint32_t>>& pcie_devices_per_tray,
    std::unordered_map<uint32_t, ASICLocation>& pcie_id_to_asic_location) {
    auto* cluster_desc = cluster.get_cluster_description();
    if (cluster_desc->get_board_type(chip_id) == BoardType::UBB_WORMHOLE ||
        cluster_desc->get_board_type(chip_id) == BoardType::UBB_BLACKHOLE) {
        constexpr std::string_view ubb_mobo_name = "S7T-MB";

        TT_FATAL(
            using_mock_cluster_desc || get_mobo_name() == ubb_mobo_name, "UBB systems must use S7T-MB motherboard.");
        auto ubb_id = tt::tt_fabric::get_ubb_id(cluster, chip_id);
        auto pcie_id = cluster_desc->get_chips_with_mmio().at(chip_id);
        pcie_devices_per_tray[ubb_id.tray_id].insert(pcie_id);
        pcie_id_to_asic_location[pcie_id] = ASICLocation{ubb_id.asic_id};
        return {TrayID{ubb_id.tray_id}, ASICLocation{ubb_id.asic_id}};
    }
    auto tray_id = get_tray_id_for_chip(cluster, chip_id, get_mobo_name(), using_mock_cluster_desc);
    ASICLocation asic_location;
    tt::ARCH arch = cluster_desc->get_arch(chip_id);
    if (arch == tt::ARCH::WORMHOLE_B0) {
        // Derive ASIC Location based on the tunnel depth for Wormhole systems
        // TODO: Remove this once UMD populates the ASIC Location for WH systems.
        auto mmio_device = cluster_desc->get_closest_mmio_capable_chip(chip_id);
        auto tunnels_from_mmio_device = llrt::discover_tunnels_from_mmio_device(cluster);
        const auto& tunnels = tunnels_from_mmio_device.at(mmio_device);
        for (const auto& devices_on_tunnel : tunnels) {
            auto device_it = std::find(devices_on_tunnel.begin(), devices_on_tunnel.end(), chip_id);
            if (device_it != devices_on_tunnel.end()) {
                asic_location = ASICLocation{static_cast<unsigned int>(device_it - devices_on_tunnel.begin())};
                break;
            }
        }
    } else if (arch == tt::ARCH::BLACKHOLE || arch == tt::ARCH::QUASAR) {
        // Query ASIC Location from the Cluster Descriptor for BH/QUASAR.
        asic_location = ASICLocation{cluster_desc->get_asic_location(chip_id)};
    } else {
        TT_THROW("Unrecognized Architecture. Cannot determine asic location.");
    }
    return {tray_id, asic_location};
}

bool resolve_hostname_uniqueness(
    const std::shared_ptr<distributed::multihost::DistributedContext>& distributed_context) {
    using namespace tt::tt_metal::distributed::multihost;
    constexpr uint32_t controller_rank = 0;
    auto my_rank = *(distributed_context->rank());

    bool all_hostnames_unique = true;
    if (my_rank == controller_rank) {
        std::vector<std::string> hostnames = {};
        hostnames.push_back(get_host_name());
        for (std::size_t rank = 0; rank < *(distributed_context->size()); rank++) {
            if (rank != controller_rank) {
                std::size_t peer_hostname_size = 0;
                distributed_context->recv(
                    tt::stl::Span<std::byte>(
                        reinterpret_cast<std::byte*>(&peer_hostname_size), sizeof(peer_hostname_size)),
                    Rank{static_cast<int>(rank)},
                    Tag{0});
                std::vector<uint8_t> serialized_peer_hostname(peer_hostname_size);
                distributed_context->recv(
                    tt::stl::as_writable_bytes(
                        tt::stl::Span<uint8_t>(serialized_peer_hostname.data(), serialized_peer_hostname.size())),
                    Rank{static_cast<int>(rank)},
                    Tag{0});

                hostnames.push_back(std::string(serialized_peer_hostname.begin(), serialized_peer_hostname.end()));
            }
        }
        all_hostnames_unique = std::set<std::string>(hostnames.begin(), hostnames.end()).size() == hostnames.size();

        for (std::size_t rank = 0; rank < *(distributed_context->size()); rank++) {
            if (rank != controller_rank) {
                distributed_context->send(
                    tt::stl::Span<std::byte>(
                        reinterpret_cast<std::byte*>(&all_hostnames_unique), sizeof(all_hostnames_unique)),
                    Rank{static_cast<int>(rank)},
                    Tag{0});
            }
        }
    } else {
        auto host_name = get_host_name();
        auto serialized_hostname = std::vector<uint8_t>(host_name.begin(), host_name.end());
        std::size_t serialized_hostname_size = serialized_hostname.size();
        distributed_context->send(
            tt::stl::Span<std::byte>(
                reinterpret_cast<std::byte*>(&serialized_hostname_size), sizeof(serialized_hostname_size)),
            Rank{controller_rank},
            Tag{0});
        distributed_context->send(
            tt::stl::as_writable_bytes(tt::stl::Span<uint8_t>(serialized_hostname.data(), serialized_hostname.size())),
            Rank{controller_rank},
            Tag{0});

        distributed_context->recv(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&all_hostnames_unique), sizeof(all_hostnames_unique)),
            Rank{controller_rank},
            Tag{0});
    }
    return all_hostnames_unique;
}

uint32_t get_chip_id_for_asic(const umd::ClusterDescriptor& cluster_desc, AsicID asic_id) {
    const auto& chip_unique_ids = cluster_desc.get_chip_unique_ids();
    for (const auto& [chip_id, unique_id] : chip_unique_ids) {
        if (unique_id == *asic_id) {
            return chip_id;
        }
    }
    TT_FATAL(false, "Chip ID not found for asic ID {}", asic_id);
    return 0;
}

void validate_eth_fw_versions(
    PhysicalSystemDescriptor& psd,
    const tt::umd::semver_t& peer_ethernet_firmware_version,
    const std::string& my_host_name,
    const std::string& peer_host_name) {
    TT_FATAL(
        peer_ethernet_firmware_version == psd.get_ethernet_firmware_version(),
        "Ethernet Firmware Versions are expected to be consistent across all nodes in the cluster. Hosts: {} and {} "
        "have different Ethernet Firmware Versions: {} and {}.",
        my_host_name,
        peer_host_name,
        psd.get_ethernet_firmware_version().to_string(),
        peer_ethernet_firmware_version.to_string());
}

void remove_unresolved_nodes(PhysicalSystemDescriptor& psd) {
    auto& asic_descriptors = psd.get_asic_descriptors();
    auto& system_graph = psd.get_system_graph();
    auto& exit_node_connection_table = psd.get_exit_node_connection_table();

    for (auto& [host, asic_group] : system_graph.asic_connectivity_graph) {
        for (auto& [src_asic, edges] : asic_group) {
            std::erase_if(edges, [&](const auto& pair) { return not asic_descriptors.contains(pair.first); });
        }
    }

    for (auto& [host, exit_nodes] : exit_node_connection_table) {
        std::erase_if(exit_nodes, [&](const auto& exit_node) {
            return asic_descriptors.find(exit_node.src_exit_node) == asic_descriptors.end() ||
                   asic_descriptors.find(exit_node.dst_exit_node) == asic_descriptors.end();
        });
    }
}

void generate_cross_host_connections(PhysicalSystemDescriptor& psd) {
    auto& exit_node_connection_table = psd.get_exit_node_connection_table();
    auto& system_graph = psd.get_system_graph();

    for (const auto& [host, exit_nodes] : exit_node_connection_table) {
        std::unordered_map<std::string, size_t> visited_hosts;
        for (const auto& [candidate_host, candidate_exit_nodes] : exit_node_connection_table) {
            if (host == candidate_host) {
                continue;  // Skip self connections
            }
            for (const auto& exit_node : exit_nodes) {
                for (const auto& candidate_node : candidate_exit_nodes) {
                    if (exit_node.src_exit_node == candidate_node.dst_exit_node &&
                        candidate_node.src_exit_node == exit_node.dst_exit_node &&
                        exit_node.eth_conn.src_chan == candidate_node.eth_conn.dst_chan &&
                        exit_node.eth_conn.dst_chan == candidate_node.eth_conn.src_chan) {
                        if (!visited_hosts.contains(candidate_host)) {
                            system_graph.host_connectivity_graph[host].push_back({candidate_host, {exit_node}});
                            visited_hosts[candidate_host] = system_graph.host_connectivity_graph[host].size() - 1;
                        } else {
                            system_graph.host_connectivity_graph[host][visited_hosts[candidate_host]].second.push_back(
                                exit_node);
                        }
                        break;
                    }
                }
            }
        }
    }
}

void validate_graphs(PhysicalSystemDescriptor& psd) {
    // Validate that the representation of the system is internally consistent.
    const auto& asic_descriptors = psd.get_asic_descriptors();
    auto& system_graph = psd.get_system_graph();

    for (auto& [host, asic_group] : system_graph.asic_connectivity_graph) {
        for (auto& [src_asic, edges] : asic_group) {
            // Skip if src_asic doesn't exist (shouldn't happen, but be defensive)
            if (!asic_descriptors.contains(src_asic)) {
                continue;
            }
            const auto& src_host = asic_descriptors.at(src_asic).host_name;

            // Skip if host_connectivity_graph doesn't have src_host (shouldn't happen, but be defensive)
            if (!system_graph.host_connectivity_graph.contains(src_host)) {
                continue;
            }
            const auto& src_host_edges = system_graph.host_connectivity_graph.at(src_host);

            for (auto& [dst_asic, eth_conns] : edges) {
                // Skip if dst_asic doesn't exist (shouldn't happen, but be defensive)
                if (!asic_descriptors.contains(dst_asic)) {
                    continue;
                }
                const auto& dst_host = asic_descriptors.at(dst_asic).host_name;

                bool all_local = std::all_of(
                    eth_conns.begin(), eth_conns.end(), [](const EthConnection& conn) { return conn.is_local; });

                bool all_global = std::all_of(
                    eth_conns.begin(), eth_conns.end(), [](const EthConnection& conn) { return !conn.is_local; });

                // All connections must be uniformly local or global.
                TT_FATAL(
                    all_local || all_global,
                    "Physical Discovery Error: All ethernet connections should either be local or global. "
                    "Please reset the system and try again.");

                if (all_local) {
                    // Local connections must remain within the same host.
                    TT_FATAL(
                        src_host == dst_host,
                        "Physical Discovery Error: Local Connection between {} and {} is not on the same host. "
                        "Please reset the system and try again.",
                        src_host,
                        dst_host);
                    continue;  // no need to check further
                }

                // Global connections must cross hosts
                TT_FATAL(
                    src_host != dst_host,
                    "Physical Discovery Error: Hostnames for connections marked as global should be different. "
                    "Please reset the system and try again.");

                // Validate each global ethernet connection.
                for (const auto& eth_conn : eth_conns) {
                    // Look for a host edge matching dst_host.
                    auto host_edge_it =
                        std::find_if(src_host_edges.begin(), src_host_edges.end(), [&](const auto& host_edge) {
                            return host_edge.first == dst_host;
                        });

                    TT_FATAL(
                        host_edge_it != src_host_edges.end(),
                        "Physical Discovery Error: Global Connection between {} and {} is not found in the host "
                        "connectivity graph. Please reset the system and try again.",
                        src_host,
                        dst_host);

                    const auto& exit_node_conns = host_edge_it->second;
                    bool exit_conn_found = std::any_of(
                        exit_node_conns.begin(), exit_node_conns.end(), [&](const ExitNodeConnection& exit_node_conn) {
                            return exit_node_conn.src_exit_node == src_asic &&
                                   exit_node_conn.dst_exit_node == dst_asic &&
                                   exit_node_conn.eth_conn.src_chan == eth_conn.src_chan &&
                                   exit_node_conn.eth_conn.dst_chan == eth_conn.dst_chan;
                        });

                    TT_FATAL(
                        exit_conn_found,
                        "Physical Discovery Error: Global Connection between {} and {} is not found in the "
                        "host connectivity graph. Please reset the system and try again.",
                        src_host,
                        dst_host);
                }
            }
        }
    }
}

void exchange_metadata(
    PhysicalSystemDescriptor& psd,
    const std::shared_ptr<distributed::multihost::DistributedContext>& distributed_context,
    bool issue_gather) {
    using namespace tt::tt_metal::distributed::multihost;
    constexpr uint32_t controller_rank = 0;
    if (*(distributed_context->size()) == 1) {
        return;
    }
    auto my_rank = *(distributed_context->rank());
    std::set<uint32_t> sender_ranks;
    std::set<uint32_t> receiver_ranks;

    if (issue_gather) {
        receiver_ranks.insert(controller_rank);
        for (std::size_t rank = 0; rank < *(distributed_context->size()); rank++) {
            if (rank != controller_rank) {
                sender_ranks.insert(rank);
            }
        }
    } else {
        sender_ranks.insert(controller_rank);
        for (std::size_t rank = 0; rank < *(distributed_context->size()); rank++) {
            if (rank != controller_rank) {
                receiver_ranks.insert(rank);
            }
        }
    }

    if (sender_ranks.contains(my_rank)) {
        auto serialized_desc = serialize_physical_system_descriptor_to_bytes(psd);
        std::size_t desc_size = serialized_desc.size();

        for (auto rank : receiver_ranks) {
            distributed_context->send(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&desc_size), sizeof(desc_size)),
                Rank{static_cast<int>(rank)},
                Tag{0});

            distributed_context->send(
                tt::stl::as_writable_bytes(tt::stl::Span<uint8_t>(serialized_desc.data(), serialized_desc.size())),
                Rank{static_cast<int>(rank)},
                Tag{0});
        }
    } else {
        for (auto rank : sender_ranks) {
            std::size_t peer_descriptor_size = 0;
            distributed_context->recv(
                tt::stl::Span<std::byte>(
                    reinterpret_cast<std::byte*>(&peer_descriptor_size), sizeof(peer_descriptor_size)),
                Rank{static_cast<int>(rank)},
                Tag{0});
            std::vector<uint8_t> serialized_peer_desc(peer_descriptor_size);
            distributed_context->recv(
                tt::stl::as_writable_bytes(
                    tt::stl::Span<uint8_t>(serialized_peer_desc.data(), serialized_peer_desc.size())),
                Rank{static_cast<int>(rank)},
                Tag{0});
            auto peer_desc = deserialize_physical_system_descriptor_from_bytes(serialized_peer_desc);

            // Check for empty asic_descriptors before accessing
            TT_FATAL(
                !psd.get_asic_descriptors().empty() && !peer_desc.get_asic_descriptors().empty(),
                "Cannot exchange metadata: empty ASIC descriptors");

            validate_eth_fw_versions(
                psd,
                peer_desc.get_ethernet_firmware_version(),
                psd.get_asic_descriptors().begin()->second.host_name,
                peer_desc.get_asic_descriptors().begin()->second.host_name);
            psd.merge(std::move(peer_desc));
        }
    }
    distributed_context->barrier();
}

}  // namespace

namespace discovery_impl {

PhysicalSystemDescriptor run_local_discovery(
    tt::umd::Cluster& cluster,
    const std::shared_ptr<distributed::multihost::DistributedContext>& distributed_context,
    tt::TargetDevice target_device_type,
    bool run_live_discovery,
    bool all_hostnames_unique) {
    PhysicalSystemDescriptor psd(target_device_type);

    std::unique_ptr<umd::ClusterDescriptor> cluster_desc = nullptr;
    if (!run_live_discovery || target_device_type != TargetDevice::Silicon) {
        cluster_desc = std::make_unique<tt::umd::ClusterDescriptor>(*cluster.get_cluster_description());
    } else {
        // As part of live discovery, we create a new cluster descriptor to query the latest state from UMD.
        cluster_desc = tt::umd::Cluster::create_cluster_descriptor();
    }
    const auto& chip_unique_ids = cluster_desc->get_chip_unique_ids();
    const auto& eth_connections = cluster_desc->get_ethernet_connections();
    auto cross_host_eth_connections = cluster_desc->get_ethernet_connections_to_remote_devices();

    auto my_rank = *(distributed_context->rank());
    auto hostname = get_host_name();

    // When multiple ranks exist and hostnames are not unique (e.g. mock, same machine), use hostname_rank
    // so each rank gets its own entry during merge. When hostnames are unique (different machines),
    // use hostname so graph keys match my_host_name() for lookups (e.g. get_host_neighbors).
    auto hostname_key = (*(distributed_context->size()) > 1 && !all_hostnames_unique)
                            ? (hostname + "_" + std::to_string(my_rank))
                            : hostname;

    // Set local hostname and rank (friend access allows direct access to private members)
    psd.get_host_mobo_name_map()[hostname_key] = get_mobo_name();
    psd.get_host_to_rank_map()[hostname_key] = my_rank;

    auto& asic_graph = psd.get_system_graph().asic_connectivity_graph[hostname_key];
    auto& exit_nodes = psd.get_exit_node_connection_table()[hostname_key];

    auto add_local_asic_descriptor = [&](AsicID src_unique_id, ChipId src_chip_id) {
        auto [tray_id, asic_location] = get_asic_position(
            cluster,
            src_chip_id,
            target_device_type != TargetDevice::Silicon,
            psd.get_pcie_devices_per_tray()[hostname_key],
            psd.get_pcie_id_to_asic_location()[hostname_key]);
        psd.get_asic_descriptors()[src_unique_id] = ASICDescriptor{
            TrayID{tray_id},
            asic_location,
            cluster_desc->get_board_type(src_chip_id),
            src_unique_id,
            src_chip_id,
            hostname_key};
    };

    for (const auto& [chip_id, unique_id] : chip_unique_ids) {
        add_local_asic_descriptor(AsicID{unique_id}, chip_id);
        asic_graph[AsicID{unique_id}] = {};
    }
    for (const auto& [src, conn] : eth_connections) {
        auto src_unique_id = AsicID{chip_unique_ids.at(src)};
        // Populate ASIC Descriptor with Physical Information
        add_local_asic_descriptor(src_unique_id, src);
        std::unordered_map<ChipId, size_t> visited_dst;
        // Populate ASIC Graph for Current Host
        for (const auto& [chan, dst] : conn) {
            auto dst_chip = std::get<0>(dst);
            auto dst_chan = std::get<1>(dst);
            if (!visited_dst.contains(dst_chip)) {
                // This neighbor has not been visited. Add it to the graph and mark visited.
                asic_graph[src_unique_id].push_back(
                    {AsicID{chip_unique_ids.at(dst_chip)}, {EthConnection(chan, dst_chan, true)}});
                visited_dst[dst_chip] = asic_graph[src_unique_id].size() - 1;
            } else {
                // This neighbor has already been visited. There is more than one channel to it.
                // Update the existing entry with the new channel.
                asic_graph[src_unique_id][visited_dst[dst_chip]].second.push_back(EthConnection(chan, dst_chan, true));
            }
        }
    }

    // Populate exit nodes for cross-host connections
    for (const auto& [local_chip_id, eth_link_info] : cross_host_eth_connections) {
        auto local_unique_id = AsicID{chip_unique_ids.at(local_chip_id)};
        // This ASIC has no local ethernet connections, but is connected to this host
        // and to a remote host. Add it to the ASIC Descriptor List.
        if (!psd.get_asic_descriptors().contains(local_unique_id)) {
            add_local_asic_descriptor(local_unique_id, local_chip_id);
        }
        std::unordered_map<AsicID, size_t> visited_dst;
        for (const auto& [eth_chan, remote_info] : eth_link_info) {
            auto dst_unique_id = AsicID{std::get<0>(remote_info)};
            auto dst_chan = std::get<1>(remote_info);
            if (!visited_dst.contains(dst_unique_id)) {
                asic_graph[local_unique_id].push_back({dst_unique_id, {EthConnection(eth_chan, dst_chan, false)}});
                visited_dst[dst_unique_id] = asic_graph[local_unique_id].size() - 1;
            } else {
                asic_graph[local_unique_id][visited_dst[dst_unique_id]].second.push_back(
                    EthConnection(eth_chan, dst_chan, false));
            }
            exit_nodes.push_back(ExitNodeConnection{
                .src_exit_node = local_unique_id,
                .dst_exit_node = dst_unique_id,
                .eth_conn = EthConnection(eth_chan, dst_chan, false)});
        }
    }

    psd.get_system_graph().host_connectivity_graph[hostname_key] = {};
    // Get Ethernet Firmware Version from the driver - Initialize to 0 if not available
    psd.get_ethernet_firmware_version() = cluster.get_ethernet_firmware_version().value_or(tt::umd::semver_t(0, 0, 0));

    return psd;
}

}  // namespace discovery_impl

PhysicalSystemDescriptor run_physical_system_discovery(
    tt::umd::Cluster& cluster,
    const std::shared_ptr<distributed::multihost::DistributedContext>& distributed_context,
    tt::TargetDevice target_device_type,
    bool run_global_discovery,
    bool run_live_discovery) {
    // Barrier to ensure all MPI ranks are synchronized and ready to communicate.
    distributed_context->barrier();

    // Resolve hostname uniqueness before discovery so run_local_discovery can use the right key
    // (hostname when unique, hostname_rank when not), matching my_host_name() for lookups.
    bool all_hostnames_unique = resolve_hostname_uniqueness(distributed_context);
    auto psd = discovery_impl::run_local_discovery(
        cluster, distributed_context, target_device_type, run_live_discovery, all_hostnames_unique);

    // Set local hostname and rank (friend access)
    auto my_rank = *(distributed_context->rank());
    auto hostname = get_host_name();
    psd.set_discovery_data(hostname, my_rank, all_hostnames_unique);

    if (run_global_discovery) {
        exchange_metadata(psd, distributed_context, true);
        auto my_rank_val = *(distributed_context->rank());
        constexpr uint32_t controller_rank = 0;
        if (my_rank_val == controller_rank) {
            remove_unresolved_nodes(psd);
            generate_cross_host_connections(psd);
            validate_graphs(psd);

            // With multi-rank (size > 1), run_local_discovery uses hostname_rank keys from the start,
            // so asic_connectivity_graph, exit_node_connection_table, and host_connectivity_graph
            // already have the correct keys. No rename needed.
        }
        exchange_metadata(psd, distributed_context, false);
    }

    return psd;
}

LocalEthernetMetrics query_local_ethernet_metrics(
    const PhysicalSystemDescriptor& psd, tt::umd::Cluster& cluster, const Hal* hal) {
    const auto& local_asics = psd.get_asics_connected_to_host(psd.my_host_name());
    const auto& local_asic_graph = psd.get_asic_topology(psd.my_host_name());
    std::unordered_map<AsicID, std::unordered_map<uint8_t, EthernetMetrics>> local_ethernet_metrics;

    auto retrain_count_addr = hal->get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::RETRAIN_COUNT);
    auto crc_addr =
        hal->get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::CRC_ERR);
    auto corr_addr =
        hal->get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::CORR_CW);
    auto uncorr_addr =
        hal->get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNCORR_CW);

    auto* cluster_desc = cluster.get_cluster_description();
    bool arch_blackhole = cluster_desc->get_arch(0) == tt::ARCH::BLACKHOLE;
    // Memory layout for 64B metrics is different on WH vs BH systems/
    uint64_t hi_offset = arch_blackhole ? sizeof(uint32_t) : 0;
    uint64_t lo_offset = arch_blackhole ? 0 : sizeof(uint32_t);

    for (const auto& asic : local_asics) {
        const auto& asic_connections = local_asic_graph.at(asic);
        for (const auto& [dst_asic, eth_connections] : asic_connections) {
            for (const auto& eth_connection : eth_connections) {
                uint32_t retrain_count_val = 0, crc_error_val = 0, corr_val_lo = 0, corr_val_hi = 0, uncorr_val_lo = 0,
                         uncorr_val_hi = 0;

                auto src_eth_chan = eth_connection.src_chan;
                auto src_chip_id = get_chip_id_for_asic(*cluster_desc, asic);
                const auto& soc_desc = cluster.get_soc_descriptor(src_chip_id);
                const auto& translated_eth_core =
                    soc_desc.get_eth_core_for_channel(src_eth_chan, CoordSystem::TRANSLATED);

                cluster.read_from_device(
                    &retrain_count_val, src_chip_id, translated_eth_core, retrain_count_addr, sizeof(uint32_t));
                cluster.read_from_device(&crc_error_val, src_chip_id, translated_eth_core, crc_addr, sizeof(uint32_t));
                cluster.read_from_device(
                    &corr_val_hi, src_chip_id, translated_eth_core, corr_addr + hi_offset, sizeof(uint32_t));
                cluster.read_from_device(
                    &corr_val_lo, src_chip_id, translated_eth_core, corr_addr + lo_offset, sizeof(uint32_t));
                cluster.read_from_device(
                    &uncorr_val_hi, src_chip_id, translated_eth_core, uncorr_addr + hi_offset, sizeof(uint32_t));
                cluster.read_from_device(
                    &uncorr_val_lo, src_chip_id, translated_eth_core, uncorr_addr + lo_offset, sizeof(uint32_t));

                local_ethernet_metrics[asic][src_eth_chan] = {
                    .retrain_count = retrain_count_val,
                    .crc_error_count = crc_error_val,
                    .corrected_codeword_count =
                        (static_cast<uint64_t>(corr_val_hi) << 32) | static_cast<uint64_t>(corr_val_lo),
                    .uncorrected_codeword_count =
                        (static_cast<uint64_t>(uncorr_val_hi) << 32) | static_cast<uint64_t>(uncorr_val_lo)};
            }
        }
    }
    return local_ethernet_metrics;
}

}  // namespace tt::tt_metal
