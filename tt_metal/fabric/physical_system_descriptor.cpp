// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <set>
#include <fstream>

#include <umd/device/cluster.hpp>
#include <umd/device/soc_descriptor.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-logger/tt-logger.hpp>

#include "tt_metal/llrt/tunnels_from_mmio_device.hpp"
#include "tt_metal/llrt/hal.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include "tt_metal/fabric/serialization/physical_system_descriptor_serialization.hpp"

namespace tt::tt_metal {

const std::unique_ptr<tt::umd::Cluster> PhysicalSystemDescriptor::null_cluster = nullptr;

/**************************************************************************************************
 Discovery helper functions
**************************************************************************************************/

namespace {

std::string get_host_name() {
    char hostname[HOST_NAME_MAX + 1];
    gethostname(hostname, sizeof(hostname));
    return std::string(hostname);
}

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
    tt::umd::Cluster& cluster,
    ChipId chip_id,
    const std::string& mobo_name,
    bool using_mock_cluster_desc) {
    static const std::unordered_map<std::string, std::vector<uint16_t>> mobo_to_bus_ids = {
        {"SIENAD8-2L2T", {0xc1, 0x01, 0x41, 0x42}},
        {"X12DPG-QT6", {0xb1, 0xca, 0x31, 0x4b}},
    };

    if (using_mock_cluster_desc || mobo_to_bus_ids.find(mobo_name) == mobo_to_bus_ids.end()) {
        return TrayID{0};
    }
    const auto& ordered_bus_ids = mobo_to_bus_ids.at(mobo_name);
    auto bus_id = tt::tt_fabric::get_bus_id(cluster, chip_id);
    auto bus_id_it = std::find(ordered_bus_ids.begin(), ordered_bus_ids.end(), bus_id);
    TT_FATAL(bus_id_it != ordered_bus_ids.end(), "Bus ID {} not found.", bus_id);
    auto tray_id = std::distance(ordered_bus_ids.begin(), bus_id_it) + 1;
    return TrayID{tray_id};
}

std::pair<TrayID, ASICLocation> get_asic_position(
    tt::umd::Cluster& cluster, ChipId chip_id, bool using_mock_cluster_desc) {
    auto cluster_desc = cluster.get_cluster_description();
    if (cluster_desc->get_board_type(chip_id) == BoardType::UBB_WORMHOLE ||
        cluster_desc->get_board_type(chip_id) == BoardType::UBB_BLACKHOLE) {
        constexpr std::string_view ubb_mobo_name = "S7T-MB";

        TT_FATAL(
            using_mock_cluster_desc || get_mobo_name() == ubb_mobo_name, "UBB systems must use S7T-MB motherboard.");
        auto ubb_id = tt::tt_fabric::get_ubb_id(cluster, chip_id);
        return {TrayID{ubb_id.tray_id}, ASICLocation{ubb_id.asic_id}};
    } else {
        auto tray_id = get_tray_id_for_chip(cluster, chip_id, get_mobo_name(), using_mock_cluster_desc);
        ASICLocation asic_location;
        tt::ARCH arch = cluster_desc->get_arch(chip_id);
        if (arch == tt::ARCH::WORMHOLE_B0) {
            // Derive ASIC Location based on the tunnel depth for Wormhole systems
            // TODO: Remove this once UMD populates the ASIC Location for WH systems.
            auto mmio_device = cluster_desc->get_closest_mmio_capable_chip(chip_id);
            auto tunnels_from_mmio_device = llrt::discover_tunnels_from_mmio_device(cluster);
            const auto& tunnels = tunnels_from_mmio_device.at(mmio_device);
            for (auto tunnel = 0; tunnel < tunnels.size(); tunnel++) {
                const auto& devices_on_tunnel = tunnels[tunnel];
                auto device_it = std::find(devices_on_tunnel.begin(), devices_on_tunnel.end(), chip_id);
                if (device_it != devices_on_tunnel.end()) {
                    asic_location = ASICLocation{device_it - devices_on_tunnel.begin()};
                    break;
                }
            }
        } else if (arch == tt::ARCH::BLACKHOLE) {
            // Query ASIC Location from the Cluster Descriptor for BH.
            asic_location = ASICLocation{cluster_desc->get_asic_location(chip_id)};
        } else if (arch == tt::ARCH::QUASAR) {
            // Query ASIC Location from the Cluster Descriptor for QUASAR.
            asic_location = ASICLocation{cluster_desc->get_asic_location(chip_id)};
        } else {
            TT_THROW("Unrecognized Architecture. Cannot determine asic location.");
        }
        return {tray_id, asic_location};
    }
}

struct EthEndpoint {
    AsicID board_id;
    uint8_t chan_id;

    auto operator<=>(const EthEndpoint&) const = default;
};

}  // namespace

/**************************************************************************************************
 PhysicalSystemDescriptor implementation
**************************************************************************************************/

PhysicalSystemDescriptor::PhysicalSystemDescriptor(
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const std::shared_ptr<distributed::multihost::DistributedContext>& distributed_context,
    const Hal* hal,
    const llrt::RunTimeOptions& rtoptions,
    bool run_discovery) :
    PhysicalSystemDescriptor(cluster, distributed_context, hal, rtoptions.get_target_device(), run_discovery) {}

PhysicalSystemDescriptor::PhysicalSystemDescriptor(
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const std::shared_ptr<distributed::multihost::DistributedContext>& distributed_context,
    const Hal* hal,
    tt::TargetDevice target_device_type,
    bool run_discovery) :
    cluster_(cluster), distributed_context_(distributed_context), hal_(hal), target_device_type_(target_device_type) {
    if (run_discovery) {
        // When constructing the PhysicalSystemDescriptor, we run local and global discovery.
        // We do not run "live" discovery since the cluster descriptor is already populated
        // with accurate state from UMD.
        this->run_discovery(true, false);
    }
}

PhysicalSystemDescriptor::PhysicalSystemDescriptor(const std::string& mock_proto_desc_path) :
    cluster_(null_cluster), distributed_context_(nullptr), hal_(nullptr), target_device_type_(TargetDevice::Silicon) {
    auto proto_desc = deserialize_physical_system_descriptor_from_text_proto_file(mock_proto_desc_path);
    this->merge(std::move(proto_desc));
}

PhysicalSystemDescriptor::~PhysicalSystemDescriptor() = default;

void PhysicalSystemDescriptor::resolve_hostname_uniqueness() {
    using namespace tt::tt_metal::distributed::multihost;
    constexpr uint32_t controller_rank = 0;
    auto my_rank = *(distributed_context_->rank());
    auto world_size = *(distributed_context_->size());

    log_info(tt::LogFabric, "Rank {}: Starting resolve_hostname_uniqueness (world_size={})", my_rank, world_size);

    // Warm-up communication to establish MPI connections and synchronize.
    // This is critical when using rankfiles with hostnames, as MPI connections
    // may not be fully established when this function is called.
    // We use bidirectional point-to-point communication to both "warm up" connections
    // and synchronize all ranks, avoiding the barrier which may hang with rankfiles.
    if (world_size > 1) {
        int warmup_msg = 1;
        if (my_rank == controller_rank) {
            // Rank 0: bidirectional communication with each other rank for synchronization
            for (std::size_t rank = 1; rank < world_size; rank++) {
                // First, receive from other rank (ensures they're ready)
                log_info(tt::LogFabric, "Rank {}: Sync: waiting to receive from rank {}", my_rank, rank);
                int recv_msg = 0;
                distributed_context_->recv(
                    tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&recv_msg), sizeof(recv_msg)),
                    Rank{rank},
                    Tag{999});  // Use a different tag for sync
                log_info(tt::LogFabric, "Rank {}: Sync: received from rank {}", my_rank, rank);

                // Then, send back to acknowledge (completes synchronization)
                log_info(tt::LogFabric, "Rank {}: Sync: sending acknowledgment to rank {}", my_rank, rank);
                distributed_context_->send(
                    tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&warmup_msg), sizeof(warmup_msg)),
                    Rank{rank},
                    Tag{998});  // Different tag for acknowledgment
                log_info(tt::LogFabric, "Rank {}: Sync: sent acknowledgment to rank {}", my_rank, rank);
            }
            log_info(tt::LogFabric, "Rank {}: Sync completed, all ranks synchronized", my_rank);
        } else {
            // Other ranks: send to rank 0, then receive acknowledgment
            log_info(tt::LogFabric, "Rank {}: Sync: sending ready signal to rank 0", my_rank);
            distributed_context_->send(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&warmup_msg), sizeof(warmup_msg)),
                Rank{controller_rank},
                Tag{999});
            log_info(tt::LogFabric, "Rank {}: Sync: sent ready signal, waiting for acknowledgment", my_rank);

            int recv_msg = 0;
            distributed_context_->recv(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&recv_msg), sizeof(recv_msg)),
                Rank{controller_rank},
                Tag{998});
            log_info(tt::LogFabric, "Rank {}: Sync: received acknowledgment, synchronized", my_rank);
        }
    }
    log_info(tt::LogFabric, "Rank {}: Starting hostname exchange", my_rank);

    if (my_rank == controller_rank) {
        std::vector<std::string> hostnames = {};
        auto my_hostname = get_host_name();
        hostnames.push_back(my_hostname);
        log_info(
            tt::LogFabric,
            "Rank {} (controller): My hostname is '{}', waiting to receive from {} ranks",
            my_rank,
            my_hostname,
            world_size - 1);

        for (std::size_t rank = 0; rank < world_size; rank++) {
            if (rank != controller_rank) {
                log_info(
                    tt::LogFabric,
                    "Rank {} (controller): Waiting to receive hostname size from rank {}",
                    my_rank,
                    rank);
                std::size_t peer_hostname_size = 0;
                distributed_context_->recv(
                    tt::stl::Span<std::byte>(
                        reinterpret_cast<std::byte*>(&peer_hostname_size), sizeof(peer_hostname_size)),
                    Rank{rank},
                    Tag{0});
                log_info(
                    tt::LogFabric,
                    "Rank {} (controller): Received hostname size {} from rank {}",
                    my_rank,
                    peer_hostname_size,
                    rank);

                std::vector<uint8_t> serialized_peer_hostname(peer_hostname_size);
                log_info(
                    tt::LogFabric,
                    "Rank {} (controller): Waiting to receive hostname data from rank {}",
                    my_rank,
                    rank);
                distributed_context_->recv(
                    tt::stl::as_writable_bytes(
                        tt::stl::Span<uint8_t>(serialized_peer_hostname.data(), serialized_peer_hostname.size())),
                    Rank{rank},
                    Tag{0});
                auto peer_hostname = std::string(serialized_peer_hostname.begin(), serialized_peer_hostname.end());
                log_info(
                    tt::LogFabric,
                    "Rank {} (controller): Received hostname '{}' from rank {}",
                    my_rank,
                    peer_hostname,
                    rank);
                hostnames.push_back(peer_hostname);
            }
        }
        all_hostnames_unique_ = std::set<std::string>(hostnames.begin(), hostnames.end()).size() == hostnames.size();
        log_info(
            tt::LogFabric, "Rank {} (controller): All hostnames received. Unique: {}", my_rank, all_hostnames_unique_);

        for (std::size_t rank = 0; rank < world_size; rank++) {
            if (rank != controller_rank) {
                log_info(tt::LogFabric, "Rank {} (controller): Sending uniqueness result to rank {}", my_rank, rank);
                distributed_context_->send(
                    tt::stl::Span<std::byte>(
                        reinterpret_cast<std::byte*>(&all_hostnames_unique_), sizeof(all_hostnames_unique_)),
                    Rank{rank},
                    Tag{0});
            }
        }
        log_info(tt::LogFabric, "Rank {} (controller): Finished resolve_hostname_uniqueness", my_rank);
    } else {
        auto host_name = get_host_name();
        log_info(
            tt::LogFabric,
            "Rank {}: My hostname is '{}', sending to controller rank {}",
            my_rank,
            host_name,
            controller_rank);

        auto serialized_hostname = std::vector<uint8_t>(host_name.begin(), host_name.end());
        std::size_t serialized_hostname_size = serialized_hostname.size();
        log_info(tt::LogFabric, "Rank {}: Sending hostname size {} to controller", my_rank, serialized_hostname_size);
        distributed_context_->send(
            tt::stl::Span<std::byte>(
                reinterpret_cast<std::byte*>(&serialized_hostname_size), sizeof(serialized_hostname_size)),
            Rank{controller_rank},
            Tag{0});

        log_info(tt::LogFabric, "Rank {}: Sending hostname '{}' to controller", my_rank, host_name);
        distributed_context_->send(
            tt::stl::as_writable_bytes(tt::stl::Span<uint8_t>(serialized_hostname.data(), serialized_hostname.size())),
            Rank{controller_rank},
            Tag{0});

        log_info(tt::LogFabric, "Rank {}: Waiting to receive uniqueness result from controller", my_rank);
        distributed_context_->recv(
            tt::stl::Span<std::byte>(
                reinterpret_cast<std::byte*>(&all_hostnames_unique_), sizeof(all_hostnames_unique_)),
            Rank{controller_rank},
            Tag{0});
        log_info(
            tt::LogFabric,
            "Rank {}: Received uniqueness result: {}, finished resolve_hostname_uniqueness",
            my_rank,
            all_hostnames_unique_);
    }
}

void PhysicalSystemDescriptor::run_discovery(bool run_global_discovery, bool run_live_discovery) {
    auto my_rank = *(distributed_context_->rank());
    log_info(tt::LogFabric, "Rank {}: Starting run_discovery, calling resolve_hostname_uniqueness", my_rank);
    this->resolve_hostname_uniqueness();
    log_info(tt::LogFabric, "Rank {}: Finished resolve_hostname_uniqueness, starting local discovery", my_rank);
    this->run_local_discovery(run_live_discovery);
    log_info(tt::LogFabric, "Rank {}: Finished local discovery", my_rank);
    if (run_global_discovery) {
        log_info(tt::LogFabric, "Rank {}: Starting global discovery", my_rank);
        this->run_global_discovery();
        log_info(tt::LogFabric, "Rank {}: Finished global discovery", my_rank);
    }
}

void PhysicalSystemDescriptor::clear() {
    // Erase all contents in all data structures
    system_graph_.asic_connectivity_graph.clear();
    system_graph_.host_connectivity_graph.clear();
    asic_descriptors_.clear();
    host_to_mobo_name_.clear();
    host_to_rank_.clear();
    exit_node_connection_table_.clear();
}

void PhysicalSystemDescriptor::run_local_discovery(bool run_live_discovery) {
    auto my_rank = *(distributed_context_->rank());
    log_info(tt::LogFabric, "Rank {}: run_local_discovery: starting", my_rank);
    this->clear();
    log_info(tt::LogFabric, "Rank {}: run_local_discovery: cleared", my_rank);

    if (!run_live_discovery || target_device_type_ != TargetDevice::Silicon) {
        TT_FATAL(cluster_ != nullptr, "PhysicalSystemDescriptor must be initialized with a valid UMD cluster reference in order to run live discovery");
        tt::umd::Cluster& cluster = *cluster_;
        log_info(
            tt::LogFabric, "Rank {}: run_local_discovery: creating cluster descriptor from existing cluster", my_rank);
        cluster_desc_ = std::make_unique<tt::umd::ClusterDescriptor>(*cluster.get_cluster_description());
    } else {
        // As part of live discovery, we create a new cluster descriptor to query the latest state from UMD.
        // Otherwise, we use the existing cluster descriptor, which may be stale with respect to the state of
        // the hardware.
        log_info(tt::LogFabric, "Rank {}: run_local_discovery: creating new cluster descriptor", my_rank);
        cluster_desc_ = tt::umd::Cluster::create_cluster_descriptor();
    }
    log_info(tt::LogFabric, "Rank {}: run_local_discovery: got cluster descriptor", my_rank);
    const auto& chip_unique_ids = cluster_desc_->get_chip_unique_ids();
    const auto& eth_connections = cluster_desc_->get_ethernet_connections();
    auto cross_host_eth_connections = cluster_desc_->get_ethernet_connections_to_remote_devices();
    log_info(
        tt::LogFabric,
        "Rank {}: run_local_discovery: got connections, chip_unique_ids size={}",
        my_rank,
        chip_unique_ids.size());

    auto hostname = this->my_host_name();
    log_info(tt::LogFabric, "Rank {}: run_local_discovery: my_host_name()={}", my_rank, hostname);
    host_to_mobo_name_[hostname] = get_mobo_name();
    host_to_rank_[hostname] = my_rank;

    auto& asic_graph = system_graph_.asic_connectivity_graph[hostname];
    auto& exit_nodes = exit_node_connection_table_[hostname];

    auto add_local_asic_descriptor = [&](AsicID src_unique_id, ChipId src_chip_id) {
        TT_FATAL(cluster_ != nullptr, "PhysicalSystemDescriptor must be initialized with a valid UMD cluster reference in order to run live discovery");
        tt::umd::Cluster& cluster = *cluster_;
        auto [tray_id, asic_location] =
            get_asic_position(cluster, src_chip_id, target_device_type_ != TargetDevice::Silicon);
        asic_descriptors_[src_unique_id] = ASICDescriptor{
            TrayID{tray_id}, asic_location, cluster_desc_->get_board_type(src_chip_id), src_unique_id, hostname};
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
        for (auto& [chan, dst] : conn) {
            auto dst_chip = std::get<0>(dst);
            auto dst_chan = std::get<1>(dst);
            if (visited_dst.find(dst_chip) == visited_dst.end()) {
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
        if (asic_descriptors_.find(local_unique_id) == asic_descriptors_.end()) {
            add_local_asic_descriptor(local_unique_id, local_chip_id);
        }
        std::unordered_map<AsicID, size_t> visited_dst;
        for (const auto& [eth_chan, remote_info] : eth_link_info) {
            auto dst_unique_id = AsicID{std::get<0>(remote_info)};
            auto dst_chan = std::get<1>(remote_info);
            if (visited_dst.find(dst_unique_id) == visited_dst.end()) {
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

    system_graph_.host_connectivity_graph[hostname] = {};
    // Get Ethernet Firmware Version from the driver - Initialize to 0 if not available
    log_info(tt::LogFabric, "Rank {}: run_local_discovery: getting ethernet firmware version", my_rank);
    ethernet_firmware_version_ = cluster_->get_ethernet_firmware_version().value_or(tt::umd::semver_t(0, 0, 0));
    log_info(tt::LogFabric, "Rank {}: run_local_discovery: completed", my_rank);
}

void PhysicalSystemDescriptor::run_global_discovery() {
    using namespace tt::tt_metal::distributed::multihost;
    constexpr uint32_t controller_rank = 0;
    auto my_rank = *(distributed_context_->rank());
    log_info(tt::LogFabric, "Rank {}: run_global_discovery: calling exchange_metadata(true)", my_rank);
    this->exchange_metadata(true);
    log_info(tt::LogFabric, "Rank {}: run_global_discovery: exchange_metadata(true) completed", my_rank);
    if (my_rank == controller_rank) {
        log_info(tt::LogFabric, "Rank {}: run_global_discovery: controller: removing unresolved nodes", my_rank);
        this->remove_unresolved_nodes();
        log_info(
            tt::LogFabric, "Rank {}: run_global_discovery: controller: generating cross host connections", my_rank);
        this->generate_cross_host_connections();
        log_info(tt::LogFabric, "Rank {}: run_global_discovery: controller: validating graphs", my_rank);
        this->validate_graphs();
        log_info(tt::LogFabric, "Rank {}: run_global_discovery: controller: validation completed", my_rank);
    }
    log_info(tt::LogFabric, "Rank {}: run_global_discovery: calling exchange_metadata(false)", my_rank);
    this->exchange_metadata(false);
    log_info(tt::LogFabric, "Rank {}: run_global_discovery: exchange_metadata(false) completed", my_rank);
}

void PhysicalSystemDescriptor::merge(PhysicalSystemDescriptor&& other) {
    for (auto& [host_name, asic_graph] : other.system_graph_.asic_connectivity_graph) {
        system_graph_.asic_connectivity_graph[host_name] = std::move(asic_graph);
    }
    for (auto& [host_name, host_connectivity] : other.system_graph_.host_connectivity_graph) {
        system_graph_.host_connectivity_graph[host_name] = std::move(host_connectivity);
    }
    for (auto& [asic_id, asic_desc] : other.get_asic_descriptors()) {
        asic_descriptors_[asic_id] = std::move(asic_desc);
    }
    for (auto& [host_name, mobo_name] : other.get_host_mobo_name_map()) {
        host_to_mobo_name_[host_name] = std::move(mobo_name);
    }
    for (auto& [host_name, rank] : other.get_host_to_rank_map()) {
        host_to_rank_[host_name] = rank;
    }
    for (auto& [host_name, exit_connections] : other.exit_node_connection_table_) {
        exit_node_connection_table_[host_name] = std::move(exit_connections);
    }

    // Merging PhysicalSystemDescriptors using mock and real clusters is undefined and unsupported
    TT_FATAL(
        target_device_type_ == other.target_device_type_,
        "Cannot merge physical and mock/simulation cluster physical system descriptors.");
}

void PhysicalSystemDescriptor::remove_unresolved_nodes() {
    for (auto& [host, asic_group] : system_graph_.asic_connectivity_graph) {
        for (auto& [src_asic, edges] : asic_group) {
            std::erase_if(edges, [&](const auto& pair) { return not asic_descriptors_.contains(pair.first); });
        }
    }

    for (auto& [host, exit_nodes] : exit_node_connection_table_) {
        std::erase_if(exit_nodes, [&](const auto& exit_node) {
            return asic_descriptors_.find(exit_node.src_exit_node) == asic_descriptors_.end() ||
                   asic_descriptors_.find(exit_node.dst_exit_node) == asic_descriptors_.end();
        });
    }
}

void PhysicalSystemDescriptor::validate_eth_fw_versions(
    const tt::umd::semver_t& peer_ethernet_firmware_version,
    const std::string& my_host_name,
    const std::string& peer_host_name) {
    TT_FATAL(
        peer_ethernet_firmware_version == ethernet_firmware_version_,
        "Ethernet Firmware Versions are expected to be consistent across all nodes in the cluster. Hosts: {} and {} "
        "have different Ethernet Firmware Versions: {} and {}.",
        my_host_name,
        peer_host_name,
        ethernet_firmware_version_.to_string(),
        peer_ethernet_firmware_version.to_string());
}

void PhysicalSystemDescriptor::exchange_metadata(bool issue_gather) {
    using namespace tt::tt_metal::distributed::multihost;
    constexpr uint32_t controller_rank = 0;
    if (*(distributed_context_->size()) == 1) {
        return;
    }
    auto my_rank = *(distributed_context_->rank());
    log_info(tt::LogFabric, "Rank {}: exchange_metadata: starting, issue_gather={}", my_rank, issue_gather);
    std::set<uint32_t> sender_ranks;
    std::set<uint32_t> receiver_ranks;

    if (issue_gather) {
        receiver_ranks.insert(controller_rank);
        for (std::size_t rank = 0; rank < *(distributed_context_->size()); rank++) {
            if (rank != controller_rank) {
                sender_ranks.insert(rank);
            }
        }
        log_info(
            tt::LogFabric,
            "Rank {}: exchange_metadata: gather mode - sender_ranks={}, receiver_ranks={}",
            my_rank,
            sender_ranks.size(),
            receiver_ranks.size());
    } else {
        sender_ranks.insert(controller_rank);
        for (std::size_t rank = 0; rank < *(distributed_context_->size()); rank++) {
            if (rank != controller_rank) {
                receiver_ranks.insert(rank);
            }
        }
        log_info(
            tt::LogFabric,
            "Rank {}: exchange_metadata: scatter mode - sender_ranks={}, receiver_ranks={}",
            my_rank,
            sender_ranks.size(),
            receiver_ranks.size());
    }

    if (sender_ranks.find(my_rank) != sender_ranks.end()) {
        log_info(tt::LogFabric, "Rank {}: exchange_metadata: I am a sender, serializing descriptor", my_rank);
        auto serialized_desc = serialize_physical_system_descriptor_to_bytes(*this);
        std::size_t desc_size = serialized_desc.size();
        log_info(
            tt::LogFabric,
            "Rank {}: exchange_metadata: serialized descriptor size={}, sending to {} receivers",
            my_rank,
            desc_size,
            receiver_ranks.size());

        for (auto rank : receiver_ranks) {
            log_info(tt::LogFabric, "Rank {}: exchange_metadata: sending size to rank {}", my_rank, rank);
            distributed_context_->send(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&desc_size), sizeof(desc_size)),
                Rank{rank},
                Tag{0});

            log_info(tt::LogFabric, "Rank {}: exchange_metadata: sending descriptor data to rank {}", my_rank, rank);
            distributed_context_->send(
                tt::stl::as_writable_bytes(tt::stl::Span<uint8_t>(serialized_desc.data(), serialized_desc.size())),
                Rank{rank},
                Tag{0});
            log_info(tt::LogFabric, "Rank {}: exchange_metadata: sent descriptor to rank {}", my_rank, rank);
        }
        log_info(tt::LogFabric, "Rank {}: exchange_metadata: finished sending to all receivers", my_rank);
    } else {
        log_info(
            tt::LogFabric,
            "Rank {}: exchange_metadata: I am a receiver, waiting for {} senders",
            my_rank,
            sender_ranks.size());
        for (auto rank : sender_ranks) {
            log_info(tt::LogFabric, "Rank {}: exchange_metadata: waiting to receive size from rank {}", my_rank, rank);
            std::size_t peer_descriptor_size = 0;
            distributed_context_->recv(
                tt::stl::Span<std::byte>(
                    reinterpret_cast<std::byte*>(&peer_descriptor_size), sizeof(peer_descriptor_size)),
                Rank{rank},
                Tag{0});
            log_info(
                tt::LogFabric,
                "Rank {}: exchange_metadata: received size {} from rank {}",
                my_rank,
                peer_descriptor_size,
                rank);
            std::vector<uint8_t> serialized_peer_desc(peer_descriptor_size);
            log_info(
                tt::LogFabric,
                "Rank {}: exchange_metadata: waiting to receive descriptor data from rank {}",
                my_rank,
                rank);
            distributed_context_->recv(
                tt::stl::as_writable_bytes(
                    tt::stl::Span<uint8_t>(serialized_peer_desc.data(), serialized_peer_desc.size())),
                Rank{rank},
                Tag{0});
            log_info(
                tt::LogFabric,
                "Rank {}: exchange_metadata: received descriptor from rank {}, deserializing",
                my_rank,
                rank);
            auto peer_desc = deserialize_physical_system_descriptor_from_bytes(serialized_peer_desc);
            log_info(
                tt::LogFabric,
                "Rank {}: exchange_metadata: validating and merging descriptor from rank {}",
                my_rank,
                rank);
            this->validate_eth_fw_versions(
                peer_desc.get_ethernet_firmware_version(),
                asic_descriptors_.begin()->second.host_name,
                peer_desc.get_asic_descriptors().begin()->second.host_name);
            this->merge(std::move(peer_desc));
            log_info(tt::LogFabric, "Rank {}: exchange_metadata: merged descriptor from rank {}", my_rank, rank);
        }
        log_info(tt::LogFabric, "Rank {}: exchange_metadata: finished receiving from all senders", my_rank);
    }

    // Synchronize all ranks using point-to-point communication instead of barrier.
    // Barriers hang with rankfiles, so we use the same synchronization pattern as resolve_hostname_uniqueness.
    auto world_size = *(distributed_context_->size());
    if (world_size > 1) {
        int sync_msg = 1;
        if (my_rank == controller_rank) {
            // Rank 0: bidirectional communication with each other rank for synchronization
            for (std::size_t rank = 1; rank < world_size; rank++) {
                log_info(
                    tt::LogFabric, "Rank {}: exchange_metadata sync: waiting to receive from rank {}", my_rank, rank);
                int recv_msg = 0;
                distributed_context_->recv(
                    tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&recv_msg), sizeof(recv_msg)),
                    Rank{rank},
                    Tag{997});  // Use a different tag for exchange_metadata sync
                log_info(tt::LogFabric, "Rank {}: exchange_metadata sync: received from rank {}", my_rank, rank);

                log_info(
                    tt::LogFabric, "Rank {}: exchange_metadata sync: sending acknowledgment to rank {}", my_rank, rank);
                distributed_context_->send(
                    tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&sync_msg), sizeof(sync_msg)),
                    Rank{rank},
                    Tag{996});  // Different tag for acknowledgment
                log_info(
                    tt::LogFabric, "Rank {}: exchange_metadata sync: sent acknowledgment to rank {}", my_rank, rank);
            }
            log_info(tt::LogFabric, "Rank {}: exchange_metadata sync: all ranks synchronized", my_rank);
        } else {
            // Other ranks: send to rank 0, then receive acknowledgment
            log_info(tt::LogFabric, "Rank {}: exchange_metadata sync: sending ready signal to rank 0", my_rank);
            distributed_context_->send(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&sync_msg), sizeof(sync_msg)),
                Rank{controller_rank},
                Tag{997});
            log_info(
                tt::LogFabric,
                "Rank {}: exchange_metadata sync: sent ready signal, waiting for acknowledgment",
                my_rank);

            int recv_msg = 0;
            distributed_context_->recv(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&recv_msg), sizeof(recv_msg)),
                Rank{controller_rank},
                Tag{996});
            log_info(tt::LogFabric, "Rank {}: exchange_metadata sync: received acknowledgment, synchronized", my_rank);
        }
    }
    log_info(tt::LogFabric, "Rank {}: exchange_metadata: synchronization completed", my_rank);
}

void PhysicalSystemDescriptor::generate_cross_host_connections() {
    for (const auto& [host, exit_nodes] : exit_node_connection_table_) {
        std::unordered_map<std::string, size_t> visited_hosts;
        for (const auto& [candidate_host, candidate_exit_nodes] : exit_node_connection_table_) {
            if (host == candidate_host) {
                continue;  // Skip self connections
            }
            for (const auto& exit_node : exit_nodes) {
                for (const auto& candidate_node : candidate_exit_nodes) {
                    if (exit_node.src_exit_node == candidate_node.dst_exit_node &&
                        candidate_node.src_exit_node == exit_node.dst_exit_node &&
                        exit_node.eth_conn.src_chan == candidate_node.eth_conn.dst_chan &&
                        exit_node.eth_conn.dst_chan == candidate_node.eth_conn.src_chan) {
                        if (visited_hosts.find(candidate_host) == visited_hosts.end()) {
                            system_graph_.host_connectivity_graph[host].push_back({candidate_host, {exit_node}});
                            visited_hosts[candidate_host] = system_graph_.host_connectivity_graph[host].size() - 1;
                        } else {
                            system_graph_.host_connectivity_graph[host][visited_hosts[candidate_host]].second.push_back(
                                exit_node);
                        }
                        break;
                    }
                }
            }
        }
    }
}

YAML::Node PhysicalSystemDescriptor::generate_yaml_node() const {
    YAML::Node root;
    YAML::Node compute_nodes;
    YAML::Node local_eth_connections(YAML::NodeType::Sequence);
    YAML::Node global_eth_connections(YAML::NodeType::Sequence);

    std::set<std::pair<EthEndpoint, EthEndpoint>> processed_connections;
    for (const auto& [host_name, mobo_name] : host_to_mobo_name_) {
        YAML::Node host_node;
        YAML::Node tray_groups;
        host_node["motherboard"] = mobo_name;

        std::map<TrayID, std::vector<ASICDescriptor>> grouped_asics;

        for (const auto& asic : system_graph_.asic_connectivity_graph.at(host_name)) {
            AsicID asic_id = asic.first;
            TrayID tray_id = asic_descriptors_.at(asic_id).tray_id;
            grouped_asics[tray_id].push_back(asic_descriptors_.at(asic_id));
        }

        for (const auto& group : grouped_asics) {
            YAML::Node tray_group;
            tray_group["tray_id"] = *(group.first);  // tray_id
            tray_group["board_type"] = enchantum::to_string(group.second.front().board_type);
            std::vector<ASICDescriptor> sorted_asics = group.second;
            std::sort(sorted_asics.begin(), sorted_asics.end(), [](const ASICDescriptor& a, const ASICDescriptor& b) {
                return a.asic_location < b.asic_location;
            });
            // Create asics array
            YAML::Node asics_array;
            for (const auto& asic : sorted_asics) {
                YAML::Node asic_node;
                asic_node["asic_location"] = *(asic.asic_location);
                asic_node["asic_id"] = *(asic.unique_id);
                asics_array.push_back(asic_node);
            }
            tray_group["asics"] = asics_array;
            tray_groups.push_back(tray_group);
        }
        host_node["asic_info"] = tray_groups;
        compute_nodes[host_name] = host_node;

        for (const auto& asic : system_graph_.asic_connectivity_graph.at(host_name)) {
            auto src_asic_id = asic.first;
            const auto& src_asic_desc = asic_descriptors_.at(src_asic_id);
            for (const auto& edge : asic.second) {
                auto dst_asic_id = edge.first;
                const auto& dst_asic_desc = asic_descriptors_.at(dst_asic_id);
                for (const auto& eth_conn : edge.second) {
                    EthEndpoint src_id{src_asic_id, eth_conn.src_chan};
                    EthEndpoint dst_id{dst_asic_id, eth_conn.dst_chan};
                    auto connection_key = std::make_pair(std::min(src_id, dst_id), std::max(src_id, dst_id));

                    if (processed_connections.find(connection_key) != processed_connections.end()) {
                        continue;
                    }
                    processed_connections.insert(connection_key);

                    YAML::Node src_conn_node;
                    YAML::Node dst_conn_node;
                    YAML::Node connection_pair(YAML::NodeType::Sequence);
                    connection_pair.SetStyle(YAML::EmitterStyle::Flow);
                    src_conn_node["host_name"] = src_asic_desc.host_name;
                    dst_conn_node["host_name"] = dst_asic_desc.host_name;
                    src_conn_node["tray_id"] = *(src_asic_desc.tray_id);
                    src_conn_node["asic_location"] = *(src_asic_desc.asic_location);
                    dst_conn_node["tray_id"] = *(dst_asic_desc.tray_id);
                    dst_conn_node["asic_location"] = *(dst_asic_desc.asic_location);
                    src_conn_node["chan_id"] = +eth_conn.src_chan;
                    dst_conn_node["chan_id"] = +eth_conn.dst_chan;

                    connection_pair.push_back(src_conn_node);
                    connection_pair.push_back(dst_conn_node);

                    if (eth_conn.is_local) {
                        local_eth_connections.push_back(connection_pair);
                    } else {
                        global_eth_connections.push_back(connection_pair);
                    }
                }
            }
        }
    }

    root["compute_node_specs"] = compute_nodes;
    root["local_eth_connections"] = local_eth_connections;
    root["global_eth_connections"] = global_eth_connections;

    return root;
}

void PhysicalSystemDescriptor::dump_to_yaml(const std::optional<std::string>& path_to_yaml) const {
    YAML::Node root = generate_yaml_node();

    if (path_to_yaml.has_value()) {
        std::ofstream fout(path_to_yaml.value());
        if (!fout.is_open()) {
            TT_THROW("Failed to open file for writing: {}", path_to_yaml.value());
        }
        fout << root;
        if (fout.fail()) {
            TT_THROW("Failed to write YAML content to file: {}", path_to_yaml.value());
        }
    } else {
        std::cout << root << std::endl;
    }
}

void PhysicalSystemDescriptor::emit_to_text_proto(const std::optional<std::string>& file_path) {
    emit_physical_system_descriptor_to_text_proto(*this, file_path);
}

void PhysicalSystemDescriptor::validate_graphs() {
    // Validate that the representation of the system is internally consistent.
    for (const auto& [host, asic_group] : system_graph_.asic_connectivity_graph) {
        for (const auto& [src_asic, edges] : asic_group) {
            const auto& src_host = asic_descriptors_.at(src_asic).host_name;
            const auto& src_host_edges = system_graph_.host_connectivity_graph.at(src_host);

            for (const auto& [dst_asic, eth_conns] : edges) {
                const auto& dst_host = asic_descriptors_.at(dst_asic).host_name;

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

                // Global connections must cross hosts.
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

std::vector<AsicID> PhysicalSystemDescriptor::get_asic_neighbors(AsicID asic_id) const {
    for (const auto& [host, asic_group] : system_graph_.asic_connectivity_graph) {
        if (asic_group.find(asic_id) != asic_group.end()) {
            std::vector<AsicID> neighbors;
            for (const auto& edge : asic_group.at(asic_id)) {
                neighbors.push_back(edge.first);
            }
            return neighbors;
        }
    }
    return {};
}

std::vector<EthConnection> PhysicalSystemDescriptor::get_eth_connections(AsicID src_asic, AsicID dst_asic) const {
    for (const auto& [host, asic_group] : system_graph_.asic_connectivity_graph) {
        if (asic_group.find(src_asic) != asic_group.end()) {
            for (const auto& edge : asic_group.at(src_asic)) {
                if (edge.first == dst_asic) {
                    return edge.second;
                }
            }
        }
    }
    return {};
}

const AsicTopology& PhysicalSystemDescriptor::get_asic_topology(const std::string& hostname) const {
    TT_FATAL(
        system_graph_.asic_connectivity_graph.find(hostname) != system_graph_.asic_connectivity_graph.end(),
        "No ASIC topology found for host {}",
        hostname);
    return system_graph_.asic_connectivity_graph.at(hostname);
}

TrayID PhysicalSystemDescriptor::get_tray_id(AsicID asic_id) const {
    TT_FATAL(
        asic_descriptors_.find(asic_id) != asic_descriptors_.end(), "No ASIC descriptor found for asic_id {}", asic_id);
    return asic_descriptors_.at(asic_id).tray_id;
}

ASICLocation PhysicalSystemDescriptor::get_asic_location(AsicID asic_id) const {
    TT_FATAL(
        asic_descriptors_.find(asic_id) != asic_descriptors_.end(), "No ASIC descriptor found for asic_id {}", asic_id);
    return asic_descriptors_.at(asic_id).asic_location;
}

std::vector<AsicID> PhysicalSystemDescriptor::get_asics_connected_to_host(const std::string& hostname) const {
    std::vector<AsicID> asics;
    if (system_graph_.asic_connectivity_graph.find(hostname) != system_graph_.asic_connectivity_graph.end()) {
        for (const auto& [asic_id, _] : system_graph_.asic_connectivity_graph.at(hostname)) {
            asics.push_back(asic_id);
        }
    }
    return asics;
}

bool PhysicalSystemDescriptor::is_cross_host_eth_link(AsicID asic_id, uint8_t chan_id) const {
    for (const auto& [host, asic_group] : system_graph_.asic_connectivity_graph) {
        if (this->get_host_name_for_asic(asic_id) != host) {
            continue;
        }
        const auto& connections = asic_group.at(asic_id);
        auto connection_it = std::find_if(connections.begin(), connections.end(), [&](const auto& connection) {
            // Check if this chan_id is a src_chan in any of the eth_connections
            return std::find_if(connection.second.begin(), connection.second.end(), [&](const auto& eth_conn) {
                       return eth_conn.src_chan == chan_id;
                   }) != connection.second.end();
        });
        TT_FATAL(
            connection_it != connections.end(),
            "Channel {} not found in asic connectivity graph for asic {}",
            chan_id,
            asic_id);
        auto connected_asic = connection_it->first;
        return this->get_host_name_for_asic(connected_asic) != host;
    }
    TT_THROW("Asic {} not found in any host's asic connectivity graph", asic_id);
    return false;
}

std::vector<std::string> PhysicalSystemDescriptor::get_host_neighbors(const std::string& hostname) const {
    TT_FATAL(
        system_graph_.host_connectivity_graph.find(hostname) != system_graph_.host_connectivity_graph.end(),
        "No Host connectivity found for host {}",
        hostname);
    std::vector<std::string> neighbors;
    for (const auto& edge : system_graph_.host_connectivity_graph.at(hostname)) {
        neighbors.push_back(edge.first);
    }
    return neighbors;
}

std::vector<ExitNodeConnection> PhysicalSystemDescriptor::get_connecting_exit_nodes(
    const std::string& src_host, const std::string& dst_host) const {
    TT_FATAL(
        system_graph_.host_connectivity_graph.find(src_host) != system_graph_.host_connectivity_graph.end(),
        "No Host connectivity found for host {}",
        src_host);
    for (const auto& edge : system_graph_.host_connectivity_graph.at(src_host)) {
        if (edge.first == dst_host) {
            return edge.second;
        }
    }
    return {};
}

uint32_t PhysicalSystemDescriptor::get_chip_id_for_asic(AsicID asic_id) const {
    const auto& chip_unique_ids = cluster_desc_->get_chip_unique_ids();
    for (const auto& [chip_id, unique_id] : chip_unique_ids) {
        if (unique_id == *asic_id) {
            return chip_id;
        }
    }
    TT_FATAL(false, "Chip ID not found for asic ID {}", asic_id);
    return 0;
}

std::pair<AsicID, uint8_t> PhysicalSystemDescriptor::get_connected_asic_and_channel(
    AsicID asic_id, uint8_t chan_id) const {
    auto host = asic_descriptors_.at(asic_id).host_name;
    auto asic_graph = system_graph_.asic_connectivity_graph.at(host);
    for (const auto& [src_asic, edges] : asic_graph) {
        if (src_asic != asic_id) {
            continue;
        }
        for (const auto& edge : edges) {
            auto dst_asic = edge.first;

            for (const auto& eth_conn : edge.second) {
                if (eth_conn.src_chan == chan_id) {
                    return {dst_asic, eth_conn.dst_chan};
                }
            }
        }
    }
    TT_FATAL(false, "No connected ASIC and channel found for asic ID {} and channel ID {}", asic_id, chan_id);
    return {AsicID{0}, 0};
}

AsicID PhysicalSystemDescriptor::get_asic_id(
    const std::string& hostname, TrayID tray_id, ASICLocation asic_location) const {
    for (const auto& [asic_id, asic_descriptor] : asic_descriptors_) {
        if (asic_descriptor.host_name == hostname && asic_descriptor.tray_id == tray_id &&
            asic_descriptor.asic_location == asic_location) {
            return asic_id;
        }
    }
    TT_THROW("No ASIC ID found at hostname {}, tray ID {}, and ASIC location {}", hostname, *tray_id, *asic_location);
    return AsicID{0};
}

LocalEthernetMetrics PhysicalSystemDescriptor::query_local_ethernet_metrics() const {
    TT_FATAL(cluster_ != nullptr, "PhysicalSystemDescriptor must be initialized with a valid UMD cluster reference in order to query Ethernet metrics");
    tt::umd::Cluster& cluster = *cluster_;

    const auto& local_asics = get_asics_connected_to_host(my_host_name());
    const auto& local_asic_graph = get_asic_topology(my_host_name());
    std::unordered_map<AsicID, std::unordered_map<uint8_t, EthernetMetrics>> local_ethernet_metrics;

    auto retrain_count_addr = hal_->get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::RETRAIN_COUNT);
    auto crc_addr =
        hal_->get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::CRC_ERR);
    auto corr_addr =
        hal_->get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::CORR_CW);
    auto uncorr_addr = hal_->get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNCORR_CW);

    for (const auto& asic : local_asics) {
        const auto& asic_connections = local_asic_graph.at(asic);
        for (const auto& [dst_asic, eth_connections] : asic_connections) {
            for (const auto& eth_connection : eth_connections) {
                uint32_t retrain_count_val = 0, crc_error_val = 0, corr_val_lo = 0, corr_val_hi = 0, uncorr_val_lo = 0,
                         uncorr_val_hi = 0;

                auto src_eth_chan = eth_connection.src_chan;
                auto src_chip_id = get_chip_id_for_asic(asic);
                const auto& soc_desc = cluster.get_soc_descriptor(src_chip_id);
                const auto& translated_eth_core =
                    soc_desc.get_eth_core_for_channel(src_eth_chan, CoordSystem::TRANSLATED);

                cluster.read_from_device(
                    &retrain_count_val, src_chip_id, translated_eth_core, retrain_count_addr, sizeof(uint32_t));
                cluster.read_from_device(
                    &crc_error_val, src_chip_id, translated_eth_core, crc_addr, sizeof(uint32_t));
                cluster.read_from_device(&corr_val_hi, src_chip_id, translated_eth_core, corr_addr, sizeof(uint32_t));
                cluster.read_from_device(
                    &corr_val_lo, src_chip_id, translated_eth_core, corr_addr + 4, sizeof(uint32_t));
                cluster.read_from_device(
                    &uncorr_val_hi, src_chip_id, translated_eth_core, uncorr_addr, sizeof(uint32_t));
                cluster.read_from_device(
                    &uncorr_val_lo, src_chip_id, translated_eth_core, uncorr_addr + 4, sizeof(uint32_t));

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

const HostTopology& PhysicalSystemDescriptor::get_host_topology() const {
    return system_graph_.host_connectivity_graph;
}

std::vector<std::string> PhysicalSystemDescriptor::get_all_hostnames() const {
    std::vector<std::string> hostnames;
    hostnames.reserve(system_graph_.asic_connectivity_graph.size());
    for (const auto& [host, _] : system_graph_.asic_connectivity_graph) {
        hostnames.push_back(host);
    }
    return hostnames;
}

std::string PhysicalSystemDescriptor::my_host_name() const {
    if (all_hostnames_unique_) {
        return get_host_name();
    }
    auto my_rank = *(distributed_context_->rank());
    return get_host_name() + "_" + std::to_string(my_rank);
}

uint32_t PhysicalSystemDescriptor::get_rank_for_hostname(const std::string& host_name) const {
    TT_FATAL(host_to_rank_.find(host_name) != host_to_rank_.end(), "Rank for host {} not found", host_name);
    return host_to_rank_.at(host_name);
}

std::string PhysicalSystemDescriptor::get_host_name_for_asic(AsicID asic_id) const {
    TT_FATAL(
        asic_descriptors_.find(asic_id) != asic_descriptors_.end(), "No ASIC descriptor found for asic_id {}", asic_id);
    return asic_descriptors_.at(asic_id).host_name;
}

UID PhysicalSystemDescriptor::get_u_id(const std::string& /*hostname*/) {
    TT_THROW("Querying Host UID requires the Cable Spec which is not currently supported.");
}

RackID PhysicalSystemDescriptor::get_rack_id(const std::string& /*hostname*/) {
    TT_THROW("Querying Host Rack ID requires the Cable Spec which is not currently supported.");
}

AisleID PhysicalSystemDescriptor::get_aisle_id(const std::string& /*hostname*/) {
    TT_THROW("Querying Host Aisle ID requires the Cable Spec which is not currently supported.");
}

HallID PhysicalSystemDescriptor::get_hall_id(const std::string& /*hostname*/) {
    TT_THROW("Querying Host Hall ID requires the Cable Spec which is not currently supported.");
}

}  // namespace tt::tt_metal
