// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include "tools/scaleout/validation/utils/cluster_validation_utils.hpp"
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include "tt_metal/fabric/link_retrainer.hpp"
#include "tt_metal/fabric/physical_system_discovery.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include <tt-umd/types/xy_pair.hpp>
#include <tt-umd/types/wormhole_eth.hpp>
#include <tt-umd/types/blackhole_eth.hpp>
#include <factory_system_descriptor/utils.hpp>
#include "protobuf/factory_system_descriptor.pb.h"
#include <google/protobuf/text_format.h>
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>
#include <fmt/format.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <type_traits>

namespace tt::scaleout_tools {

// Returns the L1 address of the ETH training status register for the given arch.
// WH: ETH_TRAIN_STATUS_ADDR (0x1104)
// BH: port_status in boot_results_t (BOOT_RESULTS_ADDR + 4).
// Both use the same encoding: 0 = IN_PROGRESS, 1 = SUCCESS, 2 = FAIL.
[[nodiscard]] uint32_t get_eth_train_status_addr(const tt::Cluster& cluster) {
    if (cluster.arch() == tt::ARCH::WORMHOLE_B0) {
        return tt::umd::wormhole::ETH_TRAIN_STATUS_ADDR;
    }
    if (cluster.arch() == tt::ARCH::BLACKHOLE) {
        static_assert(std::is_standard_layout_v<tt::umd::blackhole::eth_status_t>);
        return tt::umd::blackhole::BOOT_RESULTS_ADDR +
               static_cast<uint32_t>(offsetof(tt::umd::blackhole::eth_status_t, port_status));
    }
    TT_THROW("Unsupported architecture for ETH training status check: {}", cluster.arch());
}

struct LinkDescriptors {
    std::string host;
    uint32_t tray_id{};
    uint32_t asic_location{};
    uint32_t channel{};
    ChipId chip_id{};
    tt_xy_pair coord;
};

[[nodiscard]] tt_xy_pair get_eth_core_coord(const tt::Cluster& cluster, ChipId chip_id, uint8_t channel) {
    const auto logical_coord = cluster.get_soc_desc(chip_id).get_eth_core_for_channel(channel, CoordSystem::LOGICAL);
    return cluster.get_virtual_coordinate_from_logical_coordinates(
        chip_id, tt_xy_pair(logical_coord.x, logical_coord.y), CoreType::ETH);
}

void set_link_training_status(const tt::Cluster& cluster, ChipId chip_id, const tt_xy_pair& coord, uint32_t status) {
    const std::vector<uint32_t> status_data{status};
    cluster.write_core(chip_id, coord, status_data, get_eth_train_status_addr(cluster));
    cluster.l1_barrier(chip_id);
}

class DirectedRetrainingFixture : public ::testing::Test {
protected:
    tt::tt_metal::MetalContext* context_{};
    const tt::Cluster* cluster_{};
    std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> distributed_context_;
    std::unique_ptr<tt::tt_metal::PhysicalSystemDescriptor> physical_system_descriptor_;
    std::unordered_map<uint64_t, ChipId> asic_id_to_chip_id_;

    void SetUp() override {
        context_ = &tt::tt_metal::MetalContext::instance();
        cluster_ = &context_->get_cluster();
        distributed_context_ = context_->get_distributed_context_ptr();

        if (!cluster_->supports_ethernet_link_retraining()) {
            GTEST_SKIP() << "Link retraining not supported on this system (requires WH or BH with ETH FW >= 1.9.0)";
        }

        // Initialize physical system descriptor
        auto psd = tt::tt_metal::run_physical_system_discovery(
            *cluster_->get_cluster_desc(), distributed_context_, context_->rtoptions().get_target_device());
        physical_system_descriptor_ = std::make_unique<tt::tt_metal::PhysicalSystemDescriptor>(std::move(psd));

        // Populate asic_id_to_chip_id map
        for (const auto& [chip_id, asic_id] : cluster_->get_unique_chip_ids()) {
            asic_id_to_chip_id_[asic_id] = chip_id;
        }
    }

public:
    const tt::Cluster& get_cluster() const { return *cluster_; }
    tt::tt_metal::PhysicalSystemDescriptor& get_physical_system_descriptor() { return *physical_system_descriptor_; }
    const tt::tt_metal::PhysicalSystemDescriptor& get_physical_system_descriptor() const {
        return *physical_system_descriptor_;
    }
    const std::unordered_map<uint64_t, ChipId>& get_asic_id_to_chip_id() const { return asic_id_to_chip_id_; }
    std::string get_cabling_descriptor_path() const { return "tools/tests/scaleout/cabling_descriptors/t3k.textproto"; }
};

[[nodiscard]] uint32_t get_link_training_status(const tt::Cluster& cluster, ChipId chip_id, const tt_xy_pair& coord) {
    std::vector<uint32_t> status(1, 0);
    cluster.read_core(status, sizeof(uint32_t), tt_cxy_pair(chip_id, coord), get_eth_train_status_addr(cluster));
    return status[0];
}

void validate_connectivity(
    PhysicalSystemDescriptor& physical_system_descriptor, const std::string& cabling_descriptor_path) {
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();

    const auto gsd_yaml_node = physical_system_descriptor.generate_yaml_node();

    // Build FSD from cabling descriptor + hostnames and validate
    const auto hostnames = physical_system_descriptor.get_all_hostnames();

    log_output_rank0("Validating Factory System Descriptor against Global System Descriptor");
    const auto missing_physical_connections = validate_cabling_descriptor_against_gsd(
        cabling_descriptor_path,
        hostnames,
        gsd_yaml_node,
        true /* strict_validation */,
        true /* fail_on_warning */,
        *distributed_context.rank() == 0 /* log_output */);

    const auto missing_topology =
        generate_asic_topology_from_connections(missing_physical_connections, physical_system_descriptor);

    EXPECT_TRUE(missing_topology.empty()) << "Missing connections detected after link reset";
    log_output_rank0("Factory System Descriptor Validation Complete");
}

// Helper function to process ethernet connections for a given operation.
// `operation` is invoked once per matching ethernet connection, so take it by const ref
// (forwarding-then-calling in a loop would risk use-after-move on iterations past the first).
template <typename Operation>
void process_ethernet_connections(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::unordered_map<uint64_t, ChipId>& asic_id_to_chip_id,
    const tt::Cluster& cluster,
    const tt::umd::ClusterDescriptor& cluster_desc,
    const Operation& operation) {
    const auto& asic_topology = physical_system_descriptor.get_asic_topology(physical_system_descriptor.my_host_name());
    for (const auto& [asic_id, asic_connections] : asic_topology) {
        for (const auto& [dst_asic_id, eth_connections] : asic_connections) {
            const auto src_chip_id = asic_id_to_chip_id.at(*asic_id);
            const auto dst_chip_id = asic_id_to_chip_id.at(*dst_asic_id);

            const bool both_mmio =
                cluster_desc.is_chip_mmio_capable(src_chip_id) && cluster_desc.is_chip_mmio_capable(dst_chip_id);
            const bool both_non_mmio =
                !cluster_desc.is_chip_mmio_capable(src_chip_id) && !cluster_desc.is_chip_mmio_capable(dst_chip_id);

            if (both_mmio || both_non_mmio) {
                for (const auto& eth_connection : eth_connections) {
                    operation(src_chip_id, get_eth_core_coord(cluster, src_chip_id, eth_connection.src_chan));
                }
            }
        }
    }
}

TEST_F(DirectedRetrainingFixture, TestActiveEthRetraining) {
    validate_connectivity(get_physical_system_descriptor(), get_cabling_descriptor_path());

    // Take down MMIO-to-MMIO and non-MMIO-to-non-MMIO links
    process_ethernet_connections(
        get_physical_system_descriptor(),
        get_asic_id_to_chip_id(),
        get_cluster(),
        *get_cluster().get_cluster_desc(),
        [&](ChipId chip_id, const tt_xy_pair& coord) { set_link_training_status(get_cluster(), chip_id, coord, 0); });

    reset_ethernet_links(
        get_physical_system_descriptor(),
        get_physical_system_descriptor().get_asic_topology(get_physical_system_descriptor().my_host_name()));

    process_ethernet_connections(
        get_physical_system_descriptor(),
        get_asic_id_to_chip_id(),
        get_cluster(),
        *get_cluster().get_cluster_desc(),
        [&](ChipId chip_id, const tt_xy_pair& coord) {
            EXPECT_EQ(get_link_training_status(get_cluster(), chip_id, coord), 1);
        });

    // Re-run discovery
    get_physical_system_descriptor().clear();
    auto new_psd = tt::tt_metal::run_physical_system_discovery(
        *get_cluster().get_cluster_desc(), distributed_context_, context_->rtoptions().get_target_device(), true, true);
    get_physical_system_descriptor().merge(std::move(new_psd));

    validate_connectivity(get_physical_system_descriptor(), get_cabling_descriptor_path());
}

// NOTE: This test is disabled because exit node retraining only works on WH with Ethernet FW 7.2.0 and above.
// The FW is currently not upgraded on the dual T3K system.
TEST_F(DirectedRetrainingFixture, DISABLED_TestExitNodeRetraining) {
    const auto& my_hostname = get_physical_system_descriptor().my_host_name();
    const auto all_hostnames = get_physical_system_descriptor().get_all_hostnames();

    // Take down exit node links to all remote hosts
    for (const auto& host : all_hostnames) {
        if (host == my_hostname) {
            continue;
        }
        const auto exit_nodes = get_physical_system_descriptor().get_connecting_exit_nodes(my_hostname, host);
        log_info(tt::LogTest, "Taking {} exit node links down on host {}", exit_nodes.size(), host);

        for (const auto& exit_node : exit_nodes) {
            const auto chip_id = get_asic_id_to_chip_id().at(*exit_node.src_exit_node);
            const auto coord = get_eth_core_coord(get_cluster(), chip_id, exit_node.eth_conn.src_chan);
            set_link_training_status(get_cluster(), chip_id, coord, 0);
        }
    }

    // Reset all ethernet links
    const auto& asic_topology = get_physical_system_descriptor().get_asic_topology(my_hostname);
    reset_ethernet_links(get_physical_system_descriptor(), asic_topology);

    // Verify exit node links are back up
    for (const auto& host : all_hostnames) {
        if (host == my_hostname) {
            continue;
        }
        const auto exit_nodes = get_physical_system_descriptor().get_connecting_exit_nodes(my_hostname, host);

        for (const auto& exit_node : exit_nodes) {
            const auto chip_id = get_asic_id_to_chip_id().at(*exit_node.src_exit_node);
            const auto coord = get_eth_core_coord(get_cluster(), chip_id, exit_node.eth_conn.src_chan);
            EXPECT_EQ(get_link_training_status(get_cluster(), chip_id, coord), 1);
        }
    }

    distributed_context_->barrier();
    // Re-run discovery
    get_physical_system_descriptor().clear();
    auto new_psd = tt::tt_metal::run_physical_system_discovery(
        *get_cluster().get_cluster_desc(), distributed_context_, context_->rtoptions().get_target_device(), true, true);
    get_physical_system_descriptor().merge(std::move(new_psd));
}

[[nodiscard]] std::vector<LinkDescriptors> collect_mmio_link_params(
    const DirectedRetrainingFixture& fixture, const tt::tt_metal::AsicTopology& asic_topology) {
    constexpr size_t MAX_LINKS_TO_RESET = 4;

    auto* const cluster_desc = fixture.get_cluster().get_cluster_desc();
    const auto& asic_descriptors = fixture.get_physical_system_descriptor().get_asic_descriptors();

    std::vector<LinkDescriptors> local_links;
    std::vector<LinkDescriptors> remote_links;
    local_links.reserve(MAX_LINKS_TO_RESET);
    remote_links.reserve(MAX_LINKS_TO_RESET);

    for (const auto& [asic_id, asic_connections] : asic_topology) {
        const auto src_chip_id = fixture.get_asic_id_to_chip_id().at(*asic_id);
        if (!cluster_desc->is_chip_mmio_capable(src_chip_id)) {
            continue;
        }

        for (const auto& [dst_id, eth_connections] : asic_connections) {
            if (eth_connections.empty()) {
                continue;
            }

            const auto dst_chip_id = fixture.get_asic_id_to_chip_id().at(*dst_id);
            if (!cluster_desc->is_chip_mmio_capable(dst_chip_id)) {
                continue;
            }

            const auto& src_asic_desc = asic_descriptors.at(asic_id);
            const auto [dst_asic_id, dst_channel] =
                fixture.get_physical_system_descriptor().get_connected_asic_and_channel(
                    asic_id, eth_connections.front().src_chan);
            const bool is_local = (src_asic_desc.host_name == asic_descriptors.at(dst_asic_id).host_name);

            const auto src_coord =
                get_eth_core_coord(fixture.get_cluster(), src_chip_id, eth_connections.front().src_chan);

            const LinkDescriptors link{
                .host = src_asic_desc.host_name,
                .tray_id = *src_asic_desc.tray_id,
                .asic_location = *src_asic_desc.asic_location,
                .channel = eth_connections.front().src_chan,
                .chip_id = src_chip_id,
                .coord = src_coord};

            (is_local ? local_links : remote_links).push_back(link);
        }
    }

    // Select up to MAX_LINKS_TO_RESET links, prioritizing local
    std::vector<LinkDescriptors> selected;
    const auto num_local = std::min(local_links.size(), MAX_LINKS_TO_RESET);
    const auto num_remote = std::min(remote_links.size(), MAX_LINKS_TO_RESET - num_local);

    selected.insert(selected.end(), local_links.begin(), local_links.begin() + num_local);
    selected.insert(selected.end(), remote_links.begin(), remote_links.begin() + num_remote);

    log_info(tt::LogTest, "Found {} links to test ({} local, {} remote)", selected.size(), num_local, num_remote);
    return selected;
}

TEST_F(DirectedRetrainingFixture, TestOnDemandCableRestart) {
    validate_connectivity(get_physical_system_descriptor(), get_cabling_descriptor_path());

    const auto& asic_topology =
        get_physical_system_descriptor().get_asic_topology(get_physical_system_descriptor().my_host_name());
    ASSERT_FALSE(asic_topology.empty()) << "No links available for testing";

    const auto links = collect_mmio_link_params(*this, asic_topology);
    ASSERT_FALSE(links.empty()) << "No MMIO-to-MMIO links found";

    for (const auto& link : links) {
        log_info(
            tt::LogTest,
            "Testing link: host={}, tray={}, asic={}, channel={}",
            link.host,
            link.tray_id,
            link.asic_location,
            link.channel);

        // Take down the link
        set_link_training_status(get_cluster(), link.chip_id, link.coord, 0);
        EXPECT_EQ(get_link_training_status(get_cluster(), link.chip_id, link.coord), 0);

        // Reset the single link we are currently processing
        perform_link_reset(link.host, link.tray_id, link.asic_location, link.channel, get_physical_system_descriptor());

        // Verify the link is back up
        EXPECT_EQ(get_link_training_status(get_cluster(), link.chip_id, link.coord), 1);
    }

    validate_connectivity(get_physical_system_descriptor(), get_cabling_descriptor_path());
}

TEST_F(DirectedRetrainingFixture, TestLinkRetrainingWithDescriptorRefresh) {
    validate_connectivity(get_physical_system_descriptor(), get_cabling_descriptor_path());

    // Take down MMIO-to-MMIO and non-MMIO-to-non-MMIO links
    process_ethernet_connections(
        get_physical_system_descriptor(),
        get_asic_id_to_chip_id(),
        get_cluster(),
        *get_cluster().get_cluster_desc(),
        [&](ChipId chip_id, const tt_xy_pair& coord) { set_link_training_status(get_cluster(), chip_id, coord, 0); });

    // Retrain all downed links
    reset_ethernet_links(
        get_physical_system_descriptor(),
        get_physical_system_descriptor().get_asic_topology(get_physical_system_descriptor().my_host_name()));

    // Verify links are back up at the register level
    process_ethernet_connections(
        get_physical_system_descriptor(),
        get_asic_id_to_chip_id(),
        get_cluster(),
        *get_cluster().get_cluster_desc(),
        [&](ChipId chip_id, const tt_xy_pair& coord) {
            EXPECT_EQ(get_link_training_status(get_cluster(), chip_id, coord), 1);
        });

    // Refresh the cluster descriptor so the driver sees the recovered topology
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    cluster.rediscover_ethernet_links();

    // Re-run physical system discovery with the refreshed descriptor
    get_physical_system_descriptor().clear();
    auto new_psd = tt::tt_metal::run_physical_system_discovery(
        *get_cluster().get_cluster_desc(), distributed_context_, context_->rtoptions().get_target_device(), true, true);
    get_physical_system_descriptor().merge(std::move(new_psd));

    // Validate that the refreshed topology matches the expected cabling
    validate_connectivity(get_physical_system_descriptor(), get_cabling_descriptor_path());
}

// A factory system descriptor listing exactly the cables live discovery found on this host, so a cable faked
// down afterwards is the only factory cable missing.
void write_factory_descriptor_matching_live(
    const tt::tt_metal::PhysicalSystemDescriptor& live, const std::filesystem::path& path) {
    using CableEnd = std::tuple<uint32_t, uint32_t, uint32_t>;  // tray, ASIC location, channel
    std::map<uint32_t, BoardType> tray_board_types;
    std::set<std::pair<CableEnd, CableEnd>> cables;

    const auto& asics = live.get_asic_descriptors();
    for (const auto& [src, edges] : live.get_asic_topology(live.my_host_name())) {
        const auto& src_desc = asics.at(src);
        tray_board_types[*src_desc.tray_id] = src_desc.board_type;
        for (const auto& [dst, eth_connections] : edges) {
            const auto dst_desc = asics.find(dst);
            if (dst_desc == asics.end()) {
                continue;
            }
            for (const auto& eth_connection : eth_connections) {
                const CableEnd a{*src_desc.tray_id, *src_desc.asic_location, eth_connection.src_chan};
                const CableEnd b{*dst_desc->second.tray_id, *dst_desc->second.asic_location, eth_connection.dst_chan};
                cables.emplace(std::min(a, b), std::max(a, b));
            }
        }
    }

    fsd::proto::FactorySystemDescriptor fsd;
    fsd.add_hosts()->set_hostname(live.my_host_name());
    for (const auto& [tray_id, board_type] : tray_board_types) {
        auto* location = fsd.mutable_board_types()->add_board_locations();
        location->set_host_id(0);
        location->set_tray_id(tray_id);
        location->set_board_type(tt::board_type_to_string(board_type));
    }
    const auto set_endpoint = [](fsd::proto::FactorySystemDescriptor::EndPoint* endpoint, const CableEnd& end) {
        endpoint->set_host_id(0);
        endpoint->set_tray_id(std::get<0>(end));
        endpoint->set_asic_location(std::get<1>(end));
        endpoint->set_chan_id(std::get<2>(end));
    };
    for (const auto& [a, b] : cables) {
        auto* connection = fsd.mutable_eth_connections()->add_connection();
        set_endpoint(connection->mutable_endpoint_a(), a);
        set_endpoint(connection->mutable_endpoint_b(), b);
    }

    std::string text;
    TT_FATAL(google::protobuf::TextFormat::PrintToString(fsd, &text), "Failed to print the factory system descriptor");
    std::ofstream(path) << text;
}

// Retraining factory cables through the control plane. Blackhole silicon on a single host: the control plane is
// built with auto-discovery, and a cable is faked down by overwriting the training status of both of its ends.
// FAIL rather than IN_PROGRESS, which UMD discovery would wait out before treating the link as down.
class FactoryCableRetrainFixture : public ::testing::Test {
protected:
    static constexpr uint32_t kEthTrainSuccess = 1;
    static constexpr uint32_t kEthTrainFail = 2;

    struct LinkEnd {
        ChipId chip_id;
        uint8_t channel;
    };

    tt::tt_metal::MetalContext* context_{};
    tt::Cluster* cluster_{};
    std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> distributed_context_;
    std::unique_ptr<tt::tt_fabric::ClusterLinkRetrainer> retrainer_;
    std::array<LinkEnd, 2> link_{};
    std::filesystem::path fsd_path_ = std::filesystem::temp_directory_path() / "factory_cable_retrain_fsd.textproto";
    std::optional<std::string> saved_fsd_path_;

    void SetUp() override {
        context_ = &tt::tt_metal::MetalContext::instance();
        cluster_ = &context_->get_cluster();
        distributed_context_ = context_->get_distributed_context_ptr();

        if (cluster_->arch() != tt::ARCH::BLACKHOLE ||
            cluster_->get_target_device_type() != tt::TargetDevice::Silicon ||
            !cluster_->supports_ethernet_link_retraining()) {
            GTEST_SKIP() << "Workload link retraining requires Blackhole silicon with ETH FW >= 1.9.0";
        }
        if (*distributed_context_->size() != 1) {
            GTEST_SKIP() << "Control plane auto-discovery is single host only";
        }

        const auto live = tt::tt_metal::run_physical_system_discovery(
            *cluster_->get_cluster_desc(), distributed_context_, cluster_->get_target_device_type());
        const auto cable = find_cable_between_mmio_chips(live);
        if (!cable.has_value()) {
            GTEST_SKIP() << "No cable between two MMIO chips on this host";
        }
        link_ = *cable;
        log_info(
            tt::LogTest,
            "Faking down the cable between chip {} channel {} and chip {} channel {}",
            link_[0].chip_id,
            link_[0].channel,
            link_[1].chip_id,
            link_[1].channel);

        write_factory_descriptor_matching_live(live, fsd_path_);
        auto& rtoptions = context_->rtoptions();
        saved_fsd_path_ = rtoptions.get_factory_system_descriptor_path();
        rtoptions.set_factory_system_descriptor_path(fsd_path_.string());

        retrainer_ =
            std::make_unique<tt::tt_fabric::ClusterLinkRetrainer>(*cluster_, context_->hal(), distributed_context_);
    }

    void TearDown() override {
        // Later tests in the process expect the cable up and the cluster to know it.
        if (retrainer_ != nullptr && !link_trained()) {
            retrainer_->retrain(link_endpoints());
            cluster_->rediscover_ethernet_links();
        }
        if (saved_fsd_path_.has_value()) {
            auto& rtoptions = context_->rtoptions();
            rtoptions.set_factory_system_descriptor_path(*saved_fsd_path_);
            rtoptions.set_fsd_link_retrain_attempts(0);
            std::filesystem::remove(fsd_path_);
        }
    }

    std::optional<std::array<LinkEnd, 2>> find_cable_between_mmio_chips(
        const tt::tt_metal::PhysicalSystemDescriptor& live) const {
        const auto* cluster_desc = cluster_->get_cluster_desc();
        const auto& asics = live.get_asic_descriptors();
        for (const auto& [src, edges] : live.get_asic_topology(live.my_host_name())) {
            const ChipId src_chip = asics.at(src).umd_unique_id;
            for (const auto& [dst, eth_connections] : edges) {
                const auto dst_desc = asics.find(dst);
                if (dst_desc == asics.end() || eth_connections.empty()) {
                    continue;
                }
                const ChipId dst_chip = dst_desc->second.umd_unique_id;
                if (cluster_desc->is_chip_mmio_capable(src_chip) && cluster_desc->is_chip_mmio_capable(dst_chip)) {
                    return std::array{
                        LinkEnd{src_chip, eth_connections.front().src_chan},
                        LinkEnd{dst_chip, eth_connections.front().dst_chan}};
                }
            }
        }
        return std::nullopt;
    }

    std::vector<tt::tt_fabric::EthLinkEndpoint> link_endpoints() const {
        std::vector<tt::tt_fabric::EthLinkEndpoint> endpoints;
        for (const auto& end : link_) {
            endpoints.push_back(
                {end.chip_id, end.channel, fmt::format("chip {} channel {}", end.chip_id, end.channel)});
        }
        return endpoints;
    }

    void fake_link_down() const {
        for (const auto& end : link_) {
            set_link_training_status(
                *cluster_, end.chip_id, get_eth_core_coord(*cluster_, end.chip_id, end.channel), kEthTrainFail);
        }
    }

    bool link_trained() const {
        return std::all_of(link_.begin(), link_.end(), [this](const LinkEnd& end) {
            return get_link_training_status(
                       *cluster_, end.chip_id, get_eth_core_coord(*cluster_, end.chip_id, end.channel)) ==
                   kEthTrainSuccess;
        });
    }

    std::unique_ptr<tt::tt_fabric::ControlPlane> make_control_plane(uint32_t retrain_attempts) {
        auto& rtoptions = context_->rtoptions();
        rtoptions.set_fsd_link_retrain_attempts(retrain_attempts);
        return std::make_unique<tt::tt_fabric::ControlPlane>(
            *cluster_,
            rtoptions,
            context_->hal(),
            *distributed_context_,
            tt::tt_metal::FabricSettings{
                .reliability_mode = tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE},
            retrainer_.get());
    }

    static std::size_t missing_link_records(const tt::tt_fabric::ControlPlane& control_plane) {
        return control_plane.get_downed_links().size() + control_plane.get_unused_downed_links().size();
    }

    std::filesystem::path unretrainable_channels_report() const {
        return std::filesystem::path(context_->rtoptions().get_logs_dir()) / "generated" / "fabric" /
               "unretrainable_channels.yaml";
    }
};

TEST_F(FactoryCableRetrainFixture, ControlPlaneRetrainsCableMissingAtStartup) {
    fake_link_down();
    cluster_->rediscover_ethernet_links();
    ASSERT_FALSE(link_trained());

    const auto control_plane = make_control_plane(2);

    EXPECT_TRUE(link_trained());
    EXPECT_EQ(missing_link_records(*control_plane), 0u);
    EXPECT_FALSE(std::filesystem::exists(unretrainable_channels_report()));
}

TEST_F(FactoryCableRetrainFixture, RefreshSeesCableGoDownAndComeBack) {
    const auto control_plane = make_control_plane(0);
    ASSERT_EQ(missing_link_records(*control_plane), 0u);

    fake_link_down();
    control_plane->refresh_connectivity_diff();
    // One cable, recorded from each of its ends.
    EXPECT_EQ(missing_link_records(*control_plane), 2u);

    EXPECT_TRUE(retrainer_->retrain(link_endpoints()).empty());
    control_plane->refresh_connectivity_diff();
    EXPECT_EQ(missing_link_records(*control_plane), 0u);
}

}  // namespace tt::scaleout_tools
