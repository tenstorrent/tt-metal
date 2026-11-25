// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Offline tests for retraining the factory cables live discovery did not find. Descriptors are built in
// memory and the retrainer is a fake, so this runs on one rank with no cluster.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/physical_node_id.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include "llrt/tt_target_device.hpp"
#include "tt_metal/fabric/link_retrainer.hpp"

namespace tt::tt_fabric {
namespace {

using tt::tt_metal::ASICDescriptor;
using tt::tt_metal::AsicID;
using tt::tt_metal::ASICLocation;
using tt::tt_metal::EthConnection;
using tt::tt_metal::PhysicalSystemDescriptor;
using tt::tt_metal::PortType;
using tt::tt_metal::TrayID;
using tt::tt_metal::experimental::make_physical_node_id;

struct AsicSpec {
    std::string host;
    uint32_t loc = 0;
    uint64_t label = 0;
    ChipId chip = 0;
    uint32_t rank = 0;
};

struct CableSpec {
    uint64_t src_label = 0;
    uint8_t src_chan = 0;
    uint64_t dst_label = 0;
    uint8_t dst_chan = 0;
};

// Each cable is recorded from both ends, as discovery does. This rank is rank 0.
PhysicalSystemDescriptor make_descriptor(const std::vector<AsicSpec>& asics, const std::vector<CableSpec>& cables) {
    PhysicalSystemDescriptor descriptor(tt::TargetDevice::Silicon);
    descriptor.set_discovery_data(asics.front().host, /*local_rank=*/0);
    auto& graph = descriptor.get_system_graph().asic_connectivity_graph;
    auto host_of = [&asics](uint64_t label) {
        for (const auto& asic : asics) {
            if (asic.label == label) {
                return asic.host;
            }
        }
        return std::string{};
    };

    for (const auto& asic : asics) {
        const AsicID id{asic.label};
        descriptor.add_asic_descriptor(
            id,
            ASICDescriptor{
                TrayID{1},
                ASICLocation{asic.loc},
                BoardType::P150,
                id,
                asic.chip,
                asic.host,
                make_physical_node_id(asic.host, TrayID{1}, ASICLocation{asic.loc}),
                asic.rank});
        graph[asic.host][id];
    }
    for (const auto& cable : cables) {
        const bool local = host_of(cable.src_label) == host_of(cable.dst_label);
        const auto port = local ? PortType::TRACE : PortType::QSFP_DD;
        graph[host_of(cable.src_label)][AsicID{cable.src_label}].emplace_back(
            AsicID{cable.dst_label},
            std::vector<EthConnection>{EthConnection{cable.src_chan, cable.dst_chan, local, port}});
        graph[host_of(cable.dst_label)][AsicID{cable.dst_label}].emplace_back(
            AsicID{cable.src_label},
            std::vector<EthConnection>{EthConnection{cable.dst_chan, cable.src_chan, local, port}});
    }
    return descriptor;
}

// Two chips on host-a (rank 0) and one on host-b (rank 1): a-b cabled locally, b-c across hosts. The
// factory descriptor labels ASICs 1..3, live labels them with unique ids, as the real pair does.
constexpr uint64_t kFactoryBase = 1;
constexpr uint64_t kLiveBase = 1000;

std::vector<AsicSpec> asics(uint64_t base) {
    return {
        AsicSpec{"host-a", 0, base + 0, 10, 0},
        AsicSpec{"host-a", 1, base + 1, 11, 0},
        AsicSpec{"host-b", 0, base + 2, 20, 1},
    };
}

CableSpec local_cable(uint64_t base) { return CableSpec{base + 0, 0, base + 1, 1}; }
CableSpec cross_host_cable(uint64_t base) { return CableSpec{base + 1, 2, base + 2, 3}; }

PhysicalSystemDescriptor factory() {
    return make_descriptor(asics(kFactoryBase), {local_cable(kFactoryBase), cross_host_cable(kFactoryBase)});
}

PhysicalSystemDescriptor live(const std::vector<CableSpec>& cables) {
    return make_descriptor(asics(kLiveBase), cables);
}

PhysicalSystemDescriptor healthy_live() { return live({local_cable(kLiveBase), cross_host_cable(kLiveBase)}); }

CableEnd end(const std::string& host, uint32_t loc, uint8_t channel) {
    return CableEnd{make_physical_node_id(host, TrayID{1}, ASICLocation{loc}), channel};
}

const auto& world() { return *tt::tt_metal::distributed::multihost::DistributedContext::get_current_world(); }

// Rediscovery hands back whatever the test scripts for that round, and the retrains are recorded.
class FakeLinkRetrainer final : public LinkRetrainer {
public:
    explicit FakeLinkRetrainer(std::function<PhysicalSystemDescriptor(int round)> rediscover) :
        rediscover_(std::move(rediscover)) {}

    bool can_retrain() const override { return true; }

    std::vector<EthLinkEndpoint> retrain(const std::vector<EthLinkEndpoint>& local_endpoints) override {
        retrains.push_back(local_endpoints);
        return {};
    }

    PhysicalSystemDescriptor rediscover() override { return rediscover_(static_cast<int>(retrains.size())); }

    std::vector<std::vector<EthLinkEndpoint>> retrains;

private:
    std::function<PhysicalSystemDescriptor(int)> rediscover_;
};

TEST(FactoryLinkRetrain, AMatchingLiveDescriptorHasNothingToRetrain) {
    EXPECT_TRUE(retrainable_missing_cables(factory(), healthy_live()).empty());
}

// Joined on address, not label, and reported once per cable however many directions the diff lists.
TEST(FactoryLinkRetrain, AMissingCableIsReportedOnceByAddress) {
    const auto missing = retrainable_missing_cables(factory(), live({cross_host_cable(kLiveBase)}));

    ASSERT_EQ(missing.size(), 1u);
    EXPECT_EQ(missing[0].first, end("host-a", 0, 0));
    EXPECT_EQ(missing[0].second, end("host-a", 1, 1));
}

// A channel that trained to the wrong peer is miscabled. Retraining it would take down a working link.
TEST(FactoryLinkRetrain, AChannelUpTowardAnotherPeerIsNotRetrained) {
    const auto missing = retrainable_missing_cables(
        factory(), live({cross_host_cable(kLiveBase), CableSpec{kLiveBase + 0, 0, kLiveBase + 2, 7}}));

    EXPECT_TRUE(missing.empty());
}

// Only the ends on chips this rank discovered, addressed by the chip id discovery recorded.
TEST(FactoryLinkRetrain, EndpointsAreTheEndsThisRankOwns) {
    const auto descriptor = live({local_cable(kLiveBase)});
    const auto missing = retrainable_missing_cables(factory(), descriptor);
    ASSERT_EQ(missing.size(), 1u);

    const auto endpoints = local_endpoints_for_cables(missing, descriptor);

    ASSERT_EQ(endpoints.size(), 1u);
    EXPECT_EQ(endpoints[0].chip_id, 11);
    EXPECT_EQ(endpoints[0].channel, 2u);
}

TEST(FactoryLinkRetrain, BothEndsOfASameRankCableAreRetrained) {
    const auto descriptor = live({cross_host_cable(kLiveBase)});
    const auto endpoints = local_endpoints_for_cables(retrainable_missing_cables(factory(), descriptor), descriptor);

    ASSERT_EQ(endpoints.size(), 2u);
    EXPECT_EQ(endpoints[0].chip_id, 10);
    EXPECT_EQ(endpoints[0].channel, 0u);
    EXPECT_EQ(endpoints[1].chip_id, 11);
    EXPECT_EQ(endpoints[1].channel, 1u);
}

TEST(FactoryLinkRetrain, StopsOnceTheCableComesBack) {
    FakeLinkRetrainer retrainer([](int) { return healthy_live(); });
    auto descriptor = std::make_unique<PhysicalSystemDescriptor>(live({cross_host_cable(kLiveBase)}));

    const auto result = retrain_missing_factory_cables(retrainer, world(), factory(), descriptor, 5);

    EXPECT_EQ(result.rounds, 1u);
    EXPECT_EQ(result.missing_before.size(), 1u);
    EXPECT_TRUE(result.missing_after.empty());
    ASSERT_EQ(retrainer.retrains.size(), 1u);
    EXPECT_EQ(retrainer.retrains[0].size(), 2u);
    // The caller's descriptor is the rediscovered one.
    EXPECT_TRUE(retrainable_missing_cables(factory(), *descriptor).empty());
}

TEST(FactoryLinkRetrain, ACableThatNeverComesBackSpendsEveryRound) {
    FakeLinkRetrainer retrainer([](int) { return live({cross_host_cable(kLiveBase)}); });
    auto descriptor = std::make_unique<PhysicalSystemDescriptor>(live({cross_host_cable(kLiveBase)}));

    const auto result = retrain_missing_factory_cables(retrainer, world(), factory(), descriptor, 3);

    EXPECT_EQ(result.rounds, 3u);
    EXPECT_EQ(result.missing_after, result.missing_before);
    EXPECT_EQ(retrainer.retrains.size(), 3u);
}

// No progress rule: a retrain that fails once can succeed the next time, so a round that recovers
// nothing does not end the loop.
TEST(FactoryLinkRetrain, ARoundThatRecoversNothingDoesNotStopTheNext) {
    FakeLinkRetrainer retrainer(
        [](int round) { return round < 2 ? live({cross_host_cable(kLiveBase)}) : healthy_live(); });
    auto descriptor = std::make_unique<PhysicalSystemDescriptor>(live({cross_host_cable(kLiveBase)}));

    const auto result = retrain_missing_factory_cables(retrainer, world(), factory(), descriptor, 5);

    EXPECT_EQ(result.rounds, 2u);
    EXPECT_TRUE(result.missing_after.empty());
}

// Each round retrains what is missing now. A cable that came back is left alone; one that dropped is added.
TEST(FactoryLinkRetrain, EachRoundRetrainsWhatIsMissingThen) {
    FakeLinkRetrainer retrainer([](int round) { return round == 1 ? live({local_cable(kLiveBase)}) : healthy_live(); });
    auto descriptor = std::make_unique<PhysicalSystemDescriptor>(live({cross_host_cable(kLiveBase)}));

    const auto result = retrain_missing_factory_cables(retrainer, world(), factory(), descriptor, 5);

    EXPECT_EQ(result.rounds, 2u);
    ASSERT_EQ(retrainer.retrains.size(), 2u);
    ASSERT_EQ(retrainer.retrains[0].size(), 2u);
    EXPECT_EQ(retrainer.retrains[0][0].chip_id, 10);
    // Round two: the cross-host cable, whose only end on this rank is chip 11 channel 2.
    ASSERT_EQ(retrainer.retrains[1].size(), 1u);
    EXPECT_EQ(retrainer.retrains[1][0].chip_id, 11);
    EXPECT_EQ(retrainer.retrains[1][0].channel, 2u);
}

TEST(FactoryLinkRetrain, NothingMissingMeansNoRetrain) {
    FakeLinkRetrainer retrainer([](int) { return healthy_live(); });
    auto descriptor = std::make_unique<PhysicalSystemDescriptor>(healthy_live());

    const auto result = retrain_missing_factory_cables(retrainer, world(), factory(), descriptor, 3);

    EXPECT_EQ(result.rounds, 0u);
    EXPECT_TRUE(retrainer.retrains.empty());
}

TEST(FactoryLinkRetrain, ReportListsEveryEndInTheValidationToolFormat) {
    const auto descriptor = live({cross_host_cable(kLiveBase)});
    const auto missing = retrainable_missing_cables(factory(), descriptor);
    const auto path = std::filesystem::temp_directory_path() / "factory_link_retrain_test" / "unretrainable.yaml";

    write_unretrainable_channels(path, missing, descriptor, 4);

    std::ifstream in(path);
    ASSERT_TRUE(in.is_open());
    const std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    std::filesystem::remove_all(path.parent_path());

    EXPECT_NE(contents.find("total_retrain_iterations: 4\n"), std::string::npos) << contents;
    EXPECT_NE(
        contents.find("  - host: host-a\n"
                      "    tray_id: 1\n"
                      "    asic_location: 0\n"
                      "    channel: 0\n"
                      "    asic_unique_id: 0x00000003e8\n"),
        std::string::npos)
        << contents;
    EXPECT_NE(
        contents.find("  - host: host-a\n"
                      "    tray_id: 1\n"
                      "    asic_location: 1\n"
                      "    channel: 1\n"
                      "    asic_unique_id: 0x00000003e9\n"),
        std::string::npos)
        << contents;
}

}  // namespace
}  // namespace tt::tt_fabric
