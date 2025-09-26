// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/control_plane.hpp>
#include "fabric_fixture.hpp"
#include <hostdevcommon/fabric_common.h>
#include <tt-metalium/routing_table_generator.hpp>  // FabricNodeId
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/fabric/control_channel_interface.hpp"
#include <vector>
#include <chrono>

namespace tt::tt_fabric {
namespace fabric_router_tests {
namespace fabric_control_channel_tests {

using FabricControlChannelBaseFixture = BaseFabricFixture;
using Fabric1DControlChannelFixture = Fabric1DFixture;
using Fabric2DControlChannelFixture = Fabric2DFixture;
using Fabric2DDynamicControlChannelFixture = Fabric2DDynamicFixture;

enum class TestType : uint8_t {
    POINT_TO_POINT = 0,
    POINT_TO_POINT_RANDOM = 1,
    ALL_TO_ALL = 1,
};

struct TestEndPoint {
    FabricNodeId node_id;
    chan_id_t channel_id;
};

uint32_t generate_sequence_id() { return std::chrono::system_clock::now().time_since_epoch().count(); }

std::vector<std::pair<TestEndPoint, TestEndPoint>> generate_test_endpoints(
    FabricControlChannelBaseFixture* fixture, TestType test_type) {
    std::vector<std::pair<TestEndPoint, TestEndPoint>> test_endpoints;

    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    std::optional<TestEndPoint> initiator_endpoint = std::nullopt;
    std::optional<TestEndPoint> target_endpoint = std::nullopt;

    // for now just need to find two endpoints
    for (const auto& device : fixture->get_devices()) {
        const auto physical_chip_id = device->id();
        const auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(physical_chip_id);
        const auto& active_fabric_eth_channels = control_plane.get_active_fabric_eth_channels(fabric_node_id);
        for (const auto& [eth_chan, _] : active_fabric_eth_channels) {
            if (!initiator_endpoint.has_value()) {
                initiator_endpoint = TestEndPoint{fabric_node_id, eth_chan};
            } else if (!target_endpoint.has_value()) {
                target_endpoint = TestEndPoint{fabric_node_id, eth_chan};
                break;
            }
        }
        if (initiator_endpoint.has_value() && target_endpoint.has_value()) {
            break;
        }
    }

    TT_FATAL(initiator_endpoint.has_value() && target_endpoint.has_value(), "Could not find two endpoints");

    log_info(
        tt::LogTest,
        "Found endpoints: node: {}, eth_chan: {} -> node: {}, eth_chan: {}",
        initiator_endpoint.value().node_id,
        initiator_endpoint.value().channel_id,
        target_endpoint.value().node_id,
        target_endpoint.value().channel_id);

    test_endpoints.push_back({initiator_endpoint.value(), target_endpoint.value()});

    return test_endpoints;
}

void run_heartbeat_check_point_to_point(
    FabricControlChannelBaseFixture* fixture, TestEndPoint initiator_endpoint, TestEndPoint target_endpoint) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& control_channel_interface = control_plane.get_control_channel_interface();

    const auto sequence_id = generate_sequence_id();
    log_info(tt::LogTest, "Requesting heartbeat check for sequence id: {}", sequence_id);
    const auto result = control_channel_interface.request_remote_heartbeat_check(
        initiator_endpoint.node_id,
        initiator_endpoint.channel_id,
        target_endpoint.node_id,
        target_endpoint.channel_id,
        sequence_id);

    log_info(tt::LogTest, "Heartbeat check result: {}", result);

    // poll for completion on the initiator endpoint
    const auto initiator_completion_result = control_channel_interface.poll_for_remote_heartbeat_request_completion(
        initiator_endpoint.node_id, initiator_endpoint.channel_id, sequence_id, 10000, 200);
    EXPECT_EQ(initiator_completion_result, ControlChannelResult::SUCCESS);
    log_info(tt::LogTest, "Initiator result: {}", initiator_completion_result);

    // also check for completion on the target endpoint
    // no need to poll for completion on the target endpoint
    const bool target_completed = control_channel_interface.check_remote_heartbeat_request_completed(
        target_endpoint.node_id, target_endpoint.channel_id, sequence_id);
    EXPECT_EQ(target_completed, true);
}

void run_heartbeat_check(
    FabricControlChannelBaseFixture* fixture, std::vector<std::pair<TestEndPoint, TestEndPoint>> test_endpoints) {
    for (const auto& [initiator_endpoint, target_endpoint] : test_endpoints) {
        run_heartbeat_check_point_to_point(fixture, initiator_endpoint, target_endpoint);
    }
}

TEST_F(Fabric1DControlChannelFixture, TestFabric1DControlChannelPointToPoint) {
    run_heartbeat_check(this, generate_test_endpoints(this, TestType::POINT_TO_POINT));
}

TEST_F(Fabric2DControlChannelFixture, TestFabric2DControlChannelPointToPoint) {
    run_heartbeat_check(this, generate_test_endpoints(this, TestType::POINT_TO_POINT));
}

TEST_F(Fabric2DDynamicControlChannelFixture, TestFabric2DDynamicControlChannelPointToPoint) {
    run_heartbeat_check(this, generate_test_endpoints(this, TestType::POINT_TO_POINT));
}

TEST_F(Fabric1DControlChannelFixture, TestFabric1DControlChannelAllToAll) {
    run_heartbeat_check(this, generate_test_endpoints(this, TestType::ALL_TO_ALL));
}

TEST_F(Fabric2DControlChannelFixture, TestFabric2DControlChannelAllToAll) {
    run_heartbeat_check(this, generate_test_endpoints(this, TestType::ALL_TO_ALL));
}

TEST_F(Fabric2DDynamicControlChannelFixture, TestFabric2DDynamicControlChannelAllToAll) {
    run_heartbeat_check(this, generate_test_endpoints(this, TestType::ALL_TO_ALL));
}

}  // namespace fabric_control_channel_tests
}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
