// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/mesh_graph.hpp>

namespace tt::tt_fabric {
namespace fabric_tests {

// create memory maps

// this also has to consider the memory map which has addresses for synchronization etc

struct TestTrafficDataConfig {
    ChipSendType chip_send_type;
    NocSendType noc_send_type;
    size_t num_packets;
    size_t payload_size_bytes;

    void validate() const;
};

struct TestTrafficConfig {
    TestTrafficDataConfig data_config;
    chip_id_t src_phys_chip_id;
    std::optional<std::vector<chip_id_t>> dst_phys_chip_ids;
    std::optional<std::unordered_map<RoutingDirection, uint32_t>> hops;
    std::optional<CoreCoord> src_logical_core;
    std::optional<CoreCoord> dst_logical_core;
    std::optional<std::string_view> sender_kernel_src;
    std::optional<std::string_view> receiver_kernel_src;
    // TODO: add later
    // mode - BW, latency etc

    void validate() const;
};

struct TestTrafficSenderConfig {
    TestTrafficDataConfig data_config;
    std::vector<chip_id_t> dst_phys_chip_ids;
    std::unordered_map<RoutingDirection, uint32_t> hops;
    CoreCoord dst_logical_core;
    size_t target_address;
};

struct TestTrafficReceiverConfig {
    TestTrafficDataConfig data_config;
    uint32_t sender_id;
    size_t target_address;
};

inline void TestTrafficDataConfig::validate() const {
    // TODO remove these once converted to enum classes
    if (this->chip_send_type > ChipSendType::CHIP_SEND_TYPE_LAST) {
        tt::log_fatal(tt::LogTest, "Unknown chip send type: {}", this->chip_send_type);
        throw std::runtime_error("Unexpected traffic data config");
    }
    if (this->noc_send_type > NocSendType::NOC_SEND_TYPE_LAST) {
        tt::log_fatal(tt::LogTest, "Unknown noc send type: {}", this->noc_send_type);
        throw std::runtime_error("Unexpected traffic data config");
    }

    if (this->num_packets == 0) {
        tt::log_fatal(tt::LogTest, "Number of packets should be greater than 0");
        throw std::runtime_error("Unexpected traffic data config");
    }

    // TODO: get the max payload size
    size_t max_payload_size_bytes = 0;
    if (this->payload_size_bytes > max_payload_size_bytes) {
        tt::log_fatal(
            tt::LogTest,
            "Max allowed payload size is: {}, but got: {}",
            max_payload_size_bytes,
            this->payload_size_bytes);
        throw std::runtime_error("Unexpected traffic data config");
    }

    // payload size can only be 0 for NOC_UNICAST_ATOMIC_INC
    if (this->payload_size_bytes == 0 && this->noc_send_type != NocSendType::NOC_UNICAST_ATOMIC_INC) {
        tt::log_fatal(tt::LogTest, "Payload size can only be 0 for NOC_UNICAST_ATOMIC_INC");
        throw std::runtime_error("Unexpected traffic data config");
    }
}

inline void TestTrafficConfig::validate() const {
    this->data_config.validate();

    // validate only one of dst chip ids or hops present
    if (!this->dst_phys_chip_ids.has_value() && !this->hops.has_value()) {
        tt::log_fatal(tt::LogTest, "One of dst phys chip ids or num hops should be present, none specified");
        throw std::runtime_error("Unexpected traffic config");
    } else if (this->dst_phys_chip_ids.has_value() && this->hops.has_value()) {
        tt::log_fatal(tt::LogTest, "Only one of dst phys chip ids or num hops should be present, both specified");
        throw std::runtime_error("Unexpected traffic config");
    }

    // validate for mcast dst chip ids should not be present -> only for ucast
    if (this->data_config.chip_send_type == ChipSendType::CHIP_MULTICAST && this->dst_phys_chip_ids.has_value()) {
        tt::log_fatal(tt::LogTest, "For multicast dst chip ids shouldnt be specified, only hops");
        throw std::runtime_error("Unexpected traffic config");
    }
}

}  // namespace fabric_tests
}  // namespace tt::tt_fabric
