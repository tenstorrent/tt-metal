// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

#include "tt_fabric_test_config.hpp"
#include "tt_fabric_test_common.hpp"
#include "tt_fabric_test_device_setup.hpp"
#include "tt_fabric_test_traffic.hpp"

using TestPhysicalMeshes = tt::tt_fabric::fabric_tests::TestPhysicalMeshes;
using TestFabricFixture = tt::tt_fabric::fabric_tests::TestFabricFixture;
using TestDevice = tt::tt_fabric::fabric_tests::TestDevice;
using TestTrafficConfig = tt::tt_fabric::fabric_tests::TestTrafficConfig;
using TestTrafficSenderConfig = tt::tt_fabric::fabric_tests::TestTrafficSenderConfig;
using TestTrafficReceiverConfig = tt::tt_fabric::fabric_tests::TestTrafficReceiverConfig;
using TestWorkerType = tt::tt_fabric::fabric_tests::TestWorkerType;

class TestContext {
public:
    void init();
    void handle_test_config();  // parse and process test config
    void open_devices(tt::tt_metal::FabricConfig fabric_config);
    void process_traffic_config(TestTrafficConfig traffic_config);
    void close_devices();

private:
    void validate_physical_chip_id(chip_id_t physical_chip_id) const;
    CoreCoord find_receiver_core(const std::vector<chip_id_t>& phys_chip_ids);
    void add_traffic_config(const TestTrafficConfig& traffic_config);
    TestDevice& get_test_device(chip_id_t physical_chip_id);

    TestPhysicalMeshes physical_meshes_;
    TestFabricFixture fixture_;
    std::unordered_map<chip_id_t, TestDevice> test_devices_;
};

void TestContext::validate_physical_chip_id(chip_id_t physical_chip_id) const {
    if (this->test_devices_.find(physical_chip_id) == this->test_devices_.end()) {
        tt::log_fatal(tt::LogTest, "Unknown physical chip id: {}", physical_chip_id);
        throw std::runtime_error("Unexpected physical chip id");
    }
}

CoreCoord TestContext::find_receiver_core(const std::vector<chip_id_t>& phys_chip_ids) {
    std::unordered_map<CoreCoord, uint32_t> available_cores_histogram;
    for (const auto& chip_id : phys_chip_ids) {
        const auto& available_cores = this->test_devices_.at(chip_id).get_available_worker_cores();
        for (const auto& core : available_cores) {
            available_cores_histogram[core]++;
        }
    }

    std::optional<CoreCoord> available_core;
    for (const auto& [core, count] : available_cores_histogram) {
        // for now return the first core that is avaialble across all the chips
        if (count == phys_chip_ids.size()) {
            available_core = core;
            break;
        }
    }

    if (!available_core.has_value()) {
        tt::log_fatal(tt::LogTest, "Unable to find a common recv core from chips: {}", phys_chip_ids);
        throw std::runtime_error("No common cores found for allocation");
    }

    return available_core.value();
}

void TestContext::add_traffic_config(const TestTrafficConfig& traffic_config) {
    // by this point assume that the traffic config is well formed/massaged
    // for now assume that the cores are not shared
    // later we can grab the available address/core pairs

    const auto& src_phys_chip_id = traffic_config.src_phys_chip_id;
    auto& src_test_device = this->test_devices_.at(src_phys_chip_id);

    const auto& dst_phys_chip_ids = traffic_config.dst_phys_chip_ids.value();
    // this is only a representative of the dst devices used for common operations for dst devices
    auto& dst_rep_test_device = this->test_devices_.at(dst_phys_chip_ids[0]);

    CoreCoord src_logical_core;
    if (traffic_config.src_logical_core.has_value()) {
        src_logical_core = traffic_config.src_logical_core.value();
    } else {
        src_logical_core = src_test_device.allocate_worker_core();
    }
    src_test_device.reserve_core_for_worker(TestWorkerType::SENDER, src_logical_core, traffic_config.sender_kernel_src);

    CoreCoord dst_logical_core;
    if (traffic_config.dst_logical_core.has_value()) {
        dst_logical_core = traffic_config.dst_logical_core.value();
    } else {
        dst_logical_core = this->find_receiver_core(dst_phys_chip_ids);
    }
    for (const auto& dst_chip_id : dst_phys_chip_ids) {
        this->test_devices_.at(dst_chip_id)
            .reserve_core_for_worker(TestWorkerType::RECEIVER, dst_logical_core, traffic_config.receiver_kernel_src);
    }

    size_t target_address = dst_rep_test_device.allocate_address_for_sender(dst_logical_core);
    uint32_t sender_id = src_test_device.get_worker_id(src_logical_core);

    TestTrafficSenderConfig sender_config = {
        .data_config = traffic_config.data_config,
        .dst_phys_chip_ids = dst_phys_chip_ids,
        .hops = traffic_config.hops.value(),
        .dst_logical_core = dst_logical_core,
        .target_address = target_address};

    TestTrafficReceiverConfig receiver_config = {
        .data_config = traffic_config.data_config, .sender_id = sender_id, .target_address = target_address};

    src_test_device.add_sender_traffic_config(src_logical_core, sender_config);
    for (const auto& dst_chip_id : dst_phys_chip_ids) {
        this->test_devices_.at(dst_chip_id).add_receiver_traffic_config(dst_logical_core, receiver_config);
    }
}

TestDevice& TestContext::get_test_device(chip_id_t physical_chip_id) {
    this->validate_physical_chip_id(physical_chip_id);
    return this->test_devices_.at(physical_chip_id);
}

void TestContext::init() {
    this->physical_meshes_.setup_physical_meshes();
    this->fixture_.setup_devices();
    this->physical_meshes_.print_meshes();
}

void TestContext::open_devices(tt::tt_metal::FabricConfig fabric_config) {
    this->fixture_.open_devices(fabric_config);

    for (const auto& chip_id : this->fixture_.get_available_chip_ids()) {
        auto* device_handle = this->fixture_.get_device_handle(chip_id);
        this->test_devices_.emplace(chip_id, device_handle);
    }
}

void TestContext::process_traffic_config(TestTrafficConfig traffic_config) {
    traffic_config.validate();

    const auto src_chip_id = traffic_config.src_phys_chip_id;
    this->validate_physical_chip_id(src_chip_id);

    const auto& chip_send_type = traffic_config.data_config.chip_send_type;

    std::vector<chip_id_t> dst_phys_chip_ids;
    std::vector<std::unordered_map<tt::tt_fabric::RoutingDirection, uint32_t>> hops_vector;
    if (traffic_config.dst_phys_chip_ids.has_value()) {
        dst_phys_chip_ids = traffic_config.dst_phys_chip_ids.value();
        for (const auto& dst_chip_id : dst_phys_chip_ids) {
            this->validate_physical_chip_id(dst_chip_id);
        }

        for (const auto& dst_chip_id : dst_phys_chip_ids) {
            hops_vector.push_back(this->physical_meshes_.get_hops_to_chip(src_chip_id, dst_chip_id));
        }
    } else if (traffic_config.hops.has_value()) {
        // TODO: validate dst chip ids or number of hops based on the topology
        const auto& hops = traffic_config.hops.value();
        // get the dest chips based on chip send type. For unicast, hops are for the same dest chip
        // for mcast, capture each chip along the mcast hops
        dst_phys_chip_ids = this->physical_meshes_.get_chips_from_hops(src_chip_id, hops, chip_send_type);
        hops_vector.push_back(hops);
    }

    // TODO: handle src and dst logical cores
    // TODO: handle kernel srcs

    // if bidirectional - add sender for every receiver

    // additional handling - if mcast mode then add receivers for every chip in the route

    if (hops_vector.size() == 1) {
        // single sender config
        auto config = traffic_config;
        config.dst_phys_chip_ids = dst_phys_chip_ids;
        config.hops = hops_vector[0];
        this->add_traffic_config(config);
    } else {
        // each dest chip and hop makes a separate config
        for (auto idx = 0; idx < hops_vector.size(); idx++) {
            auto config = traffic_config;
            config.dst_phys_chip_ids = {dst_phys_chip_ids[idx]};
            config.hops = {hops_vector[idx]};
            this->add_traffic_config(config);
        }
    }
}

void TestContext::close_devices() { this->fixture_.close_devices(); }

// TODO: method to get random chip send type
// TODO: method to get random noc send type
// TODO: method to get random hops (based on mode - 1D/2D)
// TODO: method to get random dest chip (based on mode - 1D/2D)

void setup_fabric(
    tt::tt_fabric::fabric_tests::TestFabricSetup fabric_setup_config, std::vector<TestDevice>& test_devices) {}

/*
void setup_traffic_config(TestTrafficDataConfig data_config, chip_id_t src_phys_chip_id);

void setup_traffic_config(TestTrafficDataConfig data_config, chip_id_t src_phys_chip_id);
*/

int main(int argc, char** argv) {
    std::vector<std::string> input_args(argv, argv + argc);
    tt::tt_fabric::fabric_tests::parse_config(input_args);

    TestContext test_context;
    test_context.init();

    test_context.open_devices(tt::tt_metal::FabricConfig::FABRIC_1D);

    // fabric setup
    // setup_fabric()

    // all-to-all mode

    // workers setup

    // launch programs

    //

    test_context.close_devices();

    return 0;
}
