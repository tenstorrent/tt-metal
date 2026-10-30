// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/tt_metal/deployment/eth/common.hpp"
#include "tt_metal/tt_metal/deployment/deployment_common.hpp"
#include "tt_metal/tt_metal/deployment/kernels/sync_types.hpp"

#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "tt_metal/test_utils/stimulus.hpp"
#include "command_queue_fixture.hpp"
#include "tt_metal/tt_metal/eth/eth_test_common.hpp"
#include "distributed/mesh_device_impl.hpp"

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

static void delay_until_env() {
    char* run_at_strp = getenv("RUN_AT");
    if (run_at_strp) {
        uint64_t timestamp = std::stoll(run_at_strp);
        std::chrono::system_clock::time_point tp{std::chrono::seconds(timestamp)};

        std::time_t now_time_t = std::chrono::system_clock::to_time_t(tp);

        std::tm* utc_time = std::gmtime(&now_time_t);
        std::stringstream ss;
        ss << std::put_time(utc_time, "%Y-%m-%d %H:%M:%S");

        log_info(tt::LogTest, "Run at UTC time: {}", ss.str());

        std::this_thread::sleep_until(tp);
    }
}

struct chip_core {
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    IDevice* chip;
    CoreCoord core;
};

std::map<std::string, struct chip_core> get_coreuid_to_core(
    std::span<std::shared_ptr<distributed::MeshDevice>> devices_) {
    /* =================== */
    std::map<std::string, struct chip_core> ret;

    for (const auto& mesh_device : devices_) {
        auto* const chip = mesh_device->impl().get_devices()[0];
        for (const auto& core : chip->get_active_ethernet_cores(true)) {
            ret[get_eth_unique_id(chip, core)] = {
                .mesh_device = mesh_device,
                .chip = chip,
                .core = core,
            };
        }
    }

    return ret;
}

std::vector<struct chip_core> get_cores_to_test(
    std::span<std::shared_ptr<distributed::MeshDevice>> devices_, std::span<std::string> test_endpoints) {
    /* =================== */
    std::vector<struct chip_core> ret;

    std::map<std::string, struct chip_core> coreuid_to_core = get_coreuid_to_core(devices_);
    for (auto& uid : test_endpoints) {
        auto core = coreuid_to_core.at(uid);
        ret.push_back(core);
    }

    return ret;
}

template <typename FIXTURE>
static bool run_test_bandwidth_bidir(
    FIXTURE* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const CoreCoord& core,
    DataMovementProcessor processor0,
    span<uint32_t> inputs) {
    /* =================== */
    auto* const device = mesh_device->impl().get_devices()[0];

    TEST_PARAM(uint32_t, transfer_size, 160 * 1024, "ETH_TEST_TRANSFER_SIZE");
    TEST_PARAM(uint32_t, transfer_count, 20 << 10, "ETH_TEST_TRANSFER_COUNT");

    uint32_t num_bytes_per_send = transfer_size / 2;
    uint64_t total_transferred = (uint64_t)transfer_size * transfer_count;

    span<uint32_t> inp = inputs.subspan(0, transfer_size / sizeof inp[0]);

    struct l1_allocator alloc = new_erisc_allocator();

    uint32_t iter_l1_address = l1_alloc(&alloc, sizeof(uint64_t));
    uint32_t recv_l1_address = l1_alloc(&alloc, transfer_size);
    uint32_t send_delta_addr = l1_alloc(&alloc, sizeof(uint64_t));
    uint32_t send_l1_address = l1_alloc(&alloc, transfer_size);

    map<shared_ptr<distributed::MeshDevice>, shared_ptr<tt_metal::Program>> programs = {
        {mesh_device, make_shared<Program>()},
    };

    prepare_bidir(
        device,
        core,
        transfer_size,
        transfer_count,
        send_delta_addr,
        inp,
        processor0,
        num_bytes_per_send,
        iter_l1_address,
        send_l1_address,
        recv_l1_address,
        0,  // TODO should be swapped on two hosts
        1,  // TODO should be swapped on two hosts
        programs[mesh_device].get());

    vector<struct core_setup> cores = {
        {
            .program = programs[mesh_device],
            .mesh_device = mesh_device,
            .core = core,
            .iter_l1_addr = iter_l1_address,
            .expected_count = transfer_count,
        },
    };
    using namespace std::chrono_literals;
    wait_to_finish_eth_timeout_cores(fixture, cores, programs, 1s);

    double threshold = get_eth_bw() * 0.7;

    bool pass = true;
    pass &= bandwidth_check(device, core, send_delta_addr, total_transferred, threshold);
    pass &= data_check(device, core, recv_l1_address, inp);
    return pass;
}

TEST_F(MeshDispatchFixture, TensixDeploymentEthernetDist02BandwidthBidir) {
    if (not this->slow_dispatch_) {
        GTEST_SKIP() << "fast dispatch to active ethernet has been removed";
    }

    vector<LinkError> errors;

    vector<uint32_t> inputs = generate_uniform_random_vector<uint32_t>(0, 100, 1 << 20);

    print_detected_devices();
    ASSERT_TRUE(ensure_links(devices_));

    std::vector<std::string> test_endpoints = {
        "asic:0x2fb57e3b70ec5ed7/eth11",
    };

    delay_until_env();

    for (auto& core : get_cores_to_test(devices_, test_endpoints)) {
        log_info(tt::LogTest, "running {}", core);
        const auto processor = static_cast<DataMovementProcessor>(0);

        bool passed = run_test_bandwidth_bidir(this, core.mesh_device, core.core, processor, inputs);

        if (!passed) {
            errors.emplace_back(core.chip->id(), core.chip->id(), core.core, core.core, processor);
        }

        log_info(tt::LogTest, "passed: {}", passed);
    }

    print_summary(errors);
    ASSERT_TRUE(errors.empty());
#if 0
	const auto num_eriscs = MetalContext::instance().hal().get_num_risc_processors(HalProgrammableCoreType::ACTIVE_ETH);

    vector<LinkError> errors;
    int n = 0;

    vector<uint32_t> inputs = generate_uniform_random_vector<uint32_t>(0, 100, 1 << 20);

    for (const auto& sender_mesh_device : devices_) {
        auto* const sender_device = sender_mesh_device->impl().get_devices()[0];
        for (const auto& receiver_mesh_device : devices_) {
            auto* const receiver_device = receiver_mesh_device->impl().get_devices()[0];

            log_info(
                tt::LogTest,
                "sender device id: {} ({}, {}), receiver device id: {} ({}, {})",
                sender_device->id(),
                pci_bdf_for_device_id(sender_device->id()),
                get_ubb(sender_device),
                receiver_device->id(),
                pci_bdf_for_device_id(receiver_device->id()),
                get_ubb(receiver_device));

            for (const auto& sender_core : sender_device->get_active_ethernet_cores(true)) {
                if (!eth_core_connects_within_cluster(sender_device, sender_core)) {
                    continue;
                }
                auto [device_id, receiver_core] = sender_device->get_connected_ethernet_core(sender_core);
                if (receiver_device->id() != device_id) {
                    continue;
                }

                log_info(
                    tt::LogTest, "  {}", format_link_info(sender_device, sender_core, receiver_device, receiver_core));

                for (uint32_t erisc_idx = 0; erisc_idx < num_eriscs; erisc_idx++) {
                    const auto processor = static_cast<DataMovementProcessor>(erisc_idx);

                    log_info(tt::LogTest, "    running on {}", processor);
                    bool passed = run_test_bandwidth_bidir(
                        this, sender_mesh_device, receiver_mesh_device, sender_core, receiver_core, processor, inputs);
                    if (!passed) {
                        errors.emplace_back(
                            sender_device->id(), receiver_device->id(), sender_core, receiver_core, processor);
                    }
                    log_info(tt::LogTest, "    done");

                    n++;
                }
            }
        }
    }

    log_info(tt::LogTest, "Ran {} tests", n);

    print_summary(errors);
    ASSERT_TRUE(errors.empty());
#endif
}

}  // namespace tt::tt_metal
