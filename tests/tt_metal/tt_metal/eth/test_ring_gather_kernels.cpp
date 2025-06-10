// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "device_fixture.hpp"
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/kernel_types.hpp>
#include "llrt.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/test_utils/df/float32.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "umd/device/types/xy_pair.h"

using std::vector;
using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

constexpr std::int32_t WORD_SIZE = 16;  // 16 bytes per eth send packet

struct BankedConfig {
    size_t num_pages = 1;
    size_t size_bytes = 1 * 2 * 32 * 32;
    size_t page_size_bytes = 2 * 32 * 32;
    tt_metal::BufferType input_buffer_type = tt_metal::BufferType::L1;
    tt_metal::BufferType output_buffer_type = tt_metal::BufferType::L1;
    tt::DataFormat l1_data_format = tt::DataFormat::Float16_b;
};

std::vector<int> get_hamiltonian_cycle(vector<vector<int>>& adj, int N, int s = 0) {
    std::vector<std::vector<int>> dp(N, std::vector<int>(1 << N, -1));

    for (int i = 0; i < N; ++i) {
        if (adj[s][i]) {
            dp[i][(1 << i)] = i;
        }
    }

    for (int i = 0; i < (1 << N); ++i) {
        for (int j = 0; j < N; ++j) {
            if (i & (1 << j)) {
                for (int k = 0; k < N; ++k) {
                    if (i & (1 << k) && adj[k][j] && j != k && dp[k][i ^ (1 << j)] != -1) {
                        dp[j][i] = k;
                        break;
                    }
                }
            }
        }
    }

    for (int i = 0; i < N; ++i) {
        int m = (1 << N) - 1;

        if (dp[i][m] != -1 && i == s) {
            std::vector<int> path;
            path.reserve(N + 1);
            path.push_back(i);

            for (int j = 0; j < N - 1; ++j) {
                path.push_back(dp[*path.rbegin()][m]);
                m ^= 1 << *(path.rbegin() + 1);
            }
            path.push_back(s);
            return path;
        }
    }
    return {};
}

std::vector<tt_metal::IDevice*> get_device_ring(std::vector<tt::tt_metal::IDevice*> devices) {
    std::vector<std::vector<int>> adj(devices.size(), std::vector<int>(devices.size(), 0));
    for (uint32_t i = 0; i < devices.size(); ++i) {
        const auto& device = devices[i];
        auto ethernet_connected_device_ids =
            tt::tt_metal::MetalContext::instance().get_cluster().get_ethernet_connected_device_ids(device->id());
        for (const auto& connected_device_id : ethernet_connected_device_ids) {
            for (uint32_t j = 0; j < devices.size(); ++j) {
                if (devices[j]->id() == connected_device_id) {
                    adj[i][j] = 1;
                }
            }
        }
    }

    const auto& device_ring_idx = get_hamiltonian_cycle(adj, devices.size(), 0);
    std::vector<tt_metal::IDevice*> device_ring;
    device_ring.reserve(device_ring_idx.size());
    for (const auto& device_idx : device_ring_idx) {
        device_ring.push_back(devices[device_idx]);
    }
    return device_ring;
}

std::vector<std::tuple<tt_metal::IDevice*, tt_metal::IDevice*, CoreCoord, CoreCoord>> get_sender_receiver_cores(
    std::vector<tt::tt_metal::IDevice*> device_ring) {
    std::vector<std::tuple<tt_metal::IDevice*, tt_metal::IDevice*, CoreCoord, CoreCoord>> sender_receivers;
    sender_receivers.reserve(device_ring.size() - 1);

    // Special case for 2 devices to ensure core pairs are not the same for send and receive
    if (device_ring.size() - 1 == 2) {
        const auto& first_device = device_ring[0];
        const auto& second_device = device_ring[1];
        uint32_t i = 0;
        for (const auto& first_eth_core : first_device->get_active_ethernet_cores(true)) {
            if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(
                    first_device->id(), first_eth_core)) {
                continue;
            }
            auto [device_id, second_eth_core] = first_device->get_connected_ethernet_core(first_eth_core);
            if (second_device->id() == device_id) {
                tt_metal::IDevice *sender_device, *receiver_device;
                CoreCoord sender_eth_core, receiver_eth_core;
                if (i == 0) {
                    sender_device = first_device, receiver_device = second_device;
                    sender_eth_core = first_eth_core, receiver_eth_core = second_eth_core;
                } else {
                    sender_device = second_device, receiver_device = first_device;
                    sender_eth_core = second_eth_core, receiver_eth_core = first_eth_core;
                }
                sender_receivers.push_back({sender_device, receiver_device, sender_eth_core, receiver_eth_core});
                log_info(
                    tt::LogTest,
                    "Sender: {} Receiver: {} Sender Eth: {} Receiver Eth: {}",
                    sender_device->id(),
                    receiver_device->id(),
                    sender_eth_core.str(),
                    receiver_eth_core.str());
                if (i > 0) {
                    break;
                }
                i++;
            }
        }
    } else {
        for (uint32_t i = 0; i < device_ring.size() - 1; ++i) {
            const auto& sender_device = device_ring[i];
            const auto& receiver_device = device_ring[i + 1];
            for (const auto& sender_eth_core : sender_device->get_active_ethernet_cores(true)) {
                if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(
                        sender_device->id(), sender_eth_core)) {
                    continue;
                }
                auto [device_id, receiver_eth_core] = sender_device->get_connected_ethernet_core(sender_eth_core);
                if (receiver_device->id() == device_id) {
                    sender_receivers.push_back({sender_device, receiver_device, sender_eth_core, receiver_eth_core});
                    log_info(
                        tt::LogTest,
                        "Sender: {} Receiver: {} Sender Eth: {} Receiver Eth: {}",
                        sender_device->id(),
                        receiver_device->id(),
                        sender_eth_core.str(),
                        receiver_eth_core.str());
                    break;
                }
            }
        }
    }
    return sender_receivers;
}

namespace unit_tests::erisc::kernels {

/*
 *
 *                                         ███████╗████████╗██╗░░██╗
 *                                         ██╔════╝╚══██╔══╝██║░░██║
 *                                         █████╗░░░░░██║░░░███████║
 *                                         ██╔══╝░░░░░██║░░░██╔══██║
 *                                         ███████╗░░░██║░░░██║░░██║
 *                                         ╚══════╝░░░╚═╝░░░╚═╝░░╚═╝
 */
bool eth_direct_ring_gather_sender_receiver_kernels(
    std::vector<tt::tt_metal::IDevice* > device_ring,
    const size_t& byte_size_per_device,
    const size_t& src_eth_l1_byte_address,
    const size_t& dst_eth_l1_byte_address,
    const size_t& sem_l1_byte_address,
    uint32_t num_bytes_per_send = 16) {
    bool pass = true;
    const auto& sender_receivers = get_sender_receiver_cores(std::move(device_ring));

    // Generate inputs
    uint32_t numel = byte_size_per_device / sizeof(uint32_t);
    std::vector<std::vector<uint32_t>> inputs;
    inputs.reserve(sender_receivers.size());
    std::vector<uint32_t> all_zeros(numel * sender_receivers.size(), 0);
    std::map<chip_id_t, tt_metal::Program> programs;
    std::vector<uint32_t> full_input;
    full_input.reserve(numel * sender_receivers.size());

    for (uint32_t i = 0; i < sender_receivers.size(); ++i) {
        inputs.emplace_back(
            generate_uniform_random_vector<uint32_t>(0, 100, byte_size_per_device / sizeof(uint32_t), i));
        full_input.insert(full_input.begin() + i * numel, inputs[i].begin(), inputs[i].end());

        ////////////////////////////////////////////////////////////////////////////
        //                      Sender Device
        ////////////////////////////////////////////////////////////////////////////
        const auto& [sender_device, receiver_device, eth_sender_core, eth_receiver_core] = sender_receivers[i];
        auto& sender_program = programs[sender_device->id()];
        auto& receiver_program = programs[receiver_device->id()];
        CoreCoord sender_receiver_core;
        for (uint32_t j = 0; j < sender_receivers.size(); ++j) {
            if (std::get<1>(sender_receivers[j])->id() == sender_device->id()) {
                sender_receiver_core = sender_device->ethernet_core_from_logical_core(std::get<3>(sender_receivers[j]));
            }
        }
        auto eth_sender_kernel = tt_metal::CreateKernel(
            sender_program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_ring_gather_send.cpp",
            eth_sender_core,
            tt_metal::EthernetConfig{
                .noc = tt_metal::NOC::NOC_0,
                .compile_args = {
                    uint32_t(num_bytes_per_send),
                    uint32_t(num_bytes_per_send >> 4),
                    uint32_t(sender_receiver_core.x),
                    uint32_t(sender_receiver_core.y)}});

        tt_metal::SetRuntimeArgs(
            sender_program,
            eth_sender_kernel,
            eth_sender_core,
            {(uint32_t)(src_eth_l1_byte_address + (sender_receivers.size() - 1) * byte_size_per_device),
             (uint32_t)dst_eth_l1_byte_address,
             (uint32_t)byte_size_per_device,
             (uint32_t)sender_receivers.size() - 1,
             (uint32_t)(src_eth_l1_byte_address + i * byte_size_per_device),
             (uint32_t)i,
             (uint32_t)sem_l1_byte_address});

        llrt::write_hex_vec_to_core(
            sender_device->id(),
            sender_device->ethernet_core_from_logical_core(eth_sender_core),
            inputs[i],
            src_eth_l1_byte_address + i * byte_size_per_device);
        llrt::write_hex_vec_to_core(
            sender_device->id(),
            sender_device->ethernet_core_from_logical_core(eth_sender_core),
            std::vector{INVALID},
            sem_l1_byte_address);

        ////////////////////////////////////////////////////////////////////////////
        //                      Receiver Device
        ////////////////////////////////////////////////////////////////////////////
        // Clear expected value at ethernet L1 address
        CoreCoord receiver_sender_core;
        for (uint32_t j = 0; j < sender_receivers.size(); ++j) {
            if (std::get<0>(sender_receivers[j])->id() == receiver_device->id()) {
                receiver_sender_core =
                    receiver_device->ethernet_core_from_logical_core(std::get<2>(sender_receivers[j]));
            }
        }

        llrt::write_hex_vec_to_core(
            receiver_device->id(),
            receiver_device->ethernet_core_from_logical_core(eth_receiver_core),
            all_zeros,
            dst_eth_l1_byte_address);
        llrt::write_hex_vec_to_core(
            receiver_device->id(),
            receiver_device->ethernet_core_from_logical_core(eth_receiver_core),
            std::vector{INVALID},
            sem_l1_byte_address);
        auto eth_receiver_kernel = tt_metal::CreateKernel(
            receiver_program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_ring_gather_receive.cpp",
            eth_receiver_core,
            tt_metal::EthernetConfig{
                .noc = tt_metal::NOC::NOC_1,
                .compile_args = {
                    uint32_t(receiver_sender_core.x),
                    uint32_t(receiver_sender_core.y)}});  // probably want to use NOC_1 here

        tt_metal::SetRuntimeArgs(
            receiver_program,
            eth_receiver_kernel,
            eth_receiver_core,
            {(uint32_t)byte_size_per_device, (uint32_t)sender_receivers.size() - 1, (uint32_t)sem_l1_byte_address});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Programs
    ////////////////////////////////////////////////////////////////////////////

    std::vector<std::thread> ths;
    ths.reserve(sender_receivers.size());
    for (uint32_t i = 0; i < sender_receivers.size(); ++i) {
        const auto& device = std::get<0>(sender_receivers[i]);
        ths.emplace_back([&] { tt_metal::detail::LaunchProgram(device, programs.at(device->id())); });
    }
    for (auto& th : ths) {
        th.join();
    }
    for (uint32_t i = 0; i < sender_receivers.size(); ++i) {
        const auto& device = std::get<0>(sender_receivers[i]);
        const auto& core = std::get<2>(sender_receivers[i]);
        auto readback_vec = llrt::read_hex_vec_from_core(
            device->id(),
            device->ethernet_core_from_logical_core(core),
            src_eth_l1_byte_address,
            byte_size_per_device * sender_receivers.size());
        auto a = std::mismatch(full_input.begin(), full_input.end(), readback_vec.begin());
        bool p = (a.first == full_input.end());
        pass &= p;
        if (not p) {
            log_error(tt::LogTest, "Mismatch on Device {} at Core: {}", device->id(), core.str());
            log_error(
                tt::LogTest, "Position: {} Expected: {} Read: {}", a.first - full_input.begin(), *a.first, *a.second);
        }
    }

    return pass;
}

bool eth_interleaved_ring_gather_sender_receiver_kernels(
    std::vector<tt::tt_metal::IDevice* > device_ring,
    const BankedConfig& cfg,
    const size_t& src_eth_l1_byte_address,
    const size_t& dst_eth_l1_byte_address,
    const size_t& sem_l1_byte_address,
    uint32_t num_bytes_per_send = 16) {
    bool pass = true;
    const auto& sender_receivers = get_sender_receiver_cores(std::move(device_ring));

    // Generate inputs
    uint32_t numel = cfg.size_bytes / sizeof(uint32_t);
    std::vector<std::vector<uint32_t>> inputs;
    inputs.reserve(sender_receivers.size());
    std::vector<uint32_t> all_zeros(numel * sender_receivers.size(), 0);
    std::map<chip_id_t, tt_metal::Program> programs;
    std::vector<uint32_t> full_input;
    full_input.reserve(numel * sender_receivers.size());

    std::vector<std::shared_ptr<tt_metal::Buffer>> output_buffers;
    output_buffers.reserve(sender_receivers.size());

    for (uint32_t i = 0; i < sender_receivers.size(); ++i) {
        inputs.emplace_back(tt::test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(
            -1.0f, 1.0f, cfg.size_bytes / bfloat16::SIZEOF, i));
        full_input.insert(full_input.begin() + i * numel, inputs[i].begin(), inputs[i].end());

        const auto& device = std::get<0>(sender_receivers[i]);
        const auto& eth_sender_core = std::get<2>(sender_receivers[i]);
        CoreCoord eth_receiver_core;
        for (uint32_t j = 0; j < sender_receivers.size(); ++j) {
            if (std::get<1>(sender_receivers[j])->id() == device->id()) {
                eth_receiver_core = std::get<3>(sender_receivers[j]);
                break;
            }
        }

        auto& program = programs[device->id()];

        auto input_buffer = CreateBuffer(
            tt_metal::InterleavedBufferConfig{device, cfg.size_bytes, cfg.page_size_bytes, cfg.input_buffer_type});
        bool input_is_dram = cfg.input_buffer_type == tt_metal::BufferType::DRAM;
        tt_metal::detail::WriteToBuffer(input_buffer, inputs[i]);
        output_buffers.emplace_back(CreateBuffer(tt_metal::InterleavedBufferConfig{
            device, cfg.size_bytes * sender_receivers.size(), cfg.page_size_bytes, cfg.output_buffer_type}));
        tt_metal::detail::WriteToBuffer(output_buffers[i], all_zeros);

        auto eth_sender_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/interleaved_eth_ring_gather_send.cpp",
            eth_sender_core,
            tt_metal::EthernetConfig{
                .noc = tt_metal::NOC::NOC_0,
                .compile_args = {
                    uint32_t(num_bytes_per_send),
                    uint32_t(num_bytes_per_send >> 4),
                    uint32_t(device->ethernet_core_from_logical_core(eth_receiver_core).x),
                    uint32_t(device->ethernet_core_from_logical_core(eth_receiver_core).y),
                    uint32_t(input_buffer->buffer_type() == tt_metal::BufferType::DRAM),
                    uint32_t(output_buffers[i]->buffer_type() == tt_metal::BufferType::DRAM)}});

        tt_metal::SetRuntimeArgs(
            program,
            eth_sender_kernel,
            eth_sender_core,
            {(uint32_t)(src_eth_l1_byte_address),
             (uint32_t)dst_eth_l1_byte_address,
             (uint32_t)cfg.size_bytes + 32,  // + 32 for idx
             (uint32_t)sender_receivers.size() - 1,
             (uint32_t)(i * cfg.num_pages),
             (uint32_t)input_buffer->address(),
             (uint32_t)output_buffers[i]->address(),
             (uint32_t)cfg.num_pages,
             (uint32_t)cfg.page_size_bytes,
             (uint32_t)sem_l1_byte_address});
        llrt::write_hex_vec_to_core(
            device->id(),
            device->ethernet_core_from_logical_core(eth_sender_core),
            std::vector{INVALID},
            sem_l1_byte_address);

        llrt::write_hex_vec_to_core(
            device->id(),
            device->ethernet_core_from_logical_core(eth_receiver_core),
            std::vector{INVALID},
            sem_l1_byte_address);

        auto eth_receiver_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/interleaved_eth_ring_gather_receive.cpp",
            eth_receiver_core,
            tt_metal::EthernetConfig{
                .noc = tt_metal::NOC::NOC_1,
                .compile_args = {
                    uint32_t(device->ethernet_core_from_logical_core(eth_sender_core).x),
                    uint32_t(device->ethernet_core_from_logical_core(eth_sender_core).y),
                    uint32_t(
                        output_buffers[i]->buffer_type() ==
                        tt_metal::BufferType::DRAM)}});  // probably want to use NOC_1 here

        tt_metal::SetRuntimeArgs(
            program,
            eth_receiver_kernel,
            eth_receiver_core,
            {(uint32_t)dst_eth_l1_byte_address,
             (uint32_t)cfg.size_bytes + 32,  // + 32 for idx
             (uint32_t)sender_receivers.size() - 1,
             (uint32_t)output_buffers[i]->address(),
             (uint32_t)cfg.num_pages,
             (uint32_t)cfg.page_size_bytes,
             (uint32_t)sem_l1_byte_address});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Programs
    ////////////////////////////////////////////////////////////////////////////

    std::vector<std::thread> ths;
    ths.reserve(sender_receivers.size());
    for (uint32_t i = 0; i < sender_receivers.size(); ++i) {
        const auto& device = std::get<0>(sender_receivers[i]);
        ths.emplace_back([&] { tt_metal::detail::LaunchProgram(device, programs.at(device->id())); });
    }
    for (auto& th : ths) {
        th.join();
    }
    for (uint32_t i = 0; i < sender_receivers.size(); ++i) {
        const auto& device = std::get<0>(sender_receivers[i]);
        const auto& core = std::get<2>(sender_receivers[i]);
        std::vector<uint32_t> readback_vec;
        tt_metal::detail::ReadFromBuffer(output_buffers[i], readback_vec);
        auto a = std::mismatch(full_input.begin(), full_input.end(), readback_vec.begin());
        bool p = (a.first == full_input.end());
        pass &= p;
        if (not p) {
            log_error(tt::LogTest, "Mismatch on Device {} at Core: {}", device->id(), core.str());
            log_error(
                tt::LogTest, "Position: {} Expected: {} Read: {}", a.first - full_input.begin(), *a.first, *a.second);
        }
    }

    return pass;
}

}  // namespace unit_tests::erisc::kernels

namespace tt::tt_metal {

TEST_F(DeviceFixture, ActiveEthKernelsDirectRingGatherAllChips) {
    auto erisc_unreserved_base_addr =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const size_t src_eth_l1_byte_address = erisc_unreserved_base_addr + 32;
    const size_t dst_eth_l1_byte_address = erisc_unreserved_base_addr + 32;
    const size_t sem_l1_byte_address = erisc_unreserved_base_addr;
    const auto& device_ring = get_device_ring(devices_);
    if (device_ring.empty()) {
        GTEST_SKIP();
    }
    ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_ring_gather_sender_receiver_kernels(
        device_ring, WORD_SIZE, src_eth_l1_byte_address, dst_eth_l1_byte_address, sem_l1_byte_address));
}

TEST_F(DeviceFixture, ActiveEthKernelsInterleavedRingGatherAllChips) {
    auto erisc_unreserved_base_addr =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const size_t src_eth_l1_byte_address = erisc_unreserved_base_addr + 32;
    const size_t dst_eth_l1_byte_address = erisc_unreserved_base_addr + 32;
    const size_t sem_l1_byte_address = erisc_unreserved_base_addr;
    BankedConfig test_config =
        BankedConfig{.num_pages = 10, .size_bytes = 10 * 2 * 32 * 32, .page_size_bytes = 2 * 32 * 32};
    const auto& device_ring = get_device_ring(devices_);
    if (device_ring.empty()) {
        GTEST_SKIP();
    }
    ASSERT_TRUE(unit_tests::erisc::kernels::eth_interleaved_ring_gather_sender_receiver_kernels(
        device_ring, test_config, src_eth_l1_byte_address, dst_eth_l1_byte_address, sem_l1_byte_address));
}

}  // namespace tt::tt_metal
