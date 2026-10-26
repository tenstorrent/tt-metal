// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cstdlib>
#include <thread>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include "debug_tools_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {
using MixCase = std::tuple<uint32_t, uint32_t, bool, bool, DataMovementProcessor>;
class WatcherDramNocMix : public MeshWatcherFixture, public testing::WithParamInterface<MixCase> {};

TEST_P(WatcherDramNocMix, ChecksDirectionsWithinOneKernel) {
    if (arch_ != ARCH::BLACKHOLE || std::getenv("TT_METAL_WATCHER_ENABLE_DRAM_NOC_MIX_CHECK") == nullptr) {
        GTEST_SKIP() << "Requires Blackhole and TT_METAL_WATCHER_ENABLE_DRAM_NOC_MIX_CHECK=1";
    }
    const auto [read_id, write_id, reverse, separate_launches, processor] = GetParam();
    const auto mesh = devices_.at(0);
    const distributed::MeshCoordinate coord{0, 0};
    const CoreCoord core{0, 0};
    auto& cq = mesh->mesh_command_queue();
    const distributed::ReplicatedBufferConfig buffer_config{.size = 32};
    const distributed::DeviceLocalBufferConfig dram_config{.page_size = 32, .buffer_type = BufferType::DRAM};
    const distributed::DeviceLocalBufferConfig l1_config{.page_size = 32, .buffer_type = BufferType::L1};
    auto input = distributed::MeshBuffer::create(buffer_config, dram_config, mesh.get());
    auto output = distributed::MeshBuffer::create(buffer_config, dram_config, mesh.get());
    auto scratch = distributed::MeshBuffer::create(buffer_config, l1_config, mesh.get());
    std::vector<uint32_t> values(8, 0x12345678);
    std::vector<uint32_t> zeros(8, 0);
    distributed::WriteShard(cq, input, values, coord);
    distributed::WriteShard(cq, output, zeros, coord);
    auto* device = mesh->get_devices().at(0);
    detail::WriteToDeviceL1(device, core, scratch->address(), values);
    const auto dram_core = device->virtual_core_from_logical_core(device->logical_core_from_dram_channel(0), CoreType::DRAM);
    const bool should_trip = !separate_launches && read_id == write_id;

    auto launch = [&](uint32_t mode) {
        Program program;
        auto kernel = CreateKernel(
            program, "tests/tt_metal/tt_metal/test_kernels/dataflow/watcher_dram_noc_mix.cpp", core,
            DataMovementConfig{.processor = processor, .noc = NOC::NOC_0, .noc_mode = NOC_MODE::DM_DYNAMIC_NOC});
        SetRuntimeArgs(program, kernel, core,
                       {scratch->address(), input->address(), output->address(), dram_core.x, dram_core.y,
                        read_id, write_id, static_cast<uint32_t>(reverse), mode});
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(coord, coord), std::move(program));
        if (should_trip) {
            RunProgramExpectingWatcherError(mesh, workload);
        } else {
            RunProgram(mesh, workload, true);
        }
    };
    if (separate_launches) {
        launch(1);
        launch(2);
    } else {
        launch(0);
    }
    auto& server = MetalContext::instance().watcher_server();
    if (should_trip) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        std::string error;
        do {
            error = server->exception_message();
            if (!error.empty()) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        } while (std::chrono::steady_clock::now() < deadline);
        EXPECT_NE(error.find("kernel uses both DRAM reads and writes on one NoC"), std::string::npos) << error;
    } else {
        EXPECT_TRUE(server->exception_message().empty()) << server->exception_message();
        std::vector<uint32_t> actual;
        distributed::ReadShard(cq, actual, output, coord);
        EXPECT_EQ(actual, values);
    }
}

std::vector<MixCase> mix_cases() {
    std::vector<MixCase> result;
    for (auto processor : {DataMovementProcessor::RISCV_0, DataMovementProcessor::RISCV_1}) {
        for (uint32_t noc = 0; noc < 2; ++noc) {
            result.emplace_back(noc, noc, false, false, processor);
            result.emplace_back(noc, noc, true, false, processor);
            result.emplace_back(noc, 1 - noc, false, false, processor);
            result.emplace_back(noc, noc, false, true, processor);
        }
    }
    return result;
}
INSTANTIATE_TEST_SUITE_P(Blackhole, WatcherDramNocMix, testing::ValuesIn(mix_cases()));
}  // namespace
