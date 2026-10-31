// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The resident DRAM-core kernel API (tt-metalium/experimental/dram_core_kernels.hpp): topology queries, and a kernel
// launched beside fast dispatch that serves host requests until told to stop, while the command queue keeps working.

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/dram_core_kernels.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>

#include "device_fixture.hpp"

namespace tt::tt_metal {
namespace {

using experimental::DramCoreInfo;

class DramCoreKernelsFixture : public UnitMeshFastDispatchFixture {
protected:
    void SetUp() override {
        UnitMeshFastDispatchFixture::SetUp();
        if (devices_.empty()) {
            return;  // parent was skipped
        }
        if (!experimental::AreDramCoreKernelsSupported(*devices_.front())) {
            GTEST_SKIP() << "No programmable DRAM cores (or the streaming profiler holds them)";
        }
    }

    distributed::MeshDevice& mesh() { return *devices_.front(); }
    const distributed::MeshCoordinate coord_{0, 0};
};

// One usable core of every bank, the region fits the zone layout, the topology fields agree with each other.
TEST_F(DramCoreKernelsFixture, Topology) {
    const auto region = experimental::GetDramCoreKernelL1Region(mesh());
    EXPECT_GT(region.size, 0u);
    EXPECT_EQ(region.base % 16, 0u);
    const uint32_t banks = mesh().dram_grid_size().x;
    ASSERT_GT(banks, 0u);
    for (uint32_t bank = 0; bank < banks; ++bank) {
        const std::vector<DramCoreInfo> cores = experimental::GetDramCores(mesh(), coord_, bank);
        ASSERT_FALSE(cores.empty());
        uint32_t usable = 0, free = 0;
        for (uint32_t i = 0; i < cores.size(); ++i) {
            const DramCoreInfo& c = cores[i];
            EXPECT_EQ(c.logical_core, CoreCoord(bank, i));
            EXPECT_EQ(c.virtual_core, mesh().virtual_core_from_logical_core(c.logical_core, CoreType::DRAM));
            EXPECT_GE(c.mpfe_port, 1u);
            EXPECT_LE(c.mpfe_port, 3u);
            usable += c.usable;
            free += c.usable && c.endpoint_noc_mask == 0;
        }
        // index 0 is the NOC0 endpoint: never a kernel's, always a NOC0 endpoint
        EXPECT_FALSE(cores[0].usable);
        EXPECT_TRUE(cores[0].endpoint_noc_mask & 1u);
        EXPECT_GT(usable, 0u) << "bank " << bank;
        EXPECT_GT(free, 0u) << "bank " << bank << ": no core with both NIUs in stream mode";
    }
    // worker NOC0 positions are distinct
    const CoreCoord a = experimental::GetWorkerNoc0Core(mesh(), coord_, CoreCoord(0, 0));
    const CoreCoord b = experimental::GetWorkerNoc0Core(mesh(), coord_, CoreCoord(1, 0));
    EXPECT_NE(a, b);
}

// A kernel per usable core of bank 0 stays resident, answers host requests while the command queue runs a buffer
// round trip, and exits on the stop word.
TEST_F(DramCoreKernelsFixture, ResidentKernelServesUntilStopped) {
    constexpr uint32_t kReadyMagic = 0xD815C0DE, kStop = 0, kReady = 1, kPosted = 2, kServed = 3, kEcho = 4;
    constexpr auto kTimeout = std::chrono::seconds(10);
    const uint32_t base = experimental::GetDramCoreKernelL1Region(mesh()).base;
    std::vector<CoreCoord> cores;
    for (const auto& c : experimental::GetDramCores(mesh(), coord_, 0)) {
        if (c.usable) {
            cores.push_back(c.logical_core);
        }
    }
    ASSERT_FALSE(cores.empty());

    const std::array<uint32_t, 5> zeros{};
    for (const auto& c : cores) {
        experimental::WriteDramCoreL1(mesh(), c, base, zeros);
    }
    auto kernels = experimental::LaunchDramCoreKernels(mesh(), [&](const distributed::MeshCoordinate&) {
        std::vector<experimental::DramCoreKernelConfig> out;
        for (uint32_t i = 0; i < cores.size(); ++i) {
            out.push_back(experimental::DramCoreKernelConfig{
                .kernel_file = "tests/tt_metal/tt_metal/test_kernels/misc/drisc_resident_echo.cpp",
                .core = cores[i],
                .noc = NOC::NOC_0,
                .runtime_args = {base, 1000u * (i + 1)}});
        }
        return out;
    });
    for (const auto& c : cores) {
        ASSERT_TRUE(experimental::WaitForDramCoreL1Word(mesh(), c, base + 4 * kReady, kReadyMagic, kTimeout));
    }

    // fast dispatch keeps working beside the resident kernels
    const uint32_t n = 1024;
    const distributed::ReplicatedBufferConfig rcfg{.size = n * sizeof(uint32_t)};
    const distributed::DeviceLocalBufferConfig lcfg{.page_size = n * sizeof(uint32_t), .buffer_type = BufferType::DRAM};
    auto buffer = distributed::MeshBuffer::create(rcfg, lcfg, &mesh());
    std::vector<uint32_t> src(n), dst;
    for (uint32_t i = 0; i < n; ++i) {
        src[i] = i * 7 + 3;
    }
    distributed::EnqueueWriteMeshBuffer(mesh().mesh_command_queue(), buffer, src, false);
    distributed::ReadShard(mesh().mesh_command_queue(), dst, buffer, coord_, true);
    EXPECT_EQ(dst, src);

    for (uint32_t request = 1; request <= 3; ++request) {
        for (uint32_t i = 0; i < cores.size(); ++i) {
            const std::array<uint32_t, 1> posted{request};
            experimental::WriteDramCoreL1(mesh(), cores[i], base + 4 * kPosted, posted);
            ASSERT_TRUE(experimental::WaitForDramCoreL1Word(mesh(), cores[i], base + 4 * kServed, request, kTimeout));
            EXPECT_EQ(
                experimental::ReadDramCoreL1(mesh(), coord_, cores[i], base + 4 * kEcho, 1)[0],
                request + 1000u * (i + 1));
        }
    }

    const std::array<uint32_t, 1> stop{1};
    for (const auto& c : cores) {
        experimental::WriteDramCoreL1(mesh(), c, base + 4 * kStop, stop);
    }
    for (const auto& c : cores) {
        ASSERT_TRUE(experimental::WaitForDramCoreL1Word(mesh(), c, base + 4 * kReady, 0, kTimeout));
    }
    kernels->wait_done();
}

}  // namespace
}  // namespace tt::tt_metal
