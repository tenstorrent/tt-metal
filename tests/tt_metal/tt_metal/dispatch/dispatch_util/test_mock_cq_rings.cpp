// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The host-side CQ rings of a mock device with two hardware CQs: each CQ has its own memory and wraps within it.

#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>

#include "impl/dispatch/system_memory_manager.hpp"

namespace tt::tt_metal {
namespace {

constexpr uint8_t kNumCqs = 2;
// Issue queue size the mock stubs reported before the mock backing, which mock kernel configs rely on.
constexpr uint32_t kMockIssueQueueSize = 65536;
constexpr uint32_t kChunkSize = 4096;
constexpr uint32_t kChunksPerLap = kMockIssueQueueSize / kChunkSize;

class MockTwoCqRingsTest : public ::testing::Test {
protected:
    void SetUp() override {
        env_ = std::make_unique<MetalEnv>(MetalEnvDescriptor{
            .mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1)});
        mesh_device_ = distributed::MeshDevice::open_mesh_device(
            *env_,
            distributed::MeshDeviceConfig(distributed::MeshShape{1, 1}),
            distributed::OpenMeshDeviceOptions{.num_command_queues = kNumCqs});
        manager_ = &mesh_device_->get_devices().at(0)->sysmem_manager();
    }

    void TearDown() override {
        mesh_device_.reset();
        env_.reset();
    }

    // Writes `laps` full laps of kChunkSize commands filled with `fill` into the CQ's issue queue, checking that
    // every reservation stays inside [region_start, region_start + kMockIssueQueueSize).
    void write_laps(uint8_t cq_id, char* region_start, uint32_t laps, char fill) {
        for (uint32_t i = 0; i < laps * kChunksPerLap; i++) {
            auto* cmd = static_cast<char*>(manager_->issue_queue_reserve(kChunkSize, cq_id));
            EXPECT_GE(cmd, region_start) << "CQ " << int(cq_id) << " chunk " << i;
            EXPECT_LE(cmd + kChunkSize, region_start + kMockIssueQueueSize) << "CQ " << int(cq_id) << " chunk " << i;
            std::memset(cmd, fill, kChunkSize);
            manager_->issue_queue_push_back(kChunkSize, cq_id);
        }
    }

    std::unique_ptr<MetalEnv> env_;
    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    SystemMemoryManager* manager_ = nullptr;
};

TEST_F(MockTwoCqRingsTest, CPU_EachCqHasItsOwnRing) {
    std::vector<char*> region_starts;
    std::vector<uint32_t> start_ptrs;
    for (uint8_t cq_id = 0; cq_id < kNumCqs; cq_id++) {
        // Sizes and free space match the stubs the mock backing replaced.
        EXPECT_EQ(manager_->get_issue_queue_size(cq_id), kMockIssueQueueSize);
        EXPECT_EQ(manager_->get_completion_queue_size(cq_id), kMockIssueQueueSize);
        const uint32_t write_ptr = manager_->get_issue_queue_write_ptr(cq_id);
        // Each CQ is idle at the start of its region, so the first reservation is the region start.
        ASSERT_EQ(manager_->get_issue_queue_limit(cq_id) - write_ptr, kMockIssueQueueSize)
            << "CQ " << int(cq_id) << " wasn't idle at the start";
        EXPECT_EQ(
            manager_->get_completion_queue_limit(cq_id) - manager_->get_completion_queue_read_ptr(cq_id),
            kMockIssueQueueSize);
        start_ptrs.push_back(write_ptr);
        region_starts.push_back(static_cast<char*>(manager_->issue_queue_reserve(kChunkSize, cq_id)));
    }

    // The two issue queues don't overlap.
    EXPECT_TRUE(
        region_starts[0] + kMockIssueQueueSize <= region_starts[1] ||
        region_starts[1] + kMockIssueQueueSize <= region_starts[0]);

    for (uint8_t cq_id = 0; cq_id < kNumCqs; cq_id++) {
        write_laps(cq_id, region_starts[cq_id], /*laps=*/2, static_cast<char>(cq_id + 1));
        // After whole laps the write pointer is back at the CQ's own start.
        EXPECT_EQ(manager_->get_issue_queue_write_ptr(cq_id), start_ptrs[cq_id]);
    }

    // Neither CQ wrote into the other's queue.
    for (uint8_t cq_id = 0; cq_id < kNumCqs; cq_id++) {
        std::vector<char> expected(kMockIssueQueueSize, static_cast<char>(cq_id + 1));
        EXPECT_EQ(std::memcmp(region_starts[cq_id], expected.data(), kMockIssueQueueSize), 0) << "CQ " << int(cq_id);
    }
}

TEST_F(MockTwoCqRingsTest, CPU_CqsCanBeWrittenFromTwoThreads) {
    std::vector<char*> region_starts;
    for (uint8_t cq_id = 0; cq_id < kNumCqs; cq_id++) {
        // Each CQ is idle at the start of its region, so the first reservation is the region start.
        ASSERT_EQ(
            manager_->get_issue_queue_limit(cq_id) - manager_->get_issue_queue_write_ptr(cq_id), kMockIssueQueueSize)
            << "CQ " << int(cq_id) << " wasn't idle at the start";
        region_starts.push_back(static_cast<char*>(manager_->issue_queue_reserve(kChunkSize, cq_id)));
    }

    // One thread per CQ, as with a dispatch thread per mesh command queue.
    std::vector<std::thread> writers;
    writers.reserve(kNumCqs);
    for (uint8_t cq_id = 0; cq_id < kNumCqs; cq_id++) {
        writers.emplace_back(
            [this, cq_id, start = region_starts[cq_id]] { write_laps(cq_id, start, /*laps=*/4, char(cq_id + 1)); });
    }
    for (auto& writer : writers) {
        writer.join();
    }

    for (uint8_t cq_id = 0; cq_id < kNumCqs; cq_id++) {
        std::vector<char> expected(kMockIssueQueueSize, static_cast<char>(cq_id + 1));
        EXPECT_EQ(std::memcmp(region_starts[cq_id], expected.data(), kMockIssueQueueSize), 0) << "CQ " << int(cq_id);
    }
}

}  // namespace
}  // namespace tt::tt_metal
