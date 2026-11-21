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

#include "impl/context/metal_context.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"
#include "impl/dispatch/host_cq.hpp"

namespace tt::tt_metal {
namespace {

constexpr uint8_t kNumCqs = 2;
// Issue and completion queue size of each mock CQ.
constexpr uint32_t kMockIssueQueueSize = 256 * 1024;
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
        device_ = mesh_device_->get_devices().at(0);
    }

    void TearDown() override {
        mesh_device_.reset();
        env_.reset();
    }

    // Writes `laps` full laps of kChunkSize commands filled with `fill` into the CQ's issue queue, checking that
    // every reservation stays inside [region_start, region_start + kMockIssueQueueSize).
    void write_laps(uint8_t cq_id, char* region_start, uint32_t laps, char fill) {
        for (uint32_t i = 0; i < laps * kChunksPerLap; i++) {
            auto* cmd = static_cast<char*>(host_cq(*device_, cq_id).issue_queue_reserve(kChunkSize));
            EXPECT_GE(cmd, region_start) << "CQ " << int(cq_id) << " chunk " << i;
            EXPECT_LE(cmd + kChunkSize, region_start + kMockIssueQueueSize) << "CQ " << int(cq_id) << " chunk " << i;
            std::memset(cmd, fill, kChunkSize);
            host_cq(*device_, cq_id).issue_queue_push_back(kChunkSize);
        }
    }

    std::unique_ptr<MetalEnv> env_;
    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    IDevice* device_ = nullptr;
};

TEST_F(MockTwoCqRingsTest, CPU_EachCqHasItsOwnRing) {
    std::vector<char*> region_starts;
    std::vector<uint32_t> start_ptrs;
    for (uint8_t cq_id = 0; cq_id < kNumCqs; cq_id++) {
        EXPECT_EQ(host_cq(*device_, cq_id).issue_queue_size(), kMockIssueQueueSize);
        EXPECT_EQ(host_cq(*device_, cq_id).completion_queue_size(), kMockIssueQueueSize);
        const uint32_t write_ptr = host_cq(*device_, cq_id).issue_queue_write_ptr();
        // Each CQ is idle at the start of its region, so the first reservation is the region start.
        ASSERT_EQ(host_cq(*device_, cq_id).issue_queue_limit() - write_ptr, kMockIssueQueueSize)
            << "CQ " << int(cq_id) << " wasn't idle at the start";
        EXPECT_EQ(
            host_cq(*device_, cq_id).completion_queue_limit() - host_cq(*device_, cq_id).completion_queue_read_ptr(),
            kMockIssueQueueSize);
        start_ptrs.push_back(write_ptr);
        region_starts.push_back(static_cast<char*>(host_cq(*device_, cq_id).issue_queue_reserve(kChunkSize)));
    }

    // The two issue queues don't overlap.
    EXPECT_TRUE(
        region_starts[0] + kMockIssueQueueSize <= region_starts[1] ||
        region_starts[1] + kMockIssueQueueSize <= region_starts[0]);

    for (uint8_t cq_id = 0; cq_id < kNumCqs; cq_id++) {
        write_laps(cq_id, region_starts[cq_id], /*laps=*/2, static_cast<char>(cq_id + 1));
        // After whole laps the write pointer is back at the CQ's own start.
        EXPECT_EQ(host_cq(*device_, cq_id).issue_queue_write_ptr(), start_ptrs[cq_id]);
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
            host_cq(*device_, cq_id).issue_queue_limit() - host_cq(*device_, cq_id).issue_queue_write_ptr(),
            kMockIssueQueueSize)
            << "CQ " << int(cq_id) << " wasn't idle at the start";
        region_starts.push_back(static_cast<char*>(host_cq(*device_, cq_id).issue_queue_reserve(kChunkSize)));
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

// A max-size prefetch command fits each CQ's issue queue, including after the write pointer wraps.
TEST_F(MockTwoCqRingsTest, CPU_MaxSizeCommandFitsTheIssueQueue) {
    const uint32_t max_command_size_B = cq_backing(*device_).context().dispatch_mem_map().max_prefetch_command_size();
    for (uint8_t cq_id = 0; cq_id < kNumCqs; cq_id++) {
        auto* region_start = static_cast<char*>(host_cq(*device_, cq_id).issue_queue_reserve(kChunkSize));
        for (int i = 0; i < 3; i++) {
            auto* cmd = static_cast<char*>(host_cq(*device_, cq_id).issue_queue_reserve(max_command_size_B));
            EXPECT_GE(cmd, region_start) << "CQ " << int(cq_id) << " command " << i;
            EXPECT_LE(cmd + max_command_size_B, region_start + kMockIssueQueueSize)
                << "CQ " << int(cq_id) << " command " << i;
            host_cq(*device_, cq_id).issue_queue_push_back(max_command_size_B);
        }
    }
}

// Each lazily created mock CQ has its own event counters.
TEST_F(MockTwoCqRingsTest, CPU_EachCqHasItsOwnEventCounters) {
    ASSERT_EQ(device_->num_hw_cqs(), kNumCqs);
    host_cq(*device_, 1).set_last_issued_event(7);
    host_cq(*device_, 1).set_last_completed_event(5);
    EXPECT_EQ(host_cq(*device_, 1).last_issued_event(), 7);
    EXPECT_EQ(host_cq(*device_, 1).last_completed_event(), 5);
    EXPECT_EQ(host_cq(*device_, 0).last_issued_event(), 0);
    EXPECT_EQ(host_cq(*device_, 0).last_completed_event(), 0);
    host_cq(*device_, 1).reset_events(0);
    EXPECT_EQ(host_cq(*device_, 1).last_issued_event(), 0);
    EXPECT_EQ(host_cq(*device_, 1).last_completed_event(), 0);
}

}  // namespace
}  // namespace tt::tt_metal
