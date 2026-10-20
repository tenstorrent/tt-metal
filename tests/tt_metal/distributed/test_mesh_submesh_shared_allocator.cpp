// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/sub_device.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

using SharedAllocatorSubmeshTest = GenericMeshDeviceFixture;

constexpr uint32_t kPageSize = 2048;
constexpr uint32_t kBufferSize = 16 * kPageSize;

std::shared_ptr<MeshBuffer> make_buffer(MeshDevice* mesh, BufferType type, uint32_t size = kBufferSize) {
    return MeshBuffer::create(
        ReplicatedBufferConfig{.size = size},
        DeviceLocalBufferConfig{.page_size = kPageSize, .buffer_type = type},
        mesh);
}

// Per-bank address ranges of two buffers of the same type intersect.
bool overlaps(const MeshBuffer& a, const MeshBuffer& b) {
    const auto a_end = a.address() + a.get_backing_buffer()->aligned_size_per_bank();
    const auto b_end = b.address() + b.get_backing_buffer()->aligned_size_per_bank();
    return a.address() < b_end && b.address() < a_end;
}

MeshWorkload blank_workload(const MeshShape& shape) {
    Program program = CreateProgram();
    CreateKernel(program, "tests/tt_metal/tt_metal/test_kernels/compute/blank.cpp", CoreCoord{0, 0}, ComputeConfig{});
    MeshWorkload workload;
    workload.add_program(MeshCoordinateRange(shape), std::move(program));
    return workload;
}

std::vector<uint32_t> pattern(uint32_t seed) {
    std::vector<uint32_t> data(kBufferSize / sizeof(uint32_t));
    std::iota(data.begin(), data.end(), seed);
    return data;
}

// A (1, 1) submesh on the last device of the parent, so it overlaps the parent but not its first device.
std::shared_ptr<MeshDevice> last_device_submesh(MeshDevice& parent, bool share_allocator) {
    const auto& shape = parent.shape();
    return parent.create_submesh(
        MeshShape(1, 1), MeshCoordinate(shape[0] - 1, shape[1] - 1), /*share_allocator=*/share_allocator);
}

}  // namespace

TEST_F(SharedAllocatorSubmeshTest, DefaultSubmeshAllocatesOverParentBuffers) {
    if (mesh_device_->num_devices() < 2 || mesh_device_->shape().dims() != 2) {
        GTEST_SKIP() << "Requires a 2D mesh of at least 2 devices";
    }
    // The control for the tests below: an independent allocator hands out the parent's addresses again.
    auto submesh = last_device_submesh(*mesh_device_, /*share_allocator=*/false);
    for (auto type : {BufferType::DRAM, BufferType::L1}) {
        auto parent_buffer = make_buffer(mesh_device_.get(), type);
        auto submesh_buffer = make_buffer(submesh.get(), type);
        EXPECT_TRUE(overlaps(*parent_buffer, *submesh_buffer)) << "buffer type " << static_cast<int>(type);
    }
}

TEST_F(SharedAllocatorSubmeshTest, SharedSubmeshBuffersNeverOverlapParentBuffers) {
    if (mesh_device_->num_devices() < 2 || mesh_device_->shape().dims() != 2) {
        GTEST_SKIP() << "Requires a 2D mesh of at least 2 devices";
    }
    auto submesh = last_device_submesh(*mesh_device_, /*share_allocator=*/true);
    EXPECT_EQ(submesh->allocator().get(), mesh_device_->allocator().get());

    for (auto type : {BufferType::DRAM, BufferType::L1}) {
        std::vector<std::shared_ptr<MeshBuffer>> buffers;
        for (int i = 0; i < 4; i++) {
            buffers.push_back(make_buffer(mesh_device_.get(), type, kBufferSize * (i + 1)));
            buffers.push_back(make_buffer(submesh.get(), type, kBufferSize * (4 - i)));
        }
        // Freeing a submesh buffer returns its range to the shared allocator for the parent to reuse.
        const auto freed_address = buffers[1]->address();
        buffers[1].reset();
        buffers.push_back(make_buffer(mesh_device_.get(), type, kBufferSize));
        if (type == BufferType::DRAM) {
            EXPECT_EQ(buffers.back()->address(), freed_address);
        }
        buffers.erase(buffers.begin() + 1);
        for (size_t i = 0; i < buffers.size(); i++) {
            for (size_t j = i + 1; j < buffers.size(); j++) {
                EXPECT_FALSE(overlaps(*buffers[i], *buffers[j]))
                    << "buffer type " << static_cast<int>(type) << ": buffers " << i << " and " << j << " overlap";
            }
        }
    }
}

TEST_F(SharedAllocatorSubmeshTest, AlternatingParentAndSharedSubmeshKeepsData) {
    if (mesh_device_->num_devices() < 2 || mesh_device_->shape().dims() != 2) {
        GTEST_SKIP() << "Requires a 2D mesh of at least 2 devices";
    }
    auto submesh = last_device_submesh(*mesh_device_, /*share_allocator=*/true);
    auto& parent_cq = mesh_device_->mesh_command_queue();
    auto& submesh_cq = submesh->mesh_command_queue();
    auto parent_workload = blank_workload(mesh_device_->shape());
    auto submesh_workload = blank_workload(submesh->shape());

    for (auto type : {BufferType::DRAM, BufferType::L1}) {
        auto parent_data = pattern(0);
        auto submesh_data = pattern(1 << 20);
        auto parent_buffer = make_buffer(mesh_device_.get(), type);
        EnqueueWriteMeshBuffer(parent_cq, parent_buffer, parent_data, /*blocking=*/false);
        EnqueueMeshWorkload(parent_cq, parent_workload, /*blocking=*/false);
        mesh_device_->quiesce_devices();

        auto submesh_buffer = make_buffer(submesh.get(), type);
        EnqueueWriteMeshBuffer(submesh_cq, submesh_buffer, submesh_data, /*blocking=*/false);
        EnqueueMeshWorkload(submesh_cq, submesh_workload, /*blocking=*/false);
        // The read overwrites the prefetcher's ring on the shared device, where the parent's workload binary was
        // cached, so the parent's next run must fetch it again or it hangs.
        std::vector<uint32_t> readback;
        ReadShard(submesh_cq, readback, submesh_buffer, MeshCoordinate(0, 0));
        EXPECT_EQ(readback, submesh_data);
        mesh_device_->quiesce_devices();

        EnqueueMeshWorkload(parent_cq, parent_workload, /*blocking=*/false);
        for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
            ReadShard(parent_cq, readback, parent_buffer, coord);
            EXPECT_EQ(readback, parent_data) << "parent buffer corrupted on " << coord;
        }
        mesh_device_->quiesce_devices();
    }
}

TEST_F(SharedAllocatorSubmeshTest, OverlappingMeshWithoutQuiesceThrows) {
    if (mesh_device_->num_devices() < 2 || mesh_device_->shape().dims() != 2) {
        GTEST_SKIP() << "Requires a 2D mesh of at least 2 devices";
    }
    auto submesh = last_device_submesh(*mesh_device_, /*share_allocator=*/true);
    auto& parent_cq = mesh_device_->mesh_command_queue();
    auto& submesh_cq = submesh->mesh_command_queue();
    auto parent_workload = blank_workload(mesh_device_->shape());
    auto submesh_workload = blank_workload(submesh->shape());

    // Submesh then parent.
    EnqueueMeshWorkload(submesh_cq, submesh_workload, /*blocking=*/false);
    EXPECT_THROW(EnqueueMeshWorkload(parent_cq, parent_workload, /*blocking=*/false), std::runtime_error);
    EXPECT_THROW(Finish(parent_cq), std::runtime_error);
    mesh_device_->quiesce_devices();

    // Parent then submesh; the failed attempts leave both meshes usable after a quiesce.
    EnqueueMeshWorkload(parent_cq, parent_workload, /*blocking=*/false);
    auto buffer = make_buffer(submesh.get(), BufferType::DRAM);
    auto data = pattern(7);
    EXPECT_THROW(EnqueueWriteMeshBuffer(submesh_cq, buffer, data, /*blocking=*/true), std::runtime_error);
    EXPECT_THROW(EnqueueMeshWorkload(submesh_cq, submesh_workload, /*blocking=*/false), std::runtime_error);
    mesh_device_->quiesce_devices();
    EnqueueWriteMeshBuffer(submesh_cq, buffer, data, /*blocking=*/false);
    EnqueueMeshWorkload(submesh_cq, submesh_workload, /*blocking=*/false);
    std::vector<uint32_t> readback;
    ReadShard(submesh_cq, readback, buffer, MeshCoordinate(0, 0));
    EXPECT_EQ(readback, data);
    mesh_device_->quiesce_devices();
}

TEST_F(SharedAllocatorSubmeshTest, OnlyOverlappingSiblingsConflict) {
    if (mesh_device_->num_devices() < 2 || mesh_device_->shape().dims() != 2) {
        GTEST_SKIP() << "Requires a 2D mesh of at least 2 devices";
    }
    // Split the parent in two along a dimension of at least 2, plus a sibling that covers both halves.
    const auto& shape = mesh_device_->shape();
    const bool split_cols = shape[1] >= 2;
    const MeshShape half = split_cols ? MeshShape(shape[0], 1) : MeshShape(1, shape[1]);
    const MeshShape pair = split_cols ? MeshShape(shape[0], 2) : MeshShape(2, shape[1]);
    const MeshCoordinate second = split_cols ? MeshCoordinate(0, 1) : MeshCoordinate(1, 0);
    auto left = mesh_device_->create_submesh(half, MeshCoordinate(0, 0), /*share_allocator=*/true);
    auto right = mesh_device_->create_submesh(half, second, /*share_allocator=*/true);
    auto both = mesh_device_->create_submesh(pair, MeshCoordinate(0, 0), /*share_allocator=*/true);
    auto left_workload = blank_workload(left->shape());
    auto right_workload = blank_workload(right->shape());
    auto both_workload = blank_workload(both->shape());

    // Disjoint siblings may be in use together.
    EnqueueMeshWorkload(left->mesh_command_queue(), left_workload, /*blocking=*/false);
    EnqueueMeshWorkload(right->mesh_command_queue(), right_workload, /*blocking=*/false);
    // A sibling that overlaps either of them may not.
    EXPECT_THROW(
        EnqueueMeshWorkload(both->mesh_command_queue(), both_workload, /*blocking=*/false), std::runtime_error);
    mesh_device_->quiesce_devices();
    EnqueueMeshWorkload(both->mesh_command_queue(), both_workload, /*blocking=*/false);
    Finish(both->mesh_command_queue());
    mesh_device_->quiesce_devices();
}

TEST_F(SharedAllocatorSubmeshTest, SharedSubmeshRejectsTracesAndSubDeviceManagers) {
    if (mesh_device_->num_devices() < 2 || mesh_device_->shape().dims() != 2) {
        GTEST_SKIP() << "Requires a 2D mesh of at least 2 devices";
    }
    auto submesh = last_device_submesh(*mesh_device_, /*share_allocator=*/true);
    EXPECT_THROW(submesh->begin_mesh_trace(submesh->mesh_command_queue()), std::runtime_error);
    const auto grid = submesh->compute_with_storage_grid_size();
    SubDevice sub_device(std::array{CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}))});
    EXPECT_THROW(submesh->create_sub_device_manager({sub_device}, 0), std::runtime_error);
}

TEST_F(SharedAllocatorSubmeshTest, ClosingParentClosesSharedSubmesh) {
    if (mesh_device_->num_devices() < 2 || mesh_device_->shape().dims() != 2) {
        GTEST_SKIP() << "Requires a 2D mesh of at least 2 devices";
    }
    auto shared = last_device_submesh(*mesh_device_, /*share_allocator=*/true);
    // Outlives the parent's allocator; its free must be skipped rather than reach freed memory.
    auto buffer = make_buffer(shared.get(), BufferType::DRAM);
    mesh_device_->close();
    EXPECT_FALSE(shared->is_initialized());
    buffer.reset();
}

}  // namespace tt::tt_metal::distributed::test
