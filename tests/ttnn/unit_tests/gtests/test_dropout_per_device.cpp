// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Dropout with use_per_device_seed on a mesh runs one program with the device seed offset set per device (merged); when
// per-device run args are not available it runs one program per device (split). Both draw the same masks.

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/operations/experimental/dropout/device/dropout_device_operation.hpp"
#include "ttnn/operations/experimental/dropout/dropout.hpp"
#include "ttnn/per_device_run_args.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace dropout_per_device_test {

using ttnn::Tensor;
using DropoutOp = ttnn::experimental::prim::DropoutDeviceOperation;
using DropoutMeshOp = ttnn::device_operation::MeshDeviceOperationAdapter<DropoutOp>;
using DropoutAdapter =
    DropoutMeshOp::MeshWorkloadSpecFactoryAdapter<ttnn::experimental::prim::DropoutMeshWorkloadFactory>;

class DropoutPerDeviceTest : public tt::tt_metal::MeshDeviceFixtureBase {
protected:
    DropoutPerDeviceTest() : MeshDeviceFixtureBase(Config{.mesh_shape = ttnn::MeshShape{2, 4}, .num_cqs = 1}) {}

    void SetUp() override {
        MeshDeviceFixtureBase::SetUp();
        if (IsSkipped()) {
            return;
        }
        if (!tt::tt_metal::distributed::PerDeviceRuntimeArgsSupported(
                *mesh_device_, ttnn::MeshCoordinateRange(mesh_device_->shape()))) {
            GTEST_SKIP() << "per-device runtime args are not supported on this mesh";
        }
    }

    // The number of programs in the cached workload of `input`'s per-device-seed dropout.
    size_t cached_programs(const Tensor& input, uint32_t seed) {
        const DropoutOp::operation_attributes_t attrs{
            .output_dtype = tt::tt_metal::DataType::BFLOAT16,
            .output_memory_config = input.memory_config(),
            .seed = seed,
            .use_per_device_seed = true,
            .prob = 0.5f,
            .scale = 2.0f};
        const DropoutOp::tensor_args_t args{.input = input, .preallocated_output = std::nullopt};
        const auto lookup =
            ttnn::device_operation::detail::look_up_program_cache<DropoutMeshOp>(attrs, args, mesh_device_.get());
        EXPECT_TRUE(lookup.hit);
        return mesh_device_->get_program_cache()
            .get(lookup.key)
            .cached_program.template get<DropoutAdapter::cached_mesh_workload_t>()
            .workload.get_programs()
            .size();
    }

    // Each device's output words, in row-major mesh order, for every seed.
    std::vector<std::vector<std::vector<float>>> run(const Tensor& input, const std::vector<uint32_t>& seeds) {
        std::vector<std::vector<std::vector<float>>> outputs;
        for (const uint32_t seed : seeds) {
            const auto output = ttnn::experimental::dropout(input, 0.5f, 2.0f, seed, /*use_per_device_seed=*/true);
            std::vector<std::vector<float>> per_device;
            for (const auto& shard : ttnn::distributed::get_device_tensors(output.cpu())) {
                per_device.push_back(shard.to_vector<float>());
            }
            outputs.push_back(std::move(per_device));
        }
        return outputs;
    }
};

TEST_F(DropoutPerDeviceTest, SplitMatchesMerged) {
    const auto grid = mesh_device_->compute_with_storage_grid_size();
    for (const uint32_t num_tiles : {1u, static_cast<uint32_t>(grid.x * grid.y + 3)}) {
        const tt::tt_metal::TensorSpec spec(
            ttnn::Shape({1, 1, num_tiles * 32, 32}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                tt::tt_metal::MemoryConfig{}));
        const auto input = Tensor::from_vector(std::vector<float>(num_tiles * 32 * 32, 1.0f), spec, mesh_device_.get());
        const std::vector<uint32_t> seeds = {11, 12, 11};

        mesh_device_->clear_program_cache();
        const auto merged = run(input, seeds);
        EXPECT_EQ(cached_programs(input, seeds.back()), 1u) << "one program for the mesh";
        std::vector<std::vector<std::vector<float>>> split;
        {
            ttnn::device_operation::per_device_run_args::ScopedForceSplit force_split(*mesh_device_);
            split = run(input, seeds);
            EXPECT_EQ(cached_programs(input, seeds.back()), mesh_device_->shape().mesh_size()) << "one per device";
        }
        ASSERT_EQ(merged.size(), split.size());
        for (size_t s = 0; s < seeds.size(); ++s) {
            ASSERT_EQ(merged[s].size(), mesh_device_->shape().mesh_size());
            for (size_t d = 0; d < merged[s].size(); ++d) {
                EXPECT_EQ(merged[s][d], split[s][d])
                    << "tiles " << num_tiles << ", seed " << seeds[s] << ", device " << d;
            }
            for (size_t d = 1; d < merged[s].size(); ++d) {
                EXPECT_NE(merged[s][0], merged[s][d])
                    << "tiles " << num_tiles << ": devices 0 and " << d << " share a mask";
            }
        }
        EXPECT_EQ(merged[0], merged[2]) << "a fixed seed reproduces the masks on a cache hit";
        EXPECT_NE(merged[0], merged[1]) << "a new seed changes the masks";
    }
}

// Two submeshes of one mesh calling with the same seed draw different masks: each device's seed offset is its device
// id, not its coordinate in the submesh (which would repeat in every submesh).
TEST_F(DropoutPerDeviceTest, SubmeshesWithTheSameSeedDrawDifferentMasks) {
    const auto submeshes = mesh_device_->create_submeshes(ttnn::MeshShape{1, 4});
    ASSERT_GE(submeshes.size(), 2u);
    std::vector<std::vector<std::vector<float>>> masks;  // [submesh][device]
    for (const auto& submesh : submeshes) {
        const tt::tt_metal::TensorSpec spec(
            ttnn::Shape({1, 1, 4 * 32, 32}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                tt::tt_metal::MemoryConfig{}));
        const auto input = Tensor::from_vector(std::vector<float>(4 * 32 * 32, 1.0f), spec, submesh.get());
        const auto output = ttnn::experimental::dropout(input, 0.5f, 2.0f, /*seed=*/42, /*use_per_device_seed=*/true);
        std::vector<std::vector<float>> per_device;
        for (const auto& shard : ttnn::distributed::get_device_tensors(output.cpu())) {
            per_device.push_back(shard.to_vector<float>());
        }
        masks.push_back(std::move(per_device));
    }
    for (size_t d = 0; d < masks[0].size(); ++d) {
        EXPECT_NE(masks[0][d], masks[1][d]) << "device " << d << " of submeshes 0 and 1 draws the same mask";
    }
}

}  // namespace dropout_per_device_test
