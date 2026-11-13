// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// paged_fill_cache on a mesh with mesh_coords: one program per rectangle of coordinates on the same side of
// mesh_coords (not one per device), and the hash-excluded `noop` attribute applies on cache hits.

#include <gtest/gtest.h>

#include <cstdint>
#include <set>
#include <vector>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/mesh_device_operation_utils.hpp"
#include "ttnn/operations/experimental/paged_cache/device/fill_cache/paged_fill_cache_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace paged_fill_cache_mesh_programs_test {

using ttnn::Tensor;
using Op = ttnn::experimental::prim::PagedFillCacheDeviceOperation;
using MeshOp = ttnn::device_operation::MeshDeviceOperationAdapter<Op>;
using Adapter = MeshOp::MeshWorkloadSpecFactoryAdapter<ttnn::experimental::prim::PagedFillCacheMeshWorkloadFactory>;

constexpr uint32_t kBlocks = 4;
constexpr uint32_t kBlockSize = 32;
constexpr uint32_t kHeadDim = 32;
constexpr uint32_t kSeqLen = 64;  // fills blocks 0 and 1

class PagedFillCacheMeshProgramsTest : public tt::tt_metal::MeshDeviceFixtureBase {
protected:
    PagedFillCacheMeshProgramsTest() :
        MeshDeviceFixtureBase(Config{.mesh_shape = ttnn::MeshShape{2, 4}, .num_cqs = 1}) {}

    Tensor make(const ttnn::Shape& shape, tt::tt_metal::DataType dtype, tt::tt_metal::Layout layout, auto values) {
        const tt::tt_metal::TensorSpec spec(
            shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(layout), tt::tt_metal::MemoryConfig{}));
        return Tensor::from_vector(std::move(values), spec, mesh_device_.get());
    }

    static std::vector<float> cache_values(float first) {
        std::vector<float> values(kBlocks * kBlockSize * kHeadDim);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = first + static_cast<float>(i % 64);
        }
        return values;
    }
    static std::vector<float> fill_values() {
        std::vector<float> values(kSeqLen * kHeadDim);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = 128.0f + static_cast<float>(i % 64);  // exact in bfloat16
        }
        return values;
    }
};

TEST_F(PagedFillCacheMeshProgramsTest, OneProgramPerRectangleAndNoopOnHits) {
    const auto shape = mesh_device_->shape();
    const ttnn::MeshCoordinateRange all(shape);
    const std::set<ttnn::MeshCoordinate> selected = {
        ttnn::MeshCoordinate(0, 0), ttnn::MeshCoordinate(0, 1), ttnn::MeshCoordinate(1, 3)};
    const auto page_table = make(
        ttnn::Shape({1, kBlocks}),
        tt::tt_metal::DataType::INT32,
        tt::tt_metal::Layout::ROW_MAJOR,
        std::vector<int32_t>{0, 1, 2, 3});
    const auto fill = make(
        ttnn::Shape({1, 1, kSeqLen, kHeadDim}),
        tt::tt_metal::DataType::BFLOAT16,
        tt::tt_metal::Layout::TILE,
        fill_values());

    namespace utils = ttnn::device_operation::mesh_device_operation_utils;
    const size_t expected_programs =
        utils::coalesce_coordinates(all, [&](const auto& coord) { return selected.contains(coord); }).size() +
        utils::coalesce_coordinates(all, [&](const auto& coord) { return !selected.contains(coord); }).size();
    ASSERT_LT(expected_programs, shape.mesh_size());

    mesh_device_->clear_program_cache();
    // noop is excluded from the program hash: one cache entry serves all three calls.
    for (const bool noop : {true, false, true}) {
        const float first = noop ? 1.0f : 2.0f;
        const Op::tensor_args_t args{
            .cache_tensor = make(
                ttnn::Shape({kBlocks, 1, kBlockSize, kHeadDim}),
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::Layout::TILE,
                cache_values(first)),
            .input_tensor = fill,
            .page_table = page_table,
        };
        const Op::operation_attributes_t attrs{.batch_idx_fallback = 0, .mesh_coords = selected, .noop = noop};
        const auto output = ttnn::device_operation::launch<Op>(attrs, args);

        EXPECT_EQ(mesh_device_->get_program_cache().num_entries(), 1u) << "noop " << noop;
        const auto lookup =
            ttnn::device_operation::detail::look_up_program_cache<MeshOp>(attrs, args, mesh_device_.get());
        ASSERT_TRUE(lookup.hit);
        EXPECT_EQ(
            mesh_device_->get_program_cache()
                .get(lookup.key)
                .cached_program.template get<Adapter::cached_mesh_workload_t>()
                .workload.get_programs()
                .size(),
            expected_programs);

        const auto initial = cache_values(first);
        const auto filled = fill_values();
        const auto shards = ttnn::distributed::get_device_tensors(output.cpu());
        ASSERT_EQ(shards.size(), shape.mesh_size());
        size_t index = 0;
        for (const auto& coord : all) {
            const auto values = shards[index++].to_vector<float>();
            const bool written = !noop && selected.contains(coord);
            for (size_t i = 0; i < values.size(); ++i) {
                const bool in_filled_blocks = i < filled.size();
                const float expected = written && in_filled_blocks ? filled[i] : initial[i];
                ASSERT_EQ(values[i], expected) << "device " << coord << ", element " << i << ", noop " << noop;
            }
        }
    }
}

}  // namespace paged_fill_cache_mesh_programs_test
