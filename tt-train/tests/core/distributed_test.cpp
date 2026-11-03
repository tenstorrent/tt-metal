// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "core/distributed/distributed.hpp"

#include <gtest/gtest.h>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xbuilder.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "test_utils/mesh_utils.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"

using namespace ttml;

namespace {

ttnn::Tensor distribute(const ttnn::distributed::TensorToMesh& mapper) {
    auto* device = &autograd::ctx().get_device();
    const xt::xarray<float> values = xt::zeros<float>({1U, 1U, 32U, 64U});
    return core::from_xtensor(values, device, ttnn::Layout::TILE, &mapper);
}

}  // namespace

class IsShardedOnAxisTest : public test_utils::Mesh1x2SuiteFixture {};

TEST_F(IsShardedOnAxisTest, ReadsThePlacementOfEachAxis) {
    using ttnn::distributed::MeshMapperConfig;
    auto& device = autograd::ctx().get_device();
    const auto mapper = ttnn::distributed::create_mesh_mapper(
        device, MeshMapperConfig{.placements = {MeshMapperConfig::Replicate{}, MeshMapperConfig::Shard{3}}});
    const auto tensor = distribute(*mapper);
    ASSERT_EQ(tensor.tensor_topology().placements().size(), 2U);

    EXPECT_FALSE(core::distributed::is_sharded_on_axis(tensor, 0U));
    EXPECT_TRUE(core::distributed::is_sharded_on_axis(tensor, 1U));
}

// A tensor sharded without a cluster axis carries a single placement for the whole mesh, which covers every axis.
TEST_F(IsShardedOnAxisTest, CollapsedShardIsShardedOnEveryAxis) {
    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(autograd::ctx().get_device(), /*dim=*/3);
    const auto tensor = distribute(*mapper);
    ASSERT_EQ(tensor.tensor_topology().placements().size(), 1U);

    EXPECT_TRUE(core::distributed::is_sharded_on_axis(tensor, 0U));
    EXPECT_TRUE(core::distributed::is_sharded_on_axis(tensor, 1U));
}

TEST_F(IsShardedOnAxisTest, ReplicatedIsNotSharded) {
    const auto mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(autograd::ctx().get_device());
    const auto tensor = distribute(*mapper);

    EXPECT_FALSE(core::distributed::is_sharded_on_axis(tensor, 0U));
    EXPECT_FALSE(core::distributed::is_sharded_on_axis(tensor, 1U));
}
