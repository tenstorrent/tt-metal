// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Device-free smoke test of the matmul config model's input side (matmul_model_features.hpp).

#include <gtest/gtest.h>

#include <cmath>

#include "ttnn/operations/matmul/device/config/model/matmul_model_features.hpp"

namespace ttnn::operations::matmul::model {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

TEST(MatmulModelFeatures, Smoke) {
    MatmulProblem prob;
    prob.M = 1024;
    prob.K = 4096;
    prob.N = 4096;
    prob.b_dtype = DataType::kBfloat8B;
    prob.fidelity = Fidelity::kHiFi2;
    prob.packer_l1_acc = true;
    MatmulCandidate c;
    c.family = Family::k2D;
    c.grid_x = 11;
    c.grid_y = 10;
    c.in0_block_w = 16;
    c.out_subblock_h = 2;
    c.out_subblock_w = 4;
    c.per_core_M = 4;
    c.per_core_N = 12;
    c.fuse_batch = true;
    const auto& dev = device_params(Arch::kBlackholeP150);
    const ModelInput in = compute_model_input(prob, c, dev);
    EXPECT_EQ(in.status, ScoreStatus::kOk);
    EXPECT_TRUE(std::isfinite(in.base_ns));
    EXPECT_GT(in.base_ns, 0.0);
    ModelCache cache;
    const ModelInput cached = compute_model_input(prob, c, dev, &cache);
    EXPECT_EQ(cached.base_ns, in.base_ns);
    for (std::size_t i = 0; i < kNumFeatures; ++i) {
        EXPECT_TRUE(
            cached.features[i] == in.features[i] || (std::isnan(cached.features[i]) && std::isnan(in.features[i])))
            << kFeatureNames[i];
    }
    c.grid_x = 12;  // wider than the device
    EXPECT_EQ(compute_model_input(prob, c, dev).status, ScoreStatus::kGridExceedsDevice);
    prob.M = 0;
    EXPECT_EQ(compute_model_input(prob, c, dev).status, ScoreStatus::kInvalidProblem);
}

TEST(MatmulModelFeatures, GalaxyGrid) {
    EXPECT_EQ(blackhole_arch_for_grid(11, 10), Arch::kBlackholeP150);
    EXPECT_EQ(blackhole_arch_for_grid(8, 8), Arch::kBlackholeP150);
    EXPECT_EQ(blackhole_arch_for_grid(12, 10), Arch::kBlackholeGalaxy);
    EXPECT_FALSE(blackhole_arch_for_grid(13, 10).has_value());
    EXPECT_FALSE(blackhole_arch_for_grid(11, 11).has_value());

    const auto& p150 = device_params(Arch::kBlackholeP150);
    const auto& glx = device_params(Arch::kBlackholeGalaxy);
    EXPECT_EQ(glx.worker_grid_x, 12u);
    EXPECT_EQ(glx.worker_cols[11], 16u);
    for (std::size_t i = 0; i < p150.worker_grid_x; ++i) {
        EXPECT_EQ(glx.worker_cols[i], p150.worker_cols[i]);
    }

    MatmulProblem prob;
    prob.M = 1024;
    prob.K = 4096;
    prob.N = 4096;
    prob.fidelity = Fidelity::kHiFi2;
    prob.packer_l1_acc = true;
    MatmulCandidate c;
    c.family = Family::k2D;
    c.grid_x = 12;
    c.grid_y = 10;
    c.in0_block_w = 16;
    c.out_subblock_h = 2;
    c.out_subblock_w = 4;
    c.per_core_M = 4;
    c.per_core_N = 12;
    c.fuse_batch = true;
    const ModelInput in = compute_model_input(prob, c, glx);
    EXPECT_EQ(in.status, ScoreStatus::kOk);
    EXPECT_TRUE(std::isfinite(in.base_ns));
    EXPECT_EQ(compute_model_input(prob, c, p150).status, ScoreStatus::kGridExceedsDevice);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace
}  // namespace ttnn::operations::matmul::model
