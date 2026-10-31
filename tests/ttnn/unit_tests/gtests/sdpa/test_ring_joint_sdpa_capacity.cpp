// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-only: each logical_n capacity branch (plain, sliding window, stripe split, circular) and that
// dynamic_axes uses that one helper.

#include <optional>

#include <tt-metalium/core_coord.hpp>

#include "gtest/gtest.h"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.hpp"

namespace {

using ttnn::prim::RingJointSDPADeviceOperation;
using ttnn::prim::RingJointSDPAInputs;
using ttnn::prim::RingJointSDPAParams;

ttnn::Tensor seq(uint32_t n) {
    return ttnn::zeros(ttnn::Shape({1, 1, n, 32}), tt::tt_metal::DataType::BFLOAT16, ttnn::TILE_LAYOUT);
}

RingJointSDPAParams make_params(std::size_t ring, bool sliding, bool circular, uint32_t stripe) {
    const tt::tt_metal::MemoryConfig mem{};
    auto all_gather = ttnn::experimental::prim::RingAttentionAllGatherAsyncParams(
        /*dim=*/2,
        /*num_links=*/1,
        /*ring_size=*/static_cast<uint32_t>(ring),
        mem,
        ttnn::ccl::Topology::Ring,
        {},
        std::nullopt,
        std::nullopt);
    return RingJointSDPAParams(
        "rear",
        std::nullopt,
        false,
        false,
        false,
        /*logical_n=*/64,
        /*logical_l=*/0,
        ring,
        mem,
        std::nullopt,
        ttnn::DeviceComputeKernelConfig{},
        all_gather,
        {},
        tt::tt_metal::CoreCoord{0, 0},
        std::nullopt,
        std::nullopt,
        /*latent_v_head_dim=*/0,
        /*kv_cache_num_layers=*/1,
        /*kv_cache_layer_idx=*/0,
        sliding ? std::optional<uint32_t>{32} : std::nullopt,
        circular,
        stripe);
}

std::size_t axis_capacity(const RingJointSDPAParams& args, const RingJointSDPAInputs& inputs) {
    const auto axes = RingJointSDPADeviceOperation::dynamic_axes(args, inputs);
    EXPECT_EQ(axes.size(), 2);
    EXPECT_EQ(axes[0].name, "logical_n");
    EXPECT_EQ(axes[0].capacity, ttnn::prim::logical_n_capacity(args, inputs));
    return axes[0].capacity;
}

TEST(RingJointSDPACapacity, PlainUsesGatheredSequence) {
    RingJointSDPAInputs inputs{.input_q = seq(32), .input_k = seq(32), .gathered_k = seq(128)};
    const auto args = make_params(/*ring=*/4, /*sliding=*/false, /*circular=*/false, /*stripe=*/1);
    EXPECT_EQ(axis_capacity(args, inputs), 128);
}

TEST(RingJointSDPACapacity, SlidingWindowUsesLocalTimesRing) {
    RingJointSDPAInputs inputs{.input_q = seq(32), .input_k = seq(64), .gathered_k = seq(512)};
    const auto args = make_params(/*ring=*/4, /*sliding=*/true, /*circular=*/false, /*stripe=*/1);
    EXPECT_EQ(axis_capacity(args, inputs), 64u * 4u);
}

TEST(RingJointSDPACapacity, StripeSplitUsesLocalTimesRing) {
    RingJointSDPAInputs inputs{.input_q = seq(32), .input_k = seq(48), .gathered_k = seq(512)};
    const auto args = make_params(/*ring=*/4, /*sliding=*/false, /*circular=*/false, /*stripe=*/2);
    EXPECT_EQ(axis_capacity(args, inputs), 48u * 4u);
}

TEST(RingJointSDPACapacity, CircularIsUnbounded) {
    RingJointSDPAInputs inputs{.input_q = seq(32), .input_k = seq(64), .gathered_k = seq(128)};
    const auto args = make_params(/*ring=*/4, /*sliding=*/true, /*circular=*/true, /*stripe=*/1);
    EXPECT_EQ(axis_capacity(args, inputs), ttnn::dynamic_length::DynamicAxis::kUnbounded);
}

}  // namespace
