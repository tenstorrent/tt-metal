// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/cross_entropy_bw/device/cross_entropy_bw_device_operation.hpp"
#include "metal/ops/cross_entropy_fw/device/cross_entropy_fw_device_operation.hpp"
#include "metal/ops/frobenius_normalize/device/frobenius_normalize_device_operation.hpp"
#include "metal/ops/profiler_no_op/device/profiler_no_op_device_operation.hpp"
#include "metal/ops/rmsnorm_bw/device/rmsnorm_bw_device_operation.hpp"
#include "metal/ops/rmsnorm_fw/device/rmsnorm_fw_device_operation.hpp"
#include "metal/ops/select_target_logit/device/select_target_logit_device_operation.hpp"
#include "metal/ops/silu_bw/device/silu_bw_device_operation.hpp"
#include "metal/ops/softmax/device/softmax_device_operation.hpp"
#include "metal/ops/subtract_at_target/device/subtract_at_target_device_operation.hpp"
#include "metal/ops/swiglu_elemwise_bw/swiglu_elemwise_bw.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

// A program's compile-time args bake in where each tensor lives (TensorAccessorArgs), and a program-cache hit only
// patches buffer addresses. So an op launched on an L1 input after a DRAM input of the same shape must not reuse the
// DRAM program, whose kernels would read the L1 address in DRAM. Each test launches an op on DRAM tensors, then again
// with its main input moved to L1. Every other tensor, outputs included, stays in DRAM, so a reused program could only
// read the wrong memory, never write it.

class ProgramCachePlacementTest : public ttml::test_utils::DeviceSuiteTest {};

namespace {

using ttml::test_utils::zeros_on_device;

// [N, 1, S, V]: one tile row of two tiles, and its one-value-per-row reduction.
const std::vector<uint32_t> kShape = {1U, 1U, 32U, 64U};
const std::vector<uint32_t> kRowShape = {1U, 1U, 32U, 1U};

ttnn::Tensor random_input(const std::vector<uint32_t>& shape, uint32_t seed, ttnn::Layout layout = ttnn::Layout::TILE) {
    return ttml::core::from_xtensor(
        ttml::test_utils::make_uniform_xarray<float>(shape, 0.5F, 1.5F, seed),
        &ttml::autograd::ctx().get_device(),
        layout);
}

// [N, S] target classes in [0, V), UINT32 and ROW_MAJOR.
ttnn::Tensor random_targets(uint32_t seed) {
    return ttml::core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        ttml::test_utils::make_uniform_xarray<uint32_t>(
            std::vector<uint32_t>{kShape[0], kShape[2]}, 0U, kShape[3] - 1U, seed),
        &ttml::autograd::ctx().get_device(),
        ttnn::Layout::ROW_MAJOR);
}

// `launch` runs the op on the main input it is given, with every other tensor and every output in DRAM.
template <typename Launch>
void expect_l1_launch_throws_after_dram(const ttnn::Tensor& dram_input, const Launch& launch) {
    ttml::test_utils::device_with_empty_program_cache();
    ASSERT_NO_THROW(launch(dram_input)) << "the DRAM launch";
    const auto l1_input = ttml::ttnn_fixed::to_l1_interleaved(dram_input);
    // The op supports DRAM only: an L1 input must fail loudly instead of reusing the program built for the DRAM input.
    EXPECT_ANY_THROW(launch(l1_input));
}

}  // namespace

TEST_F(ProgramCachePlacementTest, CrossEntropyFw_L1InputAfterDramDoesNotReuseTheProgram) {
    const auto target = random_targets(2U);
    expect_l1_launch_throws_after_dram(random_input(kShape, 1U), [&](const ttnn::Tensor& input) {
        ttnn::prim::ttml_cross_entropy_fw(input, target, zeros_on_device(kRowShape));
    });
}

TEST_F(ProgramCachePlacementTest, CrossEntropyBw_L1InputAfterDramDoesNotReuseTheProgram) {
    const auto target = random_targets(2U);
    expect_l1_launch_throws_after_dram(random_input(kShape, 1U), [&](const ttnn::Tensor& input) {
        ttnn::prim::ttml_cross_entropy_bw(input, target, /*scaler=*/1.0F, zeros_on_device(kShape));
    });
}

TEST_F(ProgramCachePlacementTest, RmsNormFw_L1InputAfterDramDoesNotReuseTheProgram) {
    const auto gamma = random_input({1U, 1U, 1U, kShape[3]}, 2U);
    expect_l1_launch_throws_after_dram(random_input(kShape, 1U), [&](const ttnn::Tensor& input) {
        ttnn::prim::ttml_rmsnorm_fw(
            input, gamma, /*return_intermediates=*/false, /*epsilon=*/1e-6F, std::nullopt, zeros_on_device(kShape));
    });
}

TEST_F(ProgramCachePlacementTest, RmsNormBw_L1InputAfterDramDoesNotReuseTheProgram) {
    const auto gamma = random_input({1U, 1U, 1U, kShape[3]}, 2U);
    const auto rms = random_input(kRowShape, 3U);
    const auto dL_dout = random_input(kShape, 4U);
    expect_l1_launch_throws_after_dram(random_input(kShape, 1U), [&](const ttnn::Tensor& input) {
        ttnn::prim::ttml_rmsnorm_bw(
            input, gamma, rms, dL_dout, /*epsilon=*/1e-6F, zeros_on_device(kShape), zeros_on_device(kShape));
    });
}

TEST_F(ProgramCachePlacementTest, SiluBw_L1InputAfterDramDoesNotReuseTheProgram) {
    const auto dL_dout = random_input(kShape, 2U);
    expect_l1_launch_throws_after_dram(random_input(kShape, 1U), [&](const ttnn::Tensor& input) {
        ttnn::prim::ttml_silu_bw(input, dL_dout, zeros_on_device(kShape));
    });
}

TEST_F(ProgramCachePlacementTest, Softmax_L1InputAfterDramDoesNotReuseTheProgram) {
    expect_l1_launch_throws_after_dram(random_input(kShape, 1U), [](const ttnn::Tensor& input) {
        ttnn::prim::ttml_softmax(input, /*dim=*/3, zeros_on_device(kShape));
    });
}

TEST_F(ProgramCachePlacementTest, ProfilerNoOp_L1InputAfterDramDoesNotReuseTheProgram) {
    expect_l1_launch_throws_after_dram(
        random_input(kShape, 1U, ttnn::Layout::ROW_MAJOR), [](const ttnn::Tensor& input) {
            ttnn::prim::ttml_profiler_no_op(input, "placement", zeros_on_device(kShape));
        });
}

TEST_F(ProgramCachePlacementTest, FrobeniusNormalize_L1InputAfterDramDoesNotReuseTheProgram) {
    expect_l1_launch_throws_after_dram(random_input(kShape, 1U), [](const ttnn::Tensor& input) {
        ttnn::prim::ttml_frobenius_normalize(input, /*epsilon=*/1e-7F, zeros_on_device(kShape));
    });
}

TEST_F(ProgramCachePlacementTest, SelectTargetLogit_L1InputAfterDramDoesNotReuseTheProgram) {
    const auto target = random_targets(2U);
    expect_l1_launch_throws_after_dram(random_input(kShape, 1U), [&](const ttnn::Tensor& logit) {
        ttnn::prim::ttml_select_target_logit(
            logit,
            target,
            /*local_V=*/kShape[3],
            /*cluster_axis=*/std::nullopt,
            /*first_v=*/0U,
            zeros_on_device(kRowShape));
    });
}

TEST_F(ProgramCachePlacementTest, SubtractAtTarget_L1InputAfterDramDoesNotReuseTheProgram) {
    const auto target = random_targets(2U);
    expect_l1_launch_throws_after_dram(random_input(kShape, 1U), [&](const ttnn::Tensor& input) {
        ttnn::prim::ttml_subtract_at_target(
            input,
            target,
            /*local_V=*/kShape[3],
            /*cluster_axis=*/std::nullopt,
            /*first_v=*/0U,
            zeros_on_device(kShape));
    });
}

// swiglu_elemwise_bw takes L1 inputs: the L1 launch must build its own program and compute what the DRAM launch did.
TEST_F(ProgramCachePlacementTest, SwigluElemwiseBw_L1InputAfterDramDoesNotReuseTheProgram) {
    using ttml::core::to_xtensor;
    using ttml::test_utils::count_bit_mismatches;
    auto& device = ttml::test_utils::device_with_empty_program_cache();
    const auto linear1 = random_input(kShape, 1U);
    const auto gate = random_input(kShape, 2U);
    const auto dL_dprod = random_input(kShape, 3U);
    const auto dram =
        ttml::metal::swiglu_elemwise_bw(linear1, gate, dL_dprod, zeros_on_device(kShape), zeros_on_device(kShape));

    const auto l1_linear1 = ttml::ttnn_fixed::to_l1_interleaved(linear1);
    const auto dL_dlinear1 = zeros_on_device(kShape);
    const auto dL_dgate = zeros_on_device(kShape);
    const auto entries_before = device.num_program_cache_entries();
    ttml::metal::swiglu_elemwise_bw(l1_linear1, gate, dL_dprod, dL_dlinear1, dL_dgate);
    EXPECT_GT(device.num_program_cache_entries(), entries_before) << "the L1 launch reused the DRAM launch's program";
    EXPECT_EQ(count_bit_mismatches(to_xtensor(dram.dL_dlinear1), to_xtensor(dL_dlinear1)), 0U) << "dL_dlinear1";
    EXPECT_EQ(count_bit_mismatches(to_xtensor(dram.dL_dgate), to_xtensor(dL_dgate)), 0U) << "dL_dgate";
}
