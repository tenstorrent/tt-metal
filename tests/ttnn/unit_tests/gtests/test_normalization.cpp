// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Smoke tests for ttnn::batch_norm: one test per program factory, plus the
// front-end code path that picks between the FPU and SFPU compute kernels.
// batch_norm one-sided running stats (#51230) is excluded with an issue
// reference in the section comment. Each test uses small deterministic inputs
// with exact closed-form expected outputs, so a kernel that runs but produces
// garbage fails.
//
// Each test states in a comment which program factory or code path it covers.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/shape.hpp>
#include "smoke_test_utils.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/normalization/batch_norm/batch_norm.hpp"
#include "ttnn/operations/normalization/batch_norm/device/running_statistics_device_operation.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::normalization::test {

class NormalizationSmoke : public TTNNFixtureWithSuiteDevice<NormalizationSmoke> {};

namespace detail {

using ttnn::test_utils::expect_close;
using ttnn::test_utils::make_device_tensor;
using ttnn::test_utils::make_device_tensor_mc;
using ttnn::test_utils::to_float_vector;

}  // namespace detail
// ---------------------------------------------------------------------------
// BATCH_NORM cells
// Op under test: ttnn::batch_norm (batch_norm/batch_norm.hpp)
//   Tensor batch_norm(input, running_mean, running_var, training, eps, momentum, weight, bias,
//                     output, memory_config, compute_kernel_config)
// Validation (batch_norm_device_operation.cpp): input/stat/weight/bias/output must be rank-4 TILE
// on device, BFLOAT16 or FLOAT32, INTERLEAVED, with shape[1] == C; stats/weight/bias are logical
// [1, C, 1, 1] (padded to tiles). running_mean/running_var are REQUIRED in inference mode.
// Kernel choice (batch_norm_program_factory.cpp): use_sfpu_kernel = fp32_dest_acc_en || any_float32;
// the op's default compute config (batch_norm_utils.cpp) sets fp32_dest_acc_en = true, so the FPU
// kernel is reachable only with an explicit config + all-bf16 tensors.
// ---------------------------------------------------------------------------

// Covers: BatchNormOperation::BatchNormFactory with the FPU compute kernel
// (device/kernels/compute/batch_norm_kernel.cpp). Explicit fp32_dest_acc_en=false plus all-BFLOAT16
// tensors pins use_sfpu_kernel = fp32_dest_acc_en || any_float32 to false (default config would
// force the SFPU kernel).
TEST_F(NormalizationSmoke, BatchNormInferenceFpu) {
    auto& device = *device_;
    const ttnn::Shape x_shape({2, 1, 32, 32});
    const ttnn::Shape stat_shape({1, 1, 1, 1});  // logical [1, C, 1, 1] with C = 1; padded to one tile
    auto x = detail::make_device_tensor(
        device, x_shape, std::vector<float>(2 * 32 * 32, 3.0f), ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
    auto running_mean = detail::make_device_tensor(
        device, stat_shape, std::vector<float>{1.0f}, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
    auto running_var = detail::make_device_tensor(
        device, stat_shape, std::vector<float>{4.0f}, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);

    const DeviceComputeKernelConfig fpu_cfg{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .math_approx_mode = false,
        .fp32_dest_acc_en = false,  // pins the FPU kernel (no float32 tensors involved)
        .packer_l1_acc = false,
        .dst_full_sync_en = false};
    auto y = ttnn::batch_norm(
        x,
        running_mean,
        running_var,
        /*training=*/false,
        /*eps=*/0.0f,
        /*momentum=*/0.1f,
        /*weight=*/std::nullopt,
        /*bias=*/std::nullopt,
        /*output=*/std::nullopt,
        /*memory_config=*/std::nullopt,
        fpu_cfg);

    // y = (3 - 1) / sqrt(4 + 0) = 1.0; every operand and intermediate (2, 4, 0.5, 1) is a small
    // integer or power of two, exact in bf16 -> exact golden.
    detail::expect_close(detail::to_float_vector(y), std::vector<float>(2 * 32 * 32, 1.0f), 0.0f, 0.0f);
}

// Covers: BatchNormOperation::BatchNormFactory with the SFPU compute kernel
// (device/kernels/compute/batch_norm_sfpu_kernel.cpp): both triggers at once -- fp32_dest_acc_en=true
// AND FLOAT32 tensors (any_float32). HiFi3 per Wormhole HW bug #38306 (HiFi4 + fp32 acc can be
// inaccurate; matches the op's own fp32-acc default). Also covers the weight/bias affine stage.
TEST_F(NormalizationSmoke, BatchNormInferenceSfpu) {
    auto& device = *device_;
    const ttnn::Shape x_shape({2, 1, 32, 32});
    const ttnn::Shape stat_shape({1, 1, 1, 1});
    auto x = detail::make_device_tensor(
        device, x_shape, std::vector<float>(2 * 32 * 32, 3.0f), ttnn::DataType::FLOAT32, ttnn::Layout::TILE);
    auto running_mean = detail::make_device_tensor(
        device, stat_shape, std::vector<float>{1.0f}, ttnn::DataType::FLOAT32, ttnn::Layout::TILE);
    auto running_var = detail::make_device_tensor(
        device, stat_shape, std::vector<float>{4.0f}, ttnn::DataType::FLOAT32, ttnn::Layout::TILE);
    // weight/bias share the params' dtype family (all params must have identical dtype).
    auto weight = detail::make_device_tensor(
        device, stat_shape, std::vector<float>{3.0f}, ttnn::DataType::FLOAT32, ttnn::Layout::TILE);
    auto bias = detail::make_device_tensor(
        device, stat_shape, std::vector<float>{-1.0f}, ttnn::DataType::FLOAT32, ttnn::Layout::TILE);

    const DeviceComputeKernelConfig sfpu_cfg{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi3,
        .math_approx_mode = false,
        .fp32_dest_acc_en = true,  // SFPU kernel; float32 tensors would force it regardless
        .packer_l1_acc = false,
        .dst_full_sync_en = false};
    auto y = ttnn::batch_norm(
        x,
        running_mean,
        running_var,
        /*training=*/false,
        /*eps=*/0.0f,
        /*momentum=*/0.1f,
        weight,
        bias,
        /*output=*/std::nullopt,
        /*memory_config=*/std::nullopt,
        sfpu_cfg);

    // y = ((3 - 1) / sqrt(4)) * 3 + (-1) = 2.0 -- all steps exact in fp32.
    detail::expect_close(detail::to_float_vector(y), std::vector<float>(2 * 32 * 32, 2.0f), 0.0f, 0.0f);
}

// Covers: RunningStatistics::RunningStatisticsProgramFactory
// (running_statistics_program_factory.cpp) updating running_mean/running_var IN PLACE:
//   new = (1 - momentum) * old + momentum * batch.
// The prim is driven DIRECTLY rather than through ttnn::batch_norm(training=true). The training
// composite (batch_norm.cpp) reaches this prim only after mean_NHW -- two chained ttnn::mean
// reductions -- plus ttnn::subtract and ttnn::square, and those are reduction/eltwise factories
// owned by ReductionSmoke, not normalization factories. Routing through them cost 51 JIT kernel
// builds (13.6 s cold on BH p100a) against this file's 13 (1.7 s), for no normalization coverage
// this file does not already have: BatchNormOperation::BatchNormFactory is covered by the four
// inference cells above. Merge-gate rule is 5 s per test case (merge-gate.yaml), so the composite
// wiring -- that batch_norm(training=true) feeds mean_NHW's output into this prim -- is left to
// the post-merge suites, where the budget is 10-25x larger.
//
// bf16 with momentum 0.25: every term below is exact in bf16, so the readbacks are exact rather
// than tolerance-based. Values are chosen so both terms of both accumulators are distinct
// (1.5 != 8.0, and neither equals any input), which a swapped mean/var or a swapped
// momentum/(1-momentum) would break:
//   running_mean = 0.75 * 1 + 0.25 * 3 = 0.75 + 0.75 = 1.5
//   running_var  = 0.75 * 9 + 0.25 * 5 = 6.75 + 1.25 = 8.0
TEST_F(NormalizationSmoke, BatchNormRunningStatsUpdate) {
    auto& device = *device_;
    const ttnn::Shape stat_shape({1, 1, 1, 1});
    auto make = [&](float v) {
        return detail::make_device_tensor(
            device, stat_shape, std::vector<float>{v}, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
    };
    const auto batch_mean = make(3.0f);
    const auto batch_var = make(5.0f);
    auto running_mean = make(1.0f);
    auto running_var = make(9.0f);

    ttnn::prim::running_statistics(batch_mean, batch_var, /*momentum=*/0.25f, running_mean, running_var);

    // Read back from the SAME tensor handles the prim wrote in place.
    const auto rm = detail::to_float_vector(running_mean);
    const auto rv = detail::to_float_vector(running_var);
    ASSERT_EQ(rm.size(), 1u);
    ASSERT_EQ(rv.size(), 1u);
    EXPECT_FLOAT_EQ(rm[0], 1.5f);
    EXPECT_FLOAT_EQ(rv[0], 8.0f);
}

// Covers: per-channel parameter indexing in BatchNormOperation::BatchNormFactory -- C = 2, so
// mean/var/weight/bias each carry one tile per channel ([1, C, 1, 1] logical) and the reader must
// pair the right stat tile with each input channel. Distinct exact goldens per channel:
//   ch0: (5 - 4)/sqrt(1) * 1 + 0 = 1.0        ch1: (1 - 0)/sqrt(4) * 2 + 1 = 2.0
// Any cross-channel mix-up of mean/var/weight/bias produces a different (still exact) value.
TEST_F(NormalizationSmoke, BatchNormMultiChannel) {
    auto& device = *device_;
    const ttnn::Shape x_shape({1, 2, 32, 32});
    const ttnn::Shape stat_shape({1, 2, 1, 1});
    std::vector<float> x_data(2 * 32 * 32, 5.0f);             // channel 0 = 5.0
    std::fill(x_data.begin() + 32 * 32, x_data.end(), 1.0f);  // channel 1 = 1.0 (NCHW row-major)
    auto x = detail::make_device_tensor(device, x_shape, x_data, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
    auto running_mean = detail::make_device_tensor(
        device, stat_shape, std::vector<float>{4.0f, 0.0f}, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
    auto running_var = detail::make_device_tensor(
        device, stat_shape, std::vector<float>{1.0f, 4.0f}, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
    auto weight = detail::make_device_tensor(
        device, stat_shape, std::vector<float>{1.0f, 2.0f}, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
    auto bias = detail::make_device_tensor(
        device, stat_shape, std::vector<float>{0.0f, 1.0f}, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);

    auto y = ttnn::batch_norm(
        x, running_mean, running_var, /*training=*/false, /*eps=*/0.0f, /*momentum=*/0.1f, weight, bias);

    std::vector<float> expected(2 * 32 * 32, 1.0f);
    std::fill(expected.begin() + 32 * 32, expected.end(), 2.0f);
    detail::expect_close(detail::to_float_vector(y), expected, 0.0f, 0.0f);
}

// Covers: preallocated-output path (BatchNormOperation::create_output_tensors returns the provided
// tensor instead of allocating). Output must match input dtype; result is read back from the
// PREALLOCATED handle, proving the kernel wrote into the caller's buffer.
TEST_F(NormalizationSmoke, BatchNormPreallocatedOutput) {
    auto& device = *device_;
    const ttnn::Shape x_shape({2, 1, 32, 32});
    const ttnn::Shape stat_shape({1, 1, 1, 1});
    auto x = detail::make_device_tensor(
        device, x_shape, std::vector<float>(2 * 32 * 32, 3.0f), ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
    auto running_mean = detail::make_device_tensor(
        device, stat_shape, std::vector<float>{1.0f}, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
    auto running_var = detail::make_device_tensor(
        device, stat_shape, std::vector<float>{4.0f}, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
    auto out = detail::make_device_tensor(
        device, x_shape, std::vector<float>(2 * 32 * 32, 0.0f), ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);

    auto ret = ttnn::batch_norm(
        x,
        running_mean,
        running_var,
        /*training=*/false,
        /*eps=*/0.0f,
        /*momentum=*/0.1f,
        /*weight=*/std::nullopt,
        /*bias=*/std::nullopt,
        /*output=*/out);

    // Same golden as BatchNormInferenceFpu: (3 - 1)/sqrt(4) = 1.0 exact; check the caller's tensor.
    const std::vector<float> expected(2 * 32 * 32, 1.0f);
    detail::expect_close(detail::to_float_vector(out), expected, 0.0f, 0.0f);
    detail::expect_close(detail::to_float_vector(ret), expected, 0.0f, 0.0f);
}

}  // namespace ttnn::operations::normalization::test
