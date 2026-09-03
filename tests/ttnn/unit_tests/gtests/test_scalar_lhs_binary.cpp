// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <array>
#include <optional>
#include <random>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::binary::test {

// ttnn.divide is rebound to the composite div in Python, so the scalar-first
// ttnn::divide overload is reachable only from C++ and no Python test covers it.
class ScalarLhsBinaryFixture : public TTNNFixtureWithSuiteDevice<ScalarLhsBinaryFixture> {};

TEST_F(ScalarLhsBinaryFixture, DivideScalarByTensorIsNotCommutative) {
    auto& device = *device_;
    std::array<uint32_t, 2> dimensions = {32, 64};
    ttnn::Shape shape(dimensions);

    const auto input_tensor = ttnn::full(shape, 4.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

    const auto scalar_lhs = ttnn::divide(operations::unary::ScalarVariant{2.0f}, input_tensor);
    const auto expected = ttnn::full(shape, 0.5f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    TT_FATAL(
        ttnn::allclose<::bfloat16>(ttnn::from_device(expected), ttnn::from_device(scalar_lhs)),
        "divide(2, full(4)) should be 0.5");

    // 2/4 and 4/2 differ, so a dropped mirror flag cannot pass this.
    const auto tensor_lhs = ttnn::divide(input_tensor, operations::unary::ScalarVariant{2.0f});
    const auto reversed = ttnn::full(shape, 2.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    TT_FATAL(
        ttnn::allclose<::bfloat16>(ttnn::from_device(reversed), ttnn::from_device(tensor_lhs)),
        "divide(full(4), 2) should be 2.0");
}

TEST_F(ScalarLhsBinaryFixture, MultiplyScalarByBlockFloatIgnoresFastApproximateMode) {
    auto& device = *device_;
    std::array<uint32_t, 2> dimensions = {32, 64};
    ttnn::Shape shape(dimensions);

    // Varied values are required: a constant times a constant is exact on either unit and
    // would agree even with the wrong kernel selected.
    std::mt19937 rng(6);
    std::uniform_real_distribution<float> dist(0.5f, 1.5f);
    std::vector<float> values(shape.volume());
    for (auto& v : values) {
        v = dist(rng);
    }
    const tt::tt_metal::TensorSpec spec(
        shape, tt::tt_metal::TensorLayout(DataType::BFLOAT8_B, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
    const auto input_tensor = Tensor::from_vector(values, spec).to_device(&device, MemoryConfig{}, ttnn::QueueId(0));

    // Block float runs on the FPU only, so the block format overrides the caller's flag and the
    // result must not depend on it. Without that override an unset or false flag picks the SFPU.
    // Comparing the two flag values rather than the two operand orders is deliberate: mirroring
    // swaps the FPU's srcA/srcB, which perturbs rounding for any dtype, so the operand orders are
    // not bit-identical even when both are correct.
    const auto forced = ttnn::multiply(
        operations::unary::ScalarVariant{3.7f},
        input_tensor,
        DataType::BFLOAT16,
        std::nullopt,
        std::nullopt,
        {},
        {},
        {},
        /*fast_and_approximate_mode=*/true);
    const auto defaulted = ttnn::multiply(
        operations::unary::ScalarVariant{3.7f},
        input_tensor,
        DataType::BFLOAT16,
        std::nullopt,
        std::nullopt,
        {},
        {},
        {},
        /*fast_and_approximate_mode=*/false);
    TT_FATAL(
        ttnn::allclose<::bfloat16>(ttnn::from_device(forced), ttnn::from_device(defaulted)),
        "block-float multiply must select the FPU kernel regardless of fast_and_approximate_mode");
}

TEST_F(ScalarLhsBinaryFixture, SubtractScalarFromBlockFloatTensorIsMirrored) {
    auto& device = *device_;
    std::array<uint32_t, 2> dimensions = {32, 64};
    ttnn::Shape shape(dimensions);

    // The block-float multiply test above compares two flag values on one operand order, and
    // multiply is commutative, so it cannot detect a dropped mirror. Subtract on the same
    // block formats can: 2 - 6 and 6 - 2 differ in sign, and this checks an absolute value
    // rather than a second execution of the same implementation.
    for (const auto dtype : {DataType::BFLOAT8_B, DataType::BFLOAT4_B}) {
        const auto input_tensor = ttnn::full(shape, 6.0f, dtype, ttnn::TILE_LAYOUT, device);

        const auto scalar_lhs =
            ttnn::subtract(operations::unary::ScalarVariant{2.0f}, input_tensor, DataType::BFLOAT16);
        const auto expected = ttnn::full(shape, -4.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
        TT_FATAL(
            ttnn::allclose<::bfloat16>(ttnn::from_device(expected), ttnn::from_device(scalar_lhs)),
            "subtract(2, full(6)) on {} should be -4",
            dtype);
    }
}

TEST_F(ScalarLhsBinaryFixture, OmittedFastApproximateModeMatchesTensorFirstDefault) {
    auto& device = *device_;
    std::array<uint32_t, 2> dimensions = {32, 64};
    ttnn::Shape shape(dimensions);

    // add and subtract default an unset mode to true, keeping the FPU kernel, and the scalar-first
    // overloads must default it the same way as their tensor-first counterparts. Python cannot show
    // this: its bindings always pass an explicit value.
    //
    // Asserted on program-cache entries rather than output values. The mode selects a different
    // compiled kernel, so an omitted mode sharing an entry with an explicit true is exact evidence
    // that it resolved to true, whereas comparing outputs would pass even if FPU and SFPU happen to
    // agree bit-for-bit on this operand pair.
    device.enable_program_cache();
    device.clear_program_cache();

    const auto input_tensor = ttnn::full(shape, 3.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    const operations::unary::ScalarVariant scalar{2.0f};

    ttnn::subtract(scalar, input_tensor, std::nullopt, std::nullopt, std::nullopt, {}, {}, {}, /*fast=*/true);
    const auto after_explicit_true = device.num_program_cache_entries();

    ttnn::subtract(scalar, input_tensor);
    TT_FATAL(
        device.num_program_cache_entries() == after_explicit_true,
        "an omitted fast_and_approximate_mode must reuse the explicit-true program, got {} entries vs {}",
        device.num_program_cache_entries(),
        after_explicit_true);

    // The counterpart: an explicit false is a different kernel, so it must add an entry. Without this
    // the assertion above would also hold if the mode had stopped affecting kernel selection at all.
    ttnn::subtract(scalar, input_tensor, std::nullopt, std::nullopt, std::nullopt, {}, {}, {}, /*fast=*/false);
    TT_FATAL(
        device.num_program_cache_entries() == after_explicit_true + 1,
        "an explicit false must compile its own program, got {} entries vs {}",
        device.num_program_cache_entries(),
        after_explicit_true + 1);

    device.disable_and_clear_program_cache();
}

}  // namespace ttnn::operations::binary::test
