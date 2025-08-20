// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/types.h>

#include <cassert>
#include <core/ttnn_all_includes.hpp>
#include <cstddef>
#include <cstdint>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/tensor/shape/shape.hpp>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

class ProfilerNoOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(ProfilerNoOpTest, ProfilerNoOpTest_Batch) {
    using namespace ttml;

    const uint32_t N = 2U, C = 1U, H = 91U, W = 187U;

    xt::xarray<float> input_tensor = xt::empty<float>({N, C, H, W});
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    ttml::core::parallel_generate(
        std::span{input_tensor.data(), input_tensor.size()},
        []() { return std::uniform_real_distribution<float>(-10.0F, 10.0F); },
        seed);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    std::cout << "Input Logits:\n";
    input.print();

    auto result = ttml::metal::profiler_no_op(input, "identifier");
    std::cout << "Profiler_no_op_test:\nResult:\n";
    result.print();
}
