// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"

enum class ExpectedResult { OK, ERROR };

struct MatmulInput {
    ttnn::Shape shape_a;
    ttnn::Shape shape_b;
    bool transpose_a{false};
    bool transpose_b{false};
};

struct MatmulTest {
    MatmulInput input;
    ExpectedResult expected_result;
};

class GPT2SBatch64Test : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

// Matmul tests are based on GPT2-S model with batch size 64
TEST_F(GPT2SBatch64Test, Matmul) {
    std::vector<MatmulTest> tests = {
        {{{64, 12, 64, 1024}, {64, 12, 1024, 64}, false, false}, ExpectedResult::OK},
        {{{64, 12, 1024, 64}, {64, 12, 1024, 64}, false, true}, ExpectedResult::OK},
        {{{64, 12, 1024, 64}, {64, 12, 1024, 64}, true, false}, ExpectedResult::OK},
        {{{64, 12, 1024, 64}, {64, 12, 64, 1024}, false, false}, ExpectedResult::OK},
        {{{64, 12, 1024, 1024}, {64, 12, 1024, 64}, false, false}, ExpectedResult::OK},
        {{{768, 65536}, {65536, 96}, false, false}, ExpectedResult::OK},
        {{{65536, 768}, {65536, 96}, true, false}, ExpectedResult::OK},
        {{{65536, 96}, {1, 1, 96, 768}, false, false}, ExpectedResult::OK},
        {{{65536, 96}, {1, 1, 768, 96}, false, true}, ExpectedResult::OK},
        {{{3072, 65536}, {65536, 768}, false, false}, ExpectedResult::OK},
        {{{65536, 3072}, {65536, 768}, true, false}, ExpectedResult::OK},
        {{{65536, 768}, {1, 1, 768, 3072}, false, false}, ExpectedResult::OK},
        {{{65536, 768}, {1, 1, 3072, 768}, false, true}, ExpectedResult::OK},
        {{{768, 65536}, {65536, 3072}, false, false}, ExpectedResult::OK},
        {{{65536, 768}, {65536, 3072}, true, false}, ExpectedResult::OK},
        {{{65536, 3072}, {1, 1, 3072, 768}, false, false}, ExpectedResult::OK},
        {{{65536, 3072}, {1, 1, 768, 3072}, false, true}, ExpectedResult::OK},
        {{{65536, 3072}, {3072, 768}, false, false}, ExpectedResult::OK},
        {{{65536, 3072}, {768, 3072}, false, true}, ExpectedResult::OK},
        {{{768, 65536}, {65536, 768}, false, false}, ExpectedResult::OK},
        {{{65536, 768}, {65536, 768}, true, false}, ExpectedResult::OK},
        {{{65536, 768}, {1, 1, 768, 768}, false, false}, ExpectedResult::OK},
        {{{768, 65536}, {1, 1, 768, 768}, true, false}, ExpectedResult::OK},
        {{{768, 65536}, {65536, 2304}, false, false}, ExpectedResult::OK},
        {{{65536, 768}, {65536, 2304}, true, false}, ExpectedResult::OK},
        {{{65536, 768}, {768, 50257}, false, false}, ExpectedResult::OK},
        {{{65536, 768}, {50304, 768}, false, true}, ExpectedResult::OK},
        {{{65536, 50304}, {50304, 768}, false, false}, ExpectedResult::OK},
    };

    auto run_matmul = [](auto& a, auto& b, bool transpose_a, bool transpose_b) {
        fmt::println(
            "Running matmul with shapes {} and {}, tranpose_a {} transpose_b {}",
            a.get_shape(),
            b.get_shape(),
            transpose_a,
            transpose_b);
        [[maybe_unused]] auto c = ttnn::matmul(
            a,
            b,
            transpose_a,
            transpose_b,
            /* memory_config */ std::nullopt,
            /* dtype */ std::nullopt,
            /* program_config */ std::nullopt,
            /* activation */ std::nullopt,
            /* compute_kernel_config */
            ttml::core::ComputeKernelConfig::matmul(),
            /* core_grid */ ttnn::CoreGrid{7, 8},
            /* output_tile */ std::nullopt);
    };
    for (const auto& [input, expected_result] : tests) {
        auto [shape_a, shape_b, transpose_a, transpose_b] = input;

        auto* device = &ttml::autograd::ctx().get_device();
        auto a = ttml::core::empty(shape_a, device, {});
        auto b = ttml::core::empty(shape_b, device, {});

        if (expected_result == ExpectedResult::OK) {
            EXPECT_NO_THROW(run_matmul(a, b, transpose_a, transpose_b));
        } else {
            EXPECT_ANY_THROW(run_matmul(a, b, transpose_a, transpose_b));
        }
    }
}
