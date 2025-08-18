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
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

class ReduceRowOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

xt::xarray<float> sum_and_reduce_over_dim(const xt::xarray<float>& input) {
    xt::xarray<float> result = xt::sum(input, -1, xt::keep_dims);
    return result;
}

TEST_F(ReduceRowOpTest, ReduceRowOpTest_Comp_One_Tile) {
    using namespace ttml;

    // const uint32_t N = 1U, C = 1U, H = 32U, W = 32U;
    const uint32_t N = 1U, C = 1U, H = 1U, W = 1024U;

    std::random_device rd;
    std::mt19937 gen(42);
    xt::xarray<float> input_tensor = xt::random::rand<float>({N, C, H, W}, 0.0F, 1.0F, gen);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());
    std::cout << "Input:\n";
    input.print();

    auto result_reduce = ttml::metal::reduce_row_test_op(input, false);
    std::cout << "Profiler_no_op_test:\nResult:\n";
    result_reduce.print();

    auto result_matmul = ttml::metal::reduce_row_test_op(input, true);
    std::cout << "Profiler_no_op_test:\nResult with matmul:\n";
    result_matmul.print();

    auto expected_result = sum_and_reduce_over_dim(input_tensor);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result_reduce);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 1e-2F, 1e-2F));
}

// The following function writes results to a file.
// These results are later used for plotting to visualize improvements in stability
// and to compare this approach with previous implementations.

// Helper: Convert vector to a JSON-like string
std::string vector_to_json(const std::string& name, const std::vector<float>& vec) {
    std::ostringstream oss;
    oss << "\"" << name << "\": [";
    for (size_t i = 0; i < vec.size(); ++i) {
        oss << std::fixed << std::setprecision(6) << vec[i];
        if (i + 1 < vec.size())
            oss << ", ";
    }
    oss << "]";
    return oss.str();
}

TEST_F(ReduceRowOpTest, ReduceRowOpTest_Print_Result) {
    using namespace ttml;
    const uint32_t N = 1U, C = 1U, H = 1U, W = 1024U;

    const auto shape = ttnn::SmallVector<uint32_t>{N, C, H, W};
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint32_t> class_dist(0, W - 1);

    const std::string output_path = "reduce_tile_test_output.jsonl";
    std::ofstream file(output_path);
    if (!file.is_open()) {
        fmt::print("Failed to open file: {}\n", output_path);
        return;
    }

    fmt::print("Writing to: {}\n", std::filesystem::current_path());

    auto write_result = [&](const std::vector<float>& baseline_res,
                            const std::vector<float>& reduce_res,
                            const std::vector<float>& matmul_res) {
        file << "{" << vector_to_json("baseline_res", baseline_res) << ", " << vector_to_json("reduce_res", reduce_res)
             << ", " << vector_to_json("matmul_res", matmul_res) << "}\n";
    };

    const uint32_t test_size = 1000U;

    for (uint32_t i = 0; i < test_size; ++i) {
        xt::xarray<float> input_tensor = xt::random::rand<float>({N, C, H, W}, 0.0F, 1.0F, gen);
        auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

        auto expected_result = sum_and_reduce_over_dim(input_tensor);
        auto baseline_res_view = xt::view(expected_result, 0, 0, 0, xt::all());
        std::vector<float> baseline_res_vec(baseline_res_view.begin(), baseline_res_view.end());

        auto result_reduce = ttml::metal::reduce_row_test_op(input, false);
        auto result_matmul = ttml::metal::reduce_row_test_op(input, true);

        auto reduce_res_vec = core::to_vector(result_reduce);
        auto matmul_vec = core::to_vector(result_matmul);

        write_result(baseline_res_vec, reduce_res_vec, matmul_vec);
    }

    file.close();
    fmt::print("Results written to {}\n", output_path);

    EXPECT_TRUE(false);  // For debug inspection
}
