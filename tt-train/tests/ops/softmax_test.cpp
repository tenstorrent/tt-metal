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

// used for moreh softmax
#include <cmath>

#include "core/compute_kernel_config.hpp"

class SoftmaxTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

xt::xarray<float> xt_softmax(const xt::xarray<float>& input, uint32_t dim = 3U) {
    xt::xarray<float> max_value = xt::amax(input, dim, xt::keep_dims);
    xt::xarray<float> shifted_input = input - max_value;  // for numerical stability
    xt::xarray<float> exp_shifted_input = xt::exp(shifted_input);
    xt::xarray<float> exp_sum = xt::sum(exp_shifted_input, dim, xt::keep_dims);
    xt::xarray<float> result = exp_shifted_input / exp_sum;
    return result;
}

TEST_F(SoftmaxTest, SoftmaxTest_Batch) {
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 59U, W = 197U;
    const auto shape = ttnn::SmallVector<uint32_t>{N, C, H, W};
    int32_t dim = 3U;

    std::random_device rd;
    std::mt19937 gen(42);
    xt::xarray<float> input_tensor = xt::random::rand<float>({N, C, H, W}, -10.0F, 10.0F, gen);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input.print();

    auto result = ttml::metal::softmax(input, dim);
    std::cout << "Sofrmax_test:\nResult:\n";
    result.print();

    auto ttnn_softmax = ttnn_fixed::softmax(input, dim);
    auto ttnn_softmax_xtensor = core::to_xtensor(ttnn_softmax);

    auto expected_result = xt_softmax(input_tensor, dim);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(SoftmaxTest, SoftmaxTest_Big_Batch) {
    using namespace ttml;

    const uint32_t N = 1U, C = 1U, H = 32U, W = 128007U;
    const auto shape = ttnn::SmallVector<uint32_t>{N, C, H, W};
    int32_t dim = 3U;

    std::random_device rd;
    std::mt19937 gen(42);
    xt::xarray<float> input_tensor = xt::random::rand<float>({N, C, H, W}, -10.0F, 10.0F, gen);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input.print();

    auto result = ttml::metal::softmax(input, dim);
    std::cout << "CrossEntropyBackward_Test:\nResult:\n";
    result.print();

    auto expected_result = xt_softmax(input_tensor, dim);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(SoftmaxTest, SoftmaxTest_Huge_Batch) {
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 32U, W = 128000U;
    const auto shape = ttnn::SmallVector<uint32_t>{N, C, H, W};
    int32_t dim = 3U;

    std::random_device rd;
    std::mt19937 gen(42);
    xt::xarray<float> input_tensor = xt::random::rand<float>({N, C, H, W}, -10.0F, 10.0F, gen);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input.print();

    auto result = ttml::metal::softmax(input, dim);
    std::cout << "CrossEntropyBackward_Test:\nResult:\n";
    result.print();

    auto expected_result = xt_softmax(input_tensor, dim);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(SoftmaxTest, SoftmaxTest_Large_Values) {
    using namespace ttml;

    const uint32_t N = 1U, C = 1U, H = 1U, W = 256U;
    const auto shape = ttnn::SmallVector<uint32_t>{N, C, H, W};
    int32_t dim = 3U;

    xt::xarray<float> input_tensor = {
        {{{5.36871e+08,  -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08}}}};

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input.print();

    auto result = ttml::metal::softmax(input, dim);
    std::cout << "Sofrmax_test:\nResult:\n";
    result.print();

    auto ttnn_softmax = ttnn_fixed::softmax(input, dim);
    auto ttnn_softmax_xtensor = core::to_xtensor(ttnn_softmax);

    auto expected_result = xt_softmax(input_tensor, dim);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

// The following function writes results to a file.
// These results are later used for plotting to visualize improvements in stability
// and to compare this approach with previous implementations.

// // Helper: Convert vector to a JSON-like string
// std::string vector_to_json(const std::string& name, const std::vector<float>& vec) {
//     std::ostringstream oss;
//     oss << "\"" << name << "\": [";
//     for (size_t i = 0; i < vec.size(); ++i) {
//         oss << std::fixed << std::setprecision(6) << vec[i];
//         if (i + 1 < vec.size())
//             oss << ", ";
//     }
//     oss << "]";
//     return oss.str();
// }

// TEST_F(SoftmaxTest, Softmax_Print_Result) {
//     using namespace ttml;
//     const uint32_t N = 1U, C = 1U, H = 1U, W = 2048U;

//     const auto shape = ttnn::SmallVector<uint32_t>{N, C, H, W};
//     std::random_device rd;
//     std::mt19937 gen(42);  // gen(rd());
//     std::uniform_int_distribution<uint32_t> class_dist(0, W - 1);

//     const std::string output_path = "softmax_test_output.jsonl";
//     std::ofstream file(output_path);
//     if (!file.is_open()) {
//         fmt::print("Failed to open file: {}\n", output_path);
//     }
//     fmt::print("Writing to: {}\n ", std::filesystem::current_path());

//     auto write_result = [&](const std::vector<float>& xtensor_res,
//                             const std::vector<float>& op_res,
//                             const std::vector<float>& ttnn_softmax_res,
//                             const std::vector<float>& moreh_softmax_res) {
//         // file << "{" << vector_to_json("input", input) << ", " << vector_to_json("xtensor_res", xtensor_res) << ",
//         "
//         //      << vector_to_json("op_res", op_res) << ", " << vector_to_json("custom_res", custom_res) << "}\n";
//         file << "{" << vector_to_json("xtensor_baseline", xtensor_res) << ", " << vector_to_json("op_res", op_res)
//              << ", " << vector_to_json("ttnn_fixed_softmax", ttnn_softmax_res) << ", "
//              << vector_to_json("moreh_softmax", moreh_softmax_res) << "}\n";
//     };

//     const uint32_t test_size = 1000U;

//     xt::xarray<float> grad_tensor = xt::ones<float>({1U, 1U, 1U, 1U});
//     auto grad = core::from_xtensor(grad_tensor, &autograd::ctx().get_device());

//     for (uint32_t i = 0; i < test_size; ++i) {
//         xt::xarray<float> input_tensor = xt::random::rand<float>({N, C, H, W}, -10.0F, 10.0F, gen);
//         auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

//         xt::xarray<float> xtensor_softmax = xt_softmax(input_tensor, 3U);
//         auto result = ttml::metal::softmax(input, 3U);
//         auto ttnn_softmax = ttnn_fixed::softmax(input, 3U);

//         auto moreh_softmax = ttnn::moreh_softmax(
//             input,
//             /* axis */ 3U,
//             /* output */ std::nullopt,
//             ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOp::SOFTMAX,
//             ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOpParallelizationStrategy::NONE,
//             /* output_mem_config */ std::nullopt,
//             /* compute_kernel_config */ core::ComputeKernelConfig::softmax());

//         // auto input_view = xt::view(input_tensor, 0, 0, 0, xt::all());
//         // std::vector<float> input_vec(input_view.begin(), input_view.end());

//         auto xtensor_res_view = xt::view(xtensor_softmax, 0, 0, 0, xt::all());
//         std::vector<float> xtensor_res_vec(xtensor_res_view.begin(), xtensor_res_view.end());

//         auto op_res_vec = core::to_vector(result);
//         auto ttnn_softmax_vec = core::to_vector(ttnn_softmax);
//         auto moreh_softmax_vec = core::to_vector(moreh_softmax);

//         write_result(xtensor_res_vec, op_res_vec, ttnn_softmax_vec, moreh_softmax_vec);
//     }

//     file.close();
//     fmt::print("Results written to {}\n", output_path);

//     EXPECT_TRUE(false);
// }
