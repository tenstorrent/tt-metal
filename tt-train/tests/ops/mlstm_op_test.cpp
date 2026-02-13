// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/mlstm_op.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <core/ttnn_all_includes.hpp>
#include <cstdint>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"

class MLSTMOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

namespace {

/**
 * @brief Reference mLSTM forward pass implementation in CPU.
 *
 * This is a direct translation of the JAX implementation from mlstm_kernels.
 */
xt::xarray<float> mlstm_parallel_reference(
    const xt::xarray<float>& matQ,
    const xt::xarray<float>& matK,
    const xt::xarray<float>& matV,
    const xt::xarray<float>& vecI,
    const xt::xarray<float>& vecF,
    float eps = 1e-6f) {
    // Get dimensions
    size_t B = matQ.shape()[0];
    size_t NH = matQ.shape()[1];
    size_t S = matQ.shape()[2];
    size_t DHQK = matQ.shape()[3];
    size_t DHV = matV.shape()[3];

    // Step 1: log_sigmoid of forget gate
    xt::xarray<float> vecLogSigF = xt::log(1.0f / (1.0f + xt::exp(-vecF)));

    // Step 2: Cumulative sum
    xt::xarray<float> vecLogSigF_cumsum = xt::cumsum(vecLogSigF, 2);

    // Step 3: Create log forget gate matrix (S x S)
    // matLogSigF[i,j] = vecLogSigF_cumsum[i] - vecLogSigF_cumsum[j]
    xt::xarray<float> matLogSigF = xt::zeros<float>({B, NH, S, S});
    for (size_t b = 0; b < B; ++b) {
        for (size_t nh = 0; nh < NH; ++nh) {
            for (size_t i = 0; i < S; ++i) {
                for (size_t j = 0; j < S; ++j) {
                    matLogSigF(b, nh, i, j) = vecLogSigF_cumsum(b, nh, i) - vecLogSigF_cumsum(b, nh, j);
                }
            }
        }
    }

    // Step 4: Apply lower triangular mask
    for (size_t b = 0; b < B; ++b) {
        for (size_t nh = 0; nh < NH; ++nh) {
            for (size_t i = 0; i < S; ++i) {
                for (size_t j = 0; j < S; ++j) {
                    if (i < j) {
                        matLogSigF(b, nh, i, j) = -1e9f;
                    }
                }
            }
        }
    }

    // Step 5: Add input gate
    xt::xarray<float> matLogD = xt::zeros<float>({B, NH, S, S});
    for (size_t b = 0; b < B; ++b) {
        for (size_t nh = 0; nh < NH; ++nh) {
            for (size_t i = 0; i < S; ++i) {
                for (size_t j = 0; j < S; ++j) {
                    matLogD(b, nh, i, j) = matLogSigF(b, nh, i, j) + vecI(b, nh, j);
                }
            }
        }
    }

    // Step 6: Stabilization - row-wise max
    xt::xarray<float> vecM = xt::amax(matLogD, {3});  // (B, NH, S)

    // Step 7: Stabilized D matrix
    xt::xarray<float> matD = xt::zeros<float>({B, NH, S, S});
    for (size_t b = 0; b < B; ++b) {
        for (size_t nh = 0; nh < NH; ++nh) {
            for (size_t i = 0; i < S; ++i) {
                for (size_t j = 0; j < S; ++j) {
                    matD(b, nh, i, j) = std::exp(matLogD(b, nh, i, j) - vecM(b, nh, i));
                }
            }
        }
    }

    // Step 8: Scaled dot product Q @ K^T / sqrt(d)
    float scale = 1.0f / std::sqrt(static_cast<float>(DHQK));
    xt::xarray<float> matS = xt::zeros<float>({B, NH, S, S});
    for (size_t b = 0; b < B; ++b) {
        for (size_t nh = 0; nh < NH; ++nh) {
            for (size_t i = 0; i < S; ++i) {
                for (size_t j = 0; j < S; ++j) {
                    float dot = 0.0f;
                    for (size_t k = 0; k < DHQK; ++k) {
                        dot += matQ(b, nh, i, k) * matK(b, nh, j, k);
                    }
                    matS(b, nh, i, j) = dot * scale;
                }
            }
        }
    }

    // Step 9: Gated attention
    xt::xarray<float> matCtilde = matS * matD;

    // Step 10: Normalizer
    xt::xarray<float> sumCtilde = xt::sum(matCtilde, {3});  // (B, NH, S)
    xt::xarray<float> vecN = xt::zeros<float>({B, NH, S});
    for (size_t b = 0; b < B; ++b) {
        for (size_t nh = 0; nh < NH; ++nh) {
            for (size_t i = 0; i < S; ++i) {
                float absSum = std::abs(sumCtilde(b, nh, i));
                float expNegM = std::exp(-vecM(b, nh, i));
                vecN(b, nh, i) = std::max(absSum, expNegM);
            }
        }
    }

    // Step 11: Normalize
    xt::xarray<float> matC = xt::zeros<float>({B, NH, S, S});
    for (size_t b = 0; b < B; ++b) {
        for (size_t nh = 0; nh < NH; ++nh) {
            for (size_t i = 0; i < S; ++i) {
                for (size_t j = 0; j < S; ++j) {
                    matC(b, nh, i, j) = matCtilde(b, nh, i, j) / (vecN(b, nh, i) + eps);
                }
            }
        }
    }

    // Step 12: Output C @ V
    xt::xarray<float> matH = xt::zeros<float>({B, NH, S, DHV});
    for (size_t b = 0; b < B; ++b) {
        for (size_t nh = 0; nh < NH; ++nh) {
            for (size_t i = 0; i < S; ++i) {
                for (size_t k = 0; k < DHV; ++k) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < S; ++j) {
                        sum += matC(b, nh, i, j) * matV(b, nh, j, k);
                    }
                    matH(b, nh, i, k) = sum;
                }
            }
        }
    }

    return matH;
}

float compute_mse(const xt::xarray<float>& expected, const xt::xarray<float>& result) {
    xt::xarray<float> diff = expected - result;
    float mse = xt::mean(xt::square(diff))();
    return mse;
}

}  // namespace

TEST_F(MLSTMOpTest, MLSTMForwardTest_Small) {
    using namespace ttml;

    const uint32_t B = 1U;
    const uint32_t NH = 2U;
    const uint32_t S = 32U;
    const uint32_t DHQK = 32U;
    const uint32_t DHV = 32U;

    // Generate random input tensors
    auto& rng = autograd::ctx().get_generator();
    uint32_t seed = rng();

    xt::xarray<float> query_data = xt::empty<float>({B, NH, S, DHQK});
    xt::xarray<float> key_data = xt::empty<float>({B, NH, S, DHQK});
    xt::xarray<float> value_data = xt::empty<float>({B, NH, S, DHV});
    xt::xarray<float> input_gate_data = xt::empty<float>({B, NH, S});
    xt::xarray<float> forget_gate_data = xt::empty<float>({B, NH, S});

    ttml::core::parallel_generate(
        std::span{query_data.data(), query_data.size()},
        []() { return std::uniform_real_distribution<float>(-0.5f, 0.5f); },
        seed);
    ttml::core::parallel_generate(
        std::span{key_data.data(), key_data.size()},
        []() { return std::uniform_real_distribution<float>(-0.5f, 0.5f); },
        seed + 1);
    ttml::core::parallel_generate(
        std::span{value_data.data(), value_data.size()},
        []() { return std::uniform_real_distribution<float>(-0.5f, 0.5f); },
        seed + 2);
    ttml::core::parallel_generate(
        std::span{input_gate_data.data(), input_gate_data.size()},
        []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); },
        seed + 3);
    ttml::core::parallel_generate(
        std::span{forget_gate_data.data(), forget_gate_data.size()},
        []() { return std::uniform_real_distribution<float>(0.0f, 3.0f); },  // Bias towards keeping (high forget gate)
        seed + 4);

    // Create device tensors
    auto query = autograd::create_tensor(core::from_xtensor(query_data, &autograd::ctx().get_device()));
    auto key = autograd::create_tensor(core::from_xtensor(key_data, &autograd::ctx().get_device()));
    auto value = autograd::create_tensor(core::from_xtensor(value_data, &autograd::ctx().get_device()));
    auto input_gate = autograd::create_tensor(core::from_xtensor(input_gate_data, &autograd::ctx().get_device()));
    auto forget_gate = autograd::create_tensor(core::from_xtensor(forget_gate_data, &autograd::ctx().get_device()));

    // Run mLSTM forward pass
    auto result = ops::mlstm_parallel(query, key, value, input_gate, forget_gate);

    // Get result back to CPU
    xt::xarray<float> result_xtensor = core::to_xtensor(result->get_value());

    // Compute reference
    xt::xarray<float> expected =
        mlstm_parallel_reference(query_data, key_data, value_data, input_gate_data, forget_gate_data);

    // Check shapes match
    ASSERT_EQ(result_xtensor.shape(), expected.shape()) << "Output shapes do not match";

    // Check values are close
    float mse = compute_mse(expected, result_xtensor);
    EXPECT_LT(mse, 1e-2f) << "MSE between reference and kernel result is too high: " << mse;

    // Also check with allclose
    EXPECT_TRUE(xt::allclose(result_xtensor, expected, 5e-2f, 5e-2f))
        << "Result does not match reference implementation";
}

TEST_F(MLSTMOpTest, MLSTMBackwardTest_GradientCheck) {
    using namespace ttml;

    const uint32_t B = 1U;
    const uint32_t NH = 1U;
    const uint32_t S = 32U;
    const uint32_t DHQK = 32U;
    const uint32_t DHV = 32U;

    // Generate random input tensors
    auto& rng = autograd::ctx().get_generator();
    uint32_t seed = rng();

    xt::xarray<float> query_data = xt::empty<float>({B, NH, S, DHQK});
    xt::xarray<float> key_data = xt::empty<float>({B, NH, S, DHQK});
    xt::xarray<float> value_data = xt::empty<float>({B, NH, S, DHV});
    xt::xarray<float> input_gate_data = xt::empty<float>({B, NH, S});
    xt::xarray<float> forget_gate_data = xt::empty<float>({B, NH, S});

    ttml::core::parallel_generate(
        std::span{query_data.data(), query_data.size()},
        []() { return std::uniform_real_distribution<float>(-0.5f, 0.5f); },
        seed);
    ttml::core::parallel_generate(
        std::span{key_data.data(), key_data.size()},
        []() { return std::uniform_real_distribution<float>(-0.5f, 0.5f); },
        seed + 1);
    ttml::core::parallel_generate(
        std::span{value_data.data(), value_data.size()},
        []() { return std::uniform_real_distribution<float>(-0.5f, 0.5f); },
        seed + 2);
    ttml::core::parallel_generate(
        std::span{input_gate_data.data(), input_gate_data.size()},
        []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); },
        seed + 3);
    ttml::core::parallel_generate(
        std::span{forget_gate_data.data(), forget_gate_data.size()},
        []() { return std::uniform_real_distribution<float>(0.0f, 3.0f); },
        seed + 4);

    // Create device tensors
    auto query = autograd::create_tensor(core::from_xtensor(query_data, &autograd::ctx().get_device()));
    auto key = autograd::create_tensor(core::from_xtensor(key_data, &autograd::ctx().get_device()));
    auto value = autograd::create_tensor(core::from_xtensor(value_data, &autograd::ctx().get_device()));
    auto input_gate = autograd::create_tensor(core::from_xtensor(input_gate_data, &autograd::ctx().get_device()));
    auto forget_gate = autograd::create_tensor(core::from_xtensor(forget_gate_data, &autograd::ctx().get_device()));

    // Run mLSTM forward pass
    auto result = ops::mlstm_parallel(query, key, value, input_gate, forget_gate);

    // Compute a scalar loss (sum of all outputs)
    auto loss_tensor = ttnn::sum(result->get_value());
    auto loss = autograd::create_tensor(loss_tensor);

    // Run backward pass
    loss->backward();

    // Check that all gradients are computed and have correct shapes
    EXPECT_TRUE(query->is_grad_initialized()) << "Query gradient should be initialized";
    EXPECT_TRUE(key->is_grad_initialized()) << "Key gradient should be initialized";
    EXPECT_TRUE(value->is_grad_initialized()) << "Value gradient should be initialized";
    EXPECT_TRUE(input_gate->is_grad_initialized()) << "Input gate gradient should be initialized";
    EXPECT_TRUE(forget_gate->is_grad_initialized()) << "Forget gate gradient should be initialized";

    // Check gradient shapes
    EXPECT_EQ(query->get_grad().logical_shape(), query->get_value().logical_shape()) << "Query gradient shape mismatch";
    EXPECT_EQ(key->get_grad().logical_shape(), key->get_value().logical_shape()) << "Key gradient shape mismatch";
    EXPECT_EQ(value->get_grad().logical_shape(), value->get_value().logical_shape()) << "Value gradient shape mismatch";
    EXPECT_EQ(input_gate->get_grad().logical_shape(), input_gate->get_value().logical_shape())
        << "Input gate gradient shape mismatch";
    EXPECT_EQ(forget_gate->get_grad().logical_shape(), forget_gate->get_value().logical_shape())
        << "Forget gate gradient shape mismatch";

    // Check that gradients are not all zeros (basic sanity check)
    auto q_grad = core::to_xtensor(query->get_grad());
    auto k_grad = core::to_xtensor(key->get_grad());
    auto v_grad = core::to_xtensor(value->get_grad());

    EXPECT_GT(xt::sum(xt::abs(q_grad))(), 0.0f) << "Query gradient should not be all zeros";
    EXPECT_GT(xt::sum(xt::abs(k_grad))(), 0.0f) << "Key gradient should not be all zeros";
    EXPECT_GT(xt::sum(xt::abs(v_grad))(), 0.0f) << "Value gradient should not be all zeros";
}
