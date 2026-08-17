#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>
#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/optimizers/adamw/adamw.hpp"

namespace {
constexpr float kLr = 1e-3F, kB1 = 0.9F, kB2 = 0.999F, kEps = 1e-8F, kWd = 0.01F;

class AdamWNoHostTest : public ::testing::Test {
public:
    static void SetUpTestSuite() { ttml::autograd::ctx().open_device(); }
    static void TearDownTestSuite() { ttml::autograd::ctx().close_device(); }
protected:
    static ttnn::Tensor rand_t(const ttnn::Shape& s, float lo, float hi, uint32_t seed) {
        std::mt19937 g(seed);
        std::uniform_real_distribution<float> d(lo, hi);
        std::vector<float> v(s.volume());
        for (float& x : v) x = d(g);
        return ttml::core::from_vector<float, ttnn::DataType::FLOAT32>(v, s, &ttml::autograd::ctx().get_device());
    }
    static ttnn::Tensor rand_bf16(const ttnn::Shape& s, float lo, float hi, uint32_t seed) {
        std::mt19937 g(seed);
        std::uniform_real_distribution<float> d(lo, hi);
        std::vector<float> v(s.volume());
        for (float& x : v) x = d(g);
        return ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(v, s, &ttml::autograd::ctx().get_device());
    }
    static ttnn::Tensor scalar(float v) {
        return ttml::core::from_vector<float, ttnn::DataType::FLOAT32>(
            std::vector<float>{v}, ttnn::Shape({1U, 1U, 1U, 1U}), &ttml::autograd::ctx().get_device());
    }
};

TEST_F(AdamWNoHostTest, MatchesFloatOverload) {
    const ttnn::Shape shape({1U, 1U, 64U, 64U});
    const uint32_t step = 7U;
    const float b1p = std::pow(kB1, static_cast<float>(step));
    const float b2p = std::pow(kB2, static_cast<float>(step));

    auto p0 = rand_t(shape, -1.F, 1.F, 2);
    auto g0 = rand_bf16(shape, -1.F, 1.F, 1);
    auto m0 = rand_t(shape, -1.F, 1.F, 3), v0 = rand_t(shape, 0.F, 1.F, 4);
    auto p1 = rand_t(shape, -1.F, 1.F, 2);
    auto g1 = rand_bf16(shape, -1.F, 1.F, 1);
    auto m1 = rand_t(shape, -1.F, 1.F, 3), v1 = rand_t(shape, 0.F, 1.F, 4);

    auto expected = ttml::metal::adamw(p0, g0, m0, v0, std::nullopt, kLr, kB1, kB2, b1p, b2p, kEps, kWd);
    auto actual = ttml::metal::adamw(
        p1, g1, m1, v1, std::nullopt, kB1, kB2,
        scalar(kLr / (1.0F - b1p)), scalar(1.0F / std::sqrt(1.0F - b2p)), scalar(1.0F - kLr * kWd), kEps);

    auto e = ttml::core::to_vector(expected);
    auto a = ttml::core::to_vector(actual);
    ASSERT_EQ(a.size(), e.size());
    for (size_t i = 0; i < e.size(); ++i) ASSERT_FLOAT_EQ(a[i], e[i]) << "at " << i;
}
}  // namespace
