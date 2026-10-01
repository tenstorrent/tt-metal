// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Same-seed reproducibility of AdamW stochastic rounding (SFPSTOCHRND).

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"

namespace {

constexpr uint32_t kSeed = 0x12345678U;
constexpr uint32_t kRepeats = 5U;
constexpr std::array<size_t, 4> kShape = {1, 1, 256, 1024};

std::vector<uint32_t> to_bits(const ttnn::Tensor& tensor) {
    auto values = ttml::core::to_xtensor(tensor);
    std::vector<uint32_t> bits;
    bits.reserve(values.size());
    for (float v : values) {
        bits.push_back(std::bit_cast<uint32_t>(v));
    }
    return bits;
}

size_t count_mismatches(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        n += static_cast<size_t>(a[i] != b[i]);
    }
    return n;
}

xt::xarray<float> make_pattern(float base, float amplitude, float freq) {
    xt::xarray<float> x = xt::zeros<float>(kShape);
    for (size_t i = 0; i < x.size(); ++i) {
        x.flat(i) = base + amplitude * std::sin(freq * static_cast<float>(i));
    }
    return x;
}

std::vector<uint32_t> run_adamw_once(ttml::metal::StochasticRounding sr, uint32_t seed = kSeed) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto param = ttml::core::from_xtensor(make_pattern(1.0F, 0.25F, 0.37F), device);
    auto grad = ttml::core::from_xtensor(make_pattern(0.0F, 1.0F, 0.11F), device);
    auto exp_avg = ttml::core::zeros_like(param);
    auto exp_avg_sq = ttml::core::zeros_like(param);
    auto out = ttml::metal::adamw(
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        std::nullopt,
        /*lr=*/1e-3F,
        /*beta1=*/0.9F,
        /*beta2=*/0.999F,
        /*beta1_pow=*/0.9F,
        /*beta2_pow=*/0.999F,
        /*epsilon=*/1e-8F,
        /*weight_decay=*/0.0F,
        sr,
        sr == ttml::metal::StochasticRounding::Enabled ? std::optional<uint32_t>{seed} : std::nullopt);
    return to_bits(out);
}

void expect_repeats_identical(std::string_view name, const std::function<std::vector<uint32_t>()>& run) {
    const auto first = run();
    for (uint32_t r = 1; r < kRepeats; ++r) {
        const auto again = run();
        const size_t mismatches = count_mismatches(first, again);
        EXPECT_EQ(mismatches, 0U) << name << ": repeat " << r << " differs from repeat 0";
    }
}

}  // namespace

class AdamWStochasticRoundingReproTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }

protected:
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
    }
};

TEST_F(AdamWStochasticRoundingReproTest, AdamWWithoutStochasticRoundingIsBitIdentical) {
    expect_repeats_identical("adamw_sr_off", [] { return run_adamw_once(ttml::metal::StochasticRounding::Disabled); });
}

TEST_F(AdamWStochasticRoundingReproTest, AdamWStochasticRoundingSameSeedIsBitIdentical) {
    expect_repeats_identical("adamw_sr_on", [] { return run_adamw_once(ttml::metal::StochasticRounding::Enabled); });
}

TEST_F(AdamWStochasticRoundingReproTest, AdamWStochasticRoundingDifferentSeedDiffers) {
    const auto a = run_adamw_once(ttml::metal::StochasticRounding::Enabled, kSeed);
    const auto b = run_adamw_once(ttml::metal::StochasticRounding::Enabled, kSeed + 1U);
    const auto off = run_adamw_once(ttml::metal::StochasticRounding::Disabled);
    fmt::print(
        "[adamw_sr_seed] mismatches seed vs seed+1={} seed vs off={}\n",
        count_mismatches(a, b),
        count_mismatches(a, off));
    EXPECT_GT(count_mismatches(a, b), 0U) << "a different seed must change the rounding";
    EXPECT_GT(count_mismatches(a, off), 0U) << "stochastic rounding must differ from round-to-nearest";
}
