// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <stdexcept>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/binary_ops.hpp"

namespace ttml::ops::tests {

class BinaryOpsBackwardTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().reset_graph();
        autograd::ctx().close_device();
    }
};

// ============================================================================
// Same-shape backward tests
// ============================================================================

TEST_F(BinaryOpsBackwardTest, AddSameShape) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data_a = {{{{1.F, 2.F, 3.F, 4.F}}}};
    xt::xarray<float> data_b = {{{{5.F, 6.F, 7.F, 8.F}}}};

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device), /* requires_grad */ true);
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device), /* requires_grad */ true);

    auto out = a + b;
    out->backward();

    auto a_grad = core::to_xtensor(a->get_grad());
    auto b_grad = core::to_xtensor(b->get_grad());

    // d(a+b)/da = 1, d(a+b)/db = 1
    EXPECT_TRUE(xt::allclose(a_grad, xt::ones_like(data_a)));
    EXPECT_TRUE(xt::allclose(b_grad, xt::ones_like(data_b)));
}

TEST_F(BinaryOpsBackwardTest, SubSameShape) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data_a = {{{{1.F, 2.F, 3.F, 4.F}}}};
    xt::xarray<float> data_b = {{{{5.F, 6.F, 7.F, 8.F}}}};

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device), /* requires_grad */ true);
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device), /* requires_grad */ true);

    auto out = a - b;
    out->backward();

    auto a_grad = core::to_xtensor(a->get_grad());
    auto b_grad = core::to_xtensor(b->get_grad());

    // d(a-b)/da = 1, d(a-b)/db = -1
    EXPECT_TRUE(xt::allclose(a_grad, xt::ones_like(data_a)));
    EXPECT_TRUE(xt::allclose(b_grad, -xt::ones_like(data_b)));
}

TEST_F(BinaryOpsBackwardTest, MulSameShape) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data_a = {{{{1.F, 2.F, 3.F, 4.F}}}};
    xt::xarray<float> data_b = {{{{4.F, 3.F, 2.F, 1.F}}}};

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device), /* requires_grad */ true);
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device), /* requires_grad */ true);

    auto out = a * b;
    out->backward();

    auto a_grad = core::to_xtensor(a->get_grad());
    auto b_grad = core::to_xtensor(b->get_grad());

    // d(a*b)/da = b, d(a*b)/db = a
    EXPECT_TRUE(xt::allclose(a_grad, data_b));
    EXPECT_TRUE(xt::allclose(b_grad, data_a));
}

TEST_F(BinaryOpsBackwardTest, MulScalar) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data_a = {{{{1.F, 2.F, 3.F, 4.F}}}};

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device), /* requires_grad */ true);

    auto out = a * 3.0F;
    out->backward();

    auto a_grad = core::to_xtensor(a->get_grad());

    // d(a*c)/da = c
    EXPECT_TRUE(xt::allclose(a_grad, xt::ones_like(data_a) * 3.0F));
}

TEST_F(BinaryOpsBackwardTest, DivSameShape) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data_a = {{{{4.F, 6.F, 8.F, 10.F}}}};
    xt::xarray<float> data_b = {{{{2.F, 3.F, 4.F, 5.F}}}};

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device), /* requires_grad */ true);
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device), /* requires_grad */ true);

    auto out = a / b;
    out->backward();

    auto a_grad = core::to_xtensor(a->get_grad());
    auto b_grad = core::to_xtensor(b->get_grad());

    // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
    xt::xarray<float> expected_a_grad = 1.0F / data_b;
    xt::xarray<float> expected_b_grad = -data_a / (data_b * data_b);
    EXPECT_TRUE(xt::allclose(a_grad, expected_a_grad, /* rtol */ 1e-2F, /* atol */ 1e-2F));
    EXPECT_TRUE(xt::allclose(b_grad, expected_b_grad, /* rtol */ 1e-2F, /* atol */ 1e-2F));
}

// ============================================================================
// Parametrized broadcast backward tests
// ============================================================================

struct BroadcastCase {
    std::vector<size_t> a_shape;
    std::vector<size_t> b_shape;
    std::string name;
};

void PrintTo(const BroadcastCase& c, std::ostream* os) {
    *os << c.name;
}

static std::string BroadcastCaseName(const ::testing::TestParamInfo<BroadcastCase>& info) {
    return info.param.name;
}

class AddBroadcastBackwardTest : public BinaryOpsBackwardTest, public ::testing::WithParamInterface<BroadcastCase> {};
class SubBroadcastBackwardTest : public BinaryOpsBackwardTest, public ::testing::WithParamInterface<BroadcastCase> {};
class MulBroadcastBackwardTest : public BinaryOpsBackwardTest, public ::testing::WithParamInterface<BroadcastCase> {};

TEST_P(AddBroadcastBackwardTest, GradShapeMatchesInput) {
    auto* device = &autograd::ctx().get_device();
    const auto& p = GetParam();

    xt::xarray<float> data_a = xt::ones<float>(p.a_shape) * 5.0F;
    xt::xarray<float> data_b = xt::ones<float>(p.b_shape) * 3.0F;

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device), true);
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device), true);

    auto out = a + b;
    out->backward();

    auto a_grad = core::to_xtensor(a->get_grad());
    auto b_grad = core::to_xtensor(b->get_grad());

    // d(a+b)/da = 1
    EXPECT_TRUE(xt::allclose(a_grad, xt::ones_like(data_a)));

    // d(a+b)/db = 1, reduced over broadcast dims
    // Each b element accumulates (a_volume / b_volume) copies of 1
    auto a_volume = static_cast<float>(data_a.size());
    auto b_volume = static_cast<float>(data_b.size());
    xt::xarray<float> expected_b_grad = xt::ones_like(data_b) * (a_volume / b_volume);
    EXPECT_TRUE(xt::allclose(b_grad, expected_b_grad));
}

TEST_P(SubBroadcastBackwardTest, GradShapeMatchesInput) {
    auto* device = &autograd::ctx().get_device();
    const auto& p = GetParam();

    xt::xarray<float> data_a = xt::ones<float>(p.a_shape) * 5.0F;
    xt::xarray<float> data_b = xt::ones<float>(p.b_shape) * 3.0F;

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device), true);
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device), true);

    auto out = a - b;
    out->backward();

    auto a_grad = core::to_xtensor(a->get_grad());
    auto b_grad = core::to_xtensor(b->get_grad());

    // d(a-b)/da = 1
    EXPECT_TRUE(xt::allclose(a_grad, xt::ones_like(data_a)));

    // d(a-b)/db = -1, reduced over broadcast dims
    auto a_volume = static_cast<float>(data_a.size());
    auto b_volume = static_cast<float>(data_b.size());
    xt::xarray<float> expected_b_grad = -xt::ones_like(data_b) * (a_volume / b_volume);
    EXPECT_TRUE(xt::allclose(b_grad, expected_b_grad));
}

TEST_P(MulBroadcastBackwardTest, GradShapeMatchesInput) {
    auto* device = &autograd::ctx().get_device();
    const auto& p = GetParam();

    // Use constant tensors so expected gradients are easy to compute analytically.
    // a = 3, b = 2 everywhere.
    xt::xarray<float> data_a = xt::ones<float>(p.a_shape) * 3.0F;
    xt::xarray<float> data_b = xt::ones<float>(p.b_shape) * 2.0F;

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device), true);
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device), true);

    auto out = a * b;
    out->backward();

    auto a_grad = core::to_xtensor(a->get_grad());
    auto b_grad = core::to_xtensor(b->get_grad());

    // d(a*b)/da = b (broadcast) -> each element = 2
    EXPECT_TRUE(xt::allclose(a_grad, xt::ones_like(data_a) * 2.0F));

    // d(a*b)/db = a, reduced over broadcast dims.
    // Each b element sees (a_volume / b_volume) copies of a=3, so b_grad = 3 * (a_volume / b_volume).
    auto a_volume = static_cast<float>(data_a.size());
    auto b_volume = static_cast<float>(data_b.size());
    xt::xarray<float> expected_b_grad = xt::ones_like(data_b) * 3.0F * (a_volume / b_volume);
    EXPECT_TRUE(xt::allclose(b_grad, expected_b_grad));
}

// Same-rank broadcast: b has a 1 in exactly one dimension
static const BroadcastCase kSameRankCases[] = {
    {{2, 1, 1, 4}, {1, 1, 1, 4}, "Dim0"},
    {{1, 2, 1, 4}, {1, 1, 1, 4}, "Dim1"},
    {{1, 1, 2, 4}, {1, 1, 1, 4}, "Dim2"},
    {{1, 1, 1, 4}, {1, 1, 1, 1}, "Dim3"},
    {{2, 1, 2, 4}, {1, 1, 1, 4}, "Dim0_Dim2"},
    {{2, 2, 1, 4}, {1, 1, 1, 4}, "Dim0_Dim1"},
};

// Cross-rank broadcast: b has fewer dimensions than a
static const BroadcastCase kCrossRankCases[] = {
    {{2, 1, 1, 4}, {4}, "Rank4_vs_Rank1"},
    {{2, 1, 1, 4}, {1, 4}, "Rank4_vs_Rank2"},
    {{2, 1, 1, 4}, {1, 1, 4}, "Rank4_vs_Rank3"},
};

INSTANTIATE_TEST_SUITE_P(SameRank, AddBroadcastBackwardTest, ::testing::ValuesIn(kSameRankCases), BroadcastCaseName);
INSTANTIATE_TEST_SUITE_P(CrossRank, AddBroadcastBackwardTest, ::testing::ValuesIn(kCrossRankCases), BroadcastCaseName);

INSTANTIATE_TEST_SUITE_P(SameRank, SubBroadcastBackwardTest, ::testing::ValuesIn(kSameRankCases), BroadcastCaseName);
INSTANTIATE_TEST_SUITE_P(CrossRank, SubBroadcastBackwardTest, ::testing::ValuesIn(kCrossRankCases), BroadcastCaseName);

INSTANTIATE_TEST_SUITE_P(SameRank, MulBroadcastBackwardTest, ::testing::ValuesIn(kSameRankCases), BroadcastCaseName);
INSTANTIATE_TEST_SUITE_P(CrossRank, MulBroadcastBackwardTest, ::testing::ValuesIn(kCrossRankCases), BroadcastCaseName);

TEST_F(BinaryOpsBackwardTest, DivBroadcastThrows) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data_a = {4.F, 6.F, 8.F, 10.F, 12.F, 14.F, 16.F, 18.F};
    data_a.reshape({2, 1, 1, 4});
    xt::xarray<float> data_b = {{{{2.F, 2.F, 2.F, 2.F}}}};

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device), true);
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device), true);

    auto out = a / b;
    EXPECT_THROW(out->backward(), std::runtime_error);
}

}  // namespace ttml::ops::tests
