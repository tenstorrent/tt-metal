// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/mla_qkv_assemble_op.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "test_utils/random_data.hpp"

class MLAQKVAssembleForwardTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
    }
};

namespace {

// MLA QKV assemble dimensions for a single test case. All channel dims are
// multiples of TILE_WIDTH (32) and S is a multiple of TILE_HEIGHT (32) — the
// op validates this.
struct AssembleShape {
    std::string name;
    uint32_t batch = 0;
    uint32_t seq_len = 0;
    uint32_t n_heads = 0;
    uint32_t qk_nope_dim = 0;
    uint32_t qk_rope_dim = 0;
    uint32_t v_dim = 0;
};

// Build a [B, 1, S, W] BF16 device tensor filled with deterministic uniform data.
ttml::autograd::TensorPtr make_input(uint32_t batch, uint32_t seq_len, uint32_t width, uint32_t seed) {
    auto* device = &ttml::autograd::ctx().get_device();
    const size_t count = static_cast<size_t>(batch) * seq_len * width;
    const auto host = ttml::test_utils::make_uniform_vector<float>(count, -1.0F, 1.0F, seed);
    const ttnn::Shape shape({batch, 1U, seq_len, width});
    return ttml::autograd::create_tensor(
        ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(host, shape, device, ttnn::Layout::TILE));
}

// Reference assemble computed from the BF16-rounded inputs (read back via to_xtensor), so the only
// numeric path is a layout/copy/broadcast — the comparison against the kernel output must be exact.
//
// q[b,h,s,d]        = q_pre[b,0,s, h*qk_head + d]                       (head split)
// k[b,h,s,d<Tn*32]  = kv_up[b,0,s, h*(qk_nope+v_dim) + d]              (k_nope slice)
// k[b,h,s,d>=Tn*32] = k_pe[b,0,s, d - qk_nope]                          (k_pe broadcast across heads)
// v[b,h,s,d]        = kv_up[b,0,s, h*(qk_nope+v_dim) + qk_nope + d]    (v slice)
struct ReferenceOutputs {
    xt::xarray<float> q;
    xt::xarray<float> k;
    xt::xarray<float> v;
};

ReferenceOutputs reference_assemble(
    const ttml::autograd::TensorPtr& q_pre,
    const ttml::autograd::TensorPtr& kv_up,
    const ttml::autograd::TensorPtr& k_pe,
    const AssembleShape& shape) {
    const auto q_pre_bf = ttml::core::to_xtensor(q_pre->get_value());
    const auto kv_up_bf = ttml::core::to_xtensor(kv_up->get_value());
    const auto k_pe_bf = ttml::core::to_xtensor(k_pe->get_value());

    const uint32_t B = shape.batch;
    const uint32_t S = shape.seq_len;
    const uint32_t H = shape.n_heads;
    const uint32_t qk_head = shape.qk_nope_dim + shape.qk_rope_dim;
    const uint32_t kv_w = shape.qk_nope_dim + shape.v_dim;

    ReferenceOutputs ref;
    ref.q = xt::xarray<float>::from_shape({B, H, S, qk_head});
    ref.k = xt::xarray<float>::from_shape({B, H, S, qk_head});
    ref.v = xt::xarray<float>::from_shape({B, H, S, shape.v_dim});

    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t h = 0; h < H; ++h) {
            for (uint32_t s = 0; s < S; ++s) {
                for (uint32_t d = 0; d < qk_head; ++d) {
                    ref.q(b, h, s, d) = q_pre_bf(b, 0, s, h * qk_head + d);
                    if (d < shape.qk_nope_dim) {
                        ref.k(b, h, s, d) = kv_up_bf(b, 0, s, h * kv_w + d);
                    } else {
                        ref.k(b, h, s, d) = k_pe_bf(b, 0, s, d - shape.qk_nope_dim);
                    }
                }
                for (uint32_t d = 0; d < shape.v_dim; ++d) {
                    ref.v(b, h, s, d) = kv_up_bf(b, 0, s, h * kv_w + shape.qk_nope_dim + d);
                }
            }
        }
    }
    return ref;
}

void expect_exact(const xt::xarray<float>& actual, const xt::xarray<float>& expected, const std::string& tag) {
    ASSERT_EQ(actual.shape(), expected.shape()) << tag << ": shape mismatch";
    // Pure copy/broadcast through identical BF16 values → bit-exact on FP32 readback.
    EXPECT_TRUE(xt::allclose(actual, expected, /*rtol=*/0.0, /*atol=*/0.0)) << tag << ": value mismatch";
}

void run_case(const AssembleShape& shape) {
    auto q_pre = make_input(shape.batch, shape.seq_len, shape.n_heads * (shape.qk_nope_dim + shape.qk_rope_dim), 1001U);
    auto kv_up = make_input(shape.batch, shape.seq_len, shape.n_heads * (shape.qk_nope_dim + shape.v_dim), 2002U);
    auto k_pe = make_input(shape.batch, shape.seq_len, shape.qk_rope_dim, 3003U);

    auto [q, k, v] = ttml::ops::mla_qkv_assemble_fw(
        q_pre, kv_up, k_pe, shape.n_heads, shape.qk_nope_dim, shape.qk_rope_dim, shape.v_dim);

    const auto ref = reference_assemble(q_pre, kv_up, k_pe, shape);
    expect_exact(ttml::core::to_xtensor(q->get_value()), ref.q, shape.name + "/q");
    expect_exact(ttml::core::to_xtensor(k->get_value()), ref.k, shape.name + "/k");
    expect_exact(ttml::core::to_xtensor(v->get_value()), ref.v, shape.name + "/v");
}

}  // namespace

// Square head dims, S_t == 1 (single sequence tile row).
TEST_F(MLAQKVAssembleForwardTest, SquareDimsSingleSeqTile) {
    run_case({"square_st1", /*B=*/2, /*S=*/32, /*H=*/2, /*nope=*/32, /*rope=*/32, /*v=*/32});
}

// Square head dims, S_t > 1 (exercises the end-of-batch jump path in the writer).
TEST_F(MLAQKVAssembleForwardTest, SquareDimsMultiSeqTile) {
    run_case({"square_st2", /*B=*/2, /*S=*/64, /*H=*/2, /*nope=*/32, /*rope=*/32, /*v=*/32});
}

// Asymmetric Tn != Tr != Tv with Tr > 1 (broadcast spans multiple rope tiles).
TEST_F(MLAQKVAssembleForwardTest, AsymmetricDimsRopeMultiTile) {
    run_case({"asym_rope2", /*B=*/2, /*S=*/64, /*H=*/4, /*nope=*/128, /*rope=*/64, /*v=*/128});
}

// More heads, single batch, larger S_t.
TEST_F(MLAQKVAssembleForwardTest, ManyHeadsLargerSeq) {
    run_case({"heads8_s96", /*B=*/1, /*S=*/96, /*H=*/8, /*nope=*/64, /*rope=*/32, /*v=*/64});
}
