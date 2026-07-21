// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/mla_kv_assemble_op.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"
#include "test_utils/random_data.hpp"

class MLAKVAssembleTest : public ::testing::Test {
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

// MLA KV assemble dimensions for a single test case. All channel dims are multiples of TILE_WIDTH (32)
// and S is a multiple of TILE_HEIGHT (32).
struct AssembleShape {
    std::string name;
    uint32_t batch = 0;
    uint32_t seq_len = 0;
    uint32_t n_heads = 0;
    uint32_t qk_nope_dim = 0;
    uint32_t qk_rope_dim = 0;
    uint32_t v_dim = 0;
};

// Deterministic uniform BF16 device tensor of the given 4D shape.
ttnn::Tensor make_input(uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, uint32_t seed) {
    auto* device = &ttml::autograd::ctx().get_device();
    const size_t count = static_cast<size_t>(d0) * d1 * d2 * d3;
    const auto host = ttml::test_utils::make_uniform_vector<float>(count, -1.0F, 1.0F, seed);
    const ttnn::Shape shape({d0, d1, d2, d3});
    return ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(host, shape, device, ttnn::Layout::TILE);
}

// ── Forward reference (k, v) computed from BF16-rounded packed inputs ───────────────────────────────
struct FwOutputs {
    xt::xarray<float> k;
    xt::xarray<float> v;
};

FwOutputs reference_fw(const ttnn::Tensor& kv_up, const ttnn::Tensor& k_pe, const AssembleShape& shape) {
    const std::size_t B = shape.batch;
    const std::size_t S = shape.seq_len;
    const std::size_t H = shape.n_heads;
    const std::size_t nope = shape.qk_nope_dim;
    const std::size_t kv_w = shape.qk_nope_dim + shape.v_dim;

    xt::xarray<float> kv_up_bf = ttml::core::to_xtensor(kv_up);
    kv_up_bf.reshape({B, S, H, kv_w});

    const xt::xarray<float> kv = xt::transpose(kv_up_bf, {0, 2, 1, 3});
    const xt::xarray<float> k_nope = xt::view(kv, xt::all(), xt::all(), xt::all(), xt::range(std::size_t{0}, nope));
    const xt::xarray<float> k_pe_b =
        xt::broadcast(ttml::core::to_xtensor(k_pe), {B, H, S, static_cast<std::size_t>(shape.qk_rope_dim)});

    FwOutputs ref;
    ref.k = xt::concatenate(xt::xtuple(k_nope, k_pe_b), 3);
    ref.v = xt::view(kv, xt::all(), xt::all(), xt::all(), xt::range(nope, kv_w));
    return ref;
}

// ── Backward reference (dkv_up, dk_pe) from head-major grads ────────────────────────────────────────
// dkv_up = reverse head-split of [dK_nope | dV]; dk_pe = Σ_h dK_rope.
struct BwOutputs {
    xt::xarray<float> dkv_up;
    xt::xarray<float> dk_pe;
};

BwOutputs reference_bw(const ttnn::Tensor& dK, const ttnn::Tensor& dV, const AssembleShape& shape) {
    const std::size_t B = shape.batch;
    const std::size_t S = shape.seq_len;
    const std::size_t H = shape.n_heads;
    const std::size_t nope = shape.qk_nope_dim;
    const std::size_t qk_head = shape.qk_nope_dim + shape.qk_rope_dim;
    const std::size_t kv_w = shape.qk_nope_dim + shape.v_dim;

    const xt::xarray<float> dK_bf = ttml::core::to_xtensor(dK);  // [B, H, S, qk_head]
    const xt::xarray<float> dV_bf = ttml::core::to_xtensor(dV);  // [B, H, S, v_dim]

    BwOutputs ref;

    // dkv_up: concat([dK_nope | dV]) per head -> reverse head-split.
    const xt::xarray<float> dK_nope = xt::view(dK_bf, xt::all(), xt::all(), xt::all(), xt::range(std::size_t{0}, nope));
    xt::xarray<float> kv_head = xt::concatenate(xt::xtuple(dK_nope, dV_bf), 3);  // [B, H, S, kv_w]
    ref.dkv_up = xt::transpose(kv_head, {0, 2, 1, 3});
    ref.dkv_up.reshape({B, 1U, S, H * kv_w});

    // dk_pe: sum dK's rope suffix over the head axis -> [B, 1, S, qk_rope].
    const xt::xarray<float> dK_rope = xt::view(dK_bf, xt::all(), xt::all(), xt::all(), xt::range(nope, qk_head));
    ref.dk_pe = xt::sum(dK_rope, {1});
    ref.dk_pe.reshape({B, 1U, S, static_cast<std::size_t>(shape.qk_rope_dim)});
    return ref;
}

void run_fw(const AssembleShape& shape) {
    const auto kv_up =
        make_input(shape.batch, 1U, shape.seq_len, shape.n_heads * (shape.qk_nope_dim + shape.v_dim), 2002U);
    const auto k_pe = make_input(shape.batch, 1U, shape.seq_len, shape.qk_rope_dim, 3003U);

    auto [k, v] =
        ttml::metal::mla_kv_assemble_fw(kv_up, k_pe, shape.n_heads, shape.qk_nope_dim, shape.qk_rope_dim, shape.v_dim);

    const auto ref = reference_fw(kv_up, k_pe, shape);
    // Pure copy/broadcast → bit-exact.
    EXPECT_TRUE(xt::allclose(ttml::core::to_xtensor(k), ref.k, 0.0, 0.0)) << shape.name << " fw/k";
    EXPECT_TRUE(xt::allclose(ttml::core::to_xtensor(v), ref.v, 0.0, 0.0)) << shape.name << " fw/v";
}

void run_bw(const AssembleShape& shape) {
    const uint32_t qk_head = shape.qk_nope_dim + shape.qk_rope_dim;
    const auto dK = make_input(shape.batch, shape.n_heads, shape.seq_len, qk_head, 5005U);
    const auto dV = make_input(shape.batch, shape.n_heads, shape.seq_len, shape.v_dim, 6006U);

    auto [dkv_up, dk_pe] =
        ttml::metal::mla_kv_assemble_bw(dK, dV, shape.n_heads, shape.qk_nope_dim, shape.qk_rope_dim, shape.v_dim);

    const auto ref = reference_bw(dK, dV, shape);
    // dkv_up is a pure copy → bit-exact; dk_pe is a BF16 head-axis reduction → tolerant.
    EXPECT_TRUE(xt::allclose(ttml::core::to_xtensor(dkv_up), ref.dkv_up, 0.0, 0.0)) << shape.name << " bw/dkv_up";
    EXPECT_TRUE(xt::allclose(ttml::core::to_xtensor(dk_pe), ref.dk_pe, /*rtol=*/5e-3, /*atol=*/5e-3))
        << shape.name << " bw/dk_pe";
}

void run_autograd_wrapper_bw(const AssembleShape& shape) {
    auto kv_up = ttml::autograd::create_tensor(
        make_input(shape.batch, 1U, shape.seq_len, shape.n_heads * (shape.qk_nope_dim + shape.v_dim), 8008U),
        /*requires_grad=*/true);
    auto k_pe = ttml::autograd::create_tensor(
        make_input(shape.batch, 1U, shape.seq_len, shape.qk_rope_dim, 9009U), /*requires_grad=*/true);

    auto [k, v] =
        ttml::ops::mla_kv_assemble(kv_up, k_pe, shape.n_heads, shape.qk_nope_dim, shape.qk_rope_dim, shape.v_dim);
    auto loss = ttml::ops::add(ttml::ops::mean(k), ttml::ops::mean(v));
    loss->backward();

    const auto ref = reference_bw(k->get_grad(), v->get_grad(), shape);
    EXPECT_TRUE(xt::allclose(ttml::core::to_xtensor(kv_up->get_grad()), ref.dkv_up, 0.0, 0.0))
        << shape.name << " autograd/dkv_up";
    EXPECT_TRUE(xt::allclose(ttml::core::to_xtensor(k_pe->get_grad()), ref.dk_pe, /*rtol=*/5e-3, /*atol=*/5e-3))
        << shape.name << " autograd/dk_pe";
}

const std::vector<AssembleShape>& shapes() {
    static const std::vector<AssembleShape> cases = {
        {"square_st1", 2, 32, 2, 32, 32, 32},
        {"square_st2", 2, 64, 2, 32, 32, 32},
        {"asym_rope2", 2, 64, 4, 128, 64, 128},
        {"heads8_s96", 1, 96, 8, 64, 32, 64},
    };
    return cases;
}

}  // namespace

TEST_F(MLAKVAssembleTest, ForwardMatchesReference) {
    for (const auto& shape : shapes()) {
        run_fw(shape);
    }
}

TEST_F(MLAKVAssembleTest, BackwardMatchesReference) {
    for (const auto& shape : shapes()) {
        run_bw(shape);
    }
}

TEST_F(MLAKVAssembleTest, AutogradWrapperBackwardMatchesReference) {
    run_autograd_wrapper_bw({"wrapper_heads4", 2, 64, 4, 64, 32, 64});
}
