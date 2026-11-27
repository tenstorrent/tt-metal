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
#include "test_utils/program_cache.hpp"
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

void expect_fw_matches(
    const ttnn::Tensor& k, const ttnn::Tensor& v, const FwOutputs& expected, const std::string& label) {
    // Pure copy/broadcast → bit-exact.
    EXPECT_TRUE(xt::allclose(ttml::core::to_xtensor(k), expected.k, 0.0, 0.0)) << label << "/k";
    EXPECT_TRUE(xt::allclose(ttml::core::to_xtensor(v), expected.v, 0.0, 0.0)) << label << "/v";
}

void expect_bw_matches(
    const ttnn::Tensor& dkv_up, const ttnn::Tensor& dk_pe, const BwOutputs& expected, const std::string& label) {
    // dkv_up is a pure copy → bit-exact; dk_pe is a BF16 head-axis reduction → tolerant.
    EXPECT_TRUE(xt::allclose(ttml::core::to_xtensor(dkv_up), expected.dkv_up, 0.0, 0.0)) << label << "/dkv_up";
    EXPECT_TRUE(xt::allclose(ttml::core::to_xtensor(dk_pe), expected.dk_pe, /*rtol=*/5e-3, /*atol=*/5e-3))
        << label << "/dk_pe";
}

void run_fw(const AssembleShape& shape) {
    const auto kv_up =
        make_input(shape.batch, 1U, shape.seq_len, shape.n_heads * (shape.qk_nope_dim + shape.v_dim), 2002U);
    const auto k_pe = make_input(shape.batch, 1U, shape.seq_len, shape.qk_rope_dim, 3003U);

    auto [k, v] =
        ttml::metal::mla_kv_assemble_fw(kv_up, k_pe, shape.n_heads, shape.qk_nope_dim, shape.qk_rope_dim, shape.v_dim);

    expect_fw_matches(k, v, reference_fw(kv_up, k_pe, shape), shape.name + " fw");
}

void run_bw(const AssembleShape& shape) {
    const uint32_t qk_head = shape.qk_nope_dim + shape.qk_rope_dim;
    const auto dK = make_input(shape.batch, shape.n_heads, shape.seq_len, qk_head, 5005U);
    const auto dV = make_input(shape.batch, shape.n_heads, shape.seq_len, shape.v_dim, 6006U);

    auto [dkv_up, dk_pe] =
        ttml::metal::mla_kv_assemble_bw(dK, dV, shape.n_heads, shape.qk_nope_dim, shape.qk_rope_dim, shape.v_dim);

    expect_bw_matches(dkv_up, dk_pe, reference_bw(dK, dV, shape), shape.name + " bw");
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

    expect_bw_matches(
        kv_up->get_grad(),
        k_pe->get_grad(),
        reference_bw(k->get_grad(), v->get_grad(), shape),
        shape.name + " autograd");
}

const std::vector<AssembleShape>& shapes() {
    static const std::vector<AssembleShape> cases = {
        {"square_st1", 2, 32, 2, 32, 32, 32},
        {"square_st2", 2, 64, 2, 32, 32, 32},
        {"asym_rope2", 2, 64, 4, 128, 64, 128},
        // Tr = 3 rope tiles: the largest count the bw DST budget admits (Tr + 1 = 4
        // fp32 accumulator slots) — pins the accepted side of the rejection boundary.
        {"rope3_dst_max", 1, 32, 2, 32, 96, 32},
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

TEST_F(MLAKVAssembleTest, BackwardRejectsRopeDimBeyondDstBudget) {
    // qk_rope_dim = 128 (Tr = 4) needs Tr + 1 = 5 dst slots; with fp32 dest accumulation the
    // backward compute kernel's DST accumulator budget is 4, so the op must refuse to launch.
    constexpr uint32_t n_heads = 2U;
    constexpr uint32_t qk_nope_dim = 32U;
    constexpr uint32_t qk_rope_dim = 128U;
    constexpr uint32_t v_dim = 32U;
    constexpr uint32_t qk_head = qk_nope_dim + qk_rope_dim;

    const auto dK = make_input(1U, n_heads, 32U, qk_head, 222U);
    const auto dV = make_input(1U, n_heads, 32U, v_dim, 333U);

    EXPECT_ANY_THROW(ttml::metal::mla_kv_assemble_bw(dK, dV, n_heads, qk_nope_dim, qk_rope_dim, v_dim))
        << "mla_kv_assemble_bw should reject a rope tile count that overflows the DST accumulator";
}

// ── Program-cache hits ──────────────────────────────────────────────────────────────────────────────
// A launch whose shapes and attributes match a cached program reuses it, and only the buffer addresses of
// its tensors are patched in. Each test clears the cache so that its first launch (A) is the miss, then
// launches B on new tensors of the same specs, with other values, while every tensor of A is still alive, so
// no buffer of B can sit at an address of A: the hit must read B's inputs, write B's outputs and leave A's
// outputs untouched. Launch C then reads A's inputs again, so their addresses go back to the miss's while the
// program still holds B's: a hit that patched only the addresses that differ from the miss's, or patched them
// by their offset from the miss's, would read B's inputs.

namespace {

// Both ops split B * Ts blocks over the cores. B = 2 and Ts = cores + 1 give 2 * cores + 2 blocks: more
// than the core count and not a multiple of it, so split_work_to_cores hands out two core groups on any grid.
AssembleShape cache_hit_shape() {
    return {
        "two_core_groups",
        /*batch=*/2U,
        /*seq_len=*/32U * (ttml::test_utils::num_worker_cores() + 1U),
        /*n_heads=*/2U,
        /*qk_nope_dim=*/64U,
        /*qk_rope_dim=*/64U,
        /*v_dim=*/32U};
}

}  // namespace

TEST_F(MLAKVAssembleTest, ForwardCacheHitWithFreshTensors_TwoCoreGroups) {
    auto* device = &ttml::test_utils::device_with_empty_program_cache();
    const auto shape = cache_hit_shape();
    const uint32_t kv_up_width = shape.n_heads * (shape.qk_nope_dim + shape.v_dim);
    const auto launch = [&](const ttnn::Tensor& kv_up, const ttnn::Tensor& k_pe) {
        return ttml::metal::mla_kv_assemble_fw(
            kv_up, k_pe, shape.n_heads, shape.qk_nope_dim, shape.qk_rope_dim, shape.v_dim);
    };

    const auto kv_up_a = make_input(shape.batch, 1U, shape.seq_len, kv_up_width, 4101U);
    const auto k_pe_a = make_input(shape.batch, 1U, shape.seq_len, shape.qk_rope_dim, 4102U);
    const auto entries_before_a = device->num_program_cache_entries();
    const auto [k_a, v_a] = launch(kv_up_a, k_pe_a);
    ASSERT_GT(device->num_program_cache_entries(), entries_before_a) << "launch A must compile a program";
    const auto expected_a = reference_fw(kv_up_a, k_pe_a, shape);
    expect_fw_matches(k_a, v_a, expected_a, "launch A");

    const auto kv_up_b = make_input(shape.batch, 1U, shape.seq_len, kv_up_width, 4201U);
    const auto k_pe_b = make_input(shape.batch, 1U, shape.seq_len, shape.qk_rope_dim, 4202U);
    const auto entries_before_b = device->num_program_cache_entries();
    const auto [k_b, v_b] = launch(kv_up_b, k_pe_b);
    EXPECT_EQ(device->num_program_cache_entries(), entries_before_b) << "launch B must reuse launch A's program";
    EXPECT_NE(kv_up_b.buffer()->address(), kv_up_a.buffer()->address()) << "kv_up";
    EXPECT_NE(k_pe_b.buffer()->address(), k_pe_a.buffer()->address()) << "k_pe";
    EXPECT_NE(k_b.buffer()->address(), k_a.buffer()->address()) << "k";
    EXPECT_NE(v_b.buffer()->address(), v_a.buffer()->address()) << "v";
    const auto expected_b = reference_fw(kv_up_b, k_pe_b, shape);
    expect_fw_matches(k_b, v_b, expected_b, "launch B (cache hit)");
    expect_fw_matches(k_a, v_a, expected_a, "launch A after launch B");

    const auto entries_before_c = device->num_program_cache_entries();
    const auto [k_c, v_c] = launch(kv_up_a, k_pe_a);
    EXPECT_EQ(device->num_program_cache_entries(), entries_before_c) << "launch C must reuse launch A's program";
    EXPECT_NE(k_c.buffer()->address(), k_b.buffer()->address()) << "k C vs B";
    EXPECT_NE(v_c.buffer()->address(), v_b.buffer()->address()) << "v C vs B";
    expect_fw_matches(k_c, v_c, expected_a, "launch C (cache hit on launch A's inputs)");
    expect_fw_matches(k_b, v_b, expected_b, "launch B after launch C");
}

TEST_F(MLAKVAssembleTest, BackwardCacheHitWithFreshTensors_TwoCoreGroups) {
    auto* device = &ttml::test_utils::device_with_empty_program_cache();
    const auto shape = cache_hit_shape();
    const uint32_t qk_head = shape.qk_nope_dim + shape.qk_rope_dim;
    const auto launch = [&](const ttnn::Tensor& dK, const ttnn::Tensor& dV) {
        return ttml::metal::mla_kv_assemble_bw(
            dK, dV, shape.n_heads, shape.qk_nope_dim, shape.qk_rope_dim, shape.v_dim);
    };

    const auto dK_a = make_input(shape.batch, shape.n_heads, shape.seq_len, qk_head, 7101U);
    const auto dV_a = make_input(shape.batch, shape.n_heads, shape.seq_len, shape.v_dim, 7102U);
    const auto entries_before_a = device->num_program_cache_entries();
    const auto [dkv_up_a, dk_pe_a] = launch(dK_a, dV_a);
    ASSERT_GT(device->num_program_cache_entries(), entries_before_a) << "launch A must compile a program";
    const auto expected_a = reference_bw(dK_a, dV_a, shape);
    expect_bw_matches(dkv_up_a, dk_pe_a, expected_a, "launch A");

    const auto dK_b = make_input(shape.batch, shape.n_heads, shape.seq_len, qk_head, 7201U);
    const auto dV_b = make_input(shape.batch, shape.n_heads, shape.seq_len, shape.v_dim, 7202U);
    const auto entries_before_b = device->num_program_cache_entries();
    const auto [dkv_up_b, dk_pe_b] = launch(dK_b, dV_b);
    EXPECT_EQ(device->num_program_cache_entries(), entries_before_b) << "launch B must reuse launch A's program";
    EXPECT_NE(dK_b.buffer()->address(), dK_a.buffer()->address()) << "dK";
    EXPECT_NE(dV_b.buffer()->address(), dV_a.buffer()->address()) << "dV";
    EXPECT_NE(dkv_up_b.buffer()->address(), dkv_up_a.buffer()->address()) << "dkv_up";
    EXPECT_NE(dk_pe_b.buffer()->address(), dk_pe_a.buffer()->address()) << "dk_pe";
    const auto expected_b = reference_bw(dK_b, dV_b, shape);
    expect_bw_matches(dkv_up_b, dk_pe_b, expected_b, "launch B (cache hit)");
    expect_bw_matches(dkv_up_a, dk_pe_a, expected_a, "launch A after launch B");

    const auto entries_before_c = device->num_program_cache_entries();
    const auto [dkv_up_c, dk_pe_c] = launch(dK_a, dV_a);
    EXPECT_EQ(device->num_program_cache_entries(), entries_before_c) << "launch C must reuse launch A's program";
    EXPECT_NE(dkv_up_c.buffer()->address(), dkv_up_b.buffer()->address()) << "dkv_up C vs B";
    EXPECT_NE(dk_pe_c.buffer()->address(), dk_pe_b.buffer()->address()) << "dk_pe C vs B";
    expect_bw_matches(dkv_up_c, dk_pe_c, expected_a, "launch C (cache hit on launch A's inputs)");
    expect_bw_matches(dkv_up_b, dk_pe_b, expected_b, "launch B after launch C");
}
