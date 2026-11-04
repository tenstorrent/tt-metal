// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ops/mla_q_rope.hpp"
#include "ops/rope_op.hpp"
#include "ops/unary_ops.hpp"
#include "test_utils/mla_layout.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama.hpp"
#include "ttnn/types.hpp"

namespace {

struct MLA_QRopeShape {
    std::string name;
    uint32_t batch = 0;
    uint32_t seq_len = 0;
    uint32_t n_heads = 0;
    uint32_t qk_nope_dim = 0;
    uint32_t qk_rope_dim = 0;
};

ttnn::Tensor make_bf16_4d(uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, uint32_t seed) {
    auto* device = &ttml::autograd::ctx().get_device();
    const size_t count = static_cast<size_t>(d0) * d1 * d2 * d3;
    const auto host = ttml::test_utils::make_uniform_vector<float>(count, -1.0F, 1.0F, seed);
    return ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(
        host, ttnn::Shape{d0, d1, d2, d3}, device, ttnn::Layout::TILE);
}

ttml::ops::RotaryEmbeddingParams build_params(uint32_t seq_len, uint32_t qk_rope_dim) {
    return ttml::ops::build_rope_params(seq_len, qk_rope_dim, /*theta=*/10000.0F);
}

ttnn::Tensor slice_head_dim(
    const ttnn::Tensor& tensor, uint32_t B, uint32_t H, uint32_t S, uint32_t start_w, uint32_t end_w) {
    ttsl::SmallVector<uint32_t> step = {1, 1, 1, 1};
    return ttnn::slice(
        tensor, ttsl::SmallVector<uint32_t>{0, 0, 0, start_w}, ttsl::SmallVector<uint32_t>{B, H, S, end_w}, step);
}

// Reference: slice/rope/concat on the rope suffix. packed_input=true head-splits
// packed [B,1,S,H*D] first; packed_input=false takes head-major [B,H,S,D] as-is.
// pack_output=true packs the head-major result to [B,1,S,H*D].
ttnn::Tensor reference_q_rope(
    const ttnn::Tensor& q,
    const ttnn::Tensor& cos_cache,
    const ttnn::Tensor& sin_cache,
    const ttnn::Tensor& trans_mat,
    uint32_t n_heads,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim,
    bool packed_input = true,
    bool pack_output = false) {
    const auto shape = q.logical_shape();
    const uint32_t B = shape[0];
    const uint32_t S = shape[2];
    const uint32_t qk_head = qk_nope_dim + qk_rope_dim;

    auto q_in = packed_input ? ttml::test_utils::packed_to_head_major(q, n_heads, qk_head) : q;

    ttsl::SmallVector<uint32_t> step = {1, 1, 1, 1};
    auto q_nope = ttnn::slice(
        q_in, ttsl::SmallVector<uint32_t>{0, 0, 0, 0}, ttsl::SmallVector<uint32_t>{B, n_heads, S, qk_nope_dim}, step);
    auto q_pe = ttnn::slice(
        q_in,
        ttsl::SmallVector<uint32_t>{0, 0, 0, qk_nope_dim},
        ttsl::SmallVector<uint32_t>{B, n_heads, S, qk_head},
        step);

    auto q_pe_rot = ttnn::experimental::rotary_embedding_llama(
        q_pe,
        cos_cache,
        sin_cache,
        trans_mat,
        /*is_decode_mode=*/false,
        /*memory_config=*/std::nullopt,
        ttml::core::ComputeKernelConfig::precise());

    auto head_major = ttnn::concat(std::vector<ttnn::Tensor>{q_nope, q_pe_rot}, /*dim=*/3);
    return pack_output ? ttml::test_utils::head_major_to_packed(head_major) : head_major;
}

void expect_q_rope_matches_reference(
    const ttnn::Tensor& fused, const ttnn::Tensor& ref, const MLA_QRopeShape& shape, const std::string& label_prefix) {
    const uint32_t qk_head = shape.qk_nope_dim + shape.qk_rope_dim;
    const uint32_t B = shape.batch;
    const uint32_t H = shape.n_heads;
    const uint32_t S = shape.seq_len;

    {
        const auto actual_q_nope = ttml::core::to_xtensor(slice_head_dim(fused, B, H, S, 0U, shape.qk_nope_dim));
        const auto expected_q_nope = ttml::core::to_xtensor(slice_head_dim(ref, B, H, S, 0U, shape.qk_nope_dim));
        ASSERT_EQ(actual_q_nope.shape(), expected_q_nope.shape())
            << label_prefix << shape.name << " q_nope: shape mismatch";
        EXPECT_TRUE(xt::allclose(actual_q_nope, expected_q_nope, 0.0, 0.0))
            << label_prefix << shape.name << " q_nope: value mismatch";
    }
    {
        const auto actual_q_pe = ttml::core::to_xtensor(slice_head_dim(fused, B, H, S, shape.qk_nope_dim, qk_head));
        const auto expected_q_pe = ttml::core::to_xtensor(slice_head_dim(ref, B, H, S, shape.qk_nope_dim, qk_head));
        ASSERT_EQ(actual_q_pe.shape(), expected_q_pe.shape()) << label_prefix << shape.name << " q_pe: shape mismatch";
        EXPECT_TRUE(xt::allclose(actual_q_pe, expected_q_pe, 1e-2, 1e-2))
            << label_prefix << shape.name << " q_pe: value mismatch";
    }
}

}  // namespace

class MLA_QRopeParamTest : public ::testing::TestWithParam<MLA_QRopeShape> {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

TEST_P(MLA_QRopeParamTest, FusedMatchesReference) {
    const MLA_QRopeShape shape = GetParam();
    ASSERT_LE(shape.qk_rope_dim, 128U) << shape.name << ": mla_q_rope requires qk_rope_dim <= 128";

    const uint32_t qk_head = shape.qk_nope_dim + shape.qk_rope_dim;
    auto q_pre = make_bf16_4d(shape.batch, 1U, shape.seq_len, shape.n_heads * qk_head, /*seed=*/7U);
    auto params = build_params(shape.seq_len, shape.qk_rope_dim);

    const auto fused = ttml::metal::mla_q_rope(
        q_pre,
        params.cos_cache,
        params.sin_cache,
        params.trans_mat,
        shape.qk_nope_dim,
        shape.qk_rope_dim,
        /*packed_input=*/true);
    const auto ref = reference_q_rope(
        q_pre,
        params.cos_cache,
        params.sin_cache,
        params.trans_mat,
        shape.n_heads,
        shape.qk_nope_dim,
        shape.qk_rope_dim);

    ASSERT_EQ(fused.logical_shape(), ttnn::Shape({shape.batch, shape.n_heads, shape.seq_len, qk_head}))
        << shape.name << " output shape";
    expect_q_rope_matches_reference(fused, ref, shape, /*label_prefix=*/"");
}

TEST_P(MLA_QRopeParamTest, BackwardPacksGrad) {
    // packed_input=false: head-major dL/dout -> packed dq_pre (identity on nope;
    // inverse RoPE on the rope suffix with negated caches).
    const MLA_QRopeShape shape = GetParam();
    ASSERT_LE(shape.qk_rope_dim, 128U) << shape.name << ": mla_q_rope requires qk_rope_dim <= 128";

    const uint32_t qk_head = shape.qk_nope_dim + shape.qk_rope_dim;
    auto dL_dout = make_bf16_4d(shape.batch, shape.n_heads, shape.seq_len, qk_head, /*seed=*/11U);
    auto params = build_params(shape.seq_len, shape.qk_rope_dim);

    const auto packed = ttml::metal::mla_q_rope(
        dL_dout,
        params.neg_cos_cache,
        params.neg_sin_cache,
        params.trans_mat,
        shape.qk_nope_dim,
        shape.qk_rope_dim,
        /*packed_input=*/false);

    ASSERT_EQ(packed.logical_shape(), ttnn::Shape({shape.batch, 1U, shape.seq_len, shape.n_heads * qk_head}))
        << shape.name << " packed grad shape";

    const auto ref = reference_q_rope(
        dL_dout,
        params.neg_cos_cache,
        params.neg_sin_cache,
        params.trans_mat,
        shape.n_heads,
        shape.qk_nope_dim,
        shape.qk_rope_dim,
        /*packed_input=*/false,
        /*pack_output=*/true);
    ASSERT_EQ(packed.logical_shape(), ref.logical_shape()) << shape.name << " packed grad vs packed reference";
    const auto unpacked = ttml::test_utils::packed_to_head_major(packed, shape.n_heads, qk_head);
    const auto ref_unpacked = ttml::test_utils::packed_to_head_major(ref, shape.n_heads, qk_head);
    expect_q_rope_matches_reference(unpacked, ref_unpacked, shape, /*label_prefix=*/"bw: ");
}

INSTANTIATE_TEST_SUITE_P(
    MLA_QRopeShapes,
    MLA_QRopeParamTest,
    ::testing::Values(
        MLA_QRopeShape{.name = "small", .batch = 1, .seq_len = 32, .n_heads = 4, .qk_nope_dim = 64, .qk_rope_dim = 32},
        MLA_QRopeShape{
            .name = "mla_like", .batch = 1, .seq_len = 32, .n_heads = 8, .qk_nope_dim = 128, .qk_rope_dim = 64},
        MLA_QRopeShape{.name = "batch2", .batch = 2, .seq_len = 32, .n_heads = 4, .qk_nope_dim = 64, .qk_rope_dim = 32},
        MLA_QRopeShape{
            .name = "square_st1", .batch = 2, .seq_len = 32, .n_heads = 2, .qk_nope_dim = 32, .qk_rope_dim = 32},
        MLA_QRopeShape{
            .name = "square_st2", .batch = 2, .seq_len = 64, .n_heads = 2, .qk_nope_dim = 32, .qk_rope_dim = 32},
        MLA_QRopeShape{
            .name = "asym_rope2", .batch = 2, .seq_len = 64, .n_heads = 4, .qk_nope_dim = 128, .qk_rope_dim = 64},
        MLA_QRopeShape{
            .name = "heads8_s96", .batch = 1, .seq_len = 96, .n_heads = 8, .qk_nope_dim = 64, .qk_rope_dim = 32},
        MLA_QRopeShape{
            .name = "large_tr4",
            .batch = 1,
            .seq_len = 32,
            .n_heads = 4,
            .qk_nope_dim = 128,
            .qk_rope_dim = 128},  // Tr = 4, Th = 8
        // Large Tn: q_nope bypasses compute (reader -> writer); only Tr rope tiles hit DST.
        MLA_QRopeShape{
            .name = "chunked_nope_tn35",
            .batch = 1,
            .seq_len = 32,
            .n_heads = 2,
            .qk_nope_dim = 1120,
            .qk_rope_dim = 64},  // Tn = 35, Tr = 2
        // B * Ts = 8 blocks: exercises multi-core split and repeated batch-boundary jumps.
        MLA_QRopeShape{
            .name = "batch4_st2", .batch = 4, .seq_len = 64, .n_heads = 2, .qk_nope_dim = 32, .qk_rope_dim = 32}),
    [](const ::testing::TestParamInfo<MLA_QRopeShape>& info) { return info.param.name; });

class MLA_QRopeCacheTest : public ::testing::Test {
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

// Buffer placement is a compile-time property: TensorAccessorArgs bake it into the
// kernel binary, so the program cache must key on every input's memory config. Runs
// the op with q_pre interleaved in DRAM and again with q_pre in L1 at identical shapes,
// asserting the second run compiles a distinct program and still matches the
// reference — a cache key blind to placement reuses the DRAM program for the L1
// buffer and reads through the wrong accessor tables.
TEST_F(MLA_QRopeCacheTest, DistinctProgramPerInputPlacement) {
    auto* device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();

    const MLA_QRopeShape shape{
        .name = "cache_small", .batch = 1, .seq_len = 32, .n_heads = 4, .qk_nope_dim = 64, .qk_rope_dim = 32};
    const uint32_t qk_head = shape.qk_nope_dim + shape.qk_rope_dim;
    auto q_pre = make_bf16_4d(shape.batch, 1U, shape.seq_len, shape.n_heads * qk_head, /*seed=*/11U);
    auto params = build_params(shape.seq_len, shape.qk_rope_dim);
    const auto ref = reference_q_rope(
        q_pre,
        params.cos_cache,
        params.sin_cache,
        params.trans_mat,
        shape.n_heads,
        shape.qk_nope_dim,
        shape.qk_rope_dim);

    const auto entries_start = device->num_program_cache_entries();
    const auto fused_dram = ttml::metal::mla_q_rope(
        q_pre,
        params.cos_cache,
        params.sin_cache,
        params.trans_mat,
        shape.qk_nope_dim,
        shape.qk_rope_dim,
        /*packed_input=*/true);
    const auto entries_after_dram = device->num_program_cache_entries();
    // Guard against a vacuous test: the first call must actually populate the cache,
    // else the delta check below would pass trivially with the cache disabled.
    ASSERT_GT(entries_after_dram, entries_start) << "program cache not populated by the DRAM-placement run";
    expect_q_rope_matches_reference(fused_dram, ref, shape, /*label_prefix=*/"dram: ");

    auto q_pre_l1 = ttnn::to_memory_config(q_pre, ttnn::L1_MEMORY_CONFIG);
    const auto entries_before_l1 = device->num_program_cache_entries();
    const auto fused_l1 = ttml::metal::mla_q_rope(
        q_pre_l1,
        params.cos_cache,
        params.sin_cache,
        params.trans_mat,
        shape.qk_nope_dim,
        shape.qk_rope_dim,
        /*packed_input=*/true);
    const auto entries_after_l1 = device->num_program_cache_entries();
    EXPECT_GT(entries_after_l1, entries_before_l1)
        << "mla_q_rope reused the DRAM-placement program for an L1 input (stale cache key)";
    expect_q_rope_matches_reference(fused_l1, ref, shape, /*label_prefix=*/"l1: ");
}

TEST_F(MLA_QRopeCacheTest, AutogradWrapperBackwardMatchesReference) {
    const MLA_QRopeShape shape{
        .name = "autograd_small", .batch = 1, .seq_len = 32, .n_heads = 4, .qk_nope_dim = 64, .qk_rope_dim = 32};
    const uint32_t qk_head = shape.qk_nope_dim + shape.qk_rope_dim;
    auto params = build_params(shape.seq_len, shape.qk_rope_dim);

    auto q_pre = ttml::autograd::create_tensor(
        make_bf16_4d(shape.batch, 1U, shape.seq_len, shape.n_heads * qk_head, /*seed=*/13U), /*requires_grad=*/true);
    auto out = ttml::ops::mla_q_rope(q_pre, params, shape.qk_nope_dim, shape.qk_rope_dim);
    auto loss = ttml::ops::mean(out);
    loss->backward();

    ASSERT_EQ(q_pre->get_grad().logical_shape(), q_pre->get_value().logical_shape()) << "autograd dq_pre shape";
    const auto ref_packed = reference_q_rope(
        out->get_grad(),
        params.neg_cos_cache,
        params.neg_sin_cache,
        params.trans_mat,
        shape.n_heads,
        shape.qk_nope_dim,
        shape.qk_rope_dim,
        /*packed_input=*/false,
        /*pack_output=*/true);
    const auto unpacked = ttml::test_utils::packed_to_head_major(q_pre->get_grad(), shape.n_heads, qk_head);
    const auto ref_unpacked = ttml::test_utils::packed_to_head_major(ref_packed, shape.n_heads, qk_head);
    expect_q_rope_matches_reference(unpacked, ref_unpacked, shape, /*label_prefix=*/"autograd: ");
}

TEST_F(MLA_QRopeCacheTest, PackedInputRejectsMalformedShapes) {
    constexpr uint32_t qk_nope_dim = 64U;
    constexpr uint32_t qk_rope_dim = 32U;
    constexpr uint32_t qk_head = qk_nope_dim + qk_rope_dim;
    auto params = build_params(/*seq_len=*/32U, qk_rope_dim);

    // Logical packed width 128 is tile-aligned but not a multiple of qk_head=96.
    auto bad_width = make_bf16_4d(1U, 1U, 32U, 128U, /*seed=*/3U);
    EXPECT_ANY_THROW(ttml::metal::mla_q_rope(
        bad_width,
        params.cos_cache,
        params.sin_cache,
        params.trans_mat,
        qk_nope_dim,
        qk_rope_dim,
        /*packed_input=*/true))
        << "packed width that is not a multiple of qk_head must be rejected";

    auto bad_dim1 = make_bf16_4d(1U, 2U, 32U, 2U * qk_head, /*seed=*/5U);
    EXPECT_ANY_THROW(ttml::metal::mla_q_rope(
        bad_dim1,
        params.cos_cache,
        params.sin_cache,
        params.trans_mat,
        qk_nope_dim,
        qk_rope_dim,
        /*packed_input=*/true))
        << "packed input with dim 1 != 1 must be rejected";
}

// ── Program-cache hits ──────────────────────────────────────────────────────────────────────────────
// A launch whose shapes and attributes match a cached program reuses it, and only the buffer addresses of
// its five tensors (q_in, cos, sin, trans_mat; q_out) are patched in. Every test clears the cache, so its
// first launch of each direction is the miss, and keeps every tensor alive until it ends, so no launch can
// land on the buffers of an earlier one.

namespace {

// B = 2 and Ts = cores + 1 give B * Ts = 2 * cores + 2 blocks: more than the core count and not a multiple of
// it, so split_work_to_cores hands out two core groups on any grid.
MLA_QRopeShape cache_hit_shape() {
    return {
        .name = "two_core_groups",
        .batch = 2,
        .seq_len = 32U * (ttml::test_utils::num_worker_cores() + 1U),
        .n_heads = 2,
        .qk_nope_dim = 32,
        .qk_rope_dim = 64};
}

struct RopeInputs {
    ttnn::Tensor cos_cache;
    ttnn::Tensor sin_cache;
    ttnn::Tensor trans_mat;
};

// One direction of mla_q_rope at the cache-hit shape. The forward (packed_input) rotates packed q; the backward
// rotates head-major gradients back with the caches of the negated angle, as the backward of ttml::ops::mla_q_rope
// does. The two are distinct programs: packed_input is hashed.
class QRopeDirection {
public:
    QRopeDirection(const MLA_QRopeShape& shape, bool packed_input) : shape_(shape), packed_input_(packed_input) {
    }

    std::string name() const {
        return packed_input_ ? "forward" : "backward";
    }

    RopeInputs rope_inputs(const ttml::ops::RotaryEmbeddingParams& params, const ttnn::Tensor& trans_mat) const {
        if (packed_input_) {
            return {params.cos_cache, params.sin_cache, trans_mat};
        }
        return {params.neg_cos_cache, params.neg_sin_cache, trans_mat};
    }

    // Forward: packed [B, 1, S, H * qk_head]. Backward: head-major [B, H, S, qk_head].
    ttnn::Tensor make_q(uint32_t seed) const {
        return packed_input_ ? make_bf16_4d(shape_.batch, 1U, shape_.seq_len, shape_.n_heads * qk_head(), seed)
                             : make_bf16_4d(shape_.batch, shape_.n_heads, shape_.seq_len, qk_head(), seed);
    }

    ttnn::Tensor launch(const ttnn::Tensor& q, const RopeInputs& rope) const {
        return ttml::metal::mla_q_rope(
            q, rope.cos_cache, rope.sin_cache, rope.trans_mat, shape_.qk_nope_dim, shape_.qk_rope_dim, packed_input_);
    }

    ttnn::Tensor reference(const ttnn::Tensor& q, const RopeInputs& rope) const {
        return reference_q_rope(
            q,
            rope.cos_cache,
            rope.sin_cache,
            rope.trans_mat,
            shape_.n_heads,
            shape_.qk_nope_dim,
            shape_.qk_rope_dim,
            packed_input_,
            /*pack_output=*/!packed_input_);
    }

    // The backward's packed output is compared head-major, as in BackwardPacksGrad.
    void expect_matches(const ttnn::Tensor& out, const ttnn::Tensor& ref, const std::string& label) const {
        if (packed_input_) {
            expect_q_rope_matches_reference(out, ref, shape_, label);
            return;
        }
        expect_q_rope_matches_reference(
            ttml::test_utils::packed_to_head_major(out, shape_.n_heads, qk_head()),
            ttml::test_utils::packed_to_head_major(ref, shape_.n_heads, qk_head()),
            shape_,
            label);
    }

private:
    uint32_t qk_head() const {
        return shape_.qk_nope_dim + shape_.qk_rope_dim;
    }

    MLA_QRopeShape shape_;
    bool packed_input_ = true;
};

// Launch A compiles the program of one direction. Every later launch must reuse it with its own addresses patched
// in, and leave the outputs of the earlier ones untouched:
//  - B runs on a new q and on rope inputs with other values: caches built with theta = 500 instead of 10000 and a
//    negated rotation matrix. Every address changes, and a hit that kept reading any of A's buffers rotates by the
//    wrong angles.
//  - C runs on a new q with A's rope inputs, as every DeepSeek layer shares one set: the cos/sin/trans_mat addresses
//    go back to the miss's while the program holds B's.
//  - D runs on A's inputs again, so the q_in address goes back to the miss's as well.
// A hit that patched only the addresses that differ from the miss's, or patched them by their offset from the miss's,
// would read B's or C's inputs.
void run_cache_hits(bool packed_input) {
    auto* device = &ttml::test_utils::device_with_empty_program_cache();
    const auto shape = cache_hit_shape();
    const QRopeDirection direction(shape, packed_input);
    const auto label = [&](const std::string& launch) { return direction.name() + " " + launch + ": "; };
    const auto launch_hit = [&](const ttnn::Tensor& q, const RopeInputs& rope, const std::string& launch) {
        const auto entries_before = device->num_program_cache_entries();
        auto out = direction.launch(q, rope);
        EXPECT_EQ(device->num_program_cache_entries(), entries_before)
            << direction.name() << " " << launch << " must reuse launch A's program";
        return out;
    };

    const auto q_a = direction.make_q(/*seed=*/31U);
    const auto params_a = build_params(shape.seq_len, shape.qk_rope_dim);
    const auto rope_a = direction.rope_inputs(params_a, params_a.trans_mat);
    const auto entries_before_a = device->num_program_cache_entries();
    const auto out_a = direction.launch(q_a, rope_a);
    ASSERT_GT(device->num_program_cache_entries(), entries_before_a)
        << direction.name() << " launch A must compile a program";
    const auto ref_a = direction.reference(q_a, rope_a);
    direction.expect_matches(out_a, ref_a, label("launch A"));

    const auto q_b = direction.make_q(/*seed=*/37U);
    const auto params_b = ttml::ops::build_rope_params(shape.seq_len, shape.qk_rope_dim, /*theta=*/500.0F);
    const xt::xarray<float> negated_trans_mat = -ttml::core::to_xtensor(params_a.trans_mat);
    const auto rope_b = direction.rope_inputs(params_b, ttml::core::from_xtensor(negated_trans_mat, device));
    const auto out_b = launch_hit(q_b, rope_b, "launch B");
    EXPECT_NE(q_b.buffer()->address(), q_a.buffer()->address()) << direction.name() << " q_in";
    EXPECT_NE(rope_b.cos_cache.buffer()->address(), rope_a.cos_cache.buffer()->address()) << direction.name() << " cos";
    EXPECT_NE(rope_b.sin_cache.buffer()->address(), rope_a.sin_cache.buffer()->address()) << direction.name() << " sin";
    EXPECT_NE(rope_b.trans_mat.buffer()->address(), rope_a.trans_mat.buffer()->address())
        << direction.name() << " trans";
    EXPECT_NE(out_b.buffer()->address(), out_a.buffer()->address()) << direction.name() << " q_out";
    const auto ref_b = direction.reference(q_b, rope_b);
    direction.expect_matches(out_b, ref_b, label("launch B (cache hit)"));
    direction.expect_matches(out_a, ref_a, label("launch A after launch B"));

    const auto q_c = direction.make_q(/*seed=*/41U);
    const auto out_c = launch_hit(q_c, rope_a, "launch C");
    EXPECT_NE(out_c.buffer()->address(), out_b.buffer()->address()) << direction.name() << " q_out C vs B";
    const auto ref_c = direction.reference(q_c, rope_a);
    direction.expect_matches(out_c, ref_c, label("launch C (cache hit on launch A's rope inputs)"));
    direction.expect_matches(out_b, ref_b, label("launch B after launch C"));

    const auto out_d = launch_hit(q_a, rope_a, "launch D");
    EXPECT_NE(out_d.buffer()->address(), out_c.buffer()->address()) << direction.name() << " q_out D vs C";
    direction.expect_matches(out_d, ref_a, label("launch D (cache hit on launch A's inputs)"));
    direction.expect_matches(out_c, ref_c, label("launch C after launch D"));
}

}  // namespace

TEST_F(MLA_QRopeCacheTest, ForwardCacheHitWithFreshTensorsAndRopeCaches_TwoCoreGroups) {
    run_cache_hits(/*packed_input=*/true);
}

TEST_F(MLA_QRopeCacheTest, BackwardCacheHitWithFreshTensorsAndRopeCaches_TwoCoreGroups) {
    run_cache_hits(/*packed_input=*/false);
}

// A training step launches the forward and the backward, so both programs stay cached and hit in turn, with the
// rope inputs every layer shares. Each hit must patch its own program: state kept per op instead of per program
// would hand one direction's hit the runtime args of the other.
TEST_F(MLA_QRopeCacheTest, ForwardAndBackwardCacheHitsAlternate_TwoCoreGroups) {
    auto* device = &ttml::test_utils::device_with_empty_program_cache();
    const auto shape = cache_hit_shape();
    const auto params = build_params(shape.seq_len, shape.qk_rope_dim);
    const std::array<QRopeDirection, 2> directions = {
        QRopeDirection(shape, /*packed_input=*/true), QRopeDirection(shape, /*packed_input=*/false)};

    struct Launch {
        std::string label;
        const QRopeDirection* direction = nullptr;
        ttnn::Tensor q;
        ttnn::Tensor out;
        ttnn::Tensor ref;
    };
    std::vector<Launch> launches;
    for (uint32_t step = 0U; step < 2U; ++step) {
        for (const auto& direction : directions) {
            auto& launch = launches.emplace_back();
            launch.label = direction.name() + " step " + std::to_string(step);
            launch.direction = &direction;
            const auto rope = direction.rope_inputs(params, params.trans_mat);
            launch.q = direction.make_q(/*seed=*/51U + static_cast<uint32_t>(launches.size()));
            const auto entries_before = device->num_program_cache_entries();
            launch.out = direction.launch(launch.q, rope);
            if (step == 0U) {
                ASSERT_GT(device->num_program_cache_entries(), entries_before)
                    << launch.label << " must compile a program";
            } else {
                EXPECT_EQ(device->num_program_cache_entries(), entries_before)
                    << launch.label << " must reuse its program from step 0";
            }
            launch.ref = direction.reference(launch.q, rope);
            direction.expect_matches(launch.out, launch.ref, launch.label + ": ");
        }
    }

    for (const auto& launch : launches) {
        launch.direction->expect_matches(launch.out, launch.ref, launch.label + " after every launch: ");
    }
}
