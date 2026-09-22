// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <xtensor/misc/xsort.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/gumbel_sample/device/gumbel_sample_device_operation.hpp"
#include "metal/ops/gumbel_sample/device/gumbel_sample_program_factory.hpp"
#include "metal/ops/gumbel_sample/gumbel_sample.hpp"
#include "metal/ops/gumbel_sample/gumbel_sample_constants.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"  // to_l1_interleaved, for the buffer-placement test

// Device tests for the fused ttml::metal::gumbel_sample op (plus, for preallocated outputs, the
// underlying ttnn::prim::ttml_gumbel_sample). The device is opened and closed PER TEST, matching
// the suite these tests were extracted from: several tests stage deliberate program-cache
// collisions and cache-hit re-patches, and a per-test device keeps every test's cache
// interactions self-contained.
class GumbelSampleOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(GumbelSampleOpTest, TestSamplingZeroTemperatureNoMask) {
    // xarray of shape {1, 1, 32, 32} with max along the diagonal
    xt::xarray<float>::shape_type shape = {1, 1, 32, 32};
    xt::xarray<float> a = xt::zeros<float>(shape);
    // Set diagonal max: for each row i, set a(0,0,i,i) = 1000.0f
    for (size_t i = 0; i < 32; ++i) {
        a(0, 0, i, i) = 1000.0f;
    }
    std::vector<uint32_t> expected_b(32);
    for (size_t i = 0; i < 32; ++i) {
        expected_b[i] = i;
    }
    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());
    auto tensor_b = ttml::metal::gumbel_sample(tensor_a, 0.0F, 42);
    auto vector_b = ttml::core::to_vector<uint32_t>(tensor_b);
    EXPECT_EQ(vector_b, expected_b);
}

TEST_F(GumbelSampleOpTest, TestSamplingSubnormalTemperatureIsExactGreedy) {
    // 1e-39F is positive and finite, so it passes validation and used to select the noise kernel --
    // but 1/1e-39 overflows FLT_MAX, and the resulting +inf scale factor collapsed every positive
    // logit to the same +inf bit pattern (and every zero logit to NaN, which float32_greater never
    // picks). The "sampled" argmax then returned the FIRST positive column instead of the max one.
    // Sub-reciprocal-overflow temperatures now route to the greedy kernel, the exact limit a
    // temperature anneal approaches.
    //
    // Each row is built so the two behaviors disagree: an early positive DECOY column with a small
    // logit and a later WINNER column with the true max. The old path returns the decoy (first
    // positive), greedy returns the winner -- so exact equality here is the regression check.
    constexpr uint32_t kRows = 32U;
    constexpr uint32_t kVocab = 64U;
    xt::xarray<float>::shape_type shape = {1, 1, kRows, kVocab};
    xt::xarray<float> a = xt::zeros<float>(shape);
    a.fill(-1.0F);
    std::vector<uint32_t> expected(kRows);
    for (uint32_t i = 0; i < kRows; ++i) {
        const uint32_t decoy = i % 8U;
        const uint32_t winner = 8U + ((i * 3U) % (kVocab - 8U));
        a(0, 0, i, decoy) = 0.5F;
        a(0, 0, i, winner) = 2.0F;
        expected[i] = winner;
    }
    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());
    auto got = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(tensor_a, 1e-39F, 42));
    EXPECT_EQ(got, expected);
}

TEST_F(GumbelSampleOpTest, TestSamplingPositiveTemperatureNoMask) {
    // Test sampling with positive temperature, no mask, and xarray of shape {1, 1, 32, 64}
    xt::xarray<float>::shape_type shape = {1, 1, 32, 64};
    xt::xarray<float> a = ttml::test_utils::make_uniform_xarray<float>(shape, 0.0F, 1.0F, 42U);
    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());
    float temperature = 1.0F;
    auto tensor_b = ttml::metal::gumbel_sample(tensor_a, temperature, 42);
    auto vector_b = ttml::core::to_vector<uint32_t>(tensor_b);
    // The output should have shape {1, 1, 32} (one sample per row)
    EXPECT_EQ(vector_b.size(), 32);
    // All values should be in the range [0, 63] (since last dim is 64)
    for (auto v : vector_b) {
        EXPECT_GE(v, 0);
        EXPECT_LT(v, 64);
    }
}

TEST_F(GumbelSampleOpTest, TestSamplingPositiveTemperatureWithMask) {
    // Test sampling with positive temperature, with mask, and xarray of shape {1, 1, 32, 65}
    xt::xarray<float>::shape_type shape = {1, 1, 32, 65};
    xt::xarray<float> a = ttml::test_utils::make_uniform_xarray<float>(shape, 0.0F, 1.0F, 84U);
    // Mask out the last column. Shape is {1, 1, 1, 65}: one row broadcast across every token, which
    // is what callers build (padding columns do not depend on token position).
    xt::xarray<float>::shape_type mask_shape = {1, 1, 1, 65};
    xt::xarray<float> mask = xt::zeros<float>(mask_shape);
    mask(0, 0, 0, 64) = 1e4F;
    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());
    auto tensor_mask = ttml::core::from_xtensor(mask, &ttml::autograd::ctx().get_device());
    float temperature = 1.0F;
    auto tensor_b = ttml::metal::gumbel_sample(tensor_a, temperature, 42, /* seed_axes */ {}, tensor_mask);
    auto vector_b = ttml::core::to_vector<uint32_t>(tensor_b);
    // The output should have shape {1, 1, 32} (one sample per row)
    EXPECT_EQ(vector_b.size(), 32);
    // All values should be in the range [0, 63] (since last dim is 65, but last index is masked)
    for (auto v : vector_b) {
        EXPECT_GE(v, 0);
        EXPECT_LT(v, 64);
    }
}

TEST_F(GumbelSampleOpTest, TestSamplingGreedySingleRowWithMask) {
    // Smallest shape the masked greedy path can run on: one batch entry, one token row, one sample
    // out. Both logits and mask are [1, 1, 1, V], so every dim except the vocabulary is degenerate
    // and the whole result is a single index -- exact and deterministic at temperature 0.
    constexpr uint32_t kVocab = 64U;
    constexpr uint32_t kDecoy = 40U;   // the raw argmax; only the mask can dethrone it
    constexpr uint32_t kWinner = 17U;  // the best column once the decoy is masked

    xt::xarray<float> logits = xt::xarray<float>::from_shape({1U, 1U, 1U, kVocab});
    logits.fill(-1.0F);
    logits(0, 0, 0, kWinner) = -0.5F;
    logits(0, 0, 0, kDecoy) = 0.0F;

    xt::xarray<float> mask = xt::zeros<float>(xt::xarray<float>::shape_type{1U, 1U, 1U, kVocab});
    mask(0, 0, 0, kDecoy) = 1e4F;

    auto* device = &ttml::autograd::ctx().get_device();
    auto tensor_logits = ttml::core::from_xtensor(logits, device);
    auto tensor_mask = ttml::core::from_xtensor(mask, device);

    // Without the mask the decoy wins, which is what makes the masked call below a real assertion
    // rather than a restatement of the argmax.
    auto unmasked = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(tensor_logits, 0.0F, 42));
    ASSERT_EQ(unmasked.size(), 1U);
    EXPECT_EQ(unmasked[0], kDecoy) << "unmasked greedy must land on the decoy";

    auto masked = ttml::core::to_vector<uint32_t>(
        ttml::metal::gumbel_sample(tensor_logits, 0.0F, 42, /* seed_axes */ {}, tensor_mask));
    ASSERT_EQ(masked.size(), 1U);
    EXPECT_EQ(masked[0], kWinner) << "masked greedy must skip the decoy and take the best real column";
}

TEST_F(GumbelSampleOpTest, TestSamplingMaskManyTilesPerCore) {
    // Noise+mask with MANY tiles per core: ~8200 tiles over the grid gives every core ~32 blocks
    // and ~64 mask applications per kernel run. Every other mask test in this suite hands each
    // core a single tile, so a mask apply that only works on the FIRST DST batch still passes all
    // of them -- per-batch state leaks (stale unpacker config, DST offsets, SFPU/replay state
    // between the mask apply and the rand/gumbel/copy passes) are only visible here. The decoy
    // column outranks every active column unless the mask lands, and the suppressed columns can
    // win only if scores get corrupted, so both failure modes are separately visible.
    constexpr uint32_t kRows = 65600;
    constexpr uint32_t kVocab = 120;
    constexpr uint32_t kDecoyCol = 100U;
    xt::xarray<float>::shape_type shape = {1, 1, kRows, kVocab};
    xt::xarray<float> a = xt::zeros<float>(shape);
    a.fill(-60.0F);
    for (uint32_t r = 0; r < kRows; ++r) {
        a(0, 0, r, 5) = std::log(8.0F);
        a(0, 0, r, 52) = std::log(4.0F);
        a(0, 0, r, 70) = std::log(2.0F);
        a(0, 0, r, 115) = std::log(1.0F);
        a(0, 0, r, kDecoyCol) = std::log(1000.0F);  // outranks every active column unless masked
    }
    xt::xarray<float>::shape_type mask_shape = {1, 1, 1, kVocab};
    xt::xarray<float> mask = xt::zeros<float>(mask_shape);
    mask(0, 0, 0, kDecoyCol) = 1e4F;
    auto* device = &ttml::autograd::ctx().get_device();
    auto tensor_a = ttml::core::from_xtensor(a, device);
    auto tensor_mask = ttml::core::from_xtensor(mask, device);
    auto picks = ttml::core::to_vector<uint32_t>(
        ttml::metal::gumbel_sample(tensor_a, 1.0F, 42, /* seed_axes */ {}, tensor_mask));
    ASSERT_EQ(picks.size(), kRows);
    uint32_t decoys = 0U;
    uint32_t others = 0U;
    for (auto pick : picks) {
        if (pick == kDecoyCol) {
            ++decoys;
        } else if (pick != 5U && pick != 52U && pick != 70U && pick != 115U) {
            ++others;
        }
    }
    EXPECT_EQ(decoys, 0U) << decoys << " rows sampled the masked decoy column";
    EXPECT_EQ(others, 0U) << others << " rows sampled a suppressed column";
}

namespace {

// gumbel_sample() draws U ~ Uniform[2^-32, 1) and applies the Gumbel transform -log(-log(U)), so the noise
// added to the scaled logits is bounded to roughly [-3.1, +16.6]. The tests below size their logit
// gaps against this span so that "the scaled logits must win" is a guarantee, not a coin flip.
constexpr float kGumbelNoiseSpan = 20.0F;

// Logits with one WINNER column planted per (batch, token) row above a uniform floor, plus the
// row-major expected argmax. Winners walk the vocabulary with a per-row stride, so a row, page or
// batch-entry mix-up lands on a DIFFERENT id instead of coincidentally matching. `offset` shifts
// the walk and `winner_modulo` (0 = the full vocab) restricts which columns can win -- the
// buffer-placement test uses both to keep winners off its decoy column.
struct WinnerLogits {
    xt::xarray<float> logits;
    std::vector<uint32_t> expected;  // argmax per (batch, token) row, row-major
};

WinnerLogits make_winner_logits(
    uint32_t batch,
    uint32_t tokens,
    uint32_t vocab,
    uint32_t stride,
    uint32_t offset = 0U,
    uint32_t winner_modulo = 0U,
    float winner_value = -0.5F,
    float floor_value = -1.0F) {
    const uint32_t modulo = (winner_modulo == 0U) ? vocab : winner_modulo;
    WinnerLogits out;
    out.logits = xt::xarray<float>::from_shape({batch, 1U, tokens, vocab});
    out.logits.fill(floor_value);
    out.expected.resize(static_cast<size_t>(batch) * tokens);
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t t = 0; t < tokens; ++t) {
            const uint32_t winner = ((b * tokens + t) * stride + offset) % modulo;
            out.logits(b, 0, t, winner) = winner_value;
            out.expected[b * tokens + t] = winner;
        }
    }
    return out;
}

// Exact CPU reference for GREEDY (temperature 0) sampling: xt::argmax over (logits - mask) at each
// selected row. TIE-BREAK CONTRACT: axis-less xt::argmax is std::max_element over the row
// (xsort.hpp), so ties keep the FIRST occurrence -- the lowest column index -- which is exactly the
// tie-break the writer's scan and merge implement (strict greater, columns in increasing order),
// itself chosen to match ttnn::argmax. Exact on FLOAT32 device inputs: greedy applies no scaling
// and no noise, an absent (or zero) mask column leaves the logit bit-identical, and a banned
// column's fp32 subtract rounds the same way on host and device -- so the two argmaxes see the
// same values.
std::vector<uint32_t> greedy_reference(
    const xt::xarray<float>& logits,
    const std::optional<xt::xarray<float>>& mask,
    const std::optional<std::vector<uint32_t>>& positions) {
    const auto batch = static_cast<uint32_t>(logits.shape(0));
    const auto tokens = static_cast<uint32_t>(logits.shape(2));
    std::vector<uint32_t> expected;
    expected.reserve(positions.has_value() ? batch : static_cast<size_t>(batch) * tokens);
    for (uint32_t b = 0; b < batch; ++b) {
        // A [B, 1, 1, V] mask carries one row per entry; a [1, 1, 1, V] mask is shared by all.
        const uint32_t mask_row = (mask.has_value() && mask->shape(0) > 1U) ? b : 0U;
        const uint32_t first_token = positions.has_value() ? (*positions)[b] : 0U;
        const uint32_t last_token = positions.has_value() ? first_token + 1U : tokens;
        for (uint32_t t = first_token; t < last_token; ++t) {
            xt::xarray<float> scores = xt::view(logits, b, 0, t, xt::all());
            if (mask.has_value()) {
                scores -= xt::view(*mask, mask_row, 0, 0, xt::all());
            }
            expected.push_back(static_cast<uint32_t>(xt::argmax(scores)()));
        }
    }
    return expected;
}

}  // namespace

TEST_F(GumbelSampleOpTest, TestSamplingDoesNotMutateInputs) {
    // gumbel_sample() runs the Gumbel chain, the temperature scaling and the mask subtraction in place.
    // Only the noise buffer it allocates itself may be written -- the caller's logits and mask must
    // come back untouched. The size/bounds assertions in the tests above cannot see an in-place op
    // that picked the wrong operand, so check the inputs directly.
    constexpr uint32_t kRows = 32;
    constexpr uint32_t kVocab = 64;
    xt::xarray<float>::shape_type shape = {1, 1, kRows, kVocab};
    xt::xarray<float> a = ttml::test_utils::make_uniform_xarray<float>(shape, -2.0F, 2.0F, 42U);
    // [1, 1, 1, V]: the broadcast shape every caller passes.
    xt::xarray<float>::shape_type mask_shape = {1, 1, 1, kVocab};
    xt::xarray<float> m = xt::zeros<float>(mask_shape);
    m(0, 0, 0, kVocab - 1) = 1e4F;

    auto* device = &ttml::autograd::ctx().get_device();
    auto tensor_a = ttml::core::from_xtensor(a, device);
    auto tensor_mask = ttml::core::from_xtensor(m, device);

    const auto logits_before = ttml::core::to_vector(tensor_a);
    const auto mask_before = ttml::core::to_vector(tensor_mask);

    // Positive temperature, no mask: the fused add writes into the noise buffer, reading the logits
    // through an activation. Nothing should land back on the logits.
    (void)ttml::metal::gumbel_sample(tensor_a, 1.0F, 42);
    EXPECT_EQ(ttml::core::to_vector(tensor_a), logits_before) << "positive temperature, no mask";

    // Positive temperature with a mask: the subtract is in place, but on a buffer gumbel_sample() owns.
    (void)ttml::metal::gumbel_sample(tensor_a, 1.0F, 42, /* seed_axes */ {}, tensor_mask);
    EXPECT_EQ(ttml::core::to_vector(tensor_a), logits_before) << "positive temperature, with mask";
    EXPECT_EQ(ttml::core::to_vector(tensor_mask), mask_before) << "positive temperature, mask operand";

    // Zero temperature with a mask: nothing has been allocated yet, so the working tensor still
    // aliases the caller's logits and the subtract must NOT be in place.
    (void)ttml::metal::gumbel_sample(tensor_a, 0.0F, 42, /* seed_axes */ {}, tensor_mask);
    EXPECT_EQ(ttml::core::to_vector(tensor_a), logits_before) << "zero temperature, with mask";
    EXPECT_EQ(ttml::core::to_vector(tensor_mask), mask_before) << "zero temperature, mask operand";
}

TEST_F(GumbelSampleOpTest, TestSamplingTemperatureScalesLogitsNotNoise) {
    // gumbel_sample() computes logits/temperature + noise, with the 1/temperature factor carried as an
    // activation on the *logits* operand of a single fused add. If that factor were attached to the
    // noise operand instead, or inverted, both directions below flip.
    constexpr uint32_t kRows = 32;
    constexpr uint32_t kVocab = 64;
    constexpr float kWinnerLogit = 1.0F;
    constexpr float kBackgroundMax = 0.25F;

    xt::xarray<float>::shape_type shape = {1, 1, kRows, kVocab};
    xt::xarray<float> a = ttml::test_utils::make_uniform_xarray<float>(shape, 0.0F, kBackgroundMax, 7U);
    std::vector<uint32_t> expected(kRows);
    for (uint32_t i = 0; i < kRows; ++i) {
        // Deliberately not the row index and not a constant, so a degenerate result is still wrong.
        const uint32_t winner = (i * 7U + 3U) % kVocab;
        a(0, 0, i, winner) = kWinnerLogit;
        expected[i] = winner;
    }
    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());

    // Cold: the gap is 0.75, so at temperature 0.005 the scaled gap is 150 -- far beyond the noise
    // span. Sampling must collapse onto the argmax, deterministically.
    const float cold_temperature = (kWinnerLogit - kBackgroundMax) / (10.0F * kGumbelNoiseSpan);
    auto cold = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(tensor_a, cold_temperature, 42));
    EXPECT_EQ(cold, expected) << "low temperature must reduce to argmax";

    // Hot: the logits are scaled down to <= 1e-4 and the noise dominates, so the argmax must not win
    // every row. All 32 rows agreeing by chance would be (1/64)^32.
    auto hot = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(tensor_a, 1.0e4F, 42));
    ASSERT_EQ(hot.size(), expected.size());
    EXPECT_NE(hot, expected) << "high temperature must not reduce to argmax";
}

TEST_F(GumbelSampleOpTest, TestSamplingBroadcastPaddingMask) {
    // The padding mask every real caller builds is [1, 1, 1, V] -- ONE row, because which vocab
    // columns are padding does not depend on the token position (see _sample_logits_mask in
    // generate.py and _build_logits_mask in llama_completer.py). It must apply to every token row.
    //
    // The mask is independent of the BATCH for the same reason one level up: every sequence is
    // decoded by the same lm_head, so the same columns are padding for all of them. kBatch > 1 makes
    // that explicit -- one [1, 1, 1, V] mask has to cover both entries, and the reader addresses mask
    // tiles by column alone, with no batch stride to get wrong.
    constexpr uint32_t kBatch = 2;
    constexpr uint32_t kRows = 32;          // must exceed 1, or the unmasked rows do not exist
    constexpr uint32_t kVocab = 64;         // real vocabulary
    constexpr uint32_t kPaddedVocab = 128;  // what a TP-padded LM head actually emits
    constexpr uint32_t kBestRealId = 42;

    // Real columns are negative (kBestRealId least so); padding columns sit at 0.0, exactly as
    // zero-filled LM-head rows do. So an UNMASKED argmax lands on kVocab -- the first padding
    // column -- and a correctly masked one lands on kBestRealId.
    xt::xarray<float>::shape_type logits_shape = {kBatch, 1, kRows, kPaddedVocab};
    xt::xarray<float> logits = xt::zeros<float>(logits_shape);
    for (uint32_t b = 0; b < kBatch; ++b) {
        for (uint32_t r = 0; r < kRows; ++r) {
            for (uint32_t c = 0; c < kVocab; ++c) {
                logits(b, 0, r, c) = -1.0F;
            }
            logits(b, 0, r, kBestRealId) = -0.5F;
            // columns [kVocab, kPaddedVocab) stay at 0.0
        }
    }

    xt::xarray<float>::shape_type mask_shape = {1, 1, 1, kPaddedVocab};
    xt::xarray<float> mask = xt::zeros<float>(mask_shape);
    for (uint32_t c = kVocab; c < kPaddedVocab; ++c) {
        mask(0, 0, 0, c) = 1e4F;
    }

    auto* device = &ttml::autograd::ctx().get_device();
    auto tensor_logits = ttml::core::from_xtensor(logits, device);
    auto tensor_mask = ttml::core::from_xtensor(mask, device);

    // Greedy: exact and deterministic, so assert the strongest thing -- every row picks the best
    // REAL column. A row that missed the mask would report kVocab instead.
    auto greedy = ttml::core::to_vector<uint32_t>(
        ttml::metal::gumbel_sample(tensor_logits, 0.0F, 42, /* seed_axes */ {}, tensor_mask));
    ASSERT_EQ(greedy.size(), kBatch * kRows);
    EXPECT_EQ(greedy, std::vector<uint32_t>(kBatch * kRows, kBestRealId))
        << "a [1, 1, 1, V] mask must apply to every token row of every batch entry, not just row 0";

    // Positive temperature compiles a different kernel (the noise and the scaling are no longer
    // compiled out), so the broadcast has to hold there too. The noise makes the winner among the
    // real columns unpredictable, but the 1e4 penalty is far beyond the noise span, so a padding
    // column must never win.
    auto sampled = ttml::core::to_vector<uint32_t>(
        ttml::metal::gumbel_sample(tensor_logits, 1.0F, 4242, /* seed_axes */ {}, tensor_mask));
    ASSERT_EQ(sampled.size(), kBatch * kRows);
    for (uint32_t r = 0; r < kBatch * kRows; ++r) {
        EXPECT_LT(sampled[r], kVocab) << "row " << r << " sampled a masked padding column";
    }
}

TEST_F(GumbelSampleOpTest, TestSamplingConvertsMismatchedMaskDtype) {
    // Every in-tree mask builder (build_logits_mask in utils.py, _build_logits_mask in
    // llama_completer.py, _sample_logits_mask in generate.py) emits a BFLOAT16 mask whatever the
    // logits dtype, and the composite sample() this op replaced accepted that: ttnn::subtract
    // converted on the fly. The fused op requires matching dtypes, so ttml::metal::gumbel_sample
    // must typecast a mismatched mask before dispatch rather than reject it. Cover both directions
    // and both kernel variants (greedy and sampled): a decoy column that wins unmasked must lose
    // once the mask lands.
    constexpr uint32_t kRows = 32;
    constexpr uint32_t kVocab = 64;
    constexpr uint32_t kDecoy = kVocab - 1;  // the raw argmax; only the mask can dethrone it
    constexpr uint32_t kBestRealId = 17;

    xt::xarray<float>::shape_type logits_shape = {1, 1, kRows, kVocab};
    xt::xarray<float> logits = xt::zeros<float>(logits_shape);
    for (uint32_t r = 0; r < kRows; ++r) {
        for (uint32_t c = 0; c < kVocab; ++c) {
            logits(0, 0, r, c) = -1.0F;
        }
        logits(0, 0, r, kBestRealId) = -0.5F;
        logits(0, 0, r, kDecoy) = 0.0F;
    }
    xt::xarray<float>::shape_type mask_shape = {1, 1, 1, kVocab};
    xt::xarray<float> mask = xt::zeros<float>(mask_shape);
    mask(0, 0, 0, kDecoy) = 1e4F;

    auto* device = &ttml::autograd::ctx().get_device();
    const std::vector<uint32_t> expected(kRows, kBestRealId);

    auto check = [&](const ttnn::Tensor& tensor_logits, const ttnn::Tensor& tensor_mask, const char* what) {
        const auto mask_dtype = tensor_mask.dtype();
        ASSERT_NE(tensor_logits.dtype(), mask_dtype) << what << ": this test needs a dtype mismatch";

        // Greedy: exact, so every row must land on the best real column.
        auto greedy = ttml::core::to_vector<uint32_t>(
            ttml::metal::gumbel_sample(tensor_logits, 0.0F, 42, /* seed_axes */ {}, tensor_mask));
        EXPECT_EQ(greedy, expected) << what << ": greedy";

        // Sampled: the noise may pick any real column, but the 1e4 penalty puts the decoy out of reach.
        auto sampled = ttml::core::to_vector<uint32_t>(
            ttml::metal::gumbel_sample(tensor_logits, 1.0F, 4242, /* seed_axes */ {}, tensor_mask));
        ASSERT_EQ(sampled.size(), kRows) << what;
        for (uint32_t r = 0; r < kRows; ++r) {
            EXPECT_NE(sampled[r], kDecoy) << what << ": row " << r << " sampled the masked decoy";
        }

        // The conversion is out of place: the caller's mask is untouched.
        EXPECT_EQ(tensor_mask.dtype(), mask_dtype) << what << ": mask operand must keep its dtype";
    };

    // FLOAT32 logits with the BFLOAT16 mask every existing caller builds.
    check(
        ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(logits, device),
        ttml::core::from_xtensor(mask, device),
        "fp32 logits, bf16 mask");
    // And the other way round.
    check(
        ttml::core::from_xtensor(logits, device),
        ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(mask, device),
        "bf16 logits, fp32 mask");
}

TEST_F(GumbelSampleOpTest, TestSamplingRaggedShapes) {
    // Every other sampling test uses tile-aligned dimensions (32 or 2048 tokens, 32/64 vocab) and a
    // single batch entry, which leaves three pieces of the op untested:
    //
    //   * tokens % 32 != 0  -- the last tile row of each batch entry is partly padding, so the
    //                          writer must emit only `valid_rows` results for it.
    //   * V % 32 != 0       -- the last vocab tile is partly padding, so the argmax scan must stop
    //                          at the logical width (`cols_to_scan`).
    //   * batch > 1         -- output pages are indexed as batch_index * tokens + first_token, and
    //                          with one batch entry that term is always zero.
    //
    // Padding is what makes this a real test rather than a shape smoke test: from_xtensor zero-fills
    // the padded region, and every REAL logit here is negative, so any padding element the scan
    // wrongly visits (0.0) beats the whole row and shows up as an out-of-range index.
    constexpr uint32_t kBatch = 2U;
    constexpr uint32_t kTokens = 37U;  // 37 = 32 + 5 -> Ht = 2, last tile row has 5 valid rows
    constexpr uint32_t kVocab = 77U;   // 77 = 2*32 + 13 -> Wt = 3, last tile has 13 valid columns

    // A distinct winner per (batch, token) so a row/page mix-up cannot pass by coincidence. The
    // winner/floor values are exact in both dtypes, so one grid serves both runs below.
    const auto winners = make_winner_logits(kBatch, kTokens, kVocab, /* stride */ 7U);

    auto run = [&](const ttnn::Tensor& tensor_a, const char* what) {
        // Greedy is exact, so assert the full result vector: one id per token, in [batch, token] order.
        auto greedy = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(tensor_a, 0.0F, 7));
        ASSERT_EQ(greedy.size(), kBatch * kTokens) << what << ": one sampled id per token, across all batch entries";
        EXPECT_EQ(greedy, winners.expected) << what;

        // The scan bounds also have to hold in the sampled kernel, which is a separate binary
        // (the noise compile-time arg). The noise makes the winner unpredictable, but no index may ever leave the
        // logical vocabulary -- reaching the zero-filled padding would.
        auto sampled = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(tensor_a, 1.0F, 99));
        ASSERT_EQ(sampled.size(), kBatch * kTokens) << what;
        for (uint32_t i = 0; i < sampled.size(); ++i) {
            EXPECT_LT(sampled[i], kVocab) << what << ": index " << i << " left the logical vocabulary";
        }
    };

    // Both dtypes: bf16 tile copies ride SrcA while FLOAT32 rides the factory's unpack-to-dest
    // path, and dtype is in the program-cache key -- two distinct programs whose ragged scan and
    // write-out bounds must hold independently.
    auto* device = &ttml::autograd::ctx().get_device();
    run(ttml::core::from_xtensor(winners.logits, device), "bf16");
    run(ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(winners.logits, device), "fp32");
}

TEST_F(GumbelSampleOpTest, TestSamplingHonoursBufferPlacement) {
    // TensorAccessorArgs bakes buffer placement into the reader/writer COMPILE-TIME args: it sets
    // ArgConfig::IsDram from buffer->is_dram() and emits the buffer's aligned_page_size, both of
    // which differ between DRAM and L1 (see tensor_accessor_args.cpp). Two calls that differ ONLY in
    // placement therefore need two different programs -- but they have identical shapes, dtypes and
    // mask-ness, so they collide in the program cache unless placement is part of its key. On the
    // second call the cache hit patches addresses only, leaving accessors compiled for the wrong
    // memory space pointed at the other one's addresses.
    //
    // Order matters: the DRAM call must run FIRST so it is the entry the L1 call then collides with.
    constexpr uint32_t kRows = 32;
    constexpr uint32_t kVocab = 64;
    constexpr uint32_t kDecoy = kVocab - 1;  // always the raw argmax; only the mask can dethrone it

    auto* device = &ttml::autograd::ctx().get_device();

    // Winners walk (r * 7 + 3) % (kVocab - 1): never the decoy column, and at 1.0F they sit far
    // enough above the floor that greedy is exact.
    const auto [a, expected] = make_winner_logits(
        1U, kRows, kVocab, /* stride */ 7U, /* offset */ 3U, /* winner_modulo */ kVocab - 1U, /* winner_value */ 1.0F);

    // ---- logits placement ----
    auto dram_logits = ttml::core::from_xtensor(a, device);
    ASSERT_EQ(dram_logits.memory_config().buffer_type(), tt::tt_metal::BufferType::DRAM);
    auto from_dram = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(dram_logits, 0.0F, 42));
    EXPECT_EQ(from_dram, expected) << "DRAM logits";

    auto l1_logits = ttml::ttnn_fixed::to_l1_interleaved(dram_logits);
    ASSERT_EQ(l1_logits.memory_config().buffer_type(), tt::tt_metal::BufferType::L1);
    auto from_l1 = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(l1_logits, 0.0F, 42));
    EXPECT_EQ(from_l1, expected) << "L1 logits must sample identically to DRAM logits";

    // ---- mask placement ----
    // The mask's placement is hashed (placement_of) precisely because accessors are compiled per memory space — this
    // guards that. The decoy column outranks every real winner, so a mask read from the wrong memory space cannot go
    // unnoticed.
    xt::xarray<float> decoyed = a;
    for (uint32_t r = 0; r < kRows; ++r) {
        decoyed(0, 0, r, kDecoy) = 2.0F;
    }
    xt::xarray<float>::shape_type mask_shape = {1, 1, 1, kVocab};
    xt::xarray<float> m = xt::zeros<float>(mask_shape);
    m(0, 0, 0, kDecoy) = 1e4F;

    auto decoyed_logits = ttml::core::from_xtensor(decoyed, device);
    auto dram_mask = ttml::core::from_xtensor(m, device);
    ASSERT_EQ(dram_mask.memory_config().buffer_type(), tt::tt_metal::BufferType::DRAM);
    auto masked_dram = ttml::core::to_vector<uint32_t>(
        ttml::metal::gumbel_sample(decoyed_logits, 0.0F, 42, /* seed_axes */ {}, dram_mask));
    EXPECT_EQ(masked_dram, expected) << "DRAM mask";

    auto l1_mask = ttml::ttnn_fixed::to_l1_interleaved(dram_mask);
    ASSERT_EQ(l1_mask.memory_config().buffer_type(), tt::tt_metal::BufferType::L1);
    auto masked_l1 = ttml::core::to_vector<uint32_t>(
        ttml::metal::gumbel_sample(decoyed_logits, 0.0F, 42, /* seed_axes */ {}, l1_mask));
    EXPECT_EQ(masked_l1, expected) << "L1 mask must suppress the decoy exactly as a DRAM mask does";
}

namespace {

// Positions as the op wants them: [B, 1, 1, 1] UINT32 ROW_MAJOR. Note the explicit layout --
// core::from_vector defaults to TILE, which the op rejects (a tiled [B,1,1,1] pads to a 32x32 tile
// and would make every page read 4 KB instead of one aligned word).
ttnn::Tensor make_positions(const std::vector<uint32_t>& positions) {
    return ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        positions,
        ttnn::Shape({static_cast<uint32_t>(positions.size()), 1U, 1U, 1U}),
        &ttml::autograd::ctx().get_device(),
        ttnn::Layout::ROW_MAJOR);
}

}  // namespace

TEST_F(GumbelSampleOpTest, TestSamplingGreedyMatchesArgmaxReference) {
    // Greedy is exact, so RANDOM logits can be checked against greedy_reference (a host argmax over
    // the masked rows at the selected positions). Unlike the planted-winner tests this sweeps
    // shapes with no structure at all -- every column is a live candidate in every row -- so any
    // scan-bound, page or broadcast slip surfaces as a mismatch somewhere in the sweep.
    //
    // RAW FLOAT32 randoms, deliberately: this is the regression test for the factory's
    // UnpackToDestFp32 mode. Without it the compute kernel's tile copy rides through SrcA, which
    // holds 19-bit TF32, so fp32 columns closer than ~2^-11 relative tie on device but not on the
    // host argmax -- exactly the mismatches this test produced before the mode was set. The last
    // case runs the reference against a per-row [B, 1, 1, V] mask too.
    struct Case {
        uint32_t batch, tokens, vocab;
        bool per_row_mask;
    };
    const std::vector<Case> cases = {
        {1U, 32U, 64U, false},  // tile-aligned baseline
        {2U, 37U, 77U, false},  // ragged tokens and vocab
        {3U, 70U, 130U, true},  // multi-tile-row entries, ragged vocab, per-row mask
    };

    auto* device = &ttml::autograd::ctx().get_device();
    uint32_t seed = 1000U;
    for (const auto& c : cases) {
        const std::string what =
            "[" + std::to_string(c.batch) + ", 1, " + std::to_string(c.tokens) + ", " + std::to_string(c.vocab) + "]";
        const xt::xarray<float> logits = ttml::test_utils::make_uniform_xarray<float>(
            xt::xarray<float>::shape_type{c.batch, 1U, c.tokens, c.vocab}, -2.0F, 2.0F, seed++);
        // Ban a deterministic scattering of columns (every third, phase-shifted per mask row):
        // dense enough that the masked argmax genuinely differs from the unmasked one.
        const uint32_t mask_batch = c.per_row_mask ? c.batch : 1U;
        xt::xarray<float> mask = xt::zeros<float>(xt::xarray<float>::shape_type{mask_batch, 1U, 1U, c.vocab});
        for (uint32_t mb = 0; mb < mask_batch; ++mb) {
            for (uint32_t v = mb % 3U; v < c.vocab; v += 3U) {
                mask(mb, 0, 0, v) = 1e4F;
            }
        }
        std::vector<uint32_t> positions(c.batch);
        for (uint32_t b = 0; b < c.batch; ++b) {
            positions[b] = (b * 31U + 5U) % c.tokens;
        }

        auto tensor_logits = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(logits, device);
        auto tensor_mask = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(mask, device);

        auto no_mask = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(tensor_logits, 0.0F, 7));
        EXPECT_EQ(no_mask, greedy_reference(logits, std::nullopt, std::nullopt)) << what << ": no mask";

        auto masked = ttml::core::to_vector<uint32_t>(
            ttml::metal::gumbel_sample(tensor_logits, 0.0F, 7, /* seed_axes */ {}, tensor_mask));
        EXPECT_EQ(masked, greedy_reference(logits, mask, std::nullopt)) << what << ": mask";

        auto positioned = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(
            tensor_logits, 0.0F, 7, /* seed_axes */ {}, tensor_mask, make_positions(positions)));
        EXPECT_EQ(positioned, greedy_reference(logits, mask, positions)) << what << ": mask + positions";
    }
}

TEST_F(GumbelSampleOpTest, TestSamplingGreedyMatchesTtnnArgmax) {
    // The op's documented contract is that greedy output is identical to
    // ttnn::argmax(dim=3, keepdim=true) -- this pins it ON DEVICE, against the real op, per dtype.
    // Random logits catch broad disagreements; the fp32 section additionally plants NEAR-TIES:
    // column pairs whose gap (2^-20 at magnitude 1) is far below TF32's ~2^-10 resolution, with the
    // true winner at the HIGHER column index. Both ops break genuine ties toward the lowest index, so the
    // constructions below are unambiguous either way.
    constexpr uint32_t kBatch = 2U;
    constexpr uint32_t kTokens = 37U;  // ragged, so the comparison spans partly-padded tile rows
    constexpr uint32_t kVocab = 77U;   // ragged vocab: both ops must stop their scan mid-tile

    auto* device = &ttml::autograd::ctx().get_device();

    auto check_against_argmax = [&](const ttnn::Tensor& tensor_logits, const char* what) {
        auto greedy = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(tensor_logits, 0.0F, 7));
        auto argmax = ttml::core::to_vector<uint32_t>(ttnn::argmax(tensor_logits, /* dim */ 3, /* keepdim */ true));
        ASSERT_EQ(greedy.size(), argmax.size()) << what;
        EXPECT_EQ(greedy, argmax) << what;
    };

    // Random logits, both supported dtypes. For bf16 the quantization creates genuine exact ties,
    // so this also covers agreement on the lowest-index tie-break.
    const xt::xarray<float> random_logits = ttml::test_utils::make_uniform_xarray<float>(
        xt::xarray<float>::shape_type{kBatch, 1U, kTokens, kVocab}, -2.0F, 2.0F, 4242U);
    check_against_argmax(ttml::core::from_xtensor(random_logits, device), "bf16, random");
    check_against_argmax(
        ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(random_logits, device), "fp32, random");

    // fp32 near-ties (fp32-only: 1 + 2^-20 is not representable in bf16, so a bf16 build of this
    // grid would collapse the pair into a genuine tie and test nothing). Assert both ops against
    // the CONSTRUCTED winner, not just against each other -- two implementations agreeing on the
    // wrong column would satisfy a pure A/B comparison.
    xt::xarray<float> near_ties = xt::zeros<float>(xt::xarray<float>::shape_type{kBatch, 1U, kTokens, kVocab});
    near_ties.fill(-1.0F);
    std::vector<uint32_t> expected(kBatch * kTokens);
    for (uint32_t b = 0; b < kBatch; ++b) {
        for (uint32_t t = 0; t < kTokens; ++t) {
            const uint32_t loser = (b * kTokens + t) % (kVocab / 2U);
            const uint32_t winner = loser + (kVocab / 2U);  // always the higher column of the pair
            near_ties(b, 0, t, loser) = 1.0F;
            near_ties(b, 0, t, winner) = 1.0F + 0x1p-20F;
            expected[b * kTokens + t] = winner;
        }
    }
    auto near_tie_tensor = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(near_ties, device);
    auto greedy = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(near_tie_tensor, 0.0F, 7));
    EXPECT_EQ(greedy, expected) << "fp32 near-ties: greedy gumbel must resolve sub-TF32 gaps exactly";
    auto argmax = ttml::core::to_vector<uint32_t>(ttnn::argmax(near_tie_tensor, /* dim */ 3, /* keepdim */ true));
    EXPECT_EQ(argmax, expected) << "fp32 near-ties: ttnn::argmax must resolve them too (reference sanity)";
}

TEST_F(GumbelSampleOpTest, TestSamplingAtPerRowPositions) {
    // Prefill wants ONE token per sequence, taken at that sequence's own prompt end -- a different
    // row for every batch entry. Passing those positions makes the op read only the tiles holding
    // them and return [B, 1, 1, 1]. What has to hold is that the shortcut is exactly equivalent:
    // sampling at position p must give what sampling everything would have given at row p.
    //
    // The shapes are deliberately ragged (tokens and vocab both mid-tile) and the positions are
    // spread across tile rows -- the first tile row, a middle one, and the partly-padded last one --
    // because the position picks BOTH the source page and the row inside that tile, and a wrong
    // row/tile split would still land on a real row for tile-aligned positions.
    constexpr uint32_t kBatch = 3U;
    constexpr uint32_t kTokens = 70U;  // Ht = 3; the last tile row has 6 valid rows
    constexpr uint32_t kVocab = 77U;   // Wt = 3; the last tile has 13 valid columns

    const std::vector<uint32_t> positions = {0U, 45U, 69U};  // tile rows 0, 1, 2; rows 0, 13, 5

    const auto winners = make_winner_logits(kBatch, kTokens, kVocab, /* stride */ 11U);
    const auto& expected_all = winners.expected;
    std::vector<uint32_t> expected_at_positions(kBatch);
    for (uint32_t b = 0; b < kBatch; ++b) {
        expected_at_positions[b] = expected_all[b * kTokens + positions[b]];
    }

    auto run = [&](const ttnn::Tensor& tensor_a, const char* what) {
        // Sample everything first. Besides producing the reference, this seeds the program cache with
        // the no-positions program: the positioned call that follows has the same shapes, dtype and
        // mask-ness, so it collides with it unless the cache key knows about positions -- and a
        // collision would reuse a program whose output is [B, 1, tokens, 1].
        auto greedy_all = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(tensor_a, 0.0F, 7));
        ASSERT_EQ(greedy_all.size(), kBatch * kTokens) << what;
        EXPECT_EQ(greedy_all, expected_all) << what;

        auto greedy_at = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(
            tensor_a, 0.0F, 7, /* seed_axes */ {}, /* mask */ std::nullopt, make_positions(positions)));
        ASSERT_EQ(greedy_at.size(), kBatch) << what << ": one sampled id per batch entry, not per token";
        EXPECT_EQ(greedy_at, expected_at_positions) << what;

        // The scan bounds have to hold in the sampled kernel too, which is a separate binary
        // (the noise compile-time arg). The noise makes the winner unpredictable, but every real logit here is
        // negative while from_xtensor zero-fills the padding, so any index that leaves the logical
        // vocabulary means the scan walked into padding.
        auto sampled_at = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(
            tensor_a, 1.0F, 99, /* seed_axes */ {}, /* mask */ std::nullopt, make_positions(positions)));
        ASSERT_EQ(sampled_at.size(), kBatch) << what;
        for (uint32_t b = 0; b < kBatch; ++b) {
            EXPECT_LT(sampled_at[b], kVocab) << what << ": batch entry " << b << " left the logical vocabulary";
        }

        // Every entry pointed at the SAME row exercises the other extreme of the work split: all three
        // entries now read the same tile row of their own shard, and the boundary-merge path sees three
        // groups that each span whatever cores the split handed them.
        const std::vector<uint32_t> uniform(kBatch, kTokens - 1U);
        auto greedy_uniform = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(
            tensor_a, 0.0F, 7, /* seed_axes */ {}, /* mask */ std::nullopt, make_positions(uniform)));
        ASSERT_EQ(greedy_uniform.size(), kBatch) << what;
        for (uint32_t b = 0; b < kBatch; ++b) {
            EXPECT_EQ(greedy_uniform[b], expected_all[b * kTokens + (kTokens - 1U)]) << what;
        }
    };

    // Both dtypes: the position-mode page walk and the single-row writer path run against a bf16
    // program (tile copies through SrcA) and a FLOAT32 program (unpack-to-dest); dtype is in the
    // program-cache key, so the cache-collision setup above is staged once per dtype too.
    auto* device = &ttml::autograd::ctx().get_device();
    run(ttml::core::from_xtensor(winners.logits, device), "bf16");
    run(ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(winners.logits, device), "fp32");
}

TEST_F(GumbelSampleOpTest, TestSamplingAtPerRowPositionsAcrossTokenCounts) {
    // With positions supplied, the program is independent of the token dimension -- the work split
    // is one tile row per batch entry however many tokens the logits carry -- so the cache key
    // normalizes that dimension away and ONE program serves every prompt length. That is what makes
    // prefill affordable: a GRPO rollout rounds its prompts to a new length most generates, and each
    // distinct length used to cost a fresh JIT build of all three kernels (~6 s, against ~3 ms for
    // the dispatch itself).
    //
    // The price is that the second call below reuses the first call's program with only its RUNTIME
    // args patched -- and Ht is now one of those. If it is not re-applied, the reader resolves a
    // batch entry's tile row as entry * Ht_stale + position / 32, which for a later entry still
    // lands inside that entry's own data and inside the buffer: a real token row, no fault, just
    // the wrong one. So this test needs all three of: two different token counts, a position in the
    // last tile row, and assertions on a LATER batch entry. Entry 0 cannot see the bug at all,
    // because entry * Ht is zero whatever Ht is.
    constexpr uint32_t kBatch = 3U;
    constexpr uint32_t kVocab = 77U;

    auto run = [](uint32_t tokens, const std::vector<uint32_t>& positions) {
        const auto [a, expected_all] = make_winner_logits(kBatch, tokens, kVocab, /* stride */ 11U);
        auto tensor = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());
        auto got = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(
            tensor, 0.0F, 7, /* seed_axes */ {}, /* mask */ std::nullopt, make_positions(positions)));

        std::vector<uint32_t> expected(kBatch);
        for (uint32_t b = 0; b < kBatch; ++b) {
            expected[b] = expected_all[b * tokens + positions[b]];
        }
        return std::pair<std::vector<uint32_t>, std::vector<uint32_t>>{std::move(got), std::move(expected)};
    };

    // Builds the program at Ht = 3 (70 tokens pad to 96).
    const auto first = run(70U, {0U, 37U, 69U});
    ASSERT_EQ(first.first.size(), kBatch);
    EXPECT_EQ(first.first, first.second) << "70-token call";

    // Same batch and vocabulary, so under the normalized key this reuses the program above -- but it
    // needs Ht = 5 (134 tokens pad to 160). Entry 2 at token 133 belongs to tile row 14; replayed
    // with the stale Ht it would resolve to row 10, which is still entry 2 and still in bounds, but
    // holds token 0.
    const auto second = run(134U, {0U, 70U, 133U});
    ASSERT_EQ(second.first.size(), kBatch);
    EXPECT_EQ(second.first, second.second) << "134-token call reusing the 70-token program";
}

TEST_F(GumbelSampleOpTest, TestSamplingAtPerRowPositionsLargeBatch) {
    // Positions live in a small tensor each core stages into L1, so the batch is bounded only by memory.
    // This test also exercises the positions CB at a size where an off-by-one in its bound would trip watcher.
    constexpr uint32_t kBatch = 512U;
    constexpr uint32_t kTokens = 64U;
    constexpr uint32_t kVocab = 33U;

    xt::xarray<float>::shape_type shape = {kBatch, 1U, kTokens, kVocab};
    xt::xarray<float> a = xt::zeros<float>(shape);
    a.fill(-1.0F);

    std::vector<uint32_t> positions(kBatch);
    std::vector<uint32_t> expected(kBatch);
    for (uint32_t b = 0; b < kBatch; ++b) {
        positions[b] = (b * 13U) % kTokens;
        const uint32_t winner = (b * 7U) % kVocab;
        a(b, 0, positions[b], winner) = -0.5F;
        expected[b] = winner;
    }

    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());
    auto got = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(
        tensor_a, 0.0F, 7, /* seed_axes */ {}, /* mask */ std::nullopt, make_positions(positions)));
    ASSERT_EQ(got.size(), kBatch);
    EXPECT_EQ(got, expected);
}

TEST_F(GumbelSampleOpTest, TestSamplingRepatchesPositionsBufferOnCacheHit) {
    // The positions BUFFER ADDRESS is a runtime arg, and every prefill builds a new tensor. A cached
    // program replayed against a stale address reads whatever DRAM now occupies that region: in
    // bounds, no fault, a plausible-looking token. Nothing else in this file varies the positions
    // buffer across two calls that share a program, so a dropped re-patch passes every other test.
    //
    // The first tensor is deallocated before the second is built, so the allocator is likely to hand
    // back the same region -- which is exactly the case where a stale address looks healthy.
    constexpr uint32_t kBatch = 4U;
    constexpr uint32_t kTokens = 96U;
    constexpr uint32_t kVocab = 40U;

    const auto [a, winner_at] = make_winner_logits(kBatch, kTokens, kVocab, /* stride */ 3U);
    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());

    auto sample_at = [&](const std::vector<uint32_t>& positions) {
        auto positions_tt = make_positions(positions);
        auto got = ttml::core::to_vector<uint32_t>(
            ttml::metal::gumbel_sample(tensor_a, 0.0F, 7, /* seed_axes */ {}, /* mask */ std::nullopt, positions_tt));
        positions_tt.deallocate(/* force */ true);
        return got;
    };

    const std::vector<uint32_t> first_positions = {0U, 31U, 64U, 95U};
    const std::vector<uint32_t> second_positions = {95U, 64U, 31U, 0U};  // same shapes, different values

    auto first = sample_at(first_positions);
    auto second = sample_at(second_positions);

    for (uint32_t b = 0; b < kBatch; ++b) {
        EXPECT_EQ(first[b], winner_at[b * kTokens + first_positions[b]]) << "first call, entry " << b;
        EXPECT_EQ(second[b], winner_at[b * kTokens + second_positions[b]])
            << "second call reused the first call's program, entry " << b;
    }
}

TEST_F(GumbelSampleOpTest, TestSamplingClampsOutOfRangePosition) {
    // Positions live in device memory, so the host cannot range-check them on the dispatch path.
    // The kernels clamp each position to the LAST REAL TOKEN -- reader and writer apply the same
    // clamp to the value before splitting it into their disjoint bit fields (>> 5 and & 31), so
    // the expectations here are EXACT row identities, not just "stayed inside the vocabulary".
    // Two failure modes hide behind that weaker check:
    //
    //  * A position in the tile-padding band [tokens, Ht*32) -- e.g. the classic off-by-one of
    //    position == prompt length on a mid-tile prompt -- passes any tile-row-only bound and
    //    lands on a ZERO-FILLED padding row. Greedy argmax over zeros returns token 0: in range,
    //    silently wrong. kTokens is deliberately mid-tile so the band exists at all; a
    //    tile-aligned token count has none (which is why this test previously could not catch it).
    //  * A position past Ht*32 resolves to a page outside the logits buffer entirely: interleaved
    //    accessors bounds-check nothing, and watcher validates the whole DRAM window rather than
    //    the buffer.
    //
    // Every winner below is nonzero, so a scan of a zeroed padding row (which returns 0) can never
    // masquerade as a pass.
    if (std::getenv("TT_METAL_WATCHER") != nullptr) {
        GTEST_SKIP() << "out-of-range positions deliberately trip the kernels' watcher ASSERT; "
                        "under watcher the loud path replaces the clamp being tested here";
    }

    constexpr uint32_t kBatch = 4U;
    constexpr uint32_t kTokens = 70U;  // Ht = 3, mid-tile: the padding band is [70, 96)
    constexpr uint32_t kVocab = 40U;

    const auto [a, winner_at] = make_winner_logits(kBatch, kTokens, kVocab, /* stride */ 5U);
    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());

    // Entry 0: in-range control (the last real token itself -- the clamp must not disturb it).
    // Entry 1: the off-by-one, first value of the padding band; SAME tile row as real data, so a
    //          tile-row-only clamp passes it straight through to the padding row.
    // Entry 2: deep in the padding band, still inside the last real tile.
    // Entry 3: far past the padded extent, in a tile row that does not exist.
    const std::vector<uint32_t> positions = {kTokens - 1U, kTokens, 90U, 10U * kTokens};
    auto got = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(
        tensor_a, 0.0F, 7, /* seed_axes */ {}, /* mask */ std::nullopt, make_positions(positions)));
    ASSERT_EQ(got.size(), kBatch);

    EXPECT_EQ(got[0], winner_at[0 * kTokens + (kTokens - 1U)]) << "in-range entry disturbed by the clamp";
    for (uint32_t b = 1; b < kBatch; ++b) {
        EXPECT_EQ(got[b], winner_at[b * kTokens + (kTokens - 1U)])
            << "entry " << b << " (position " << positions[b] << ") must clamp to the last real token's row";
    }
}

TEST_F(GumbelSampleOpTest, TestSamplingRejectsOutOfRangeSeedAxis) {
    // seeded_linear_index() skips mesh axes it cannot find, so before this was validated an
    // out-of-range seed axis (a typo, or a config reused across mesh topologies) silently degraded
    // to "no axis seeded": every data-parallel device drew byte-identical noise and a GRPO rollout
    // emitted duplicate completions with zero-variance advantages. The op must reject it loudly
    // instead. Axis 7 is out of range on any mesh this suite runs on.
    xt::xarray<float>::shape_type shape = {1, 1, 32, 64};
    xt::xarray<float> a = ttml::test_utils::make_uniform_xarray<float>(shape, 0.0F, 1.0F, 42U);
    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());

    EXPECT_ANY_THROW(ttml::metal::gumbel_sample(tensor_a, 1.0F, 42, /* seed_axes */ std::vector<uint32_t>{7U}));
}

TEST_F(GumbelSampleOpTest, TestSamplingWithoutPositionsUnchangedByAccessorChain) {
    // The non-position path gained a NULL TensorAccessorArgs append so the accessor chain's length is
    // the same in both modes. If that append is dropped, or the hard-coded offsets drift, the next
    // accessor misdecodes its page size as the config flags -- silently, not as a build error. Cover
    // the reader chain at its new length both with and without a mask.
    constexpr uint32_t kBatch = 2U;
    constexpr uint32_t kTokens = 37U;
    constexpr uint32_t kVocab = 77U;

    const auto [a, expected] = make_winner_logits(kBatch, kTokens, kVocab, /* stride */ 7U);
    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());

    auto no_mask = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(tensor_a, 0.0F, 7));
    ASSERT_EQ(no_mask.size(), kBatch * kTokens) << "no-positions output must stay [B, 1, tokens, 1]";
    EXPECT_EQ(no_mask, expected);

    // A mask sits between the logits and positions accessors in the chain, so it is the case where a
    // length mismatch shows up.
    xt::xarray<float> m = xt::zeros<float>(xt::xarray<float>::shape_type{1U, 1U, 1U, kVocab});
    auto mask = ttml::core::from_xtensor(m, &ttml::autograd::ctx().get_device());
    auto with_mask =
        ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(tensor_a, 0.0F, 7, /* seed_axes */ {}, mask));
    ASSERT_EQ(with_mask.size(), kBatch * kTokens);
    EXPECT_EQ(with_mask, expected) << "an all-zero mask must not change the result";
}

TEST_F(GumbelSampleOpTest, TestSamplingPerRowMask) {
    // A [B, 1, 1, V] mask gives each batch entry its own bias row (per-request logit bias / banned
    // ids), broadcast down token positions -- served by the same program as the shared [1, 1, 1, V]
    // mask via a runtime page stride. Each entry bans a DIFFERENT column, so reading another entry's
    // mask row (the stride bug this test exists to catch: wrong entry -> wrong page, in bounds, no
    // fault) changes which token wins.
    constexpr uint32_t kBatch = 3U;
    constexpr uint32_t kTokens = 70U;  // Ht = 3, so entry != tile_row: the entry derivation is exercised
    constexpr uint32_t kVocab = 77U;   // Wt = 3, ragged last tile

    xt::xarray<float>::shape_type shape = {kBatch, 1U, kTokens, kVocab};
    xt::xarray<float> a = xt::zeros<float>(shape);
    a.fill(-1.0F);

    // Per (entry, token): best column b+1, runner-up 0. Entry e's mask bans column e+1, so with the
    // mask the winner must flip to 0 for ALL tokens of that entry -- but only that entry's rows.
    for (uint32_t b = 0; b < kBatch; ++b) {
        for (uint32_t t = 0; t < kTokens; ++t) {
            a(b, 0, t, b + 1) = -0.25F;
            a(b, 0, t, 0) = -0.5F;
        }
    }
    xt::xarray<float> m = xt::zeros<float>(xt::xarray<float>::shape_type{kBatch, 1U, 1U, kVocab});
    for (uint32_t b = 0; b < kBatch; ++b) {
        m(b, 0, 0, b + 1) = 1e4F;
    }

    auto* device = &ttml::autograd::ctx().get_device();
    auto tensor_a = ttml::core::from_xtensor(a, device);
    auto tensor_m = ttml::core::from_xtensor(m, device);

    // Unmasked: entry b picks b+1 everywhere (sanity that the setup is what we think).
    auto greedy = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(tensor_a, 0.0F, 7));
    ASSERT_EQ(greedy.size(), kBatch * kTokens);
    for (uint32_t b = 0; b < kBatch; ++b) {
        for (uint32_t t = 0; t < kTokens; ++t) {
            ASSERT_EQ(greedy[b * kTokens + t], b + 1) << "unmasked winner, entry " << b << " token " << t;
        }
    }

    // A SHARED all-zero [1, 1, 1, V] mask first, at the same logits shape. This is not a smoke
    // call: it seeds the program cache with the mask-present program built at stride 0, so the
    // per-row call below is a CACHE HIT that only works if override_runtime_arguments re-patches
    // the stride (0 -> Wt). With a stale stride every entry reads entry 0's mask row, so entries
    // 1..B-1 keep their unbanned winners -- a deterministic failure.
    xt::xarray<float> shared = xt::zeros<float>(xt::xarray<float>::shape_type{1U, 1U, 1U, kVocab});
    auto tensor_shared = ttml::core::from_xtensor(shared, device);
    auto shared_greedy = ttml::core::to_vector<uint32_t>(
        ttml::metal::gumbel_sample(tensor_a, 0.0F, 7, /* seed_axes */ {}, tensor_shared));
    ASSERT_EQ(shared_greedy.size(), kBatch * kTokens);
    for (uint32_t b = 0; b < kBatch; ++b) {
        ASSERT_EQ(shared_greedy[b * kTokens], b + 1) << "zero shared mask must not change the winner";
    }

    // Per-row masked: every entry's own winner is banned, so 0 must win everywhere -- and if entry
    // e were served entry f's mask row (f != e), e's winner e+1 would survive and this fails.
    auto masked =
        ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(tensor_a, 0.0F, 7, /* seed_axes */ {}, tensor_m));
    ASSERT_EQ(masked.size(), kBatch * kTokens);
    for (uint32_t b = 0; b < kBatch; ++b) {
        for (uint32_t t = 0; t < kTokens; ++t) {
            EXPECT_EQ(masked[b * kTokens + t], 0U) << "per-row mask missed entry " << b << " token " << t;
        }
    }

    // Back to the shared mask on the same program: the reverse stride re-patch (Wt -> 0). A stale
    // Wt stride here sends entries past the shared mask's Wt pages, so the winners it produces are
    // garbage-dependent rather than deterministic -- the assertion still holds on a correct patch
    // and the forward (0 -> Wt) direction above is the deterministic guard on the patch line.
    auto shared_again = ttml::core::to_vector<uint32_t>(
        ttml::metal::gumbel_sample(tensor_a, 0.0F, 7, /* seed_axes */ {}, tensor_shared));
    ASSERT_EQ(shared_again.size(), kBatch * kTokens);
    for (uint32_t b = 0; b < kBatch; ++b) {
        EXPECT_EQ(shared_again[b * kTokens], b + 1) << "stride must re-patch back to 0 for a shared mask";
    }

    // The per-row mask must also hold under NOISE (a separate kernel binary): the banned column
    // carries -1e4 after the subtract, so it can never win whatever the Gumbel draw.
    auto sampled =
        ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(tensor_a, 1.0F, 99, /* seed_axes */ {}, tensor_m));
    ASSERT_EQ(sampled.size(), kBatch * kTokens);
    for (uint32_t b = 0; b < kBatch; ++b) {
        for (uint32_t t = 0; t < kTokens; ++t) {
            EXPECT_NE(sampled[b * kTokens + t], b + 1) << "banned column sampled, entry " << b;
            EXPECT_LT(sampled[b * kTokens + t], kVocab);
        }
    }

    // Same mask through POSITION mode: entry derivation there is virtual_tile / Wt, a different
    // code path from the tile-row derivation above.
    const std::vector<uint32_t> positions = {0U, 37U, 69U};
    auto positioned = ttml::core::to_vector<uint32_t>(
        ttml::metal::gumbel_sample(tensor_a, 0.0F, 7, /* seed_axes */ {}, tensor_m, make_positions(positions)));
    ASSERT_EQ(positioned.size(), kBatch);
    for (uint32_t b = 0; b < kBatch; ++b) {
        EXPECT_EQ(positioned[b], 0U) << "per-row mask in position mode, entry " << b;
    }
}

TEST_F(GumbelSampleOpTest, TestSamplingGumbelMatchesSoftmaxDistribution) {
    // The Gumbel-max trick guarantees P(argmax == i) == softmax(logits / temperature)_i. This is the
    // only assertion in the suite that actually pins down the -log(-log(U)) chain: dropping or
    // reordering a step still yields in-range indices, so the shape and bounds checks stay green.
    //
    // The shape is deliberately awkward on every axis, so the distribution has to survive the same
    // padding and page arithmetic the rest of the op relies on:
    //   kBatch = 2       -> results are gathered across batch entries (page = batch * tokens + t)
    //   kRows  % 32 = 1  -> the last of 33 tile rows per batch entry contributes a SINGLE sample
    //   kVocab % 32 = 24 -> the last of 4 vocab tiles is partly padding
    // kBatch * kRows * seeds still totals 8200 samples, so the tolerances below are unchanged.
    constexpr uint32_t kBatch = 2;
    constexpr uint32_t kRows = 1025;
    constexpr uint32_t kVocab = 120;
    constexpr uint32_t kActive = 4;
    // The active columns sit in FOUR DIFFERENT vocab tiles and in both half-faces of a tile (a tile
    // is 32 wide and splits into 16-column faces). With every weight packed into column 0..3 they
    // would all land in one face of one tile, and a running argmax that failed to carry its maximum
    // across tile or face boundaries would still pass.
    //   5 -> tile 0 face 0 | 52 -> tile 1 face 1 | 70 -> tile 2 face 0 | 115 -> tile 3 face 1
    constexpr std::array<uint32_t, kActive> kActiveCols = {5U, 52U, 70U, 115U};
    // Unnormalized weights on those columns; every other column is pushed far enough down that the
    // bounded noise can never lift it (-60 + 16.6 is still well below 0 - 3.1).
    constexpr std::array<float, kActive> kWeights = {8.0F, 4.0F, 2.0F, 1.0F};
    constexpr float kWeightTotal = 15.0F;
    constexpr float kSuppressed = -60.0F;

    xt::xarray<float>::shape_type shape = {kBatch, 1, kRows, kVocab};
    xt::xarray<float> a = xt::zeros<float>(shape);
    a.fill(kSuppressed);
    for (uint32_t b = 0; b < kBatch; ++b) {
        for (uint32_t r = 0; r < kRows; ++r) {
            for (uint32_t c = 0; c < kActive; ++c) {
                a(b, 0, r, kActiveCols[c]) = std::log(kWeights[c]);
            }
        }
    }
    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());

    // Reverse map so a sampled column can be attributed to its weight.
    std::vector<int> col_to_slot(kVocab, -1);
    for (uint32_t c = 0; c < kActive; ++c) {
        col_to_slot[kActiveCols[c]] = static_cast<int>(c);
    }

    // Five sigma on Binomial(total, p): flakes at ~1e-6 per column, while a broken Gumbel chain
    // moves these counts by tens of sigma. Shared by the every-row and the positions+mask paths --
    // both draw the same number of samples, so the bounds are identical.
    //
    // LIMITATION: this bound cannot see the approximate-log BIAS.
    auto expect_counts_match_weights =
        [&](const std::array<uint32_t, kActive>& counts, uint32_t total, const char* what) {
            for (uint32_t c = 0; c < kActive; ++c) {
                const double p = static_cast<double>(kWeights[c]) / kWeightTotal;
                const double expected_count = p * total;
                const double tolerance = 5.0 * std::sqrt(total * p * (1.0 - p));
                EXPECT_NEAR(static_cast<double>(counts[c]), expected_count, tolerance)
                    << what << ": column " << kActiveCols[c] << " (weight " << kWeights[c] << ") selected " << counts[c]
                    << " / " << total;
            }
        };

    // Pool several seeds so the result does not hinge on the internal structure of one RNG stream.
    const std::vector<uint32_t> seeds = {1U, 2U, 3U, 4U};
    std::array<uint32_t, kActive> counts{};
    uint32_t total_samples = 0U;
    for (auto seed : seeds) {
        auto picks = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(tensor_a, 1.0F, seed));
        ASSERT_EQ(picks.size(), kBatch * kRows);
        for (auto pick : picks) {
            // Past kVocab is tile padding, which from_xtensor zero-fills -- reaching it would beat
            // the weight-1 column outright, so this also guards the ragged-width scan bound.
            ASSERT_LT(pick, kVocab) << "sampled index left the logical vocabulary";
            ASSERT_NE(col_to_slot[pick], -1) << "sampled column " << pick << ", whose logit was " << kSuppressed;
            ++counts[static_cast<uint32_t>(col_to_slot[pick])];
            ++total_samples;
        }
    }

    expect_counts_match_weights(counts, total_samples, "every-row path");

    // A nonzero seed is contractually reproducible, and distinct seeds must actually decorrelate --
    // both are properties the fused in-place chain could silently break.
    auto first = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(tensor_a, 1.0F, 1234U));
    auto again = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(tensor_a, 1.0F, 1234U));
    EXPECT_EQ(first, again) << "same seed must reproduce the same samples";
    auto other = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(tensor_a, 1.0F, 5678U));
    EXPECT_NE(first, other) << "different seeds must produce different samples";

    // ---- the same guarantee through the positions + mask path ----
    //
    // Position mode is a different program: its own work split (NC * Wt virtual tiles), its own RNG
    // stream layout, and the single-row writer path. Nothing above proves the distribution survives
    // it, so it is re-proven here with the BATCH as the sample axis: 2050 entries x 4 seeds is the
    // same 8200 samples, so the 5-sigma bounds carry over unchanged.
    //
    // Two tripwires ride along:
    //   * every NON-target row holds its mass on a sentinel column instead of the four weighted
    //     ones, so reading the wrong row samples the sentinel almost surely and trips col_to_slot;
    //   * a decoy column outweighs every active column but is suppressed by the padding mask, so a
    //     dropped or misapplied mask hands the decoy ~98.5% of the samples and shreds every bound.
    constexpr uint32_t kPosBatch = 2050U;
    constexpr uint32_t kPosTokens = 70U;          // Ht = 3; the last tile row keeps only 6 real rows
    constexpr uint32_t kSentinelCol = 20U;        // tile 0, face 1 -- not an active column
    constexpr uint32_t kDecoyCol = 100U;          // tile 3, face 0 -- masked below
    const float decoy_logit = std::log(1000.0F);  // outranks log(8) by far more than the noise span

    xt::xarray<float>::shape_type pos_shape = {kPosBatch, 1, kPosTokens, kVocab};
    xt::xarray<float> pos_logits = xt::zeros<float>(pos_shape);
    pos_logits.fill(kSuppressed);
    std::vector<uint32_t> entry_positions(kPosBatch);
    for (uint32_t b = 0; b < kPosBatch; ++b) {
        entry_positions[b] = (b * 13U) % kPosTokens;  // 13 is coprime with 70: every row gets hit
        for (uint32_t t = 0; t < kPosTokens; ++t) {
            if (t == entry_positions[b]) {
                for (uint32_t c = 0; c < kActive; ++c) {
                    pos_logits(b, 0, t, kActiveCols[c]) = std::log(kWeights[c]);
                }
                pos_logits(b, 0, t, kDecoyCol) = decoy_logit;
            } else {
                pos_logits(b, 0, t, kSentinelCol) = 0.0F;
            }
        }
    }

    xt::xarray<float>::shape_type mask_shape = {1, 1, 1, kVocab};
    xt::xarray<float> mask = xt::zeros<float>(mask_shape);
    mask(0, 0, 0, kDecoyCol) = 1e4F;

    auto* device = &ttml::autograd::ctx().get_device();
    auto pos_tensor = ttml::core::from_xtensor(pos_logits, device);
    auto mask_tensor = ttml::core::from_xtensor(mask, device);
    auto positions_tensor = make_positions(entry_positions);

    std::array<uint32_t, kActive> pos_counts{};
    uint32_t pos_total = 0U;
    for (auto seed : seeds) {
        auto picks = ttml::core::to_vector<uint32_t>(
            ttml::metal::gumbel_sample(pos_tensor, 1.0F, seed, /* seed_axes */ {}, mask_tensor, positions_tensor));
        ASSERT_EQ(picks.size(), kPosBatch);
        for (auto pick : picks) {
            ASSERT_LT(pick, kVocab) << "sampled index left the logical vocabulary";
            ASSERT_NE(col_to_slot[pick], -1) << "sampled column " << pick << ": a wrong row (sentinel " << kSentinelCol
                                             << "), an unmasked decoy (" << kDecoyCol << "), or a suppressed column";
            ++pos_counts[static_cast<uint32_t>(col_to_slot[pick])];
            ++pos_total;
        }
    }

    expect_counts_match_weights(pos_counts, pos_total, "positions+mask");
}

TEST_F(GumbelSampleOpTest, TestSamplingWideRowManyOwners) {
    // The other shapes in this suite split a row over at most a few cores, so a row's owner merges
    // at most ~3 foreign records. A single wide tile row spread over the whole grid is the other
    // extreme: every core holding a shard of the row sends a record to the one owner, so the
    // owner's exact-count semaphore wait, the host-assigned slot addressing and the merge loop all
    // run at grid-scale fan-in.
    constexpr uint32_t kTokens = 2U;    // one tile row (Ht = 1); both real rows ride the same merge
    constexpr uint32_t kVocab = 4100U;  // Wt = 129; the last tile keeps 4 valid columns

    xt::xarray<float>::shape_type shape = {1U, 1U, kTokens, kVocab};
    xt::xarray<float> a = xt::zeros<float>(shape);
    a.fill(-1.0F);

    // Distinct winner per token row, in different vocab tiles -- one mid-run, one inside the
    // ragged final tile -- so a merge that drops, duplicates or mis-slots records cannot pass by
    // coincidence, and the ragged scan bound is exercised through the merge path too.
    constexpr std::array<uint32_t, kTokens> kWinners = {1234U, kVocab - 1U};
    std::vector<uint32_t> expected(kTokens);
    for (uint32_t t = 0; t < kTokens; ++t) {
        a(0, 0, t, kWinners[t]) = -0.5F;
        expected[t] = kWinners[t];
    }

    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());

    // Greedy is exact, so the winners must come back verbatim through the full-fan-in merge.
    auto greedy = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(tensor_a, 0.0F, 7));
    ASSERT_EQ(greedy.size(), kTokens);
    EXPECT_EQ(greedy, expected);

    // Position mode reruns the same wide-row merge under its own work split (NC * Wt virtual
    // tiles) and the single-row writer path. One batch entry means one position per call; select
    // each row in turn.
    for (uint32_t t = 0; t < kTokens; ++t) {
        auto at = ttml::core::to_vector<uint32_t>(ttml::metal::gumbel_sample(
            tensor_a, 0.0F, 7, /* seed_axes */ {}, /* mask */ std::nullopt, make_positions(std::vector<uint32_t>{t})));
        ASSERT_EQ(at.size(), 1U);
        EXPECT_EQ(at[0], expected[t]) << "position " << t;
    }
}

namespace {

// gumbel_sfpu.h's approximate log, re-derived on the host. The four constants come straight from
// gumbel_sample_constants.hpp -- the same header the TRISC pass compiles against -- so the
// polynomial whose invariants are pinned below is, by construction, the one the kernel runs.
constexpr float kApproxNegLogLn2 = ttml::metal::sfpu::kGumbelNegLn2;
constexpr float kApproxLogB = ttml::metal::sfpu::kGumbelPolyB;
constexpr float kApproxLogC = ttml::metal::sfpu::kGumbelPolyC;
constexpr float kApproxLogD = ttml::metal::sfpu::kGumbelPolyD;

// The NEGATED mantissa polynomial q(m) = m*(m*B + C) + D = -p(m) on the octave [1, 2), in double.
// The kernel returns -ln directly (negation folded into the constants), so the mirror does too.
double approx_neg_log_poly(double m) {
    return m * (m * static_cast<double>(kApproxLogB) + static_cast<double>(kApproxLogC)) +
           static_cast<double>(kApproxLogD);
}

// The full approximation -log(v) ~= e*(-ln2) + q(m) for v = m * 2^e, m in [1, 2), matching the
// setexp/exexp split the SFPI pass performs.
template <typename T>
T approx_neg_log(T v) {
    int exponent = 0;
    const T half_mantissa = std::frexp(v, &exponent);  // v = half_mantissa * 2^exponent, in [0.5, 1)
    const T m = half_mantissa * T(2);
    const T e = static_cast<T>(exponent - 1);
    const T poly = m * (m * T(kApproxLogB) + T(kApproxLogC)) + T(kApproxLogD);
    return e * T(kApproxNegLogLn2) + poly;
}

}  // namespace

TEST(GumbelSfpuHostTest, TestGumbelApproxLogInvariants) {
    constexpr double kTwoPowNeg20 = 0x1p-20;

    // Endpoint ties, exact: q(1) = +2^-20 and q(2) = -ln2_c + 2^-20. These are what make
    // e*(-ln2_c) + q(m) continuous across octave boundaries, and the shared +2^-20 shift is what
    // keeps -log(U) strictly positive without a zero guard.
    EXPECT_EQ(approx_neg_log_poly(1.0), kTwoPowNeg20);
    EXPECT_EQ(approx_neg_log_poly(2.0), static_cast<double>(kApproxNegLogLn2) + kTwoPowNeg20);

    // q falls across a dense sweep of the octave, including the fp32 neighbours of both
    // endpoints, and stays within the fitted error bound of the exact -log. A monotone transform
    // of U cannot reorder samples, so this is the property that preserves argmax semantics.
    constexpr int kGridPoints = 1'000'000;
    std::vector<double> grid;
    grid.reserve(kGridPoints + 4);
    grid.push_back(1.0);
    grid.push_back(static_cast<double>(std::nextafterf(1.0F, 2.0F)));
    for (int i = 1; i < kGridPoints; ++i) {
        grid.push_back(1.0 + static_cast<double>(i) / kGridPoints);
    }
    grid.push_back(static_cast<double>(std::nextafterf(2.0F, 1.0F)));
    grid.push_back(2.0);
    std::sort(grid.begin(), grid.end());

    uint32_t monotonicity_violations = 0U;
    double max_error = 0.0;
    double prev = approx_neg_log_poly(grid.front());
    for (double m : grid) {
        const double q = approx_neg_log_poly(m);
        if (q > prev) {
            ++monotonicity_violations;
        }
        prev = q;
        if (m < 2.0) {
            max_error = std::max(max_error, std::abs(q + std::log(m)));
        }
    }
    EXPECT_EQ(monotonicity_violations, 0U) << "q(m) must be nonincreasing on [1, 2]";
    EXPECT_LE(max_error, 5.5e-3) << "|q(m) + ln(m)| left the fitted bound";

    // Across octave boundaries: the full approximation, evaluated at fp32-adjacent points spanning
    // powers of two, must be NONINCREASING (it is -log). The exponent range comfortably covers
    // everything the noise chain feeds it: U in [2^-32, 1) and -log(U) in [~1e-6, ~22].
    for (int e = -40; e <= 32; e += 8) {
        const float x = std::ldexp(1.0F, e);
        const float below = std::nextafterf(x, 0.0F);
        const float above = std::nextafterf(x, HUGE_VALF);
        EXPECT_GE(approx_neg_log<double>(static_cast<double>(below)), approx_neg_log<double>(static_cast<double>(x)))
            << "octave boundary below 2^" << e;
        EXPECT_GE(approx_neg_log<double>(static_cast<double>(x)), approx_neg_log<double>(static_cast<double>(above)))
            << "octave boundary above 2^" << e;
    }

    // Noise ceiling. At the raw upper bound 1 - 2^-24 the fused chain stays finite (the whole
    // point of bounding U below 1.0); the generator's ATTAINABLE top of range sits one fp32 step
    // lower still, because the factory shrinks rand's closed-interval scale by one ULP when
    // from + scale would round past the bound (compute_rand_scale_bits), and there the ceiling is
    // near 13.81 -- the approximate-log analogue of the exact log's ~16.6.
    const float u_raw_max = std::nextafterf(1.0F, 0.0F);  // kGumbelUniformUpperBound
    const float raw_inner = approx_neg_log<float>(u_raw_max);
    ASSERT_GT(raw_inner, 0.0F) << "-log(U) must stay strictly positive below 1.0";
    EXPECT_LE(approx_neg_log<float>(raw_inner), 13.9F);

    const float lower = 0x1p-32F;  // kGumbelUniformLowerBound
    // The factory's own one-ULP closed-interval guard, not a re-implementation of it: u_top is
    // whatever compute_rand_scale_bits actually hands the kernel as the attainable top of range.
    const float scale =
        std::bit_cast<float>(ttml::metal::ops::gumbel_sample::device::compute_rand_scale_bits(lower, u_raw_max));
    const float u_top = lower + scale;
    const float top_inner = approx_neg_log<float>(u_top);
    ASSERT_GT(top_inner, 0.0F);
    EXPECT_LE(approx_neg_log<float>(top_inner), 13.85F) << "noise ceiling left the documented ~13.81 cap";
}

TEST_F(GumbelSampleOpTest, TestSamplingPreallocatedOutput) {
    // The prim accepts a caller-preallocated output ([B, 1, tokens, 1] UINT32 ROW_MAJOR -- the
    // op's own output spec). The public gumbel_sample wrapper never passes one, so this is the one
    // test that calls ttnn::prim::ttml_gumbel_sample directly. The samples must land IN the
    // caller's buffer, not in a fresh allocation the caller never sees.
    constexpr uint32_t kBatch = 2U;
    constexpr uint32_t kTokens = 37U;  // mid-tile, so the ragged write-out is the path exercised
    constexpr uint32_t kVocab = 77U;

    const auto [a, expected] = make_winner_logits(kBatch, kTokens, kVocab, /* stride */ 7U);

    auto* device = &ttml::autograd::ctx().get_device();
    auto tensor_a = ttml::core::from_xtensor(a, device);

    auto preallocated = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        std::vector<uint32_t>(kBatch * kTokens, 0U),
        ttnn::Shape({kBatch, 1U, kTokens, 1U}),
        device,
        ttnn::Layout::ROW_MAJOR);
    const auto preallocated_address = preallocated.buffer()->address();

    auto out = ttnn::prim::ttml_gumbel_sample(
        tensor_a,
        /* temperature */ 0.0F,
        /* seed */ 7U,
        /* seed_axes */ {},
        /* logits_mask */ std::nullopt,
        /* positions */ std::nullopt,
        preallocated);

    // Greedy is exact: assert the returned tensor aliases the caller's buffer AND that reading the
    // preallocated tensor itself (not just the returned handle) sees the results.
    EXPECT_EQ(out.buffer()->address(), preallocated_address) << "output must reuse the preallocated buffer";
    EXPECT_EQ(ttml::core::to_vector<uint32_t>(out), expected);
    EXPECT_EQ(ttml::core::to_vector<uint32_t>(preallocated), expected) << "samples must land in the caller's tensor";
}
