// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The packed-SwiGLU metal ops and their autograd wrapper, mirroring tests/python/test_swiglu_packed.py so
// that the merge gate, which runs only ttml_tests, covers them. `packed` is [.., R, 2*I] = [gate | up];
// the ops read the two column halves in place:
//     h        = silu(gate) * up
//     dL/dgate = dL/dh * up * silu'(gate),   silu'(x) = s * (1 + x * (1 - s)),  s = sigmoid(x)
//     dL/dup   = dL/dh * silu(gate)
// The program-cache tests launch an op again with the same specs and fresh tensors while every earlier
// tensor stays alive, so the cache hit must read and write the new buffers.

#include "ops/swiglu_packed_op.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/swiglu_packed_bw/swiglu_packed_bw.hpp"
#include "metal/ops/swiglu_packed_fw/swiglu_packed_fw.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/random_data.hpp"

class SwigluPackedOpTest : public ttml::test_utils::DeviceSuiteTest {};

namespace {

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;
using ttml::test_utils::device_with_empty_program_cache;
using ttml::test_utils::expect_new_buffers;
using ttml::test_utils::expect_unchanged;
using ttml::test_utils::two_core_group_units;
using ttml::test_utils::zeros_on_device;

// Bounds of the python suite, whose observed worst case is 1.24 ULP. Per-element ULP is unbounded where
// the oracle vanishes: silu'(gate) -> 0 for negative gate drives dgate's per-element error to ~164 on a
// correct kernel while its p99 stays near 6.
constexpr double kMaxUlp = 2.5;
constexpr double kMaxUlpP99 = 2.5;
constexpr double kMaxUlpP99DGate = 10.0;

// One bf16 ULP at |x|, i.e. the gap between adjacent bf16 values in x's binade, as tests/python/bf16_ulp.py
// computes it: from the exponent, one binade up for mantissas that round into the next power of two, and
// no smaller than at the smallest normal.
double bf16_spacing(double x) {
    constexpr int kMantissaBits = 7;
    const double magnitude = std::max(std::abs(x), std::ldexp(1.0, -126));
    int exponent = 0;
    // magnitude = mantissa * 2^exponent, with mantissa in [0.5, 1)
    const double mantissa = std::frexp(magnitude, &exponent);
    const int binade = mantissa >= 1.0 - std::ldexp(1.0, -(kMantissaBits + 2)) ? exponent : exponent - 1;
    return std::ldexp(1.0, binade - kMantissaBits);
}

// |got - expected| in bf16 ULP, as assert_within_bf16_ulp measures it: the peak error in ULP at
// max |expected|, and the p99 of the per-element errors, each in ULP at its own element.
void expect_within_bf16_ulp(
    const xt::xarray<float>& got,
    const xt::xarray<double>& expected,
    const std::string& label,
    double max_ulp,
    double max_ulp_p99) {
    ASSERT_EQ(got.shape(), expected.shape()) << label;
    ASSERT_GT(expected.size(), 0U) << label;
    ASSERT_TRUE(xt::all(xt::isfinite(got))) << label << ": non-finite output";

    double peak_error = 0.0;
    double peak_magnitude = 0.0;
    std::vector<double> ulp_per_element(expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        const double error = std::abs(static_cast<double>(got.data()[i]) - expected.data()[i]);
        peak_error = std::max(peak_error, error);
        peak_magnitude = std::max(peak_magnitude, std::abs(expected.data()[i]));
        ulp_per_element[i] = error / bf16_spacing(expected.data()[i]);
    }
    const double ulp = peak_error / bf16_spacing(peak_magnitude);

    // numpy.percentile's default linear interpolation between the two closest ranks.
    const double rank = 0.99 * static_cast<double>(ulp_per_element.size() - 1U);
    const auto lower = static_cast<size_t>(std::floor(rank));
    std::nth_element(ulp_per_element.begin(), ulp_per_element.begin() + lower, ulp_per_element.end());
    const double lower_value = ulp_per_element[lower];
    const double upper_value = lower + 1U < ulp_per_element.size()
                                   ? *std::min_element(ulp_per_element.begin() + lower + 1, ulp_per_element.end())
                                   : lower_value;
    const double ulp_p99 = lower_value + (rank - static_cast<double>(lower)) * (upper_value - lower_value);

    EXPECT_LE(ulp, max_ulp) << label << ": peak error " << ulp << " bf16 ULP";
    EXPECT_LE(ulp_p99, max_ulp_p99) << label << ": p99 error " << ulp_p99 << " bf16 ULP";
}

xt::xarray<double> sigmoid(const xt::xarray<double>& x) {
    return 1.0 / (1.0 + xt::exp(-x));
}

xt::xarray<float> first_half(const xt::xarray<float>& packed) {
    const size_t inner = packed.shape().back() / 2U;
    return xt::view(packed, xt::all(), xt::all(), xt::all(), xt::range(0U, inner));
}

xt::xarray<float> second_half(const xt::xarray<float>& packed) {
    const size_t inner = packed.shape().back() / 2U;
    return xt::view(packed, xt::all(), xt::all(), xt::all(), xt::range(inner, 2U * inner));
}

std::vector<uint32_t> packed_shape(uint32_t batch, uint32_t seq, uint32_t inner) {
    return {batch, 1U, seq, 2U * inner};
}

// Both ops split blocks of a packed row over the cores, one block per tile-row when I = 64 (two tiles per
// half, block size 2).
constexpr uint32_t kTwoCoreGroupInner = 64U;

uint32_t two_core_group_seq() {
    return two_core_group_units(3U) * 32U;
}

struct PackedInputs {
    ttnn::Tensor packed;
    ttnn::Tensor dL_dh;
    xt::xarray<float> packed_stored;  // what the kernels read: the values rounded to bf16
    xt::xarray<float> dL_dh_stored;
};

// gate in [-4, 4] spans both silu saturation tails; up in [-2, 2]; dL/dh in [-1, 1].
PackedInputs make_inputs(uint32_t batch, uint32_t seq, uint32_t inner, uint32_t seed) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const std::vector<uint32_t> half_shape = {batch, 1U, seq, inner};
    const auto gate = test_utils::make_uniform_xarray<float>(half_shape, -4.0F, 4.0F, seed);
    const auto up = test_utils::make_uniform_xarray<float>(half_shape, -2.0F, 2.0F, seed + 1U);
    const xt::xarray<float> packed_host = xt::concatenate(xt::xtuple(gate, up), 3);

    PackedInputs in;
    in.packed = core::from_xtensor(packed_host, device);
    in.dL_dh = core::from_xtensor(test_utils::make_uniform_xarray<float>(half_shape, -1.0F, 1.0F, seed + 2U), device);
    in.packed_stored = core::to_xtensor(in.packed);
    in.dL_dh_stored = core::to_xtensor(in.dL_dh);
    return in;
}

void expect_forward_matches_reference(const ttnn::Tensor& h, const PackedInputs& in, const std::string& label) {
    const xt::xarray<double> gate = xt::cast<double>(first_half(in.packed_stored));
    const xt::xarray<double> up = xt::cast<double>(second_half(in.packed_stored));
    const xt::xarray<double> expected = gate * sigmoid(gate) * up;
    expect_within_bf16_ulp(ttml::core::to_xtensor(h), expected, label, kMaxUlp, kMaxUlpP99);
}

void expect_backward_matches_reference(
    const ttnn::Tensor& dL_dpacked, const PackedInputs& in, const xt::xarray<float>& dL_dh, const std::string& label) {
    const xt::xarray<double> gate = xt::cast<double>(first_half(in.packed_stored));
    const xt::xarray<double> up = xt::cast<double>(second_half(in.packed_stored));
    const xt::xarray<double> dh = xt::cast<double>(dL_dh);
    const xt::xarray<double> s = sigmoid(gate);
    const xt::xarray<double> expected_dgate = dh * up * (s * (1.0 + gate * (1.0 - s)));
    const xt::xarray<double> expected_dup = dh * gate * s;

    const auto got = ttml::core::to_xtensor(dL_dpacked);
    ASSERT_EQ(got.shape(), in.packed_stored.shape()) << label;
    expect_within_bf16_ulp(first_half(got), expected_dgate, label + " dgate", kMaxUlp, kMaxUlpP99DGate);
    expect_within_bf16_ulp(second_half(got), expected_dup, label + " dup", kMaxUlp, kMaxUlpP99);
}

void check_forward(uint32_t batch, uint32_t seq, uint32_t inner) {
    const auto in = make_inputs(batch, seq, inner, 100U + inner);
    const auto h = ttml::metal::swiglu_packed_fw(in.packed);
    ASSERT_EQ(h.logical_shape(), ttnn::Shape({batch, 1U, seq, inner}));
    expect_forward_matches_reference(h, in, "forward");
}

void check_backward(uint32_t batch, uint32_t seq, uint32_t inner) {
    const auto in = make_inputs(batch, seq, inner, 200U + inner);
    const auto dL_dpacked = ttml::metal::swiglu_packed_bw(in.packed, in.dL_dh);
    ASSERT_EQ(dL_dpacked.logical_shape(), ttnn::Shape({batch, 1U, seq, 2U * inner}));
    expect_backward_matches_reference(dL_dpacked, in, in.dL_dh_stored, "backward");
}

}  // namespace

// ============================================================================
// Section 1: accuracy against a float64 oracle
// ============================================================================
// I/32 of 1, 3 and 16 makes block sizes 1, 3 and 4 (4 blocks per row); the last shape has batch > 1. Each
// half has at least 8192 elements: with fewer, dgate's p99 rests on a handful of elements near
// silu'(gate) = 0, and its peak on whether max |dgate| happens to reach the next binade.

TEST_F(SwigluPackedOpTest, Forward_MatchesReference_1x256x32) {
    check_forward(1U, 256U, 32U);
}

TEST_F(SwigluPackedOpTest, Forward_MatchesReference_1x128x96) {
    check_forward(1U, 128U, 96U);
}

TEST_F(SwigluPackedOpTest, Forward_MatchesReference_2x64x512) {
    check_forward(2U, 64U, 512U);
}

TEST_F(SwigluPackedOpTest, Backward_MatchesReference_1x256x32) {
    check_backward(1U, 256U, 32U);
}

TEST_F(SwigluPackedOpTest, Backward_MatchesReference_1x128x96) {
    check_backward(1U, 128U, 96U);
}

TEST_F(SwigluPackedOpTest, Backward_MatchesReference_2x64x512) {
    check_backward(2U, 64U, 512U);
}

// ============================================================================
// Section 2: program-cache hits
// ============================================================================
// Each test clears the cache so that its first launch is the miss. Launch seeds are 10 apart because
// make_inputs draws gate, up and dL/dh from seed, seed + 1 and seed + 2. The program key does not see
// which buffer the backward's grad is, so an in-place launch (preallocated_dL_dpacked == packed) and a
// launch with a separate preallocated grad share one program; the backward tests switch between the two
// on cache hits in both orders.

TEST_F(SwigluPackedOpTest, Forward_CacheHitReadsFreshBuffers_TwoCoreGroups) {
    using namespace ttml;
    auto& device = device_with_empty_program_cache();
    const uint32_t seq = two_core_group_seq();

    const auto a = make_inputs(1U, seq, kTwoCoreGroupInner, 1U);
    const auto entries_before_miss = device.num_program_cache_entries();
    const auto a_h = metal::swiglu_packed_fw(a.packed);
    ASSERT_GT(device.num_program_cache_entries(), entries_before_miss) << "the first launch must compile a program";
    expect_forward_matches_reference(a_h, a, "launch A");
    const auto a_h_result = core::to_xtensor(a_h);

    // Same specs, new values, launch A's tensors still alive.
    const auto b = make_inputs(1U, seq, kTwoCoreGroupInner, 11U);
    const auto entries_before_hit = device.num_program_cache_entries();
    const auto b_h = metal::swiglu_packed_fw(b.packed);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit) << "launch B must reuse launch A's program";
    expect_new_buffers({a.packed, a_h}, {b.packed, b_h}, "launch B");
    expect_forward_matches_reference(b_h, b, "launch B (cache hit)");

    expect_unchanged(a.packed, a.packed_stored, "launch A packed");
    expect_unchanged(a_h, a_h_result, "launch A h");
}

TEST_F(SwigluPackedOpTest, Backward_CacheHitReadsFreshBuffers_TwoCoreGroups) {
    using namespace ttml;
    auto& device = device_with_empty_program_cache();
    const uint32_t seq = two_core_group_seq();

    const auto a = make_inputs(1U, seq, kTwoCoreGroupInner, 21U);
    const auto entries_before_miss = device.num_program_cache_entries();
    const auto a_dpacked = metal::swiglu_packed_bw(a.packed, a.dL_dh);
    ASSERT_GT(device.num_program_cache_entries(), entries_before_miss) << "the first launch must compile a program";
    expect_backward_matches_reference(a_dpacked, a, a.dL_dh_stored, "launch A");
    const auto a_dpacked_result = core::to_xtensor(a_dpacked);

    // Same specs, new values, launch A's tensors still alive.
    const auto b = make_inputs(1U, seq, kTwoCoreGroupInner, 31U);
    const auto entries_before_hit = device.num_program_cache_entries();
    const auto b_dpacked = metal::swiglu_packed_bw(b.packed, b.dL_dh);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit) << "launch B must reuse launch A's program";
    expect_new_buffers({a.packed, a.dL_dh, a_dpacked}, {b.packed, b.dL_dh, b_dpacked}, "launch B");
    expect_backward_matches_reference(b_dpacked, b, b.dL_dh_stored, "launch B (cache hit)");

    expect_unchanged(a.packed, a.packed_stored, "launch A packed");
    expect_unchanged(a.dL_dh, a.dL_dh_stored, "launch A dL_dh");
    expect_unchanged(a_dpacked, a_dpacked_result, "launch A dL_dpacked");
}

// A launch without preallocated_output, then two with one: the op must write into the preallocated
// tensor and return it, whether or not the program cache keys on its presence.
TEST_F(SwigluPackedOpTest, Forward_PreallocatedOutputAfterAllocatedOutput_TwoCoreGroups) {
    using namespace ttml;
    auto& device = device_with_empty_program_cache();
    const uint32_t seq = two_core_group_seq();
    const std::vector<uint32_t> h_shape = {1U, 1U, seq, kTwoCoreGroupInner};

    const auto a = make_inputs(1U, seq, kTwoCoreGroupInner, 41U);
    const auto a_h = metal::swiglu_packed_fw(a.packed);
    expect_forward_matches_reference(a_h, a, "launch A (op allocates the output)");
    const auto a_h_result = core::to_xtensor(a_h);

    const auto b = make_inputs(1U, seq, kTwoCoreGroupInner, 51U);
    const auto b_preallocated = zeros_on_device(h_shape);
    const auto b_h = metal::swiglu_packed_fw(b.packed, b_preallocated);
    EXPECT_EQ(b_h.buffer()->address(), b_preallocated.buffer()->address()) << "must return the preallocated output";
    expect_forward_matches_reference(b_preallocated, b, "launch B (preallocated)");
    const auto b_h_result = core::to_xtensor(b_preallocated);

    // The pattern of launch B again: a hit under any program-cache key.
    const auto c = make_inputs(1U, seq, kTwoCoreGroupInner, 61U);
    const auto c_preallocated = zeros_on_device(h_shape);
    const auto entries_before_hit = device.num_program_cache_entries();
    const auto c_h = metal::swiglu_packed_fw(c.packed, c_preallocated);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit) << "launch C must reuse launch B's program";
    EXPECT_EQ(c_h.buffer()->address(), c_preallocated.buffer()->address()) << "must return the preallocated output";
    expect_new_buffers({b.packed, b_preallocated}, {c.packed, c_preallocated}, "launch C");
    expect_forward_matches_reference(c_preallocated, c, "launch C (preallocated, cache hit)");

    expect_unchanged(a_h, a_h_result, "launch A h");
    expect_unchanged(b_preallocated, b_h_result, "launch B h");
    expect_unchanged(b.packed, b.packed_stored, "launch B packed");
}

// A launch without preallocated_dL_dpacked, then two with a separate one, then one in place: the op must
// write into the preallocated grad and return it, whether or not the program cache keys on its presence.
// The in-place launch hits the program of the separate preallocated grad.
TEST_F(SwigluPackedOpTest, Backward_PreallocatedGradAfterAllocatedGrad_TwoCoreGroups) {
    using namespace ttml;
    auto& device = device_with_empty_program_cache();
    const uint32_t seq = two_core_group_seq();
    const auto dpacked_shape = packed_shape(1U, seq, kTwoCoreGroupInner);

    const auto a = make_inputs(1U, seq, kTwoCoreGroupInner, 71U);
    const auto a_dpacked = metal::swiglu_packed_bw(a.packed, a.dL_dh);
    expect_backward_matches_reference(a_dpacked, a, a.dL_dh_stored, "launch A (op allocates the grad)");
    const auto a_dpacked_result = core::to_xtensor(a_dpacked);

    const auto b = make_inputs(1U, seq, kTwoCoreGroupInner, 81U);
    const auto b_preallocated = zeros_on_device(dpacked_shape);
    const auto b_dpacked = metal::swiglu_packed_bw(b.packed, b.dL_dh, b_preallocated);
    EXPECT_EQ(b_dpacked.buffer()->address(), b_preallocated.buffer()->address()) << "must return the preallocated grad";
    expect_backward_matches_reference(b_preallocated, b, b.dL_dh_stored, "launch B (preallocated)");
    const auto b_dpacked_result = core::to_xtensor(b_preallocated);

    // The pattern of launch B again: a hit under any program-cache key.
    const auto c = make_inputs(1U, seq, kTwoCoreGroupInner, 91U);
    const auto c_preallocated = zeros_on_device(dpacked_shape);
    const auto entries_before_hit = device.num_program_cache_entries();
    const auto c_dpacked = metal::swiglu_packed_bw(c.packed, c.dL_dh, c_preallocated);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit) << "launch C must reuse launch B's program";
    EXPECT_EQ(c_dpacked.buffer()->address(), c_preallocated.buffer()->address()) << "must return the preallocated grad";
    expect_new_buffers({b.packed, b.dL_dh, b_preallocated}, {c.packed, c.dL_dh, c_preallocated}, "launch C");
    expect_backward_matches_reference(c_preallocated, c, c.dL_dh_stored, "launch C (preallocated, cache hit)");
    const auto c_dpacked_result = core::to_xtensor(c_preallocated);

    // In place over packed: the key of launch C, now with packed's buffer passed twice.
    const auto d = make_inputs(1U, seq, kTwoCoreGroupInner, 101U);
    const auto d_dpacked = metal::swiglu_packed_bw(d.packed, d.dL_dh, d.packed);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit) << "launch D must reuse launch B's program";
    EXPECT_EQ(d_dpacked.buffer()->address(), d.packed.buffer()->address()) << "must write into packed";
    expect_new_buffers({c.packed, c.dL_dh, c_preallocated}, {d.packed, d.dL_dh, d.packed}, "launch D");
    expect_backward_matches_reference(d.packed, d, d.dL_dh_stored, "launch D (in place, cache hit)");

    expect_unchanged(a_dpacked, a_dpacked_result, "launch A dL_dpacked");
    expect_unchanged(b_preallocated, b_dpacked_result, "launch B dL_dpacked");
    expect_unchanged(b.packed, b.packed_stored, "launch B packed");
    expect_unchanged(b.dL_dh, b.dL_dh_stored, "launch B dL_dh");
    expect_unchanged(c_preallocated, c_dpacked_result, "launch C dL_dpacked");
    expect_unchanged(c.packed, c.packed_stored, "launch C packed");
    expect_unchanged(c.dL_dh, c.dL_dh_stored, "launch C dL_dh");
}

// preallocated_dL_dpacked == packed passes validation: the grad overwrites the forward input in place.
// The same buffer appears twice among the op's tensors, and each hit must still read and write its own
// call's buffers. The last launch hits the in-place program with a separate preallocated grad.
TEST_F(SwigluPackedOpTest, Backward_InPlaceOverPacked_CacheHit_TwoCoreGroups) {
    using namespace ttml;
    auto& device = device_with_empty_program_cache();
    const uint32_t seq = two_core_group_seq();

    const auto a = make_inputs(1U, seq, kTwoCoreGroupInner, 111U);
    const auto entries_before_miss = device.num_program_cache_entries();
    const auto a_dpacked = metal::swiglu_packed_bw(a.packed, a.dL_dh, a.packed);
    ASSERT_GT(device.num_program_cache_entries(), entries_before_miss) << "the first launch must compile a program";
    EXPECT_EQ(a_dpacked.buffer()->address(), a.packed.buffer()->address()) << "must write into packed";
    expect_backward_matches_reference(a.packed, a, a.dL_dh_stored, "launch A (in place)");
    const auto a_dpacked_result = core::to_xtensor(a.packed);

    const auto b = make_inputs(1U, seq, kTwoCoreGroupInner, 121U);
    const auto entries_before_hit = device.num_program_cache_entries();
    const auto b_dpacked = metal::swiglu_packed_bw(b.packed, b.dL_dh, b.packed);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit) << "launch B must reuse launch A's program";
    EXPECT_EQ(b_dpacked.buffer()->address(), b.packed.buffer()->address()) << "must write into packed";
    expect_new_buffers({a.packed, a.dL_dh}, {b.packed, b.dL_dh}, "launch B");
    expect_backward_matches_reference(b.packed, b, b.dL_dh_stored, "launch B (in place, cache hit)");
    const auto b_dpacked_result = core::to_xtensor(b.packed);

    // A separate preallocated grad: the key of the in-place launches, the grad now a buffer of its own.
    const auto c = make_inputs(1U, seq, kTwoCoreGroupInner, 131U);
    const auto c_preallocated = zeros_on_device(packed_shape(1U, seq, kTwoCoreGroupInner));
    const auto entries_before_c = device.num_program_cache_entries();
    const auto c_dpacked = metal::swiglu_packed_bw(c.packed, c.dL_dh, c_preallocated);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_c) << "launch C must reuse launch A's program";
    EXPECT_EQ(c_dpacked.buffer()->address(), c_preallocated.buffer()->address()) << "must return the preallocated grad";
    expect_new_buffers({b.packed, b.dL_dh, b.packed}, {c.packed, c.dL_dh, c_preallocated}, "launch C");
    expect_backward_matches_reference(c_preallocated, c, c.dL_dh_stored, "launch C (preallocated, cache hit)");

    expect_unchanged(a.packed, a_dpacked_result, "launch A packed (holds dL_dpacked)");
    expect_unchanged(a.dL_dh, a.dL_dh_stored, "launch A dL_dh");
    expect_unchanged(b.packed, b_dpacked_result, "launch B packed (holds dL_dpacked)");
    expect_unchanged(b.dL_dh, b.dL_dh_stored, "launch B dL_dh");
    expect_unchanged(c.packed, c.packed_stored, "launch C packed");
}

// ============================================================================
// Section 3: autograd wrapper
// ============================================================================
// ops::swiglu_packed is the path LlamaMLP takes; it wires both metal ops into the graph.

TEST_F(SwigluPackedOpTest, AutogradWrapper_ForwardAndBackward_2x128x128) {
    using namespace ttml;
    const uint32_t batch = 2U;
    const uint32_t seq = 128U;
    const uint32_t inner = 128U;
    const auto in = make_inputs(batch, seq, inner, 400U + inner);
    auto packed = autograd::create_tensor(in.packed, /* requires_grad */ true);

    auto out = ops::swiglu_packed(packed);
    expect_forward_matches_reference(out->get_value(), in, "autograd forward");

    out->backward();  // seeds dL/dh with ones
    ASSERT_TRUE(packed->is_grad_initialized()) << "backward did not reach the packed input";
    const xt::xarray<float> ones = xt::ones<float>(std::vector<size_t>{batch, 1U, seq, inner});
    expect_backward_matches_reference(packed->get_grad(), in, ones, "autograd backward");
    autograd::ctx().reset_graph();
}

// ============================================================================
// Section 4: validation
// ============================================================================

TEST_F(SwigluPackedOpTest, RejectsHalfWidthThatIsNotTileAligned) {
    // 96 pads to 96, whose halves are 1.5 tiles each.
    const auto packed = zeros_on_device({1U, 1U, 32U, 96U});
    EXPECT_THAT(
        [&] { (void)ttml::metal::swiglu_packed_fw(packed); },
        ThrowsMessage<std::runtime_error>(HasSubstr("last padded dim")));
}

TEST_F(SwigluPackedOpTest, RejectsUpstreamGradThatIsNotHalfThePackedWidth) {
    const auto packed = zeros_on_device({1U, 1U, 32U, 128U});
    const auto wrong = zeros_on_device({1U, 1U, 32U, 32U});  // should be 64
    EXPECT_THAT(
        [&] { (void)ttml::metal::swiglu_packed_bw(packed, wrong); },
        ThrowsMessage<std::runtime_error>(HasSubstr("halved")));
}

// 100 pads to 128, so the padded halves are tile-aligned and only the logical split is wrong: the kernel
// would read up at element 64 while the op's contract puts it at 50.
TEST_F(SwigluPackedOpTest, ForwardRejectsALogicalWidthThatPadsIntoAlignment) {
    const auto packed = zeros_on_device({1U, 1U, 32U, 100U});
    EXPECT_THAT(
        [&] { (void)ttml::metal::swiglu_packed_fw(packed); },
        ThrowsMessage<std::runtime_error>(HasSubstr("last logical dim")));
}

TEST_F(SwigluPackedOpTest, BackwardRejectsALogicalWidthThatPadsIntoAlignment) {
    const auto packed = zeros_on_device({1U, 1U, 32U, 100U});
    const auto dL_dh = zeros_on_device({1U, 1U, 32U, 50U});  // consistent with packed, both halves of a bad split
    EXPECT_THAT(
        [&] { (void)ttml::metal::swiglu_packed_bw(packed, dL_dh); },
        ThrowsMessage<std::runtime_error>(HasSubstr("last logical dim")));
}
