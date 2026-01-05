// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * GELU ULP Bug Reproducer and BFloat16 ULP Calculator Verification
 *
 * This test file contains:
 * 1. BFloat16 ULP (Units in Last Place) calculator with verification tests
 * 2. GELU precision bug reproducers for three problematic regions
 *
 * The ULP calculator is verified using a sorted BF16 value index approach:
 * - All valid BF16 values are sorted in numerical order
 * - Adjacent values in this order should have ULP distance of 1
 * - Both +0 and -0 map to the same index (ULP distance = 0)
 *
 * GELU Bug Summary:
 * - Region 1 (Deep Negative Tail, x < -5.5): Max ULP = 32,767 (hardware returns 0.0)
 * - Region 2 (Near-Zero, |x| < ~1e-4): Max ULP = 14,276 (floor value 2.98e-05)
 * - Region 3 (Transition, -5.5 to -4.0): Max ULP = 1,475 (poor polynomial fit)
 *
 * Run: ./build_Debug/test/ttnn/unit_tests_ttnn --gtest_filter="*GeluUlp*"
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <limits>

#include <tt-metalium/bfloat16.hpp>
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::test {

// =============================================================================
// BFloat16 ULP Calculator
// =============================================================================

namespace bf16_ulp {

/**
 * Convert float to BFloat16 bit representation (truncation, no rounding).
 */
inline uint16_t float_to_bf16_bits(float f) {
    uint32_t f32_bits;
    std::memcpy(&f32_bits, &f, sizeof(float));
    return static_cast<uint16_t>(f32_bits >> 16);
}

/**
 * Convert BFloat16 bits to float.
 */
inline float bf16_bits_to_float(uint16_t bits) {
    uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &f32_bits, sizeof(float));
    return f;
}

/**
 * Get the next representable BFloat16 value (increment by 1 ULP).
 */
inline float bf16_next(float f) {
    uint16_t bits = float_to_bf16_bits(f);
    if (bits == 0x7F80) {
        return std::numeric_limits<float>::infinity();  // +inf
    }
    if (bits == 0xFF80) {
        return bf16_bits_to_float(0xFF7F);  // -inf -> -max
    }
    if ((bits & 0x7FFF) == 0) {
        return bf16_bits_to_float(0x0001);  // ±0 -> smallest positive
    }
    if (bits & 0x8000) {
        // Negative: decrement magnitude
        return bf16_bits_to_float(bits - 1);
    } else {
        // Positive: increment
        return bf16_bits_to_float(bits + 1);
    }
}

/**
 * Calculate the value order index for a BFloat16 value.
 *
 * This creates a linear index where:
 * - The most negative finite value has index 0
 * - Values increase monotonically
 * - Both +0 and -0 have the same index
 * - Adjacent indices represent adjacent BF16 values (ULP distance = 1)
 *
 * The index range is [0, 65278] for all finite non-NaN BF16 values.
 */
inline int32_t bf16_value_order_index(float f) {
    uint16_t bits = float_to_bf16_bits(f);

    // Handle NaN - return -1 as invalid
    if ((bits & 0x7F80) == 0x7F80 && (bits & 0x007F) != 0) {
        return -1;
    }

    // Handle infinity
    if (bits == 0x7F80) {
        return 65279;  // +inf
    }
    if (bits == 0xFF80) {
        return -1;  // -inf (exclude from valid range)
    }

    // For signed magnitude representation, convert to ordered index:
    // Negative numbers: -max_negative (0xFF7F) -> 0, -smallest (0x8001) -> 32638, -0 (0x8000) -> 32639
    // Positive numbers: +0 (0x0000) -> 32639, +smallest (0x0001) -> 32640, +max (0x7F7F) -> 65278
    //
    // Both zeroes map to the same index (32639)

    if (bits & 0x8000) {
        // Negative: 0x8000 (-0) -> 32639, 0x8001 -> 32638, ..., 0xFF7F -> 0
        uint16_t magnitude = bits & 0x7FFF;
        if (magnitude == 0) {
            return 32639;  // -0 same as +0
        }
        return 32639 - magnitude;
    } else {
        // Positive: 0x0000 (+0) -> 32639, 0x0001 -> 32640, ..., 0x7F7F -> 65278
        return 32639 + bits;
    }
}

/**
 * Calculate ULP distance between two BFloat16 values.
 *
 * Uses signed-magnitude to linear transformation for cross-zero comparisons.
 * This matches the Python implementation for consistent bug reporting.
 *
 * For values of the same sign: simple bit difference
 * For values of different signs: transforms negative values to a "mirrored"
 * position on the positive side of the number line, then computes distance.
 */
inline int32_t ulp_distance_bf16(float a, float b) {
    uint16_t a_bits = float_to_bf16_bits(a);
    uint16_t b_bits = float_to_bf16_bits(b);

    // Handle NaN
    if ((a_bits & 0x7F80) == 0x7F80 && (a_bits & 0x007F) != 0) {
        return -1;
    }
    if ((b_bits & 0x7F80) == 0x7F80 && (b_bits & 0x007F) != 0) {
        return -1;
    }

    // Different signs - use signed-magnitude to linear transformation
    if ((a_bits >> 15) != (b_bits >> 15)) {
        int32_t a_linear = a_bits;
        int32_t b_linear = b_bits;

        // Transform negative values: 0x8000 -> 0, 0x8001 -> 0x7FFF, 0x8002 -> 0x7FFE, etc.
        if (a_bits >> 15) {
            a_linear = (a_bits == 0x8000) ? 0 : (0x8000 - (a_bits & 0x7FFF));
        }
        if (b_bits >> 15) {
            b_linear = (b_bits == 0x8000) ? 0 : (0x8000 - (b_bits & 0x7FFF));
        }
        return a_linear + b_linear;
    }

    // Same sign - simple bit difference
    return std::abs(static_cast<int32_t>(a_bits) - static_cast<int32_t>(b_bits));
}

/**
 * Exact GELU using erfc to avoid catastrophic cancellation for negative x.
 *
 * GELU(x) = 0.5 * x * (1 + erf(x/sqrt(2)))
 *
 * For negative x, we use: GELU(x) = 0.5 * x * erfc(-x/sqrt(2))
 * This avoids the (1 + erf(large_negative)) cancellation.
 */
inline double gelu_exact(double x) {
    const double SQRT_2 = std::sqrt(2.0);
    if (x >= 0) {
        return 0.5 * x * (1.0 + std::erf(x / SQRT_2));
    } else {
        return 0.5 * x * std::erfc(-x / SQRT_2);
    }
}

}  // namespace bf16_ulp

// =============================================================================
// ULP Calculator Verification Tests (No Device Required)
// =============================================================================

class BFloat16UlpTest : public ::testing::Test {};

TEST_F(BFloat16UlpTest, ZeroesHaveSameIndex) {
    // Both +0 and -0 should have the same value order index
    float pos_zero = 0.0f;
    float neg_zero = -0.0f;

    int32_t idx_pos = bf16_ulp::bf16_value_order_index(pos_zero);
    int32_t idx_neg = bf16_ulp::bf16_value_order_index(neg_zero);

    EXPECT_EQ(idx_pos, idx_neg) << "+0 and -0 should have the same index";
    EXPECT_EQ(idx_pos, 32639) << "Zero index should be 32639";
}

TEST_F(BFloat16UlpTest, ZeroesHaveUlpDistanceZero) {
    // ULP distance between +0 and -0 should be 0
    float pos_zero = 0.0f;
    float neg_zero = -0.0f;

    int32_t ulp = bf16_ulp::ulp_distance_bf16(pos_zero, neg_zero);
    EXPECT_EQ(ulp, 0) << "ULP distance between +0 and -0 should be 0";
}

TEST_F(BFloat16UlpTest, AdjacentPositiveValuesHaveUlpOne) {
    // Adjacent BF16 values should have ULP distance of 1
    // Test a few pairs of adjacent positive values
    std::vector<uint16_t> test_bits = {0x0001, 0x3F80, 0x4000, 0x7F00};

    for (uint16_t bits : test_bits) {
        float val = bf16_ulp::bf16_bits_to_float(bits);
        float next_val = bf16_ulp::bf16_bits_to_float(bits + 1);

        int32_t ulp = bf16_ulp::ulp_distance_bf16(val, next_val);
        EXPECT_EQ(ulp, 1) << "Adjacent positive BF16 values at bits 0x" << std::hex << bits
                          << " should have ULP distance 1";
    }
}

TEST_F(BFloat16UlpTest, AdjacentNegativeValuesHaveUlpOne) {
    // Adjacent negative BF16 values should have ULP distance of 1
    std::vector<uint16_t> test_bits = {0x8001, 0xBF80, 0xC000, 0xFF00};

    for (uint16_t bits : test_bits) {
        float val = bf16_ulp::bf16_bits_to_float(bits);
        float next_val = bf16_ulp::bf16_bits_to_float(bits + 1);  // More negative

        int32_t ulp = bf16_ulp::ulp_distance_bf16(val, next_val);
        EXPECT_EQ(ulp, 1) << "Adjacent negative BF16 values at bits 0x" << std::hex << bits
                          << " should have ULP distance 1";
    }
}

TEST_F(BFloat16UlpTest, SmallestPositiveToZeroIsOne) {
    // Distance from +0 to smallest positive subnormal should be 1
    float zero = 0.0f;
    float smallest_pos = bf16_ulp::bf16_bits_to_float(0x0001);

    int32_t ulp = bf16_ulp::ulp_distance_bf16(zero, smallest_pos);
    EXPECT_EQ(ulp, 1) << "ULP distance from 0 to smallest positive should be 1";
}

TEST_F(BFloat16UlpTest, SmallestNegativeToZeroIsOne) {
    // Distance from -0 to smallest negative subnormal should be 1
    float neg_zero = -0.0f;
    float smallest_neg = bf16_ulp::bf16_bits_to_float(0x8001);

    int32_t ulp = bf16_ulp::ulp_distance_bf16(neg_zero, smallest_neg);
    EXPECT_EQ(ulp, 1) << "ULP distance from -0 to smallest negative should be 1";
}

TEST_F(BFloat16UlpTest, CrossZeroDistance) {
    // Test cross-zero distance using value order index (mathematically correct ULP)
    // smallest_neg (-1 ULP from zero) to smallest_pos (+1 ULP from zero) = 2
    float smallest_pos = bf16_ulp::bf16_bits_to_float(0x0001);
    float smallest_neg = bf16_ulp::bf16_bits_to_float(0x8001);

    // Use value order index for mathematically correct ULP distance
    int32_t idx_pos = bf16_ulp::bf16_value_order_index(smallest_pos);
    int32_t idx_neg = bf16_ulp::bf16_value_order_index(smallest_neg);
    int32_t ulp_via_index = std::abs(idx_pos - idx_neg);
    EXPECT_EQ(ulp_via_index, 2) << "Value order ULP should be 2";

    // Note: ulp_distance_bf16 uses Python-compatible algorithm which sums
    // distances for different-sign values (used for GELU bug detection)
    int32_t ulp = bf16_ulp::ulp_distance_bf16(smallest_pos, smallest_neg);
    // Python algorithm: 0x0001 (positive) stays 1, 0x8001 (negative) transforms to 0x8000-0x0001=0x7FFF=32767
    // Result: 1 + 32767 = 32768
    EXPECT_EQ(ulp, 32768) << "Python-compatible ULP (sum of distances) should be 32768";
}

TEST_F(BFloat16UlpTest, MaxUlpDistanceIs65278) {
    // Distance from most negative to most positive finite value
    float max_neg = bf16_ulp::bf16_bits_to_float(0xFF7F);  // -max finite
    float max_pos = bf16_ulp::bf16_bits_to_float(0x7F7F);  // +max finite

    int32_t idx_neg = bf16_ulp::bf16_value_order_index(max_neg);
    int32_t idx_pos = bf16_ulp::bf16_value_order_index(max_pos);

    EXPECT_EQ(idx_neg, 0) << "Most negative finite value should have index 0";
    EXPECT_EQ(idx_pos, 65278) << "Most positive finite value should have index 65278";

    // Using value order index for mathematically correct ULP distance
    int32_t ulp_via_index = std::abs(idx_pos - idx_neg);
    EXPECT_EQ(ulp_via_index, 65278) << "Value order max ULP distance should be 65278";

    // Python-compatible ulp_distance_bf16 gives different result for cross-sign values
    // 0xFF7F (negative): 0x8000 - 0x7F7F = 0x0081 = 129
    // 0x7F7F (positive): stays 0x7F7F = 32639
    // Result: 129 + 32639 = 32768
    int32_t ulp = bf16_ulp::ulp_distance_bf16(max_neg, max_pos);
    EXPECT_EQ(ulp, 32768) << "Python-compatible ULP should be 32768";
}

TEST_F(BFloat16UlpTest, VerifyIndexMonotonicity) {
    // Verify that the value order index is monotonically increasing
    // across the entire BF16 range (excluding NaN and inf)
    std::vector<std::pair<float, int32_t>> values_with_indices;

    // Collect all finite BF16 values with their indices
    for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        uint16_t bf16_bits = static_cast<uint16_t>(bits);

        // Skip NaN (exponent all 1s, mantissa non-zero)
        if ((bf16_bits & 0x7F80) == 0x7F80 && (bf16_bits & 0x007F) != 0) {
            continue;
        }
        // Skip infinity
        if (bf16_bits == 0x7F80 || bf16_bits == 0xFF80) {
            continue;
        }

        float val = bf16_ulp::bf16_bits_to_float(bf16_bits);
        int32_t idx = bf16_ulp::bf16_value_order_index(val);
        values_with_indices.push_back({val, idx});
    }

    // Sort by numerical value
    std::sort(values_with_indices.begin(), values_with_indices.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // Verify indices are monotonically increasing
    for (size_t i = 1; i < values_with_indices.size(); ++i) {
        float prev_val = values_with_indices[i - 1].first;
        float curr_val = values_with_indices[i].first;
        int32_t prev_idx = values_with_indices[i - 1].second;
        int32_t curr_idx = values_with_indices[i].second;

        // Handle special case: -0 and +0 have same index
        if (prev_val == 0.0f && curr_val == 0.0f) {
            EXPECT_EQ(prev_idx, curr_idx) << "Both zeroes should have same index";
        } else {
            EXPECT_LE(prev_idx, curr_idx)
                << "Index should be monotonically increasing: val[" << i - 1 << "]=" << prev_val << " (idx=" << prev_idx
                << ") vs val[" << i << "]=" << curr_val << " (idx=" << curr_idx << ")";
        }
    }
}

TEST_F(BFloat16UlpTest, AdjacentValuesAlwaysHaveUlpOneOrZero) {
    // For ALL adjacent BF16 values (sorted numerically), ULP should be 0 or 1
    std::vector<float> sorted_values;

    // Collect all finite BF16 values
    for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        uint16_t bf16_bits = static_cast<uint16_t>(bits);
        if ((bf16_bits & 0x7F80) == 0x7F80 && (bf16_bits & 0x007F) != 0) {
            continue;  // NaN
        }
        if (bf16_bits == 0x7F80 || bf16_bits == 0xFF80) {
            continue;  // Inf
        }

        sorted_values.push_back(bf16_ulp::bf16_bits_to_float(bf16_bits));
    }

    std::sort(sorted_values.begin(), sorted_values.end());

    // Check all adjacent pairs
    int failures = 0;
    for (size_t i = 1; i < sorted_values.size() && failures < 10; ++i) {
        int32_t ulp = bf16_ulp::ulp_distance_bf16(sorted_values[i - 1], sorted_values[i]);
        if (ulp != 0 && ulp != 1) {
            ++failures;
            std::cerr << "FAIL: Adjacent values [" << i - 1 << "]=" << sorted_values[i - 1] << " and [" << i
                      << "]=" << sorted_values[i] << " have ULP=" << ulp << " (expected 0 or 1)\n";
        }
    }
    EXPECT_EQ(failures, 0) << "All adjacent sorted BF16 values should have ULP 0 or 1";
}

// =============================================================================
// GELU Bug Reproducer Tests (Require Device)
// =============================================================================

class GeluUlpBugTest : public TTNNFixtureWithDevice {};

TEST_F(GeluUlpBugTest, DeepNegativeTailReturnsZero) {
    // Region 1: Deep negative tail (x < -5.5)
    // Hardware returns 0.0 but exact GELU has tiny negative values
    // Max ULP = 32,767 (maximum possible for BF16)

    std::vector<std::pair<float, int32_t>> test_cases = {
        {-13.5f, 32000},  // Saturation boundary
        {-12.0f, 29000},
        {-10.0f, 24000},
        {-8.0f, 22000},
        {-6.0f, 20000},
        {-5.5625f, 19000}  // Just below -5.5 threshold
    };

    for (const auto& [input_val, expected_ulp_min] : test_cases) {
        // Create BF16 input tensor
        std::array<uint32_t, 4> dims = {1, 1, 32, 32};
        ttnn::Shape shape(dims);
        auto input_tensor = ttnn::full(shape, input_val, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);

        // Run accurate GELU (fast_and_approximate_mode=false)
        auto output_tensor = ttnn::gelu(input_tensor, false);
        auto output_cpu = ttnn::from_device(output_tensor);
        auto output_vec = output_cpu.to_vector<::bfloat16>();
        float actual = static_cast<float>(output_vec[0]);

        double expected = bf16_ulp::gelu_exact(input_val);
        int32_t ulp_error = bf16_ulp::ulp_distance_bf16(actual, static_cast<float>(expected));

        // Verify the bug exists: hardware returns 0.0
        EXPECT_FLOAT_EQ(actual, 0.0f) << "Deep negative x=" << input_val << " should return 0.0, got " << actual;

        EXPECT_GE(ulp_error, expected_ulp_min)
            << "x=" << input_val << " expected ULP >= " << expected_ulp_min << ", got " << ulp_error;

        std::cout << "x=" << input_val << ": expected=" << expected << ", actual=" << actual << ", ULP=" << ulp_error
                  << "\n";
    }
}

TEST_F(GeluUlpBugTest, NearZeroFloorValue) {
    // Region 2: Near-zero floor value bug
    // Hardware returns 2.98e-05 (Chebyshev c0 coefficient) for all tiny inputs
    // Max ULP = 14,276

    const float CHEBYSHEV_C0 = 2.98325768482e-05f;
    const float FLOOR_VALUE = bf16_ulp::bf16_bits_to_float(bf16_ulp::float_to_bf16_bits(CHEBYSHEV_C0));

    std::vector<float> tiny_inputs = {1e-38f, 1e-35f, 1e-30f, 1e-25f, 1e-20f, 1e-15f, 1e-10f, 1e-8f};

    for (float input_val : tiny_inputs) {
        std::array<uint32_t, 4> dims = {1, 1, 32, 32};
        ttnn::Shape shape(dims);
        auto input_tensor = ttnn::full(shape, input_val, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);

        auto output_tensor = ttnn::gelu(input_tensor, false);
        auto output_cpu = ttnn::from_device(output_tensor);
        auto output_vec = output_cpu.to_vector<::bfloat16>();
        float actual = static_cast<float>(output_vec[0]);

        float expected = 0.5f * input_val;  // GELU(x) ≈ 0.5*x for tiny x
        int32_t ulp_error = bf16_ulp::ulp_distance_bf16(actual, expected);

        // Verify the bug: actual should be close to floor value
        EXPECT_NEAR(actual, FLOOR_VALUE, 1e-7f)
            << "Tiny input " << input_val << " should return floor value " << FLOOR_VALUE << ", got " << actual;

        EXPECT_GT(ulp_error, 1000) << "x=" << input_val << " expected ULP > 1000, got " << ulp_error;

        std::cout << "x=" << input_val << ": expected=" << expected << ", actual=" << actual << ", ULP=" << ulp_error
                  << "\n";
    }
}

TEST_F(GeluUlpBugTest, TransitionRegionErrors) {
    // Region 3: Transition region (-5.5 to ~-4.0)
    // Polynomial is poorly fitted near the -5.5 boundary
    // ULP errors range from 100-1500

    std::vector<float> transition_inputs = {-5.5f, -5.4375f, -5.375f, -5.25f, -5.0f, -4.75f, -4.5f, -4.25f, -4.0f};

    int32_t max_ulp = 0;
    for (float input_val : transition_inputs) {
        std::array<uint32_t, 4> dims = {1, 1, 32, 32};
        ttnn::Shape shape(dims);
        auto input_tensor = ttnn::full(shape, input_val, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);

        auto output_tensor = ttnn::gelu(input_tensor, false);
        auto output_cpu = ttnn::from_device(output_tensor);
        auto output_vec = output_cpu.to_vector<::bfloat16>();
        float actual = static_cast<float>(output_vec[0]);

        double expected = bf16_ulp::gelu_exact(input_val);
        int32_t ulp_error = bf16_ulp::ulp_distance_bf16(actual, static_cast<float>(expected));
        max_ulp = std::max(max_ulp, ulp_error);

        std::cout << "x=" << input_val << ": expected=" << expected << ", actual=" << actual << ", ULP=" << ulp_error
                  << "\n";

        // Log high ULP errors for analysis
        if (ulp_error > 100) {
            std::cerr << "WARNING: High ULP error (" << ulp_error << ") at x=" << input_val << "\n";
        }
    }

    // Verify transition region has elevated errors
    EXPECT_GT(max_ulp, 500) << "Transition region should have at least one value with ULP > 500";
}

TEST_F(GeluUlpBugTest, SummaryStatistics) {
    // Run a subset of values and report summary statistics
    std::cout << "\n========================================\n";
    std::cout << "GELU PRECISION BUG SUMMARY\n";
    std::cout << "========================================\n";

    std::array<uint32_t, 4> dims = {1, 1, 32, 32};
    ttnn::Shape shape(dims);

    int32_t max_ulp_region1 = 0;
    int32_t max_ulp_region2 = 0;
    int32_t max_ulp_region3 = 0;

    // Region 1: Deep negative
    for (float x : {-13.5f, -10.0f, -6.0f}) {
        auto tensor = ttnn::full(shape, x, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
        auto result = ttnn::from_device(ttnn::gelu(tensor, false));
        float actual = static_cast<float>(result.to_vector<::bfloat16>()[0]);
        int32_t ulp = bf16_ulp::ulp_distance_bf16(actual, static_cast<float>(bf16_ulp::gelu_exact(x)));
        max_ulp_region1 = std::max(max_ulp_region1, ulp);
    }

    // Region 2: Near-zero
    for (float x : {1e-38f, 1e-20f, 1e-10f}) {
        auto tensor = ttnn::full(shape, x, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
        auto result = ttnn::from_device(ttnn::gelu(tensor, false));
        float actual = static_cast<float>(result.to_vector<::bfloat16>()[0]);
        int32_t ulp = bf16_ulp::ulp_distance_bf16(actual, 0.5f * x);
        max_ulp_region2 = std::max(max_ulp_region2, ulp);
    }

    // Region 3: Transition
    for (float x : {-5.5f, -5.0f, -4.0f}) {
        auto tensor = ttnn::full(shape, x, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
        auto result = ttnn::from_device(ttnn::gelu(tensor, false));
        float actual = static_cast<float>(result.to_vector<::bfloat16>()[0]);
        int32_t ulp = bf16_ulp::ulp_distance_bf16(actual, static_cast<float>(bf16_ulp::gelu_exact(x)));
        max_ulp_region3 = std::max(max_ulp_region3, ulp);
    }

    std::cout << "Region 1 (Deep Negative Tail): Max ULP = " << max_ulp_region1 << "\n";
    std::cout << "Region 2 (Near-Zero):          Max ULP = " << max_ulp_region2 << "\n";
    std::cout << "Region 3 (Transition):         Max ULP = " << max_ulp_region3 << "\n";
    std::cout << "\n";
    std::cout << "NOTE: Region 1 has the WORST error (32,767 = max possible for BF16)\n";
    std::cout << "Source: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h\n";
    std::cout << "========================================\n";

    // Verify bugs exist
    EXPECT_GT(max_ulp_region1, 20000) << "Region 1 should have Max ULP > 20000";
    EXPECT_GT(max_ulp_region2, 2000) << "Region 2 should have Max ULP > 2000";
    EXPECT_GT(max_ulp_region3, 100) << "Region 3 should have Max ULP > 100";
}

}  // namespace ttnn::test
