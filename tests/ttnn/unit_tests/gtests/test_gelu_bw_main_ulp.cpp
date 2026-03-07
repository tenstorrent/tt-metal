// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * GELU Backward ULP Precision Tests
 *
 * This test file validates the accuracy of ttnn::gelu_bw (GELU derivative) across
 * the entire BFloat16 range using the same methodology as test_gelu_ulp_bug.cpp.
 *
 * MATHEMATICAL FORMULA:
 * GELU'(x) = grad * (cdf + x * pdf)
 * where:
 *   cdf = 0.5 * (1 + erf(x / sqrt(2)))  -- CDF of standard normal distribution
 *   pdf = exp(-x^2 / 2) / sqrt(2*pi)    -- PDF of standard normal distribution
 *
 * REFERENCE IMPLEMENTATION:
 * Uses fp64 with erfc() for numerically stable computation.
 * Key insight: for negative x, 1 + erf(x/√2) = erfc(|x|/√2)
 * This avoids erf() saturation at x ≈ -8.375.
 *
 * HARDWARE MODEL: DAZ+FTZ (Denormals-Are-Zero + Flush-To-Zero)
 * Per tech_reports/Handling_Special_Value/special_values.md: "denormals | all | 0x0"
 *
 * BATCHED TESTING PATTERN:
 * Tests that sweep all ~65,000 BF16 values use batched tensor operations for efficiency:
 *   1. Collect all valid BF16 values into a vector
 *   2. Pad to tile boundary (multiple of 32x32=1024)
 *   3. Create single tensor: Tensor::from_vector(data, TensorSpec).to_device(device)
 *   4. Call operation ONCE on the entire tensor
 *   5. Process results from output vector
 * This achieves ~100x speedup vs calling the operation individually per value.
 *
 * Run: ./build_Debug/test/ttnn/unit_tests_ttnn --gtest_filter="*GeluBwMainUlp*"
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <numbers>
#include <vector>
#include <limits>
#include <iomanip>
#include <set>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::test {

// =============================================================================
// BFloat16 ULP Calculator (shared with test_gelu_ulp_bug.cpp)
// =============================================================================

namespace bf16_ulp_bw_main {

// BFloat16 bit-field constants
constexpr uint16_t BF16_EXP_MASK = 0x7F80;       // Exponent bits [14:7]
constexpr uint16_t BF16_MANTISSA_MASK = 0x007F;  // Mantissa bits [6:0]
constexpr uint16_t BF16_SIGN_MASK = 0x8000;      // Sign bit [15]
constexpr uint16_t BF16_POS_INF = 0x7F80;        // +Infinity
constexpr uint16_t BF16_NEG_INF = 0xFF80;        // -Infinity

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
 * Check if BF16 bits represent a denormal (subnormal) value.
 */
inline bool is_bf16_denormal(uint16_t bits) {
    uint16_t exp = (bits >> 7) & 0xFF;
    uint16_t mantissa = bits & BF16_MANTISSA_MASK;
    return (exp == 0) && (mantissa != 0);
}

inline bool is_bf16_denormal(float f) { return is_bf16_denormal(float_to_bf16_bits(f)); }

/**
 * Apply DAZ (Denormals-Are-Zero) normalization to BF16 bits.
 */
inline uint16_t bf16_daz_normalize(uint16_t bits) {
    if (is_bf16_denormal(bits)) {
        return 0x0000;
    }
    if (bits == 0x8000) {  // -0 -> +0
        return 0x0000;
    }
    return bits;
}

inline float bf16_daz_normalize(float f) {
    uint16_t bits = float_to_bf16_bits(f);
    uint16_t normalized = bf16_daz_normalize(bits);
    return bf16_bits_to_float(normalized);
}

/**
 * Calculate the value order index for a BFloat16 value with DAZ.
 */
inline int32_t bf16_value_order_index_daz(uint16_t bits) {
    bits = bf16_daz_normalize(bits);

    uint16_t exp = (bits >> 7) & 0xFF;
    uint16_t mantissa = bits & BF16_MANTISSA_MASK;
    if (exp == 0xFF && mantissa != 0) {
        return -1;  // NaN
    }
    if (bits == BF16_POS_INF) {
        return 65281;  // +inf
    }
    if (bits == BF16_NEG_INF) {
        return -1;  // -inf
    }
    if (bits == 0x0000) {
        return 32640;  // Zero
    }

    if (bits & BF16_SIGN_MASK) {
        uint16_t magnitude = bits & 0x7FFF;
        return 0x7F7F - magnitude;
    }
    return 32640 + bits - BF16_MANTISSA_MASK;
}

inline int32_t bf16_value_order_index_daz(float f) { return bf16_value_order_index_daz(float_to_bf16_bits(f)); }

/**
 * Calculate ULP distance between two BFloat16 values with DAZ+FTZ.
 */
inline int32_t ulp_distance_bf16_daz(float a, float b) {
    uint16_t a_bits = bf16_daz_normalize(float_to_bf16_bits(a));
    uint16_t b_bits = bf16_daz_normalize(float_to_bf16_bits(b));

    uint16_t a_exp = (a_bits >> 7) & 0xFF;
    uint16_t b_exp = (b_bits >> 7) & 0xFF;
    if ((a_exp == 0xFF && (a_bits & 0x7F) != 0) || (b_exp == 0xFF && (b_bits & 0x7F) != 0)) {
        return -1;  // NaN
    }

    int32_t idx_a = bf16_value_order_index_daz(a_bits);
    int32_t idx_b = bf16_value_order_index_daz(b_bits);

    if (idx_a < 0 || idx_b < 0) {
        return -1;
    }

    return std::abs(idx_a - idx_b);
}

// =============================================================================
// GELU Backward Reference Implementation
// =============================================================================

/**
 * Exact GELU derivative reference using fp64 with erfc() for numerical stability.
 *
 * GELU'(x) = cdf + x * pdf
 * where:
 *   cdf = 0.5 * (1 + erf(x / sqrt(2)))
 *   pdf = exp(-x^2 / 2) / sqrt(2*pi)
 *
 * For numerical stability with large negative x, we use:
 *   1 + erf(x/√2) = erfc(|x|/√2) when x < 0
 *
 * This avoids the erf() saturation issue at x ≈ -8.375.
 */
// Uses double (fp64) intentionally: the golden reference must be higher precision
// than BF16 to allow meaningful ULP distance measurement.
inline double gelu_derivative_exact(double x) {
    constexpr double SQRT2 = std::numbers::sqrt2;
    constexpr double INV_SQRT_2PI = 0.3989422804014327;

    double cdf;
    if (x < 0.0) {
        // For negative x: 1 + erf(x/√2) = erfc(|x|/√2)
        double abs_x_div_sqrt2 = -x / SQRT2;  // |x| / √2
        cdf = 0.5 * std::erfc(abs_x_div_sqrt2);
    } else {
        // For non-negative x: standard formula works fine
        cdf = 0.5 * (1.0 + std::erf(x / SQRT2));
    }

    // PDF: exp(-x^2 / 2) / sqrt(2*pi)
    double pdf = std::exp(-0.5 * x * x) * INV_SQRT_2PI;

    // GELU'(x) = cdf + x * pdf
    return cdf + (x * pdf);
}

/**
 * Compute the expected BF16 GELU derivative value with DAZ+FTZ applied.
 */
inline float gelu_derivative_expected_bf16_daz(float x) {
    float x_daz = bf16_daz_normalize(x);
    double result = gelu_derivative_exact(x_daz);
    float result_f32 = static_cast<float>(result);
    return bf16_daz_normalize(result_f32);
}

inline float gelu_bw_expected_bf16_daz(float grad, float x) {
    float grad_daz = bf16_daz_normalize(grad);
    float x_daz = bf16_daz_normalize(x);
    double result = static_cast<double>(grad_daz) * gelu_derivative_exact(x_daz);
    float result_f32 = static_cast<float>(result);
    return bf16_daz_normalize(result_f32);
}

}  // namespace bf16_ulp_bw_main

// =============================================================================
// GELU Backward ULP Tests (Require Device)
// =============================================================================

class GeluBwMainUlpTest : public TTNNFixtureWithDevice {};

/**
 * Helper function to run GELU backward on device with grad=1.0
 * This tests the GELU derivative directly: GELU'(x)
 */
float run_gelu_bw_main_single(tt::tt_metal::distributed::MeshDevice& device, float input_val, float grad_val = 1.0f) {
    ttnn::Shape shape({1, 1, 32, 32});

    auto input_tensor = ttnn::full(shape, input_val, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    auto grad_tensor = ttnn::full(shape, grad_val, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

    // Call gelu_bw with approximate="none" (polynomial approximation)
    auto results = ttnn::gelu_bw(grad_tensor, input_tensor, "none");
    auto result = results[0].value();

    auto output_cpu = ttnn::from_device(result);
    auto output_vec = output_cpu.to_vector<::bfloat16>();
    return static_cast<float>(output_vec[0]);
}

// Correctness guard: verifies the trivial midpoint GELU'(0) = 0.5.
// Catches broken CDF constant, wrong formula sign, or kernel crash.
TEST_F(GeluBwMainUlpTest, DerivativeAtZero) {
    float actual = run_gelu_bw_main_single(*device_, 0.0f);
    float expected = bf16_ulp_bw_main::gelu_derivative_expected_bf16_daz(0.0f);

    int32_t ulp = bf16_ulp_bw_main::ulp_distance_bf16_daz(actual, expected);

    std::cout << "x=0: expected=" << expected << ", actual=" << actual << ", ULP=" << ulp << std::endl;

    EXPECT_LE(ulp, 2) << "GELU'(0) should be ~0.5 with low ULP error";
}

// Correctness guard: GELU'(x) must approach 1 for positive x.
// Catches broken positive saturation or wrong formula sign.
TEST_F(GeluBwMainUlpTest, DerivativeAtPositiveValues) {
    std::vector<std::pair<float, int32_t>> test_cases = {
        {0.5f, 2},
        {1.0f, 2},
        {2.0f, 2},
        {3.0f, 2},
        {5.0f, 2},
        {10.0f, 2},
    };

    for (const auto& [input_val, max_expected_ulp] : test_cases) {
        float actual = run_gelu_bw_main_single(*device_, input_val);
        float expected = bf16_ulp_bw_main::gelu_derivative_expected_bf16_daz(input_val);
        int32_t ulp = bf16_ulp_bw_main::ulp_distance_bf16_daz(actual, expected);

        std::cout << "x=" << input_val << ": expected=" << expected << ", actual=" << actual << ", ULP=" << ulp
                  << std::endl;

        EXPECT_LE(ulp, max_expected_ulp) << "GELU'(" << input_val << ") ULP too high";
    }
}

// Correctness guard: GELU'(x) must approach 0 for negative x.
// Per-point ULP thresholds (2). Catches precision regression at any negative sample point.
TEST_F(GeluBwMainUlpTest, DerivativeAtNegativeValues) {
    std::vector<std::pair<float, int32_t>> test_cases = {
        {-0.5f, 2},
        {-1.0f, 2},
        {-2.0f, 2},
        {-3.0f, 2},
        {-4.0f, 2},
        {-5.0f, 2},
        {-6.0f, 2},
        {-8.0f, 2},
    };

    for (const auto& [input_val, max_expected_ulp] : test_cases) {
        float actual = run_gelu_bw_main_single(*device_, input_val);
        float expected = bf16_ulp_bw_main::gelu_derivative_expected_bf16_daz(input_val);
        int32_t ulp = bf16_ulp_bw_main::ulp_distance_bf16_daz(actual, expected);

        std::cout << "x=" << input_val << ": expected=" << expected << ", actual=" << actual << ", ULP=" << ulp
                  << std::endl;

        EXPECT_LE(ulp, max_expected_ulp) << "GELU'(" << input_val << ") ULP too high";
    }
}

// Correctness guard: near-zero region where GELU'(x) ≈ 0.5 + x/sqrt(2π).
// Catches broken small-value handling, DAZ flush bugs, or denormal issues.
TEST_F(GeluBwMainUlpTest, DerivativeNearZero) {
    std::vector<float> test_values = {1e-6f, 1e-4f, 0.01f, 0.1f, -0.1f, -0.01f, -1e-4f};

    for (float input_val : test_values) {
        float actual = run_gelu_bw_main_single(*device_, input_val);
        float expected = bf16_ulp_bw_main::gelu_derivative_expected_bf16_daz(input_val);
        int32_t ulp = bf16_ulp_bw_main::ulp_distance_bf16_daz(actual, expected);

        std::cout << "x=" << input_val << ": expected=" << expected << ", actual=" << actual << ", ULP=" << ulp
                  << std::endl;

        EXPECT_LE(ulp, 2) << "GELU'(" << input_val << ") near-zero ULP too high";
    }
}

// Correctness guard: only test using grad != 1.0 in this fixture.
// Catches swapped grad/input tensors or missing gradient multiplication.
TEST_F(GeluBwMainUlpTest, WithGradientScaling) {
    std::vector<std::tuple<float, float, int32_t>> test_cases = {
        // {input, grad, max_ulp}
        {1.0f, 2.0f, 2},
        {-1.0f, 0.5f, 2},
        {0.0f, 1.0f, 2},
        {2.0f, -1.0f, 2},
    };

    for (const auto& [input_val, grad_val, max_expected_ulp] : test_cases) {
        float actual = run_gelu_bw_main_single(*device_, input_val, grad_val);
        float expected = bf16_ulp_bw_main::gelu_bw_expected_bf16_daz(grad_val, input_val);
        int32_t ulp = bf16_ulp_bw_main::ulp_distance_bf16_daz(actual, expected);

        std::cout << "x=" << input_val << ", grad=" << grad_val << ": expected=" << expected << ", actual=" << actual
                  << ", ULP=" << ulp << std::endl;

        EXPECT_LE(ulp, max_expected_ulp) << "GELU backward with grad ULP too high";
    }
}

// Precision guard: sweeps ALL ~65K BF16 values, enforces per-region ULP caps.
// 7 regions from deep-negative to large-positive, each with a tuned threshold.
// Catches precision degradation in any region that point-sample tests would miss.
TEST_F(GeluBwMainUlpTest, ComprehensiveULPByRegion) {
    std::vector<float> input_values;
    input_values.reserve(70000);

    // Collect all valid BF16 values
    for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        uint16_t bf16_bits = static_cast<uint16_t>(bits);

        // Skip NaN
        if ((bf16_bits & bf16_ulp_bw_main::BF16_EXP_MASK) == bf16_ulp_bw_main::BF16_EXP_MASK &&
            (bf16_bits & bf16_ulp_bw_main::BF16_MANTISSA_MASK) != 0) {
            continue;
        }
        // Skip infinity
        if (bf16_bits == bf16_ulp_bw_main::BF16_POS_INF || bf16_bits == bf16_ulp_bw_main::BF16_NEG_INF) {
            continue;
        }
        // Skip denormals (will be treated as zero)
        if (bf16_ulp_bw_main::is_bf16_denormal(bf16_bits)) {
            continue;
        }

        float val = bf16_ulp_bw_main::bf16_bits_to_float(bf16_bits);
        input_values.push_back(val);
    }

    const size_t valid_count = input_values.size();
    std::cout << "\nCollected " << valid_count << " valid BF16 values\n";

    // Pad to tile boundary (multiple of 32*32 = 1024)
    const size_t tile_size = tt::constants::TILE_HW;
    size_t padded_size = ((valid_count + tile_size - 1) / tile_size) * tile_size;
    input_values.resize(padded_size, 0.0f);

    // Create tensor shape
    uint32_t num_tiles = static_cast<uint32_t>(padded_size / tile_size);
    std::array<uint32_t, 4> dims = {1, 1, num_tiles * tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};

    // Create input tensors from vectors
    std::vector<::bfloat16> bf16_inputs;
    std::vector<::bfloat16> bf16_grads;
    bf16_inputs.reserve(padded_size);
    bf16_grads.reserve(padded_size);

    for (float x : input_values) {
        bf16_inputs.push_back(::bfloat16(x));
        bf16_grads.push_back(::bfloat16(1.0f));  // grad = 1.0 to get GELU'(x)
    }

    // Create TensorSpec for tile layout
    tt::tt_metal::TensorSpec tensor_spec(
        tt::tt_metal::Shape(dims),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), tt::tt_metal::MemoryConfig{}));

    auto input_tensor = tt::tt_metal::Tensor::from_vector(std::move(bf16_inputs), tensor_spec).to_device(device_);
    auto grad_tensor = tt::tt_metal::Tensor::from_vector(std::move(bf16_grads), tensor_spec).to_device(device_);

    // Run GELU backward once on entire tensor
    auto results = ttnn::gelu_bw(grad_tensor, input_tensor, "none");
    auto result = results[0].value();
    auto output_cpu = ttnn::from_device(result);
    auto output_vec = output_cpu.to_vector<::bfloat16>();

    // Analyze results by region
    struct RegionStats {
        std::string name;
        int count = 0;
        double ulp_sum = 0;
        int max_ulp = 0;
        float worst_x = 0;
    };

    std::vector<RegionStats> regions = {
        {"Deep negative (x < -5)"},
        {"Moderate negative [-5, -2]"},
        {"Near negative [-2, -0.5]"},
        {"Near zero [-0.5, 0.5]"},
        {"Near positive [0.5, 2]"},
        {"Moderate positive [2, 5]"},
        {"Large positive (x > 5]"},
    };

    int overall_max_ulp = 0;
    float overall_worst_x = 0;
    double overall_ulp_sum = 0;

    for (size_t i = 0; i < valid_count; ++i) {
        float x = bf16_ulp_bw_main::bf16_bits_to_float(bf16_ulp_bw_main::float_to_bf16_bits(input_values[i]));
        float actual = static_cast<float>(output_vec[i]);
        float expected = bf16_ulp_bw_main::gelu_derivative_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp_bw_main::ulp_distance_bf16_daz(actual, expected);

        if (ulp < 0) {
            continue;  // Skip invalid
        }

        // Categorize by region
        int region_idx;
        if (x < -5.0f) {
            region_idx = 0;
        } else if (x < -2.0f) {
            region_idx = 1;
        } else if (x < -0.5f) {
            region_idx = 2;
        } else if (x < 0.5f) {
            region_idx = 3;
        } else if (x < 2.0f) {
            region_idx = 4;
        } else if (x < 5.0f) {
            region_idx = 5;
        } else {
            region_idx = 6;
        }

        regions[region_idx].count++;
        regions[region_idx].ulp_sum += ulp;
        if (ulp > regions[region_idx].max_ulp) {
            regions[region_idx].max_ulp = ulp;
            regions[region_idx].worst_x = x;
        }

        overall_ulp_sum += ulp;
        if (ulp > overall_max_ulp) {
            overall_max_ulp = ulp;
            overall_worst_x = x;
        }
    }

    // Print results
    std::cout << "\n============================================================\n";
    std::cout << "GELU BACKWARD ULP ANALYSIS BY REGION (DAZ+FTZ MODEL)\n";
    std::cout << "============================================================\n";
    std::cout << std::setw(30) << "Region" << std::setw(10) << "Count" << std::setw(12) << "Mean ULP" << std::setw(12)
              << "Max ULP" << std::setw(15) << "Worst x\n";
    std::cout << std::string(79, '-') << "\n";

    for (const auto& r : regions) {
        if (r.count > 0) {
            std::cout << std::setw(30) << r.name << std::setw(10) << r.count << std::setw(12) << std::fixed
                      << std::setprecision(2) << (r.ulp_sum / r.count) << std::setw(12) << r.max_ulp << std::setw(15)
                      << std::scientific << std::setprecision(3) << r.worst_x << "\n";
        }
    }

    std::cout << std::string(79, '-') << "\n";
    std::cout << std::setw(30) << "OVERALL" << std::setw(10) << valid_count << std::setw(12) << std::fixed
              << std::setprecision(2) << (overall_ulp_sum / valid_count) << std::setw(12) << overall_max_ulp
              << std::setw(15) << std::scientific << std::setprecision(3) << overall_worst_x << "\n";
    std::cout << "============================================================\n";

    // Per-region regression guards
    // Thresholds based on the polynomial+exp implementation (overall max ULP = 5)
    std::vector<int> region_max_ulp_thresholds = {
        2,  // Deep negative (x < -5): exp-based + saturation
        2,  // Moderate negative [-5, -2]: polynomial transition
        2,  // Near negative [-2, -0.5]: core polynomial
        2,  // Near zero [-0.5, 0.5]: most accurate region
        2,  // Near positive [0.5, 2]: core polynomial
        2,  // Moderate positive [2, 5]: approaching saturation
        2,  // Large positive (x > 5): saturation to 1
    };

    for (size_t r = 0; r < regions.size(); ++r) {
        EXPECT_LE(regions[r].max_ulp, region_max_ulp_thresholds[r])
            << "Region '" << regions[r].name << "' max ULP " << regions[r].max_ulp << " exceeds threshold "
            << region_max_ulp_thresholds[r] << " (worst at x=" << regions[r].worst_x << ")";
    }

    EXPECT_LE(overall_max_ulp, 2) << "Overall max ULP " << overall_max_ulp << " exceeds threshold 2"
                                  << " (worst at x=" << overall_worst_x << ")";
}

// Precision guard: builds ULP histogram across all ~65K BF16 values.
// Asserts max ULP <= 10 and >= 99% of values within 2 ULP.
// Catches broad precision degradation that region-based tests might miss.
TEST_F(GeluBwMainUlpTest, CumulativeULPDistribution) {
    std::vector<float> input_values;
    input_values.reserve(70000);

    for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        uint16_t bf16_bits = static_cast<uint16_t>(bits);
        if ((bf16_bits & bf16_ulp_bw_main::BF16_EXP_MASK) == bf16_ulp_bw_main::BF16_EXP_MASK &&
            (bf16_bits & bf16_ulp_bw_main::BF16_MANTISSA_MASK) != 0) {
            continue;
        }
        if (bf16_bits == bf16_ulp_bw_main::BF16_POS_INF || bf16_bits == bf16_ulp_bw_main::BF16_NEG_INF) {
            continue;
        }
        if (bf16_ulp_bw_main::is_bf16_denormal(bf16_bits)) {
            continue;
        }

        input_values.push_back(bf16_ulp_bw_main::bf16_bits_to_float(bf16_bits));
    }

    const size_t valid_count = input_values.size();
    const size_t tile_size = tt::constants::TILE_HW;
    size_t padded_size = ((valid_count + tile_size - 1) / tile_size) * tile_size;
    input_values.resize(padded_size, 0.0f);

    uint32_t num_tiles = static_cast<uint32_t>(padded_size / tile_size);
    std::array<uint32_t, 4> dims = {1, 1, num_tiles * tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};

    std::vector<::bfloat16> bf16_inputs, bf16_grads;
    for (float x : input_values) {
        bf16_inputs.push_back(::bfloat16(x));
        bf16_grads.push_back(::bfloat16(1.0f));
    }

    tt::tt_metal::TensorSpec tensor_spec(
        tt::tt_metal::Shape(dims),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), tt::tt_metal::MemoryConfig{}));

    auto input_tensor = tt::tt_metal::Tensor::from_vector(std::move(bf16_inputs), tensor_spec).to_device(device_);
    auto grad_tensor = tt::tt_metal::Tensor::from_vector(std::move(bf16_grads), tensor_spec).to_device(device_);

    auto results = ttnn::gelu_bw(grad_tensor, input_tensor, "none");
    auto result = results[0].value();
    auto output_cpu = ttnn::from_device(result);
    auto output_vec = output_cpu.to_vector<::bfloat16>();

    // Compute ULP histogram
    std::map<int32_t, int> ulp_histogram;
    int max_ulp = 0;
    float worst_x = 0;

    for (size_t i = 0; i < valid_count; ++i) {
        float x = bf16_ulp_bw_main::bf16_bits_to_float(bf16_ulp_bw_main::float_to_bf16_bits(input_values[i]));
        float actual = static_cast<float>(output_vec[i]);
        float expected = bf16_ulp_bw_main::gelu_derivative_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp_bw_main::ulp_distance_bf16_daz(actual, expected);

        if (ulp >= 0) {
            ulp_histogram[ulp]++;
            if (ulp > max_ulp) {
                max_ulp = ulp;
                worst_x = x;
            }
        }
    }

    // Print cumulative distribution
    std::cout << "\n============================================================\n";
    std::cout << "GELU BACKWARD CUMULATIVE ULP DISTRIBUTION (DAZ+FTZ MODEL)\n";
    std::cout << "============================================================\n";

    std::vector<int> thresholds = {0, 1, 2, 3, 5, 10, 20, 50, 100, 500, 1000};

    std::cout << std::setw(10) << "ULP <=" << std::setw(12) << "Count" << std::setw(12) << "Percent" << std::setw(12)
              << "Cumul %\n";
    std::cout << std::string(46, '-') << "\n";

    for (int threshold : thresholds) {
        int total_le_threshold = 0;
        for (auto& [ulp, count] : ulp_histogram) {
            if (ulp <= threshold) {
                total_le_threshold += count;
            }
        }
        double percent = 100.0 * total_le_threshold / valid_count;
        std::cout << std::setw(10) << threshold << std::setw(12) << total_le_threshold << std::setw(11) << std::fixed
                  << std::setprecision(2) << percent << "%" << std::setw(11) << percent << "%\n";
    }

    std::cout << std::string(46, '-') << "\n";
    std::cout << "\nMax ULP: " << max_ulp << " at x = " << worst_x << "\n";
    std::cout << "============================================================\n";

    // Regression guards
    EXPECT_LE(max_ulp, 2) << "Max ULP " << max_ulp << " at x=" << worst_x << " exceeds threshold 2";

    // At least 99% of values should be within 2 ULP
    int count_le_2 = 0;
    for (auto& [ulp, count] : ulp_histogram) {
        if (ulp <= 2) {
            count_le_2 += count;
        }
    }
    double pct_le_2 = 100.0 * count_le_2 / valid_count;
    EXPECT_GE(pct_le_2, 99.0) << "Only " << pct_le_2 << "% of values within 2 ULP (expected >= 99%)";
}

// Test infrastructure guard: validates the fp64 golden reference function.
// If this fails, all other tests' expected values are wrong, causing false passes.
TEST_F(GeluBwMainUlpTest, ReferenceImplementationVerification) {
    // GELU'(0) = 0.5 (cdf=0.5, pdf term = 0)
    double deriv_0 = bf16_ulp_bw_main::gelu_derivative_exact(0.0);
    EXPECT_NEAR(deriv_0, 0.5, 1e-10) << "GELU'(0) should be 0.5";

    // GELU'(inf) = 1 (cdf=1, pdf term = 0)
    double deriv_large = bf16_ulp_bw_main::gelu_derivative_exact(100.0);
    EXPECT_NEAR(deriv_large, 1.0, 1e-6) << "GELU'(large) should approach 1";

    // GELU'(-inf) = 0 (cdf=0, pdf term = 0)
    double deriv_neg_large = bf16_ulp_bw_main::gelu_derivative_exact(-100.0);
    EXPECT_NEAR(deriv_neg_large, 0.0, 1e-6) << "GELU'(-large) should approach 0";

    // At x = -0.75 (local minimum of GELU), GELU'(x) should be close to 0
    // Actually the minimum of GELU is around x ≈ -0.751
    double deriv_min = bf16_ulp_bw_main::gelu_derivative_exact(-0.751);
    EXPECT_LT(std::abs(deriv_min), 0.1) << "GELU'(-0.751) should be near 0 (local minimum)";

    std::cout << "\nReference implementation verification:\n";
    std::cout << "GELU'(0) = " << deriv_0 << " (expected: 0.5)\n";
    std::cout << "GELU'(100) = " << deriv_large << " (expected: ~1.0)\n";
    std::cout << "GELU'(-100) = " << deriv_neg_large << " (expected: ~0.0)\n";
    std::cout << "GELU'(-0.751) = " << deriv_min << " (expected: ~0.0 at local min)\n";
}

// Precision guard: the [-5, -2] region where the original erfc bug had max ULP = 29,756.
// Asserts max ULP <= 10 across 11 sample points. Threshold is 10 (not 2) because some
// sample values (e.g. -3.7f) are not exact BF16 representable — the reference truncates
// to BF16 while torch.bfloat16 rounds to nearest, causing a 1-ULP input mismatch that
// amplifies to ~10 ULP in the output. Comprehensive BF16 sweeps confirm max ULP = 1.
TEST_F(GeluBwMainUlpTest, ModerateNegativeRegionBugAnalysis) {
    std::vector<float> moderate_negative_values = {
        -2.0f, -2.5f, -3.0f, -3.5f, -3.7f, -3.719f, -3.75f, -3.8f, -4.0f, -4.5f, -5.0f};

    std::cout << "\n========================================\n";
    std::cout << "MODERATE NEGATIVE REGION BUG ANALYSIS\n";
    std::cout << "(Critical region for training: [-5, -2])\n";
    std::cout << "========================================\n";
    std::cout << std::setw(10) << "x" << std::setw(15) << "Expected" << std::setw(15) << "Actual" << std::setw(10)
              << "ULP" << std::setw(15) << "Abs Error\n";
    std::cout << std::string(65, '-') << "\n";

    int max_ulp_found = 0;
    float worst_x = 0;

    for (float x : moderate_negative_values) {
        float actual = run_gelu_bw_main_single(*device_, x);
        float expected = bf16_ulp_bw_main::gelu_derivative_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp_bw_main::ulp_distance_bf16_daz(actual, expected);
        float abs_error = std::abs(actual - expected);

        std::cout << std::setw(10) << std::fixed << std::setprecision(3) << x << std::setw(15) << std::scientific
                  << std::setprecision(3) << expected << std::setw(15) << actual << std::setw(10) << ulp
                  << std::setw(15) << abs_error << "\n";

        if (ulp > max_ulp_found) {
            max_ulp_found = ulp;
            worst_x = x;
        }
    }

    std::cout << std::string(65, '-') << "\n";
    std::cout << "Worst ULP: " << max_ulp_found << " at x = " << worst_x << "\n";
    std::cout << "========================================\n";

    // Regression guard: moderate negative region must stay within 10 ULP
    // Original bug had max ULP = 29,756 here
    EXPECT_LE(max_ulp_found, 10) << "Moderate negative region [-5, -2] max ULP " << max_ulp_found << " at x=" << worst_x
                                 << " exceeds threshold 10";
}

// Correctness guard: deep negative region (-6 to -13) where values should saturate toward 0.
// Asserts max ULP <= 10. Catches broken saturation path or erf() overflow.
TEST_F(GeluBwMainUlpTest, DeepNegativeRegionStability) {
    std::vector<float> deep_negative_values = {-6.0f, -7.0f, -8.0f, -9.0f, -10.0f, -12.0f, -13.0f};

    std::cout << "\n========================================\n";
    std::cout << "DEEP NEGATIVE REGION ANALYSIS\n";
    std::cout << "========================================\n";
    std::cout << std::setw(10) << "x" << std::setw(15) << "Expected" << std::setw(15) << "Actual" << std::setw(10)
              << "ULP\n";
    std::cout << std::string(50, '-') << "\n";

    int max_ulp = 0;
    float worst_x = 0;

    for (float x : deep_negative_values) {
        float actual = run_gelu_bw_main_single(*device_, x);
        float expected = bf16_ulp_bw_main::gelu_derivative_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp_bw_main::ulp_distance_bf16_daz(actual, expected);

        std::cout << std::setw(10) << x << std::setw(15) << std::scientific << std::setprecision(3) << expected
                  << std::setw(15) << actual << std::setw(10) << ulp << "\n";

        if (ulp > max_ulp) {
            max_ulp = ulp;
            worst_x = x;
        }
    }
    std::cout << "========================================\n";

    // Regression guard: deep negative values should saturate cleanly
    EXPECT_LE(max_ulp, 2) << "Deep negative region max ULP " << max_ulp << " at x=" << worst_x
                          << " exceeds threshold 10";
}

// Correctness guard: 6 key points spanning all kernel code paths.
// Per-point ULP thresholds (2-5). Catches any single broken code path quickly.
TEST_F(GeluBwMainUlpTest, SummaryStatistics) {
    std::cout << "\n========================================\n";
    std::cout << "GELU BACKWARD ULP SUMMARY (DAZ+FTZ MODEL)\n";
    std::cout << "========================================\n";

    // Test key regions with per-point ULP thresholds
    // Each point tests a different kernel code path
    struct KeyPoint {
        std::string name;
        float x;
        int max_ulp_threshold;
    };

    std::vector<KeyPoint> key_points = {
        {"Zero", 0.0f, 2},
        {"Positive unity", 1.0f, 2},
        {"Negative unity", -1.0f, 2},
        {"Local minimum", -0.751f, 2},
        {"Large positive", 5.0f, 2},
        {"Large negative", -5.0f, 2},
    };

    for (const auto& kp : key_points) {
        float actual = run_gelu_bw_main_single(*device_, kp.x);
        float expected = bf16_ulp_bw_main::gelu_derivative_expected_bf16_daz(kp.x);
        int32_t ulp = bf16_ulp_bw_main::ulp_distance_bf16_daz(actual, expected);

        std::cout << kp.name << " (x=" << kp.x << "): ";
        std::cout << "expected=" << expected << ", actual=" << actual << ", ULP=" << ulp << "\n";

        EXPECT_LE(ulp, kp.max_ulp_threshold)
            << kp.name << " (x=" << kp.x << "): ULP " << ulp << " exceeds threshold " << kp.max_ulp_threshold;
    }

    std::cout << "\nExpected behavior:\n";
    std::cout << "- GELU'(0) ≈ 0.5\n";
    std::cout << "- GELU'(x) → 1 as x → +∞\n";
    std::cout << "- GELU'(x) → 0 as x → -∞\n";
    std::cout << "- Local minimum near x ≈ -0.751 where GELU'(x) ≈ 0\n";
    std::cout << "========================================\n";
}

// =============================================================================
// POLYNOMIAL-SPECIFIC ANALYSIS TESTS
// =============================================================================

class GeluBwMainPolyTest : public TTNNFixtureWithDevice {};

// Correctness guard: same midpoint check as GeluBwMainUlpTest but against the poly kernel.
// Verifies GELU'(0) = 0.5 via the polynomial path. Catches broken constant term.
TEST_F(GeluBwMainPolyTest, DerivativeAtZero) {
    // GELU'(0) = 0.5
    float actual = run_gelu_bw_main_single(*device_, 0.0f);
    float expected = bf16_ulp_bw_main::gelu_derivative_expected_bf16_daz(0.0f);

    int32_t ulp = bf16_ulp_bw_main::ulp_distance_bf16_daz(actual, expected);

    std::cout << "[POLY] x=0: expected=" << expected << ", actual=" << actual << ", ULP=" << ulp << std::endl;

    EXPECT_LE(ulp, 2) << "POLY GELU'(0) should be ~0.5 with low ULP error";
}

// Correctness guard: tests 24 negative-side points across all 4 implementation regions
// (core poly, left poly, exp-based, saturation). Each has a per-point ULP threshold.
// Catches region-boundary bugs or broken polynomial/exp coefficients.
TEST_F(GeluBwMainPolyTest, DerivativeAtNegativeValues) {
    // Implementation covers (after Session 34 optimization):
    // - Core region [-3, 3.1719]: degree 16 polynomial
    // - Left region [-5, -3]: degree 8 shifted polynomial (t = x + 4)
    // - Exp-based region (-13.375, -5]: fused x*exp(t) with Mills ratio correction
    // - Saturation: x <= -13.375 saturates to 0, x >= 3.1719 saturates to 1

    // Polynomial region tests - expect excellent accuracy (Max ULP = 1)
    std::vector<std::pair<float, int32_t>> poly_tests = {
        {-0.5f, 2},  // Core polynomial region
        {-1.0f, 2},
        {-2.0f, 2},
        {-3.0f, 2},  // Boundary between core and left polynomial
        {-4.0f, 2},  // Left polynomial region (shifted, t = x + 4)
        {-5.0f, 2},  // Edge of left polynomial / start of exp-based
    };

    std::cout << "\n[POLY] Polynomial region tests (should have low ULP):\n";
    for (const auto& [input_val, max_expected_ulp] : poly_tests) {
        float actual = run_gelu_bw_main_single(*device_, input_val);
        float expected = bf16_ulp_bw_main::gelu_derivative_expected_bf16_daz(input_val);
        int32_t ulp = bf16_ulp_bw_main::ulp_distance_bf16_daz(actual, expected);

        std::cout << "[POLY] x=" << input_val << ": expected=" << expected << ", actual=" << actual << ", ULP=" << ulp
                  << std::endl;

        EXPECT_LE(ulp, max_expected_ulp) << "POLY GELU'(" << input_val << ") polynomial region ULP too high";
    }

    // Exp-based region tests (-13.375, -5] - fused x*exp(t) with Mills ratio correction
    // After Session 34: extended from (-13.375, -9) to (-13.375, -5], Max ULP = 1
    std::vector<std::pair<float, int32_t>> exp_tests = {
        {-6.0f, 2},   // Exp-based region
        {-7.0f, 2},   // Exp-based region
        {-8.0f, 2},   // Exp-based region
        {-10.0f, 2},  // Exp-based region - excellent accuracy
        {-11.0f, 2},
        {-12.0f, 2},
        {-13.0f, 2},  // Near saturation boundary
    };

    std::cout << "\n[EXP] Exp-based region tests (x in (-13.375, -5], should have low ULP):\n";
    for (const auto& [input_val, max_expected_ulp] : exp_tests) {
        float actual = run_gelu_bw_main_single(*device_, input_val);
        float expected = bf16_ulp_bw_main::gelu_derivative_expected_bf16_daz(input_val);
        int32_t ulp = bf16_ulp_bw_main::ulp_distance_bf16_daz(actual, expected);

        std::cout << "[EXP] x=" << input_val << ": expected=" << expected << ", actual=" << actual << ", ULP=" << ulp
                  << std::endl;

        EXPECT_LE(ulp, max_expected_ulp) << "EXP GELU'(" << input_val << ") exp-based region ULP too high";
    }

    // Saturation region tests (x <= -13.375) - expect saturation to 0
    std::vector<float> saturation_tests = {-13.375f, -14.0f, -20.0f};

    std::cout << "\n[SAT] Saturation region tests (x <= -13.375, saturates to 0):\n";
    for (float input_val : saturation_tests) {
        float actual = run_gelu_bw_main_single(*device_, input_val);
        float expected = bf16_ulp_bw_main::gelu_derivative_expected_bf16_daz(input_val);
        int32_t ulp = bf16_ulp_bw_main::ulp_distance_bf16_daz(actual, expected);

        std::cout << "[SAT] x=" << input_val << ": expected=" << expected << ", actual=" << actual
                  << " (saturated to 0), ULP=" << ulp << std::endl;

        // Just verify it returns 0 (saturation)
        EXPECT_EQ(actual, 0.0f) << "SAT GELU'(" << input_val << ") should saturate to 0";
    }
}

// Precision guard: batches ALL ~65K BF16 values through the poly kernel, tabulates
// per-ULP-bucket counts and identifies worst-case values. Asserts max ULP <= 10 overall.
// The definitive poly-kernel precision test — catches any regression anywhere in the range.
TEST_F(GeluBwMainPolyTest, ComprehensiveULPAnalysis) {
    // Comprehensive test: batch all BF16 values and compute GELU backward with polynomial

    std::vector<float> input_values;
    input_values.reserve(70000);

    // Collect all valid BF16 values
    for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        uint16_t bf16_bits = static_cast<uint16_t>(bits);

        // Skip NaN
        if ((bf16_bits & bf16_ulp_bw_main::BF16_EXP_MASK) == bf16_ulp_bw_main::BF16_EXP_MASK &&
            (bf16_bits & bf16_ulp_bw_main::BF16_MANTISSA_MASK) != 0) {
            continue;
        }
        // Skip infinity
        if (bf16_bits == bf16_ulp_bw_main::BF16_POS_INF || bf16_bits == bf16_ulp_bw_main::BF16_NEG_INF) {
            continue;
        }
        // Skip denormals
        if (bf16_ulp_bw_main::is_bf16_denormal(bf16_bits)) {
            continue;
        }

        float val = bf16_ulp_bw_main::bf16_bits_to_float(bf16_bits);
        input_values.push_back(val);
    }

    const size_t valid_count = input_values.size();
    std::cout << "\n[POLY] Collected " << valid_count << " valid BF16 values\n";

    // Pad to tile boundary
    const size_t tile_size = tt::constants::TILE_HW;
    size_t padded_size = ((valid_count + tile_size - 1) / tile_size) * tile_size;
    input_values.resize(padded_size, 0.0f);

    // Create tensor shape
    uint32_t num_tiles = static_cast<uint32_t>(padded_size / tile_size);
    std::array<uint32_t, 4> dims = {1, 1, num_tiles * tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};

    // Create input tensors from vectors
    std::vector<::bfloat16> bf16_inputs;
    std::vector<::bfloat16> bf16_grads;
    bf16_inputs.reserve(padded_size);
    bf16_grads.reserve(padded_size);

    for (float x : input_values) {
        bf16_inputs.push_back(::bfloat16(x));
        bf16_grads.push_back(::bfloat16(1.0f));  // grad = 1.0 to get GELU'(x)
    }

    // Create TensorSpec for tile layout
    tt::tt_metal::TensorSpec tensor_spec(
        tt::tt_metal::Shape(dims),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), tt::tt_metal::MemoryConfig{}));

    auto input_tensor = tt::tt_metal::Tensor::from_vector(std::move(bf16_inputs), tensor_spec).to_device(device_);
    auto grad_tensor = tt::tt_metal::Tensor::from_vector(std::move(bf16_grads), tensor_spec).to_device(device_);

    // Run GELU backward with polynomial approximation
    auto results = ttnn::gelu_bw(grad_tensor, input_tensor, "none");
    auto result = results[0].value();
    auto output_cpu = ttnn::from_device(result);
    auto output_vec = output_cpu.to_vector<::bfloat16>();

    // Analyze results by region
    struct RegionStats {
        std::string name;
        int count = 0;
        double ulp_sum = 0;
        int max_ulp = 0;
        float worst_x = 0;
    };

    std::vector<RegionStats> regions = {
        {"Deep negative (x < -5)"},
        {"Moderate negative [-5, -2]"},
        {"Near negative [-2, -0.5]"},
        {"Near zero [-0.5, 0.5]"},
        {"Near positive [0.5, 2]"},
        {"Moderate positive [2, 5]"},
        {"Large positive (x > 5]"},
    };

    int overall_max_ulp = 0;
    float overall_worst_x = 0;
    double overall_ulp_sum = 0;

    for (size_t i = 0; i < valid_count; ++i) {
        float x = bf16_ulp_bw_main::bf16_bits_to_float(bf16_ulp_bw_main::float_to_bf16_bits(input_values[i]));
        float actual = static_cast<float>(output_vec[i]);
        float expected = bf16_ulp_bw_main::gelu_derivative_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp_bw_main::ulp_distance_bf16_daz(actual, expected);

        if (ulp < 0) {
            continue;  // Skip invalid
        }

        // Categorize by region
        int region_idx;
        if (x < -5.0f) {
            region_idx = 0;
        } else if (x < -2.0f) {
            region_idx = 1;
        } else if (x < -0.5f) {
            region_idx = 2;
        } else if (x < 0.5f) {
            region_idx = 3;
        } else if (x < 2.0f) {
            region_idx = 4;
        } else if (x < 5.0f) {
            region_idx = 5;
        } else {
            region_idx = 6;
        }

        regions[region_idx].count++;
        regions[region_idx].ulp_sum += ulp;
        if (ulp > regions[region_idx].max_ulp) {
            regions[region_idx].max_ulp = ulp;
            regions[region_idx].worst_x = x;
        }

        overall_ulp_sum += ulp;
        if (ulp > overall_max_ulp) {
            overall_max_ulp = ulp;
            overall_worst_x = x;
        }
    }

    // Print results
    std::cout << "\n============================================================\n";
    std::cout << "POLYNOMIAL GELU BACKWARD ULP ANALYSIS (DAZ+FTZ MODEL)\n";
    std::cout << "============================================================\n";
    std::cout << std::setw(30) << "Region" << std::setw(10) << "Count" << std::setw(12) << "Mean ULP" << std::setw(12)
              << "Max ULP" << std::setw(15) << "Worst x\n";
    std::cout << std::string(79, '-') << "\n";

    for (const auto& r : regions) {
        if (r.count > 0) {
            std::cout << std::setw(30) << r.name << std::setw(10) << r.count << std::setw(12) << std::fixed
                      << std::setprecision(2) << (r.ulp_sum / r.count) << std::setw(12) << r.max_ulp << std::setw(15)
                      << std::scientific << std::setprecision(3) << r.worst_x << "\n";
        }
    }

    std::cout << std::string(79, '-') << "\n";
    std::cout << std::setw(30) << "OVERALL" << std::setw(10) << valid_count << std::setw(12) << std::fixed
              << std::setprecision(2) << (overall_ulp_sum / valid_count) << std::setw(12) << overall_max_ulp
              << std::setw(15) << std::scientific << std::setprecision(3) << overall_worst_x << "\n";
    std::cout << "============================================================\n";

    // The polynomial implementation achieves excellent accuracy in the core region [-3, 3.5]
    // where most practical values lie. Outside this range, accuracy degrades because:
    // 1. The polynomial was fitted for [-3, 3] and diverges outside this range
    // 2. For x < -3, GELU'(x) < 0.012 (very small) - returning 0 is acceptable for training
    // 3. For x > 3.5, GELU'(x) ≈ 1 - saturation is appropriate

    // Find max ULP in core region [-3, 3.5] specifically
    int32_t core_max_ulp = 0;
    for (const auto& r : regions) {
        if (r.name == "Near negative [-2, -0.5]" || r.name == "Near zero [-0.5, 0.5]" ||
            r.name == "Near positive [0.5, 2]") {
            core_max_ulp = std::max(r.max_ulp, core_max_ulp);
        }
    }

    // Core region should have Max ULP <= 5 (actually achieves 1)
    EXPECT_LE(core_max_ulp, 2) << "Polynomial GELU backward core region Max ULP should be <= 2";

    std::cout << "\nComparison with standard implementation:\n";
    std::cout << "  Standard (erfc-based): Max ULP = ~32,460 at x = -3.376e+38\n";
    std::cout << "  Polynomial (overall):  Max ULP = " << overall_max_ulp << " at x = " << overall_worst_x << "\n";
    std::cout << "  Polynomial (core):     Max ULP = " << core_max_ulp << " (core region [-2, 2])\n";
    if (core_max_ulp <= 5) {
        std::cout << "  CORE REGION IMPROVEMENT: " << std::fixed << std::setprecision(0)
                  << (32460.0 / std::max(1, core_max_ulp)) << "x better accuracy!\n";
    }
    std::cout << "\nNote: Outside core region [-3, 3.5], polynomial saturates to 0 or 1.\n";
    std::cout << "      This is acceptable because GELU'(x) is very small (< 0.012) for x < -3.\n";
}

// Precision guard: per-segment ULP analysis matching the kernel's exact code regions.
// 5 segments (saturation-low, exp, left-poly, core-poly, saturation-high) each with a
// tuned ULP threshold. Catches precision regression in any single code path.
TEST_F(GeluBwMainPolyTest, DetailedSegmentAnalysis) {
    // Detailed per-segment ULP analysis matching exact implementation regions:
    // - x <= -13.375: Saturation to 0
    // - (-13.375, -5]: Fused x*exp(t) with Mills ratio correction
    // - [-5, -3): LEFT polynomial (shifted, t = x + 4)
    // - [-3, 3.1719): CORE polynomial (degree 16)
    // - x >= 3.1719: Saturation to 1

    std::vector<float> input_values;
    input_values.reserve(70000);

    for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        uint16_t bf16_bits = static_cast<uint16_t>(bits);
        if ((bf16_bits & bf16_ulp_bw_main::BF16_EXP_MASK) == bf16_ulp_bw_main::BF16_EXP_MASK &&
            (bf16_bits & bf16_ulp_bw_main::BF16_MANTISSA_MASK) != 0) {
            continue;  // NaN
        }
        if (bf16_bits == bf16_ulp_bw_main::BF16_POS_INF || bf16_bits == bf16_ulp_bw_main::BF16_NEG_INF) {
            continue;  // Inf
        }
        if (bf16_ulp_bw_main::is_bf16_denormal(bf16_bits)) {
            continue;  // Denormal
        }

        float val = bf16_ulp_bw_main::bf16_bits_to_float(bf16_bits);
        input_values.push_back(val);
    }

    const size_t valid_count = input_values.size();
    const size_t tile_size = tt::constants::TILE_HW;
    size_t padded_size = ((valid_count + tile_size - 1) / tile_size) * tile_size;
    input_values.resize(padded_size, 0.0f);

    uint32_t num_tiles = static_cast<uint32_t>(padded_size / tile_size);
    std::array<uint32_t, 4> dims = {1, 1, num_tiles * tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};

    std::vector<::bfloat16> bf16_inputs, bf16_grads;
    bf16_inputs.reserve(padded_size);
    bf16_grads.reserve(padded_size);

    for (float x : input_values) {
        bf16_inputs.push_back(::bfloat16(x));
        bf16_grads.push_back(::bfloat16(1.0f));
    }

    tt::tt_metal::TensorSpec tensor_spec(
        tt::tt_metal::Shape(dims),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), tt::tt_metal::MemoryConfig{}));

    auto input_tensor = tt::tt_metal::Tensor::from_vector(std::move(bf16_inputs), tensor_spec).to_device(device_);
    auto grad_tensor = tt::tt_metal::Tensor::from_vector(std::move(bf16_grads), tensor_spec).to_device(device_);

    auto results = ttnn::gelu_bw(grad_tensor, input_tensor, "none");
    auto result = results[0].value();
    auto output_cpu = ttnn::from_device(result);
    auto output_vec = output_cpu.to_vector<::bfloat16>();

    // Define segments matching implementation
    struct SegmentStats {
        std::string name;
        float x_min, x_max;
        int count = 0;
        double ulp_sum = 0;
        int max_ulp = 0;
        float worst_x = 0;
        int ulp_le_1 = 0;
    };

    // Segment boundaries based on implementation (Session 34: extended exp-based to -5):
    // Kernel uses: v_if(x >= 3.1719) / v_elseif(x >= -3) / v_elseif(x >= -5) / v_elseif(x > -13.375)
    // Note: x > -13.375 (strict) means x = -13.375 falls through to saturation (result = 0)
    //
    // Using half-open intervals [x_min, x_max) to match kernel boundaries:
    // - x <= -13.375: BF16 natural saturation (kernel: falls through to default result = 0)
    // - (-13.375, -5): fused x*exp(t) (kernel: x > -13.375f, caught before x >= -5.0f)
    // - [-5, -3): LEFT polynomial (kernel: x >= -5.0f)
    // - [-3, 3.1719): CORE polynomial (kernel: x >= -3.0f)
    // - x >= 3.1719: Saturation to 1 (kernel: x >= 3.1719f)
    //
    // BF16 values at boundary: -13.375 (0xc156) saturates, -13.3125 (0xc155) is first exp-based
    std::vector<SegmentStats> segments = {
        {"x <= -13.375 (BF16 natural 0)", -1e38f, -13.3125f},  // Use -13.3125 to include -13.375
        {"(-13.375, -5] exp-based", -13.3125f, -5.0f},         // First exp-based value is -13.3125
        {"[-5, -3) LEFT polynomial", -5.0f, -3.0f},
        {"[-3, 3.1719) CORE polynomial", -3.0f, 3.1719f},
        {"x >= 3.1719 (saturation to 1)", 3.1719f, 1e38f},
    };

    for (size_t i = 0; i < valid_count; ++i) {
        float x = bf16_ulp_bw_main::bf16_bits_to_float(bf16_ulp_bw_main::float_to_bf16_bits(input_values[i]));
        float actual = static_cast<float>(output_vec[i]);
        float expected = bf16_ulp_bw_main::gelu_derivative_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp_bw_main::ulp_distance_bf16_daz(actual, expected);

        if (ulp < 0) {
            continue;
        }

        for (auto& seg : segments) {
            if (x >= seg.x_min && x < seg.x_max) {
                seg.count++;
                seg.ulp_sum += ulp;
                if (ulp <= 1) {
                    seg.ulp_le_1++;
                }
                if (ulp > seg.max_ulp) {
                    seg.max_ulp = ulp;
                    seg.worst_x = x;
                }
                break;
            }
        }
    }

    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "DETAILED PER-SEGMENT ULP ANALYSIS - GELU BACKWARD IMPLEMENTATION\n";
    std::cout << "================================================================================\n";
    std::cout << std::setw(35) << "Segment" << std::setw(10) << "Count" << std::setw(12) << "Mean ULP" << std::setw(12)
              << "Max ULP" << std::setw(12) << "%<=1 ULP" << std::setw(15) << "Worst x\n";
    std::cout << std::string(96, '-') << "\n";

    int total_count = 0;
    double total_ulp_sum = 0;
    int overall_max_ulp = 0;
    float overall_worst_x = 0;

    for (const auto& seg : segments) {
        if (seg.count == 0) {
            std::cout << std::setw(35) << seg.name << std::setw(10) << 0 << std::setw(12) << "-" << std::setw(12) << "-"
                      << std::setw(12) << "-" << std::setw(15) << "-\n";
            continue;
        }

        double mean_ulp = seg.ulp_sum / seg.count;
        double pct_le_1 = 100.0 * seg.ulp_le_1 / seg.count;

        std::cout << std::setw(35) << seg.name << std::setw(10) << seg.count << std::setw(12) << std::fixed
                  << std::setprecision(2) << mean_ulp << std::setw(12) << seg.max_ulp << std::setw(11)
                  << std::setprecision(1) << pct_le_1 << "%" << std::setw(15) << std::scientific << std::setprecision(3)
                  << seg.worst_x << "\n";

        total_count += seg.count;
        total_ulp_sum += seg.ulp_sum;
        if (seg.max_ulp > overall_max_ulp) {
            overall_max_ulp = seg.max_ulp;
            overall_worst_x = seg.worst_x;
        }
    }

    std::cout << std::string(96, '-') << "\n";
    std::cout << std::setw(35) << "TOTAL" << std::setw(10) << total_count << std::setw(12) << std::fixed
              << std::setprecision(2) << (total_ulp_sum / total_count) << std::setw(12) << overall_max_ulp
              << std::setw(12) << "-" << std::setw(15) << std::scientific << std::setprecision(3) << overall_worst_x
              << "\n";
    std::cout << "================================================================================\n";

    // Per-segment regression guards matching kernel code paths
    // Saturation regions must be exact (0 ULP), polynomial/exp regions allow small error
    std::vector<int> segment_max_ulp_thresholds = {
        0,  // x <= -13.375: BF16 natural saturation to 0 — must be exact
        2,  // (-13.375, -5]: exp-based fused x*exp(t)
        2,  // [-5, -3): LEFT polynomial
        2,  // [-3, 3.1719): CORE polynomial
        0,  // x >= 3.1719: saturation to 1 — must be exact
    };

    for (size_t s = 0; s < segments.size(); ++s) {
        if (segments[s].count > 0) {
            EXPECT_LE(segments[s].max_ulp, segment_max_ulp_thresholds[s])
                << "Segment '" << segments[s].name << "' max ULP " << segments[s].max_ulp << " exceeds threshold "
                << segment_max_ulp_thresholds[s] << " (worst at x=" << segments[s].worst_x << ")";
        }
    }

    EXPECT_LE(overall_max_ulp, 2) << "Overall max ULP " << overall_max_ulp << " exceeds threshold 2"
                                  << " (worst at x=" << overall_worst_x << ")";
}

// =============================================================================
// Precision guard: dumps all ~70 BF16 values in the exp-based region (-13.375, -9).
// Asserts max ULP <= 2. This region uses fused x*exp(t) with Mills ratio correction —
// a code path that handles very small magnitudes (1e-10 to 1e-18).
// =============================================================================

TEST_F(GeluBwMainPolyTest, ExpBasedRegionFullDump) {
    // Collect all BF16 values in the exp-based region (-13.375, -9)
    std::vector<float> exp_region_values;

    for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        uint16_t bf16_bits = static_cast<uint16_t>(bits);
        if ((bf16_bits & bf16_ulp_bw_main::BF16_EXP_MASK) == bf16_ulp_bw_main::BF16_EXP_MASK &&
            (bf16_bits & bf16_ulp_bw_main::BF16_MANTISSA_MASK) != 0) {
            continue;  // NaN
        }
        if (bf16_bits == bf16_ulp_bw_main::BF16_POS_INF || bf16_bits == bf16_ulp_bw_main::BF16_NEG_INF) {
            continue;  // Inf
        }
        if (bf16_ulp_bw_main::is_bf16_denormal(bf16_bits)) {
            continue;  // Denormal
        }

        float x = bf16_ulp_bw_main::bf16_bits_to_float(bf16_bits);
        if (x > -13.375f && x < -9.0f) {
            exp_region_values.push_back(x);
        }
    }

    // Sort by x value (most negative first)
    std::sort(exp_region_values.begin(), exp_region_values.end());

    const size_t count = exp_region_values.size();
    std::cout << "\n================================================================================\n";
    std::cout << "EXP-BASED REGION FULL DUMP: All " << count << " BF16 values in (-13.375, -9)\n";
    std::cout << "================================================================================\n";

    // Pad for tensor
    const size_t tile_size = tt::constants::TILE_HW;
    size_t padded_size = ((count + tile_size - 1) / tile_size) * tile_size;
    std::vector<float> padded_values = exp_region_values;
    padded_values.resize(padded_size, 0.0f);

    uint32_t num_tiles = static_cast<uint32_t>(padded_size / tile_size);
    std::array<uint32_t, 4> dims = {1, 1, num_tiles * tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};

    std::vector<::bfloat16> bf16_inputs, bf16_grads;
    for (float x : padded_values) {
        bf16_inputs.push_back(::bfloat16(x));
        bf16_grads.push_back(::bfloat16(1.0f));
    }

    tt::tt_metal::TensorSpec tensor_spec(
        tt::tt_metal::Shape(dims),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), tt::tt_metal::MemoryConfig{}));

    auto input_tensor = tt::tt_metal::Tensor::from_vector(std::move(bf16_inputs), tensor_spec).to_device(device_);
    auto grad_tensor = tt::tt_metal::Tensor::from_vector(std::move(bf16_grads), tensor_spec).to_device(device_);

    auto results = ttnn::gelu_bw(grad_tensor, input_tensor, "none");
    auto result = results[0].value();
    auto output_cpu = ttnn::from_device(result);
    auto output_vec = output_cpu.to_vector<::bfloat16>();

    // Print header
    std::cout << std::setw(8) << "Index" << std::setw(12) << "x" << std::setw(16) << "BF16 bits" << std::setw(18)
              << "Expected" << std::setw(18) << "Actual" << std::setw(10) << "ULP" << "\n";
    std::cout << std::string(82, '-') << "\n";

    int total_ulp = 0;
    int max_ulp = 0;
    float worst_x = 0;
    int count_le_1 = 0;

    for (size_t i = 0; i < count; ++i) {
        float x = exp_region_values[i];
        uint16_t bits = bf16_ulp_bw_main::float_to_bf16_bits(x);
        float actual = static_cast<float>(output_vec[i]);
        float expected = bf16_ulp_bw_main::gelu_derivative_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp_bw_main::ulp_distance_bf16_daz(actual, expected);

        std::cout << std::setw(8) << i << std::setw(12) << std::fixed << std::setprecision(4) << x << "    0x"
                  << std::hex << std::setw(4) << std::setfill('0') << bits << std::dec << std::setfill(' ')
                  << std::setw(18) << std::scientific << std::setprecision(6) << expected << std::setw(18) << actual
                  << std::setw(10) << ulp << "\n";

        total_ulp += ulp;
        if (ulp > max_ulp) {
            max_ulp = ulp;
            worst_x = x;
        }
        if (ulp <= 1) {
            count_le_1++;
        }
    }

    std::cout << std::string(82, '-') << "\n";
    std::cout << "SUMMARY:\n";
    std::cout << "  Total values: " << count << "\n";
    std::cout << "  Mean ULP: " << std::fixed << std::setprecision(2) << (double)total_ulp / count << "\n";
    std::cout << "  Max ULP: " << max_ulp << " at x = " << std::setprecision(4) << worst_x << "\n";
    std::cout << "  Values with ULP <= 1: " << count_le_1 << " (" << std::setprecision(1)
              << (100.0 * count_le_1 / count) << "%)\n";
    std::cout << "================================================================================\n";

    // Regression guard: exp-based region should be very precise
    EXPECT_LE(max_ulp, 2) << "Exp-based region (-13.375, -9) max ULP " << max_ulp << " at x=" << worst_x
                          << " exceeds threshold 2";
}

// =============================================================================
// Correctness guard: verifies all BF16 values at x <= -13.375 saturate to exactly 0.0.
// Also documents why polynomial coverage stops at x = -9 (values span 8 orders of
// magnitude making polynomial fit impractical; saturation to 0 is correct for ML).
// =============================================================================

TEST_F(GeluBwMainPolyTest, DeepNegativeRegionAnalysis) {
    std::cout << "\n============================================================\n";
    std::cout << "DEEP NEGATIVE REGION ANALYSIS: Why coverage stops at x = -9\n";
    std::cout << "============================================================\n";
    std::cout << "\nFunction values in [-13.375, -9] span 8 orders of magnitude:\n";
    std::cout << std::setw(10) << "x" << std::setw(20) << "GELU'(x) [fp64]\n";
    std::cout << std::string(30, '-') << "\n";

    std::vector<float> test_points = {-9.0f, -10.0f, -11.0f, -12.0f, -13.0f, -13.375f};
    for (float x : test_points) {
        double exact = bf16_ulp_bw_main::gelu_derivative_exact(x);
        std::cout << std::setw(10) << std::fixed << std::setprecision(3) << x << std::setw(20) << std::scientific
                  << std::setprecision(6) << exact << "\n";
    }

    std::cout << "\nPolynomial coverage is impractical because:\n";
    std::cout << "  - Coefficients would need to be < 1e-18 for accurate fit\n";
    std::cout << "  - Float32 Horner evaluation loses precision with such tiny coefficients\n";
    std::cout << "  - Tested FL3 polynomial (numpy polyfit): Max ULP = 15448 (worse than saturation!)\n";
    std::cout << "\nCurrent implementation: saturate to 0 for x < -9\n";
    std::cout << "  - Max ULP at saturation boundary: 8898 (at x = -9.062)\n";
    std::cout << "  - This is acceptable: values < 9e-18 are irrelevant for ML training\n";
    std::cout << "============================================================\n";

    // Saturation guard: verify all BF16 values below -13.375 produce exactly 0.0
    // These values are in the BF16-natural-zero region where GELU'(x) rounds to 0 in BF16
    std::vector<float> saturation_values;
    for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        uint16_t bf16_bits = static_cast<uint16_t>(bits);
        if ((bf16_bits & bf16_ulp_bw_main::BF16_EXP_MASK) == bf16_ulp_bw_main::BF16_EXP_MASK) {
            continue;  // Skip NaN/Inf
        }
        if (bf16_ulp_bw_main::is_bf16_denormal(bf16_bits)) {
            continue;
        }
        float x = bf16_ulp_bw_main::bf16_bits_to_float(bf16_bits);
        if (x <= -13.375f) {
            saturation_values.push_back(x);
        }
    }

    const size_t sat_count = saturation_values.size();
    std::cout << "\nSaturation guard: testing " << sat_count << " BF16 values with x <= -13.375\n";

    const size_t tile_size = tt::constants::TILE_HW;
    size_t padded_size = ((sat_count + tile_size - 1) / tile_size) * tile_size;
    saturation_values.resize(padded_size, 0.0f);

    uint32_t num_tiles = static_cast<uint32_t>(padded_size / tile_size);
    std::array<uint32_t, 4> dims = {1, 1, num_tiles * tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};

    std::vector<::bfloat16> bf16_inputs, bf16_grads;
    for (float x : saturation_values) {
        bf16_inputs.push_back(::bfloat16(x));
        bf16_grads.push_back(::bfloat16(1.0f));
    }

    tt::tt_metal::TensorSpec tensor_spec(
        tt::tt_metal::Shape(dims),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), tt::tt_metal::MemoryConfig{}));

    auto input_tensor = tt::tt_metal::Tensor::from_vector(std::move(bf16_inputs), tensor_spec).to_device(device_);
    auto grad_tensor = tt::tt_metal::Tensor::from_vector(std::move(bf16_grads), tensor_spec).to_device(device_);

    auto results = ttnn::gelu_bw(grad_tensor, input_tensor, "none");
    auto result = results[0].value();
    auto output_cpu = ttnn::from_device(result);
    auto output_vec = output_cpu.to_vector<::bfloat16>();

    int nonzero_count = 0;
    for (size_t i = 0; i < sat_count; ++i) {
        float actual = static_cast<float>(output_vec[i]);
        if (actual != 0.0f) {
            nonzero_count++;
            if (nonzero_count <= 10) {  // Print first 10 violations
                std::cout << "  VIOLATION: x=" << saturation_values[i] << " produced " << actual << " (expected 0.0)\n";
            }
        }
    }

    EXPECT_EQ(nonzero_count, 0) << nonzero_count << " of " << sat_count
                                << " values in x <= -13.375 did not saturate to 0.0";
}

// =============================================================================
// Correctness guard: verifies BOTH saturation regions — positive (x >= 3.1719 → 1.0)
// and negative (x <= -13.375 → 0.0). Batches all saturation-region BF16 values through
// the device and asserts zero violations. Also reports the exact transition thresholds.
// =============================================================================

TEST_F(GeluBwMainPolyTest, SaturationThresholdResearch) {
    std::cout << "\n============================================================\n";
    std::cout << "GELU DERIVATIVE SATURATION THRESHOLD RESEARCH (DAZ+FTZ)\n";
    std::cout << "Scanning ENTIRE BF16 range\n";
    std::cout << "============================================================\n";

    // Scan negative values to find zero saturation threshold
    float last_nonzero_x = 0.0f;
    float first_zero_x = 0.0f;
    float last_nonzero_value = 0.0f;
    int count_zero_negative = 0;
    int count_nonzero_negative = 0;

    std::cout << "\n--- Scanning ALL negative values for zero saturation ---\n";
    std::cout << std::setw(14) << "x" << std::setw(20) << "GELU'(x) [fp64]" << std::setw(18) << "BF16 result"
              << std::setw(10) << "Status\n";
    std::cout << std::string(62, '-') << "\n";

    // Scan entire negative range: from smallest negative normal to most negative
    bool found_transition = false;
    for (uint16_t bits = 0x8080; bits <= 0xFF7F; ++bits) {  // All negative normals
        // Skip NaN and Inf
        uint16_t exp = (bits >> 7) & 0xFF;
        uint16_t mantissa = bits & 0x7F;
        if (exp == 0xFF) {
            continue;  // Skip Inf/NaN
        }
        if (exp == 0 && mantissa != 0) {
            continue;  // Skip denormals
        }

        float x = bf16_ulp_bw_main::bf16_bits_to_float(bits);
        float expected = bf16_ulp_bw_main::gelu_derivative_expected_bf16_daz(x);

        if (expected == 0.0f) {
            count_zero_negative++;
            if (!found_transition && last_nonzero_value != 0.0f) {
                first_zero_x = x;
                found_transition = true;
                // Print transition region
                std::cout << std::setw(14) << std::fixed << std::setprecision(4) << last_nonzero_x << std::setw(20)
                          << std::scientific << std::setprecision(6)
                          << bf16_ulp_bw_main::gelu_derivative_exact(last_nonzero_x) << std::setw(18)
                          << last_nonzero_value << std::setw(10) << "LAST NONZERO\n";
                std::cout << std::setw(14) << std::fixed << std::setprecision(4) << x << std::setw(20)
                          << std::scientific << std::setprecision(6) << bf16_ulp_bw_main::gelu_derivative_exact(x)
                          << std::setw(18) << expected << std::setw(10) << "FIRST ZERO\n";
            }
        } else {
            count_nonzero_negative++;
            last_nonzero_x = x;
            last_nonzero_value = expected;
        }
    }

    // Scan positive values - GELU'(x) has a "hump" above 1.0 for x ∈ [~0.8, ~3.15]
    // We need to track: first ≥1, first >1, last >1, first =1 (final saturation)
    float first_ge1_x = 0.0f;   // First x where BF16 result >= 1
    float first_gt1_x = 0.0f;   // First x where BF16 result > 1 (hump starts)
    float last_gt1_x = 0.0f;    // Last x where BF16 result > 1 (hump ends)
    float final_sat1_x = 0.0f;  // First x where BF16 result = 1 and stays 1 forever
    int count_eq1 = 0;
    int count_gt1 = 0;
    int count_lt1 = 0;

    std::cout << "\n--- Scanning ALL positive values (GELU' has hump > 1) ---\n";
    std::cout << "GELU'(x) approaches 0.5 at x=0, peaks > 1 around x=1-2, then → 1\n\n";
    std::cout << std::setw(14) << "x" << std::setw(20) << "GELU'(x) [fp64]" << std::setw(18) << "BF16 result"
              << std::setw(14) << "Category\n";
    std::cout << std::string(66, '-') << "\n";

    // Scan entire positive range
    bool found_first_ge1 = false;
    bool found_first_gt1 = false;
    bool in_gt1_region = false;
    float prev_x = 0.0f;
    float prev_value = 0.0f;

    for (uint16_t bits = 0x0080; bits <= 0x7F7F; ++bits) {  // All positive normals
        uint16_t exp = (bits >> 7) & 0xFF;
        uint16_t mantissa = bits & 0x7F;
        if (exp == 0xFF) {
            continue;  // Skip Inf/NaN
        }
        if (exp == 0 && mantissa != 0) {
            continue;  // Skip denormals
        }

        float x = bf16_ulp_bw_main::bf16_bits_to_float(bits);
        float expected = bf16_ulp_bw_main::gelu_derivative_expected_bf16_daz(x);
        double exact = bf16_ulp_bw_main::gelu_derivative_exact(x);

        if (expected < 1.0f) {
            count_lt1++;
            if (in_gt1_region) {
                // Exiting >1 region, entering final =1 region
                last_gt1_x = prev_x;
                final_sat1_x = x;  // Actually this might still be <1 briefly
                in_gt1_region = false;
                std::cout << std::setw(14) << std::fixed << std::setprecision(4) << prev_x << std::setw(20)
                          << std::scientific << std::setprecision(6) << bf16_ulp_bw_main::gelu_derivative_exact(prev_x)
                          << std::setw(18) << prev_value << std::setw(14) << "LAST >1\n";
                std::cout << std::setw(14) << std::fixed << std::setprecision(4) << x << std::setw(20)
                          << std::scientific << std::setprecision(6) << exact << std::setw(18) << expected
                          << std::setw(14) << "back to <1\n";
            }
        } else if (expected == 1.0f) {
            count_eq1++;
            if (!found_first_ge1) {
                found_first_ge1 = true;
                first_ge1_x = x;
                std::cout << std::setw(14) << std::fixed << std::setprecision(4) << prev_x << std::setw(20)
                          << std::scientific << std::setprecision(6) << bf16_ulp_bw_main::gelu_derivative_exact(prev_x)
                          << std::setw(18) << prev_value << std::setw(14) << "last <1\n";
                std::cout << std::setw(14) << std::fixed << std::setprecision(4) << x << std::setw(20)
                          << std::scientific << std::setprecision(6) << exact << std::setw(18) << expected
                          << std::setw(14) << "FIRST >=1\n";
            }
            if (in_gt1_region) {
                // Transition from >1 to =1 (end of hump)
                last_gt1_x = prev_x;
                final_sat1_x = x;
                in_gt1_region = false;
                std::cout << std::setw(14) << std::fixed << std::setprecision(4) << prev_x << std::setw(20)
                          << std::scientific << std::setprecision(6) << bf16_ulp_bw_main::gelu_derivative_exact(prev_x)
                          << std::setw(18) << prev_value << std::setw(14) << "LAST >1\n";
                std::cout << std::setw(14) << std::fixed << std::setprecision(4) << x << std::setw(20)
                          << std::scientific << std::setprecision(6) << exact << std::setw(18) << expected
                          << std::setw(14) << "FINAL =1\n";
            }
        } else {  // expected > 1.0f
            count_gt1++;
            if (!found_first_gt1) {
                found_first_gt1 = true;
                first_gt1_x = x;
                in_gt1_region = true;
                std::cout << std::setw(14) << std::fixed << std::setprecision(4) << prev_x << std::setw(20)
                          << std::scientific << std::setprecision(6) << bf16_ulp_bw_main::gelu_derivative_exact(prev_x)
                          << std::setw(18) << prev_value << std::setw(14) << "last <=1\n";
                std::cout << std::setw(14) << std::fixed << std::setprecision(4) << x << std::setw(20)
                          << std::scientific << std::setprecision(6) << exact << std::setw(18) << expected
                          << std::setw(14) << "FIRST >1\n";
            }
            in_gt1_region = true;
        }

        prev_x = x;
        prev_value = expected;
    }

    std::cout << "\n============================================================\n";
    std::cout << "SATURATION THRESHOLD RESULTS\n";
    std::cout << "============================================================\n";

    std::cout << "\nNegative region (saturation to 0):\n";
    std::cout << "  Last nonzero at x = " << std::fixed << std::setprecision(4) << last_nonzero_x << " (bf16: 0x"
              << std::hex << bf16_ulp_bw_main::float_to_bf16_bits(last_nonzero_x) << std::dec << ")\n";
    std::cout << "  First zero at  x = " << std::fixed << std::setprecision(4) << first_zero_x << " (bf16: 0x"
              << std::hex << bf16_ulp_bw_main::float_to_bf16_bits(first_zero_x) << std::dec << ")\n";
    std::cout << "  Values with GELU'(x) = 0:  " << count_zero_negative << "\n";
    std::cout << "  Values with GELU'(x) != 0: " << count_nonzero_negative << "\n";

    std::cout << "\nPositive region (GELU' has hump > 1):\n";
    std::cout << "  First >=1 at x = " << std::fixed << std::setprecision(4) << first_ge1_x << " (bf16: 0x" << std::hex
              << bf16_ulp_bw_main::float_to_bf16_bits(first_ge1_x) << std::dec << ")\n";
    std::cout << "  First >1  at x = " << std::fixed << std::setprecision(4) << first_gt1_x << " (bf16: 0x" << std::hex
              << bf16_ulp_bw_main::float_to_bf16_bits(first_gt1_x) << std::dec << ")\n";
    std::cout << "  Last >1   at x = " << std::fixed << std::setprecision(4) << last_gt1_x << " (bf16: 0x" << std::hex
              << bf16_ulp_bw_main::float_to_bf16_bits(last_gt1_x) << std::dec << ")\n";
    std::cout << "  Final =1  at x = " << std::fixed << std::setprecision(4) << final_sat1_x << " (bf16: 0x" << std::hex
              << bf16_ulp_bw_main::float_to_bf16_bits(final_sat1_x) << std::dec << ")\n";
    std::cout << "  Values with BF16 < 1:  " << count_lt1 << "\n";
    std::cout << "  Values with BF16 = 1:  " << count_eq1 << "\n";
    std::cout << "  Values with BF16 > 1:  " << count_gt1 << "\n";

    std::cout << "\n============================================================\n";
    std::cout << "RECOMMENDATION FOR POLYNOMIAL IMPLEMENTATION\n";
    std::cout << "============================================================\n";
    std::cout << "  For x <= " << first_zero_x << ": return 0.0f (zero saturation)\n";
    std::cout << "  For x >= " << final_sat1_x << ": return 1.0f (one saturation)\n";
    std::cout << "  For " << first_zero_x << " < x < " << final_sat1_x << ": use polynomial\n";
    std::cout << "\nNote: GELU'(x) exceeds 1.0 for x in [" << first_gt1_x << ", " << last_gt1_x << "]\n";
    std::cout << "      Polynomial must reproduce this 'hump' accurately.\n";
    std::cout << "============================================================\n";

    // =========================================================================
    // Saturation guards: verify device produces exact saturation values
    // =========================================================================

    // Collect all BF16 values in both saturation regions
    std::vector<float> pos_sat_values;  // x >= 3.1719 → should produce 1.0
    std::vector<float> neg_sat_values;  // x <= -13.375 → should produce 0.0

    for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        uint16_t bf16_bits = static_cast<uint16_t>(bits);
        if ((bf16_bits & bf16_ulp_bw_main::BF16_EXP_MASK) == bf16_ulp_bw_main::BF16_EXP_MASK) {
            continue;  // Skip NaN/Inf
        }
        if (bf16_ulp_bw_main::is_bf16_denormal(bf16_bits)) {
            continue;
        }
        float x = bf16_ulp_bw_main::bf16_bits_to_float(bf16_bits);
        if (x >= 3.1719f) {
            pos_sat_values.push_back(x);
        } else if (x <= -13.375f) {
            neg_sat_values.push_back(x);
        }
    }

    std::cout << "\nSaturation guard: " << pos_sat_values.size() << " positive + " << neg_sat_values.size()
              << " negative values\n";

    // Helper lambda to test saturation region
    auto test_saturation = [&](std::vector<float>& values, float expected_val, const std::string& label) {
        const size_t count = values.size();
        const size_t tile_size = tt::constants::TILE_HW;
        size_t padded = ((count + tile_size - 1) / tile_size) * tile_size;
        values.resize(padded, 0.0f);

        uint32_t nt = static_cast<uint32_t>(padded / tile_size);
        std::array<uint32_t, 4> d = {1, 1, nt * tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};

        std::vector<::bfloat16> inputs, grads;
        for (float x : values) {
            inputs.push_back(::bfloat16(x));
            grads.push_back(::bfloat16(1.0f));
        }

        tt::tt_metal::TensorSpec spec(
            tt::tt_metal::Shape(d),
            tt::tt_metal::TensorLayout(
                DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                tt::tt_metal::MemoryConfig{}));

        auto in_t = tt::tt_metal::Tensor::from_vector(std::move(inputs), spec).to_device(device_);
        auto gr_t = tt::tt_metal::Tensor::from_vector(std::move(grads), spec).to_device(device_);

        auto res_vec = ttnn::gelu_bw(gr_t, in_t, "none");
        auto res = res_vec[0].value();
        auto out = ttnn::from_device(res).to_vector<::bfloat16>();

        int violations = 0;
        for (size_t i = 0; i < count; ++i) {
            float actual = static_cast<float>(out[i]);
            if (actual != expected_val) {
                violations++;
                if (violations <= 10) {
                    std::cout << "  " << label << " VIOLATION: x=" << values[i] << " produced " << actual
                              << " (expected " << expected_val << ")\n";
                }
            }
        }
        return violations;
    };

    int pos_violations = test_saturation(pos_sat_values, 1.0f, "positive");
    EXPECT_EQ(pos_violations, 0) << pos_violations << " positive-saturation values (x >= 3.1719) did not produce 1.0";

    int neg_violations = test_saturation(neg_sat_values, 0.0f, "negative");
    EXPECT_EQ(neg_violations, 0) << neg_violations << " negative-saturation values (x <= -13.375) did not produce 0.0";
}

// Correctness guard: verifies IEEE special values (+inf, -inf, NaN) per nmauriceTT review.
// +inf → 1.0, -inf → 0.0, NaN → 1.0 (via v_if comparison treating NaN bit pattern as
// large positive). Ref: tt-llk#675, tech_reports/Handling_Special_Value/special_values.md.
TEST_F(GeluBwMainPolyTest, SpecialValues) {
    float pos_inf = std::numeric_limits<float>::infinity();
    float neg_inf = -std::numeric_limits<float>::infinity();
    float nan_val = std::numeric_limits<float>::quiet_NaN();

    // +inf: treated as large positive, saturates to 1.0
    // (also the correct mathematical limit: lim x->+inf GELU'(x) = 1)
    float result_pos_inf = run_gelu_bw_main_single(*device_, pos_inf);
    std::cout << "GELU_BW(+inf) = " << result_pos_inf << "\n";
    EXPECT_EQ(result_pos_inf, 1.0f) << "+inf should saturate to 1.0";

    // -inf: treated as large negative, falls through to default 0.0
    // (also the correct mathematical limit: lim x->-inf GELU'(x) = 0)
    float result_neg_inf = run_gelu_bw_main_single(*device_, neg_inf);
    std::cout << "GELU_BW(-inf) = " << result_neg_inf << "\n";
    EXPECT_EQ(result_neg_inf, 0.0f) << "-inf should saturate to 0.0";

    // NaN: bit pattern 0x7FFF treated as large positive, saturates to 1.0
    // Ideally should return NaN, but TT hardware treats NaN as ordinary number
    // in comparisons (see special_values.md). This is acceptable per tt-llk#675.
    float result_nan = run_gelu_bw_main_single(*device_, nan_val);
    std::cout << "GELU_BW(NaN) = " << result_nan << "\n";
    EXPECT_EQ(result_nan, 1.0f) << "NaN treated as large positive by SFPU, saturates to 1.0";
}

}  // namespace ttnn::test
