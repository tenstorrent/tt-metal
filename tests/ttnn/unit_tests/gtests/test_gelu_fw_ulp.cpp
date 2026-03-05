// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * GELU Forward ULP Precision Tests
 *
 * This test file validates the accuracy of ttnn::gelu (forward) across
 * the entire BFloat16 range using the same methodology as test_gelu_bw_ulp.cpp.
 *
 * MATHEMATICAL FORMULA:
 * GELU(x) = x * Phi(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
 *
 * REFERENCE IMPLEMENTATION:
 * Uses fp64 with erfc() for numerically stable computation.
 * Key insight: for negative x, 1 + erf(x/sqrt(2)) = erfc(|x|/sqrt(2))
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
 * Run: ./build_Debug/test/ttnn/unit_tests_ttnn --gtest_filter="*GeluFwUlp*"
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <numbers>
#include <vector>
#include <limits>
#include <iomanip>
#include <map>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::test {

// =============================================================================
// BFloat16 ULP Calculator (shared with test_gelu_bw_ulp.cpp)
// =============================================================================

namespace bf16_ulp_fw {

constexpr uint16_t BF16_EXP_MASK = 0x7F80;
constexpr uint16_t BF16_MANTISSA_MASK = 0x007F;
constexpr uint16_t BF16_SIGN_MASK = 0x8000;
constexpr uint16_t BF16_POS_INF = 0x7F80;
constexpr uint16_t BF16_NEG_INF = 0xFF80;

inline uint16_t float_to_bf16_bits(float f) {
    // Round-to-nearest-even (RNE), matching hardware pack behavior.
    // See tech_reports/data_formats/data_formats.md and bfloat16::from_float().
    if (std::isnan(f)) {
        return UINT16_C(0x7FC0);
    }
    uint32_t u32;
    std::memcpy(&u32, &f, sizeof(float));
    uint32_t rounding_bias = ((u32 >> 16) & 1) + UINT32_C(0x7FFF);
    return static_cast<uint16_t>((u32 + rounding_bias) >> 16);
}

inline float bf16_bits_to_float(uint16_t bits) {
    uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &f32_bits, sizeof(float));
    return f;
}

inline bool is_bf16_denormal(uint16_t bits) {
    uint16_t exp = (bits >> 7) & 0xFF;
    uint16_t mantissa = bits & BF16_MANTISSA_MASK;
    return (exp == 0) && (mantissa != 0);
}

inline bool is_bf16_denormal(float f) { return is_bf16_denormal(float_to_bf16_bits(f)); }

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
        // Negative normals: magnitude 0x0080 (smallest) maps to 32639 (zero-1),
        // magnitude 0x7F7F (largest) maps to 128. The 0x7F offset skips denormals
        // which are DAZ-collapsed to zero, matching the positive formula symmetry.
        uint16_t magnitude = bits & 0x7FFF;
        return 32640 + BF16_MANTISSA_MASK - magnitude;
    }
    return 32640 + bits - BF16_MANTISSA_MASK;
}

inline int32_t bf16_value_order_index_daz(float f) { return bf16_value_order_index_daz(float_to_bf16_bits(f)); }

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
// GELU Forward Reference Implementation
// =============================================================================

/**
 * Exact GELU reference using fp64 with erfc() for numerical stability.
 *
 * GELU(x) = x * Phi(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
 *
 * For numerical stability with large negative x, we use:
 *   1 + erf(x/sqrt(2)) = erfc(|x|/sqrt(2)) when x < 0
 */
inline double gelu_exact(double x) {
    constexpr double SQRT2 = std::numbers::sqrt2;

    double cdf;
    if (x < 0.0) {
        double abs_x_div_sqrt2 = -x / SQRT2;
        cdf = 0.5 * std::erfc(abs_x_div_sqrt2);
    } else {
        cdf = 0.5 * (1.0 + std::erf(x / SQRT2));
    }

    return x * cdf;
}

/**
 * Compute the expected BF16 GELU value with DAZ+FTZ applied.
 */
inline float gelu_expected_bf16_daz(float x) {
    float x_daz = bf16_daz_normalize(x);
    double result = gelu_exact(x_daz);
    float result_f32 = static_cast<float>(result);
    return bf16_daz_normalize(result_f32);
}

}  // namespace bf16_ulp_fw

// =============================================================================
// GELU Forward ULP Tests (Require Device)
// =============================================================================

class GeluFwUlpTest : public TTNNFixtureWithDevice {};

/**
 * Helper function to run forward GELU on device for a single value.
 */
float run_gelu_fw_single(tt::tt_metal::distributed::MeshDevice& device, float input_val) {
    ttnn::Shape shape({1, 1, 32, 32});

    auto input_tensor = ttnn::full(shape, input_val, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

    // Call gelu with fast_and_approx=false (polynomial, not fast approximation)
    auto result = ttnn::gelu(input_tensor, false);

    auto output_cpu = ttnn::from_device(result);
    auto output_vec = output_cpu.to_vector<::bfloat16>();
    return static_cast<float>(output_vec[0]);
}

// Correctness guard: GELU(0) = 0 exactly.
// Catches broken constant term or CDF polynomial at zero.
TEST_F(GeluFwUlpTest, GeluAtZero) {
    float actual = run_gelu_fw_single(*device_, 0.0f);
    float expected = bf16_ulp_fw::gelu_expected_bf16_daz(0.0f);

    int32_t ulp = bf16_ulp_fw::ulp_distance_bf16_daz(actual, expected);

    std::cout << "x=0: expected=" << expected << ", actual=" << actual << ", ULP=" << ulp << std::endl;

    EXPECT_EQ(actual, 0.0f) << "GELU(0) must be exactly 0";
    EXPECT_LE(ulp, 0) << "GELU(0) should have ULP=0";
}

// Correctness guard: GELU(x) approaches x for large positive x.
TEST_F(GeluFwUlpTest, PositiveValues) {
    std::vector<std::pair<float, int32_t>> test_cases = {
        {0.5f, 2},
        {1.0f, 2},
        {2.0f, 2},
        {3.0f, 2},
        {5.0f, 2},
        {5.375f, 2},
        {10.0f, 2},
    };

    for (const auto& [input_val, max_expected_ulp] : test_cases) {
        float actual = run_gelu_fw_single(*device_, input_val);
        float expected = bf16_ulp_fw::gelu_expected_bf16_daz(input_val);
        int32_t ulp = bf16_ulp_fw::ulp_distance_bf16_daz(actual, expected);

        std::cout << "x=" << input_val << ": expected=" << expected << ", actual=" << actual << ", ULP=" << ulp
                  << std::endl;

        EXPECT_LE(ulp, max_expected_ulp) << "GELU(" << input_val << ") ULP too high";
    }
}

// Correctness guard: GELU(x) approaches 0 for large negative x.
TEST_F(GeluFwUlpTest, NegativeValues) {
    std::vector<std::pair<float, int32_t>> test_cases = {
        {-0.5f, 2},
        {-1.0f, 2},
        {-2.0f, 2},
        {-3.0f, 2},
        {-4.0f, 2},
        {-5.0f, 2},
        {-6.0f, 2},
        {-8.0f, 2},
        {-10.0f, 2},
        {-13.0f, 2},
    };

    for (const auto& [input_val, max_expected_ulp] : test_cases) {
        float actual = run_gelu_fw_single(*device_, input_val);
        float expected = bf16_ulp_fw::gelu_expected_bf16_daz(input_val);
        int32_t ulp = bf16_ulp_fw::ulp_distance_bf16_daz(actual, expected);

        std::cout << "x=" << input_val << ": expected=" << expected << ", actual=" << actual << ", ULP=" << ulp
                  << std::endl;

        EXPECT_LE(ulp, max_expected_ulp) << "GELU(" << input_val << ") ULP too high";
    }
}

// Correctness guard: near-zero region where GELU(x) ~ x/2.
TEST_F(GeluFwUlpTest, NearZeroRegion) {
    std::vector<float> test_values = {1e-6f, 1e-4f, 0.01f, 0.1f, -0.1f, -0.01f, -1e-4f};

    for (float input_val : test_values) {
        float actual = run_gelu_fw_single(*device_, input_val);
        float expected = bf16_ulp_fw::gelu_expected_bf16_daz(input_val);
        int32_t ulp = bf16_ulp_fw::ulp_distance_bf16_daz(actual, expected);

        std::cout << "x=" << input_val << ": expected=" << expected << ", actual=" << actual << ", ULP=" << ulp
                  << std::endl;

        EXPECT_LE(ulp, 2) << "GELU(" << input_val << ") near-zero ULP too high";
    }
}

// Precision guard: sweeps ALL ~65K BF16 values, enforces per-region ULP caps.
// Regions match the kernel's piecewise implementation:
//   x <= -13.1875: saturation to 0
//   (-13.1875, -5]: exp-based asymptotic
//   [-5, -3): left CDF polynomial
//   [-3, 3): core CDF polynomial
//   x >= 3.0: identity (return x)
//   With RNE rounding, GELU(x) rounds to x for all BF16 x >= ~2.78
TEST_F(GeluFwUlpTest, ComprehensiveULPByRegion) {
    std::vector<float> input_values;
    input_values.reserve(70000);

    for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        uint16_t bf16_bits = static_cast<uint16_t>(bits);

        if ((bf16_bits & bf16_ulp_fw::BF16_EXP_MASK) == bf16_ulp_fw::BF16_EXP_MASK &&
            (bf16_bits & bf16_ulp_fw::BF16_MANTISSA_MASK) != 0) {
            continue;  // Skip NaN
        }
        if (bf16_bits == bf16_ulp_fw::BF16_POS_INF || bf16_bits == bf16_ulp_fw::BF16_NEG_INF) {
            continue;  // Skip infinity
        }
        if (bf16_ulp_fw::is_bf16_denormal(bf16_bits)) {
            continue;  // Skip denormals
        }

        float val = bf16_ulp_fw::bf16_bits_to_float(bf16_bits);
        input_values.push_back(val);
    }

    const size_t valid_count = input_values.size();
    std::cout << "\nCollected " << valid_count << " valid BF16 values\n";

    // Pad to tile boundary
    const size_t tile_size = tt::constants::TILE_HW;
    size_t padded_size = ((valid_count + tile_size - 1) / tile_size) * tile_size;
    input_values.resize(padded_size, 0.0f);

    uint32_t num_tiles = static_cast<uint32_t>(padded_size / tile_size);
    std::array<uint32_t, 4> dims = {1, 1, num_tiles * tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};

    std::vector<::bfloat16> bf16_inputs;
    bf16_inputs.reserve(padded_size);

    for (float x : input_values) {
        bf16_inputs.push_back(::bfloat16(x));
    }

    tt::tt_metal::TensorSpec tensor_spec(
        tt::tt_metal::Shape(dims),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), tt::tt_metal::MemoryConfig{}));

    auto input_tensor = tt::tt_metal::Tensor::from_vector(std::move(bf16_inputs), tensor_spec).to_device(device_);

    // Run forward GELU once on entire tensor
    auto result = ttnn::gelu(input_tensor, false);
    auto output_cpu = ttnn::from_device(result);
    auto output_vec = output_cpu.to_vector<::bfloat16>();

    // Analyze results by region (matching kernel piecewise regions)
    struct RegionStats {
        std::string name;
        int count = 0;
        double ulp_sum = 0;
        int max_ulp = 0;
        float worst_x = 0;
    };

    std::vector<RegionStats> regions = {
        {"Saturation to 0 (x <= -13.1875)"},
        {"Exp-based (-13.1875, -5]"},
        {"Left CDF poly [-5, -3)"},
        {"Core CDF poly [-3, 3)"},
        {"Identity (x >= 3)"},
    };

    int overall_max_ulp = 0;
    float overall_worst_x = 0;
    double overall_ulp_sum = 0;

    for (size_t i = 0; i < valid_count; ++i) {
        float x = bf16_ulp_fw::bf16_bits_to_float(bf16_ulp_fw::float_to_bf16_bits(input_values[i]));
        float actual = static_cast<float>(output_vec[i]);
        float expected = bf16_ulp_fw::gelu_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp_fw::ulp_distance_bf16_daz(actual, expected);

        if (ulp < 0) {
            continue;
        }

        // Categorize by kernel region
        int region_idx;
        if (x <= -13.1875f) {
            region_idx = 0;
        } else if (x <= -5.0f) {
            region_idx = 1;
        } else if (x < -3.0f) {
            region_idx = 2;
        } else if (x < 3.0f) {
            region_idx = 3;
        } else {
            region_idx = 4;  // Identity (x >= 3.0)
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
    std::cout << "GELU FORWARD ULP ANALYSIS BY REGION (DAZ+FTZ MODEL)\n";
    std::cout << "============================================================\n";
    std::cout << std::setw(35) << "Region" << std::setw(10) << "Count" << std::setw(12) << "Mean ULP" << std::setw(12)
              << "Max ULP" << std::setw(15) << "Worst x\n";
    std::cout << std::string(84, '-') << "\n";

    for (const auto& r : regions) {
        if (r.count > 0) {
            std::cout << std::setw(35) << r.name << std::setw(10) << r.count << std::setw(12) << std::fixed
                      << std::setprecision(2) << (r.ulp_sum / r.count) << std::setw(12) << r.max_ulp << std::setw(15)
                      << std::scientific << std::setprecision(3) << r.worst_x << "\n";
        }
    }

    std::cout << std::string(84, '-') << "\n";
    std::cout << std::setw(35) << "OVERALL" << std::setw(10) << valid_count << std::setw(12) << std::fixed
              << std::setprecision(2) << (overall_ulp_sum / valid_count) << std::setw(12) << overall_max_ulp
              << std::setw(15) << std::scientific << std::setprecision(3) << overall_worst_x << "\n";
    std::cout << "============================================================\n";

    // Per-region regression guards
    // With RNE rounding model matching hardware:
    std::vector<int> region_max_ulp_thresholds = {
        0,  // Saturation to 0: exact
        2,  // Exp-based: allow 2 for hardware precision
        2,  // Left CDF poly: allow 2
        2,  // Core CDF poly: allow 2
        0,  // Identity (x >= 3.0): exact (GELU rounds to x with RNE)
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
TEST_F(GeluFwUlpTest, CumulativeULPDistribution) {
    std::vector<float> input_values;
    input_values.reserve(70000);

    for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        uint16_t bf16_bits = static_cast<uint16_t>(bits);
        if ((bf16_bits & bf16_ulp_fw::BF16_EXP_MASK) == bf16_ulp_fw::BF16_EXP_MASK &&
            (bf16_bits & bf16_ulp_fw::BF16_MANTISSA_MASK) != 0) {
            continue;
        }
        if (bf16_bits == bf16_ulp_fw::BF16_POS_INF || bf16_bits == bf16_ulp_fw::BF16_NEG_INF) {
            continue;
        }
        if (bf16_ulp_fw::is_bf16_denormal(bf16_bits)) {
            continue;
        }

        input_values.push_back(bf16_ulp_fw::bf16_bits_to_float(bf16_bits));
    }

    const size_t valid_count = input_values.size();
    const size_t tile_size = tt::constants::TILE_HW;
    size_t padded_size = ((valid_count + tile_size - 1) / tile_size) * tile_size;
    input_values.resize(padded_size, 0.0f);

    uint32_t num_tiles = static_cast<uint32_t>(padded_size / tile_size);
    std::array<uint32_t, 4> dims = {1, 1, num_tiles * tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};

    std::vector<::bfloat16> bf16_inputs;
    for (float x : input_values) {
        bf16_inputs.push_back(::bfloat16(x));
    }

    tt::tt_metal::TensorSpec tensor_spec(
        tt::tt_metal::Shape(dims),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), tt::tt_metal::MemoryConfig{}));

    auto input_tensor = tt::tt_metal::Tensor::from_vector(std::move(bf16_inputs), tensor_spec).to_device(device_);

    auto result = ttnn::gelu(input_tensor, false);
    auto output_cpu = ttnn::from_device(result);
    auto output_vec = output_cpu.to_vector<::bfloat16>();

    std::map<int32_t, int> ulp_histogram;
    int max_ulp = 0;
    float worst_x = 0;

    for (size_t i = 0; i < valid_count; ++i) {
        float x = bf16_ulp_fw::bf16_bits_to_float(bf16_ulp_fw::float_to_bf16_bits(input_values[i]));
        float actual = static_cast<float>(output_vec[i]);
        float expected = bf16_ulp_fw::gelu_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp_fw::ulp_distance_bf16_daz(actual, expected);

        if (ulp >= 0) {
            ulp_histogram[ulp]++;
            if (ulp > max_ulp) {
                max_ulp = ulp;
                worst_x = x;
            }
        }
    }

    std::cout << "\n============================================================\n";
    std::cout << "GELU FORWARD CUMULATIVE ULP DISTRIBUTION (DAZ+FTZ MODEL)\n";
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

    EXPECT_LE(max_ulp, 2) << "Max ULP " << max_ulp << " at x=" << worst_x << " exceeds threshold 2";

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
TEST_F(GeluFwUlpTest, ReferenceImplementationVerification) {
    // GELU(0) = 0 exactly
    double gelu_0 = bf16_ulp_fw::gelu_exact(0.0);
    EXPECT_EQ(gelu_0, 0.0) << "GELU(0) should be exactly 0";

    // GELU(large) ~ large (approaches identity)
    double gelu_100 = bf16_ulp_fw::gelu_exact(100.0);
    EXPECT_NEAR(gelu_100, 100.0, 1e-6) << "GELU(100) should approach 100";

    // GELU(-large) ~ 0
    double gelu_neg100 = bf16_ulp_fw::gelu_exact(-100.0);
    EXPECT_NEAR(gelu_neg100, 0.0, 1e-6) << "GELU(-100) should approach 0";

    // GELU(-0.751) ~ -0.177 (local minimum of GELU)
    double gelu_min = bf16_ulp_fw::gelu_exact(-0.751);
    EXPECT_LT(gelu_min, 0.0) << "GELU(-0.751) should be negative (local minimum)";
    EXPECT_GT(gelu_min, -0.2) << "GELU(-0.751) should be > -0.2";

    // GELU(1) ~ 0.8413
    double gelu_1 = bf16_ulp_fw::gelu_exact(1.0);
    EXPECT_NEAR(gelu_1, 0.8413, 0.001) << "GELU(1) should be ~0.8413";

    std::cout << "\nReference implementation verification:\n";
    std::cout << "GELU(0) = " << gelu_0 << " (expected: 0)\n";
    std::cout << "GELU(1) = " << gelu_1 << " (expected: ~0.8413)\n";
    std::cout << "GELU(100) = " << gelu_100 << " (expected: ~100)\n";
    std::cout << "GELU(-100) = " << gelu_neg100 << " (expected: ~0)\n";
    std::cout << "GELU(-0.751) = " << gelu_min << " (expected: local minimum ~-0.177)\n";
}

// Correctness guard: key points spanning all 6 kernel code paths.
TEST_F(GeluFwUlpTest, SummaryStatistics) {
    std::cout << "\n========================================\n";
    std::cout << "GELU FORWARD ULP SUMMARY (DAZ+FTZ MODEL)\n";
    std::cout << "========================================\n";

    struct KeyPoint {
        std::string name;
        float x;
        int max_ulp_threshold;
    };

    std::vector<KeyPoint> key_points = {
        {"Zero", 0.0f, 0},
        {"Positive unity", 1.0f, 2},
        {"Negative unity", -1.0f, 2},
        {"Local minimum (~-0.751)", -0.75f, 2},
        {"Large positive (identity)", 6.0f, 0},
        {"Moderate negative", -4.0f, 2},
        {"Deep negative (exp)", -8.0f, 2},
        {"Saturation boundary", -13.1875f, 0},
    };

    for (const auto& kp : key_points) {
        float actual = run_gelu_fw_single(*device_, kp.x);
        float expected = bf16_ulp_fw::gelu_expected_bf16_daz(kp.x);
        int32_t ulp = bf16_ulp_fw::ulp_distance_bf16_daz(actual, expected);

        std::cout << std::setw(30) << kp.name << ": x=" << std::setw(10) << std::fixed << std::setprecision(4) << kp.x
                  << ", expected=" << std::setw(12) << std::scientific << std::setprecision(3) << expected
                  << ", actual=" << std::setw(12) << actual << ", ULP=" << std::setw(5) << ulp << "\n";

        EXPECT_LE(ulp, kp.max_ulp_threshold)
            << kp.name << " (x=" << kp.x << ") ULP " << ulp << " exceeds threshold " << kp.max_ulp_threshold;
    }

    std::cout << "========================================\n";
}

}  // namespace ttnn::test
