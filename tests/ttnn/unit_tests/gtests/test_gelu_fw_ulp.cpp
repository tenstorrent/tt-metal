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
//   [-3, 2.78125): core CDF polynomial
//   x >= 2.78125: identity (return x)
//   2.78125 (BF16 0x4032) is the exact boundary where GELU rounds to x with RNE
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
        {"Core CDF poly [-3, 2.78125)"},
        {"Identity (x >= 2.78125)"},
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
        } else if (x < 2.78125f) {
            region_idx = 3;
        } else {
            region_idx = 4;  // Identity (x >= 2.78125)
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
        0,  // Identity (x >= 2.78125): exact (GELU rounds to x with RNE)
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
    bf16_inputs.reserve(input_values.size());
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

// Saturation boundary verification: finds the golden reference's natural
// saturation boundaries and verifies the kernel gives ULP=0 in both tails.
//
// The golden's boundaries are where GELU(x) naturally rounds to 0 (negative
// tail) or to x (positive tail) in BF16 with RNE rounding. These may differ
// from the kernel's hardcoded thresholds (-13.1875 and 3.0).
TEST_F(GeluFwUlpTest, SaturationBoundaryVerification) {
    // Collect all valid BF16 values, sorted
    std::vector<float> all_values;
    all_values.reserve(65536);
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
        all_values.push_back(bf16_ulp_fw::bf16_bits_to_float(bf16_bits));
    }
    std::sort(all_values.begin(), all_values.end());

    // --- Find golden saturation boundaries ---
    // There are TWO zero regions for negative x:
    //   1. Deep negative saturation: x < ~-13 where GELU(x) mathematically -> 0
    //   2. Near-zero FTZ: tiny |x| where GELU(x) is a BF16 denormal -> flushed to 0
    // We want the deep negative boundary (region 1), not the near-zero FTZ.

    // Negative tail: scan from most negative toward zero.
    // The deep saturation region has GELU == 0. At some point it becomes non-zero.
    float neg_sat_boundary_zero = 0;     // least negative x still in deep zero region
    float neg_sat_boundary_nonzero = 0;  // first x past boundary (GELU != 0)
    bool found_neg_transition = false;
    for (float x : all_values) {
        if (x >= 0.0f) {
            break;
        }
        float expected = bf16_ulp_fw::gelu_expected_bf16_daz(x);
        if (!found_neg_transition) {
            if (expected == 0.0f) {
                neg_sat_boundary_zero = x;
            } else {
                neg_sat_boundary_nonzero = x;
                found_neg_transition = true;
            }
        }
    }

    // Positive tail: scan from smallest positive upward.
    // Find the first x where GELU(x) == x in BF16, and the last where it doesn't.
    float pos_first_identity = 0;
    float pos_last_non_identity = 0;
    bool found_pos_identity = false;
    for (float x : all_values) {
        if (x <= 0.0f) {
            continue;
        }
        float expected = bf16_ulp_fw::gelu_expected_bf16_daz(x);
        float x_bf16 = bf16_ulp_fw::bf16_bits_to_float(bf16_ulp_fw::float_to_bf16_bits(x));
        if (expected == x_bf16) {
            if (!found_pos_identity) {
                pos_first_identity = x;
                found_pos_identity = true;
            }
        } else {
            pos_last_non_identity = x;
        }
    }

    std::cout << "\n============================================================\n";
    std::cout << "GELU FORWARD SATURATION BOUNDARY VERIFICATION\n";
    std::cout << "============================================================\n";
    std::cout << "\nGolden reference saturation boundaries (BF16 RNE model):\n";
    std::cout << "  Negative zero tail (deep saturation):\n";
    std::cout << "    Last x in zero region:     " << std::fixed << std::setprecision(4) << neg_sat_boundary_zero
              << " (0x" << std::hex << bf16_ulp_fw::float_to_bf16_bits(neg_sat_boundary_zero) << std::dec << ")\n";
    std::cout << "    First x past boundary:     " << std::fixed << std::setprecision(4) << neg_sat_boundary_nonzero
              << " (0x" << std::hex << bf16_ulp_fw::float_to_bf16_bits(neg_sat_boundary_nonzero) << std::dec << ")\n";
    std::cout << "    Kernel threshold:           -13.1875\n";
    std::cout << "  Positive identity tail:\n";
    std::cout << "    First x with GELU(x) == x: " << std::fixed << std::setprecision(4) << pos_first_identity << " (0x"
              << std::hex << bf16_ulp_fw::float_to_bf16_bits(pos_first_identity) << std::dec << ")\n";
    std::cout << "    Last x with GELU(x) != x:  " << std::fixed << std::setprecision(4) << pos_last_non_identity
              << " (0x" << std::hex << bf16_ulp_fw::float_to_bf16_bits(pos_last_non_identity) << std::dec << ")\n";
    std::cout << "    Kernel threshold:           2.78125\n";

    // --- Collect tail values for hardware verification ---
    // Negative tail: only deep saturation region (x <= neg_sat_boundary_zero)
    std::vector<float> neg_tail_values;
    neg_tail_values.reserve(all_values.size());
    // Positive tail: golden says identity (x >= pos_first_identity)
    std::vector<float> pos_tail_values;
    pos_tail_values.reserve(all_values.size());
    for (float x : all_values) {
        if (x < 0.0f && x <= neg_sat_boundary_zero) {
            neg_tail_values.push_back(x);
        }
        if (x > 0.0f && x >= pos_first_identity) {
            pos_tail_values.push_back(x);
        }
    }

    std::cout << "\n  Negative zero tail: " << neg_tail_values.size() << " BF16 values\n";
    std::cout << "  Positive identity tail: " << pos_tail_values.size() << " BF16 values\n";

    // Combine both tails for a single device run
    std::vector<float> tail_values;
    tail_values.insert(tail_values.end(), neg_tail_values.begin(), neg_tail_values.end());
    size_t neg_count = neg_tail_values.size();
    tail_values.insert(tail_values.end(), pos_tail_values.begin(), pos_tail_values.end());

    const size_t tile_size = tt::constants::TILE_HW;
    size_t padded_size = ((tail_values.size() + tile_size - 1) / tile_size) * tile_size;
    tail_values.resize(padded_size, 0.0f);

    uint32_t num_tiles = static_cast<uint32_t>(padded_size / tile_size);
    std::array<uint32_t, 4> dims = {1, 1, num_tiles * tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};

    std::vector<::bfloat16> bf16_inputs;
    bf16_inputs.reserve(tail_values.size());
    for (float x : tail_values) {
        bf16_inputs.push_back(::bfloat16(x));
    }

    tt::tt_metal::TensorSpec tensor_spec(
        tt::tt_metal::Shape(dims),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), tt::tt_metal::MemoryConfig{}));

    auto input_tensor = tt::tt_metal::Tensor::from_vector(std::move(bf16_inputs), tensor_spec).to_device(device_);
    auto result_tensor = ttnn::gelu(input_tensor, false);
    auto output_cpu = ttnn::from_device(result_tensor);
    auto output_vec = output_cpu.to_vector<::bfloat16>();

    // Verify negative tail: all should produce 0
    int neg_tail_max_ulp = 0;
    float neg_tail_worst_x = 0;
    int neg_tail_nonzero_count = 0;
    for (size_t i = 0; i < neg_count; ++i) {
        float actual = static_cast<float>(output_vec[i]);
        if (actual != 0.0f) {
            neg_tail_nonzero_count++;
            float expected = bf16_ulp_fw::gelu_expected_bf16_daz(neg_tail_values[i]);
            int32_t ulp = bf16_ulp_fw::ulp_distance_bf16_daz(actual, expected);
            if (ulp > neg_tail_max_ulp) {
                neg_tail_max_ulp = ulp;
                neg_tail_worst_x = neg_tail_values[i];
            }
        }
    }

    // Verify positive tail: all should produce identity (actual == x)
    int pos_tail_max_ulp = 0;
    float pos_tail_worst_x = 0;
    int pos_tail_non_identity_count = 0;
    for (size_t i = 0; i < pos_tail_values.size(); ++i) {
        float actual = static_cast<float>(output_vec[neg_count + i]);
        float x_bf16 = bf16_ulp_fw::bf16_bits_to_float(bf16_ulp_fw::float_to_bf16_bits(pos_tail_values[i]));
        if (actual != x_bf16) {
            pos_tail_non_identity_count++;
            float expected = bf16_ulp_fw::gelu_expected_bf16_daz(pos_tail_values[i]);
            int32_t ulp = bf16_ulp_fw::ulp_distance_bf16_daz(actual, expected);
            if (ulp > pos_tail_max_ulp) {
                pos_tail_max_ulp = ulp;
                pos_tail_worst_x = pos_tail_values[i];
            }
        }
    }

    std::cout << "\nHardware verification:\n";
    std::cout << "  Negative zero tail:     " << neg_tail_values.size() << " values, " << neg_tail_nonzero_count
              << " non-zero, max ULP=" << neg_tail_max_ulp << "\n";
    std::cout << "  Positive identity tail: " << pos_tail_values.size() << " values, " << pos_tail_non_identity_count
              << " non-identity, max ULP=" << pos_tail_max_ulp << "\n";
    std::cout << "============================================================\n";

    EXPECT_EQ(neg_tail_max_ulp, 0) << "Negative saturation tail should have ULP=0 (worst at x=" << neg_tail_worst_x
                                   << ")";
    EXPECT_EQ(neg_tail_nonzero_count, 0) << neg_tail_nonzero_count << " negative tail values returned non-zero";
    // Positive identity tail: kernel threshold (2.78125) matches golden boundary exactly.
    // All values x >= 2.78125 should return x with ULP=0.
    EXPECT_EQ(pos_tail_max_ulp, 0) << "Positive identity tail should have ULP=0 (worst at x=" << pos_tail_worst_x
                                   << ")";
    EXPECT_EQ(pos_tail_non_identity_count, 0)
        << pos_tail_non_identity_count << " positive tail values returned non-identity";
}

// Research test: locate every BF16 value with ULP > 1 and print full detail.
// Also analyzes the boundary neighborhood at x=-5 (exp/left-CDF boundary)
// to determine whether shifting the boundary could eliminate ULP=2 values.
TEST_F(GeluFwUlpTest, LocateULP2Values) {
    // Collect all valid BF16 values
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
    bf16_inputs.reserve(input_values.size());
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

    // Collect all ULP > 1 values
    struct ULP2Entry {
        float x;
        uint16_t x_bits;
        float actual;
        uint16_t actual_bits;
        float expected;
        uint16_t expected_bits;
        int32_t ulp;
        std::string region;
    };
    std::vector<ULP2Entry> ulp2_entries;
    ulp2_entries.reserve(256);

    // Also collect detailed info for all values near x=-5 boundary
    struct BoundaryEntry {
        float x;
        uint16_t x_bits;
        float actual;
        float expected;
        int32_t ulp;
        std::string region;
    };
    std::vector<BoundaryEntry> boundary_entries;
    boundary_entries.reserve(1024);

    for (size_t i = 0; i < valid_count; ++i) {
        float x = bf16_ulp_fw::bf16_bits_to_float(bf16_ulp_fw::float_to_bf16_bits(input_values[i]));
        float actual = static_cast<float>(output_vec[i]);
        float expected = bf16_ulp_fw::gelu_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp_fw::ulp_distance_bf16_daz(actual, expected);
        if (ulp < 0) {
            continue;
        }

        // Determine region
        std::string region;
        if (x <= -13.1875f) {
            region = "saturation";
        } else if (x <= -5.0f) {
            region = "exp-based";
        } else if (x < -3.0f) {
            region = "left-CDF";
        } else if (x < 2.78125f) {
            region = "core-CDF";
        } else {
            region = "identity";
        }

        if (ulp > 1) {
            ulp2_entries.push_back(
                {x,
                 bf16_ulp_fw::float_to_bf16_bits(x),
                 actual,
                 bf16_ulp_fw::float_to_bf16_bits(actual),
                 expected,
                 bf16_ulp_fw::float_to_bf16_bits(expected),
                 ulp,
                 region});
        }

        // Collect values in the exp/left-CDF overlap research zone
        if (x >= -7.0f && x <= -3.0f) {
            boundary_entries.push_back({x, bf16_ulp_fw::float_to_bf16_bits(x), actual, expected, ulp, region});
        }
    }

    // Print all ULP > 1 values
    std::cout << "\n============================================================\n";
    std::cout << "ALL BF16 VALUES WITH ULP > 1\n";
    std::cout << "============================================================\n";
    std::cout << std::setw(12) << "x" << std::setw(10) << "x_bits" << std::setw(15) << "expected" << std::setw(10)
              << "exp_bits" << std::setw(15) << "actual" << std::setw(10) << "act_bits" << std::setw(6) << "ULP"
              << std::setw(12) << "region\n";
    std::cout << std::string(90, '-') << "\n";

    for (const auto& e : ulp2_entries) {
        std::cout << std::setw(12) << std::scientific << std::setprecision(4) << e.x << "  0x" << std::hex
                  << std::setfill('0') << std::setw(4) << e.x_bits << std::dec << std::setfill(' ') << std::setw(15)
                  << std::scientific << std::setprecision(4) << e.expected << "  0x" << std::hex << std::setfill('0')
                  << std::setw(4) << e.expected_bits << std::dec << std::setfill(' ') << std::setw(15)
                  << std::scientific << std::setprecision(4) << e.actual << "  0x" << std::hex << std::setfill('0')
                  << std::setw(4) << e.actual_bits << std::dec << std::setfill(' ') << std::setw(6) << e.ulp
                  << std::setw(12) << e.region << "\n";
    }
    std::cout << "\nTotal ULP > 1 values: " << ulp2_entries.size() << "\n";

    // Print boundary neighborhood
    std::cout << "\n============================================================\n";
    std::cout << "BOUNDARY NEIGHBORHOOD: x in [-7, -3]\n";
    std::cout << "(exp/left-CDF boundary research zone)\n";
    std::cout << "============================================================\n";
    std::cout << std::setw(12) << "x" << std::setw(10) << "x_bits" << std::setw(15) << "expected" << std::setw(15)
              << "actual" << std::setw(6) << "ULP" << std::setw(12) << "region\n";
    std::cout << std::string(70, '-') << "\n";

    // Sort by x value for readability
    std::sort(boundary_entries.begin(), boundary_entries.end(), [](const BoundaryEntry& a, const BoundaryEntry& b) {
        return a.x < b.x;
    });

    for (const auto& e : boundary_entries) {
        std::string marker = (e.ulp > 1) ? " <<< ULP>1" : "";
        std::cout << std::setw(12) << std::fixed << std::setprecision(4) << e.x << "  0x" << std::hex
                  << std::setfill('0') << std::setw(4) << e.x_bits << std::dec << std::setfill(' ') << std::setw(15)
                  << std::scientific << std::setprecision(4) << e.expected << std::setw(15) << e.actual << std::setw(6)
                  << e.ulp << std::setw(12) << e.region << marker << "\n";
    }

    // Analyze: for each ULP>1 value, check what ULP it would get from the OTHER region's formula
    std::cout << "\n============================================================\n";
    std::cout << "BOUNDARY SHIFT ANALYSIS\n";
    std::cout << "Can shifting the x=-5 boundary eliminate ULP=2 values?\n";
    std::cout << "============================================================\n";

    // Find the range of ULP>1 values
    if (!ulp2_entries.empty()) {
        float min_x = ulp2_entries[0].x;
        float max_x = ulp2_entries[0].x;
        for (const auto& e : ulp2_entries) {
            min_x = std::min(min_x, e.x);
            max_x = std::max(max_x, e.x);
        }
        std::cout << "ULP>1 value range: [" << std::fixed << std::setprecision(4) << min_x << ", " << max_x << "]\n";

        // Count how many are on each side of x=-5
        int exp_side = 0, cdf_side = 0;
        for (const auto& e : ulp2_entries) {
            if (e.x <= -5.0f) {
                exp_side++;
            } else {
                cdf_side++;
            }
        }
        std::cout << "  In exp-based region (x <= -5): " << exp_side << " values\n";
        std::cout << "  In left-CDF region (x > -5):   " << cdf_side << " values\n";

        // Suggest: if all ULP>1 values are near x=-5, shifting might help
        if (min_x >= -5.5f && max_x <= -4.5f) {
            std::cout << "\nAll ULP>1 values cluster near x=-5 boundary.\n";
            std::cout << "Shifting the boundary might eliminate them.\n";
            std::cout << "Candidate boundaries to try:\n";

            // List BF16 values between min_x and max_x as potential boundaries
            for (const auto& e : boundary_entries) {
                if (e.x >= min_x - 0.25f && e.x <= max_x + 0.25f) {
                    std::cout << "  x=" << std::fixed << std::setprecision(4) << e.x << " (0x" << std::hex
                              << std::setfill('0') << std::setw(4) << e.x_bits << std::dec << std::setfill(' ')
                              << ") ULP=" << e.ulp << " [" << e.region << "]\n";
                }
            }
        }
    } else {
        std::cout << "No ULP>1 values found!\n";
    }
    std::cout << "============================================================\n";

    // Regression guard: ensure ULP never exceeds 2 for any BF16 value
    EXPECT_LE(ulp2_entries.size(), 2u) << "Expected at most 2 values with ULP > 1; got " << ulp2_entries.size();
    for (const auto& e : ulp2_entries) {
        EXPECT_LE(e.ulp, 2) << "ULP too high at x=" << e.x << ": " << e.ulp;
    }
}

// Special values: +inf, -inf, NaN, +0, -0
TEST_F(GeluFwUlpTest, SpecialValues) {
    struct SpecialCase {
        std::string name;
        float input;
        uint16_t expected_bits;
    };

    // GELU(+inf)=+inf, GELU(-inf)=0, GELU(0)=0
    std::vector<SpecialCase> cases = {
        {"positive zero", 0.0f, 0x0000},
        {"-inf → 0", -std::numeric_limits<float>::infinity(), 0x0000},
        {"+inf → +inf", std::numeric_limits<float>::infinity(), bf16_ulp_fw::BF16_POS_INF},
    };

    for (const auto& sc : cases) {
        float actual = run_gelu_fw_single(*device_, sc.input);
        uint16_t actual_bits = bf16_ulp_fw::float_to_bf16_bits(actual);

        EXPECT_EQ(actual_bits, sc.expected_bits)
            << sc.name << ": expected 0x" << std::hex << sc.expected_bits << ", got 0x" << actual_bits;
    }
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
