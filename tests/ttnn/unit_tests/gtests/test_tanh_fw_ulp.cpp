// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * tanh Forward ULP Precision Tests
 *
 * Validates accuracy of ttnn::tanh (forward) across the entire BFloat16 range.
 * Serves as a baseline comparison for the tanh_bw fix — forward tanh achieves
 * Max ULP = 1, proving that correct BF16 implementation is achievable.
 *
 * MATHEMATICAL FORMULA:
 * tanh(x) = (e^x - e^-x) / (e^x + e^-x)
 *
 * REFERENCE IMPLEMENTATION:
 * Uses fp64 std::tanh() — well beyond BF16 precision.
 *
 * HARDWARE MODEL: DAZ+FTZ (Denormals-Are-Zero + Flush-To-Zero)
 * Per tech_reports/Handling_Special_Value/special_values.md: "denormals | all | 0x0"
 * All denormal BF16 values are excluded from test inputs. NaN and Inf are excluded
 * per policy: "all inputs to the device should be filtered from the host not to have
 * any Inf/NaNs".
 *
 * Run: ./build_Debug/test/ttnn/unit_tests_ttnn --gtest_filter="*TanhFwUlp*"
 */

#include <gtest/gtest.h>
#include <mpfr.h>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <map>
#include <vector>
#include <iomanip>
#include <sstream>

#include <tt-logger/tt-logger.hpp>
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
// BFloat16 ULP Calculator
// =============================================================================

namespace bf16_ulp_tanh_fw {

constexpr uint16_t BF16_EXP_MASK = 0x7F80;
constexpr uint16_t BF16_MANTISSA_MASK = 0x007F;
constexpr uint16_t BF16_SIGN_MASK = 0x8000;
constexpr uint16_t BF16_POS_INF = 0x7F80;
constexpr uint16_t BF16_NEG_INF = 0xFF80;

inline uint16_t float_to_bf16_bits(float f) {
    uint32_t f32_bits;
    std::memcpy(&f32_bits, &f, sizeof(float));
    // Round-to-nearest-even (RNE): add rounding bias before truncating
    uint32_t rounding_bias = ((f32_bits >> 16) & 1) + 0x7FFF;
    f32_bits += rounding_bias;
    return static_cast<uint16_t>(f32_bits >> 16);
}

inline float bf16_bits_to_float(uint16_t bits) {
    uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &f32_bits, sizeof(float));
    return f;
}

inline bool is_bf16_denormal(uint16_t bits) { return ((bits >> 7) & 0xFF) == 0 && (bits & BF16_MANTISSA_MASK) != 0; }

inline uint16_t bf16_daz_normalize(uint16_t bits) {
    if (is_bf16_denormal(bits)) {
        return 0x0000;
    }
    if (bits == 0x8000) {
        return 0x0000;
    }
    return bits;
}

inline float bf16_daz_normalize(float f) { return bf16_bits_to_float(bf16_daz_normalize(float_to_bf16_bits(f))); }

inline int32_t bf16_value_order_index_daz(uint16_t bits) {
    bits = bf16_daz_normalize(bits);
    uint16_t exp = (bits >> 7) & 0xFF;
    uint16_t mantissa = bits & BF16_MANTISSA_MASK;
    if (exp == 0xFF && mantissa != 0) {
        return -1;
    }
    if (bits == BF16_POS_INF) {
        return 65281;
    }
    if (bits == BF16_NEG_INF) {
        return -1;
    }
    if (bits == 0x0000) {
        return 32640;
    }
    if (bits & BF16_SIGN_MASK) {
        return 0x7FFF - (bits & 0x7FFF);
    }
    return 32640 + bits - BF16_MANTISSA_MASK;
}

inline int32_t ulp_distance_bf16_daz(float a, float b) {
    uint16_t a_bits = bf16_daz_normalize(float_to_bf16_bits(a));
    uint16_t b_bits = bf16_daz_normalize(float_to_bf16_bits(b));
    int32_t idx_a = bf16_value_order_index_daz(a_bits);
    int32_t idx_b = bf16_value_order_index_daz(b_bits);
    if (idx_a < 0 || idx_b < 0) {
        return -1;
    }
    return std::abs(idx_a - idx_b);
}

inline std::vector<float> all_valid_bf16_values() {
    std::vector<float> values;
    values.reserve(65536);
    for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        uint16_t bf16_bits = static_cast<uint16_t>(bits);
        if ((bf16_bits & BF16_EXP_MASK) == BF16_EXP_MASK) {
            continue;
        }
        if (is_bf16_denormal(bf16_bits)) {
            continue;
        }
        values.push_back(bf16_bits_to_float(bf16_bits));
    }
    return values;
}

}  // namespace bf16_ulp_tanh_fw

// =============================================================================
// Reference ULP calculator: brute-force sorted table
// =============================================================================

namespace bf16_ulp_reference {

// Build a lookup table mapping every uint16_t BF16 bit pattern to its
// order index under DAZ+FTZ. The table is constructed mechanically:
//   1. Enumerate all 65,536 bit patterns
//   2. Filter out NaN and Inf
//   3. Convert to float (with DAZ: denormals → 0)
//   4. Sort by float value
//   5. Assign indices: equal values share the same index
//
// This is the ground truth — no formulas, no assumptions.

struct Bf16Entry {
    uint16_t bits;
    float value;
};

inline std::unordered_map<uint16_t, int32_t> build_order_index_table() {
    // Collect all non-NaN, non-Inf bit patterns with DAZ-normalized values
    std::vector<Bf16Entry> entries;
    entries.reserve(65536);
    for (uint32_t b = 0; b <= 0xFFFF; ++b) {
        uint16_t bits = static_cast<uint16_t>(b);
        uint16_t exp = (bits >> 7) & 0xFF;
        uint16_t mantissa = bits & 0x007F;

        // Skip NaN
        if (exp == 0xFF && mantissa != 0) {
            continue;
        }
        // Skip Inf
        if (bits == 0x7F80 || bits == 0xFF80) {
            continue;
        }

        // DAZ normalize: denormals and -0 become +0
        uint16_t norm_bits = bf16_ulp_tanh_fw::bf16_daz_normalize(bits);
        float value = bf16_ulp_tanh_fw::bf16_bits_to_float(norm_bits);

        entries.push_back({bits, value});
    }

    // Sort by float value
    std::sort(entries.begin(), entries.end(), [](const Bf16Entry& a, const Bf16Entry& b) { return a.value < b.value; });

    // Assign order indices: equal values share the same index
    std::unordered_map<uint16_t, int32_t> table;
    int32_t current_index = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        if (i > 0 && entries[i].value != entries[i - 1].value) {
            current_index++;
        }
        table[entries[i].bits] = current_index;
    }

    return table;
}

inline int32_t reference_ulp_distance(
    const std::unordered_map<uint16_t, int32_t>& table, uint16_t a_bits, uint16_t b_bits) {
    // DAZ normalize both
    a_bits = bf16_ulp_tanh_fw::bf16_daz_normalize(a_bits);
    b_bits = bf16_ulp_tanh_fw::bf16_daz_normalize(b_bits);

    auto it_a = table.find(a_bits);
    auto it_b = table.find(b_bits);
    if (it_a == table.end() || it_b == table.end()) {
        return -1;
    }
    return std::abs(it_a->second - it_b->second);
}

}  // namespace bf16_ulp_reference

// =============================================================================
// tanh reference: fp64 std::tanh (golden)
// =============================================================================

inline float tanh_expected_bf16_daz(float x) {
    float x_daz = bf16_ulp_tanh_fw::bf16_daz_normalize(x);
    double result = std::tanh(static_cast<double>(x_daz));
    return bf16_ulp_tanh_fw::bf16_daz_normalize(static_cast<float>(result));
}

// =============================================================================
// tanh reference: MPFR-256 (platinum)
// =============================================================================

inline float tanh_mpfr256_bf16_daz(float x) {
    float x_daz = bf16_ulp_tanh_fw::bf16_daz_normalize(x);

    mpfr_t mx, mr;
    mpfr_init2(mx, 256);
    mpfr_init2(mr, 256);

    mpfr_set_d(mx, static_cast<double>(x_daz), MPFR_RNDN);
    mpfr_tanh(mr, mx, MPFR_RNDN);

    float result = static_cast<float>(mpfr_get_d(mr, MPFR_RNDN));

    mpfr_clear(mx);
    mpfr_clear(mr);

    return bf16_ulp_tanh_fw::bf16_daz_normalize(result);
}

// =============================================================================
// Test Fixture
// =============================================================================

class TanhFwUlpTest : public TTNNFixtureWithDevice {};

float run_tanh_fw_single(tt::tt_metal::distributed::MeshDevice& device, float input_val) {
    ttnn::Shape shape({1, 1, 32, 32});
    auto input_tensor = ttnn::full(shape, input_val, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    auto result = ttnn::tanh(input_tensor);
    auto output_cpu = ttnn::from_device(result);
    auto output_vec = output_cpu.to_vector<::bfloat16>();
    return static_cast<float>(output_vec[0]);
}

std::vector<::bfloat16> run_tanh_fw_batched(
    tt::tt_metal::distributed::MeshDevice* device, const std::vector<float>& input_values) {
    const size_t tile_size = tt::constants::TILE_HW;
    size_t padded_size = ((input_values.size() + tile_size - 1) / tile_size) * tile_size;

    std::vector<::bfloat16> bf16_inputs;
    bf16_inputs.reserve(padded_size);
    for (size_t i = 0; i < padded_size; ++i) {
        float x = (i < input_values.size()) ? input_values[i] : 0.0f;
        bf16_inputs.push_back(::bfloat16(x));
    }

    uint32_t num_tiles = static_cast<uint32_t>(padded_size / tile_size);
    std::array<uint32_t, 4> dims = {1, 1, num_tiles * tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};

    tt::tt_metal::TensorSpec tensor_spec(
        tt::tt_metal::Shape(dims),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), tt::tt_metal::MemoryConfig{}));

    auto input_tensor = tt::tt_metal::Tensor::from_vector(std::move(bf16_inputs), tensor_spec).to_device(device);
    auto result = ttnn::tanh(input_tensor);
    auto output_cpu = ttnn::from_device(result);
    return output_cpu.to_vector<::bfloat16>();
}

// =============================================================================
// Smoke tests
// =============================================================================

// tanh(0) = 0 exactly.
TEST_F(TanhFwUlpTest, ValueAtZero) {
    float actual = run_tanh_fw_single(*device_, 0.0f);
    float expected = tanh_expected_bf16_daz(0.0f);
    int32_t ulp = bf16_ulp_tanh_fw::ulp_distance_bf16_daz(actual, expected);
    log_debug(tt::LogTest, "x=0: expected={}, actual={}, ULP={}", expected, actual, ulp);
    EXPECT_EQ(actual, 0.0f) << "tanh(0) should be exactly 0";
    EXPECT_EQ(ulp, 0);
}

// tanh(x) → 1 for positive x.
TEST_F(TanhFwUlpTest, PositiveValues) {
    std::vector<float> values = {0.5f, 1.0f, 2.0f, 3.0f, 5.0f, 10.0f};
    for (float x : values) {
        float actual = run_tanh_fw_single(*device_, x);
        float expected = tanh_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp_tanh_fw::ulp_distance_bf16_daz(actual, expected);
        log_debug(tt::LogTest, "x={}: expected={}, actual={}, ULP={}", x, expected, actual, ulp);
        EXPECT_LE(ulp, 1) << "tanh(" << x << ") ULP too high";
    }
}

// tanh(-x) = -tanh(x) — odd function symmetry.
TEST_F(TanhFwUlpTest, NegativeValues) {
    std::vector<float> values = {-0.5f, -1.0f, -2.0f, -3.0f, -5.0f, -10.0f};
    for (float x : values) {
        float actual = run_tanh_fw_single(*device_, x);
        float expected = tanh_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp_tanh_fw::ulp_distance_bf16_daz(actual, expected);
        log_debug(tt::LogTest, "x={}: expected={}, actual={}, ULP={}", x, expected, actual, ulp);
        EXPECT_LE(ulp, 1) << "tanh(" << x << ") ULP too high";
    }
}

// Near-zero region where tanh(x) ≈ x.
// Threshold is 2 because some test values (e.g. 0.01f) are not exactly representable
// in BF16, causing up to 1 ULP input rounding on top of 1 ULP computation error.
TEST_F(TanhFwUlpTest, NearZero) {
    std::vector<float> values = {1e-6f, 1e-4f, 0.01f, 0.1f, -0.1f, -0.01f, -1e-4f};
    for (float x : values) {
        float actual = run_tanh_fw_single(*device_, x);
        float expected = tanh_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp_tanh_fw::ulp_distance_bf16_daz(actual, expected);
        log_debug(tt::LogTest, "x={}: expected={}, actual={}, ULP={}", x, expected, actual, ulp);
        EXPECT_LE(ulp, 2) << "tanh(" << x << ") near-zero ULP too high";
    }
}

// =============================================================================
// Full BF16 sweep: per-segment ULP analysis
// =============================================================================

TEST_F(TanhFwUlpTest, PerSegmentULPAnalysis) {
    auto input_values = bf16_ulp_tanh_fw::all_valid_bf16_values();
    const size_t valid_count = input_values.size();

    const size_t tile_size = tt::constants::TILE_HW;
    size_t padded_size = ((valid_count + tile_size - 1) / tile_size) * tile_size;
    input_values.resize(padded_size, 0.0f);

    auto output_vec = run_tanh_fw_batched(device_, input_values);

    struct RegionStats {
        std::string name;
        int count = 0;
        double ulp_sum = 0;
        int max_ulp = 0;
        float worst_x = 0;
    };

    std::vector<RegionStats> regions = {
        {"x < -10"},
        {"-10 <= x < -5"},
        {"-5 <= x < -4"},
        {"-4 <= x < -3"},
        {"-3 <= x < -2"},
        {"-2 <= x < -1"},
        {"-1 <= x < 0"},
        {"x == 0"},
        {"0 < x < 1"},
        {"1 <= x < 2"},
        {"2 <= x < 3"},
        {"3 <= x < 4"},
        {"4 <= x < 5"},
        {"5 <= x < 10"},
        {"x >= 10"},
    };

    int overall_max_ulp = 0;
    float overall_worst_x = 0;
    double overall_ulp_sum = 0;
    int overall_count = 0;

    for (size_t i = 0; i < valid_count; ++i) {
        float x = bf16_ulp_tanh_fw::bf16_bits_to_float(bf16_ulp_tanh_fw::float_to_bf16_bits(input_values[i]));
        float actual = static_cast<float>(output_vec[i]);
        float expected = tanh_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp_tanh_fw::ulp_distance_bf16_daz(actual, expected);
        if (ulp < 0) {
            continue;
        }

        int idx;
        if (x < -10.0f) {
            idx = 0;
        } else if (x < -5.0f) {
            idx = 1;
        } else if (x < -4.0f) {
            idx = 2;
        } else if (x < -3.0f) {
            idx = 3;
        } else if (x < -2.0f) {
            idx = 4;
        } else if (x < -1.0f) {
            idx = 5;
        } else if (x < 0.0f) {
            idx = 6;
        } else if (x == 0.0f) {
            idx = 7;
        } else if (x < 1.0f) {
            idx = 8;
        } else if (x < 2.0f) {
            idx = 9;
        } else if (x < 3.0f) {
            idx = 10;
        } else if (x < 4.0f) {
            idx = 11;
        } else if (x < 5.0f) {
            idx = 12;
        } else if (x < 10.0f) {
            idx = 13;
        } else {
            idx = 14;
        }

        regions[idx].count++;
        regions[idx].ulp_sum += ulp;
        if (ulp > regions[idx].max_ulp) {
            regions[idx].max_ulp = ulp;
            regions[idx].worst_x = x;
        }
        overall_ulp_sum += ulp;
        overall_count++;
        if (ulp > overall_max_ulp) {
            overall_max_ulp = ulp;
            overall_worst_x = x;
        }
    }

    {
        std::ostringstream oss;
        oss << "\n============================================================\n"
            << "TANH (FORWARD) PER-SEGMENT ULP ANALYSIS (DAZ+FTZ MODEL)\n"
            << "============================================================\n"
            << std::setw(20) << "Region" << std::setw(10) << "Count" << std::setw(12) << "Mean ULP" << std::setw(12)
            << "Max ULP" << std::setw(15) << "Worst x\n"
            << std::string(69, '-') << "\n";
        for (const auto& r : regions) {
            if (r.count > 0) {
                oss << std::setw(20) << r.name << std::setw(10) << r.count << std::setw(12) << std::fixed
                    << std::setprecision(2) << (r.ulp_sum / r.count) << std::setw(12) << r.max_ulp << std::setw(15)
                    << std::scientific << std::setprecision(3) << r.worst_x << "\n";
            }
        }
        oss << std::string(69, '-') << "\n"
            << std::setw(20) << "OVERALL" << std::setw(10) << overall_count << std::setw(12) << std::fixed
            << std::setprecision(2) << (overall_ulp_sum / overall_count) << std::setw(12) << overall_max_ulp
            << std::setw(15) << std::scientific << std::setprecision(3) << overall_worst_x << "\n"
            << "============================================================\n";
        log_debug(tt::LogTest, "{}", oss.str());
    }

    for (const auto& r : regions) {
        EXPECT_LE(r.max_ulp, 1) << "Region '" << r.name << "' max ULP " << r.max_ulp
                                << " exceeds threshold 1 (worst at x=" << r.worst_x << ")";
    }
    EXPECT_LE(overall_max_ulp, 1) << "Overall max ULP " << overall_max_ulp << " at x=" << overall_worst_x;
}

// =============================================================================
// Full BF16 sweep: cumulative ULP histogram
// =============================================================================

TEST_F(TanhFwUlpTest, CumulativeULPDistribution) {
    auto input_values = bf16_ulp_tanh_fw::all_valid_bf16_values();
    const size_t valid_count = input_values.size();

    const size_t tile_size = tt::constants::TILE_HW;
    size_t padded_size = ((valid_count + tile_size - 1) / tile_size) * tile_size;
    input_values.resize(padded_size, 0.0f);

    auto output_vec = run_tanh_fw_batched(device_, input_values);

    std::map<int32_t, int> ulp_histogram;
    int max_ulp = 0;
    float worst_x = 0;

    for (size_t i = 0; i < valid_count; ++i) {
        float x = bf16_ulp_tanh_fw::bf16_bits_to_float(bf16_ulp_tanh_fw::float_to_bf16_bits(input_values[i]));
        float actual = static_cast<float>(output_vec[i]);
        float expected = tanh_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp_tanh_fw::ulp_distance_bf16_daz(actual, expected);
        if (ulp >= 0) {
            ulp_histogram[ulp]++;
            if (ulp > max_ulp) {
                max_ulp = ulp;
                worst_x = x;
            }
        }
    }

    {
        std::ostringstream oss;
        oss << "\n============================================================\n"
            << "TANH (FORWARD) CUMULATIVE ULP DISTRIBUTION (DAZ+FTZ MODEL)\n"
            << "============================================================\n"
            << std::setw(10) << "ULP <=" << std::setw(12) << "Count" << std::setw(12) << "Percent\n"
            << std::string(34, '-') << "\n";

        std::vector<int> thresholds = {0, 1, 2, 3, 5};
        for (int t : thresholds) {
            int count = 0;
            for (auto& [u, c] : ulp_histogram) {
                if (u <= t) {
                    count += c;
                }
            }
            oss << std::setw(10) << t << std::setw(12) << count << std::setw(11) << std::fixed << std::setprecision(2)
                << (100.0 * count / valid_count) << "%\n";
        }
        oss << std::string(34, '-') << "\n"
            << "Max ULP: " << max_ulp << " at x = " << worst_x << "\n"
            << "============================================================\n";
        log_debug(tt::LogTest, "{}", oss.str());
    }

    EXPECT_LE(max_ulp, 1) << "Max ULP " << max_ulp << " at x=" << worst_x << " exceeds threshold 1";

    // 100% of values should be within 1 ULP
    int count_le_1 = 0;
    for (auto& [u, c] : ulp_histogram) {
        if (u <= 1) {
            count_le_1 += c;
        }
    }
    double pct_le_1 = 100.0 * count_le_1 / valid_count;
    EXPECT_GE(pct_le_1, 100.0) << "Only " << pct_le_1 << "% within 1 ULP (expected 100%)";
}

// =============================================================================
// Reference implementation verification
// =============================================================================

TEST_F(TanhFwUlpTest, ReferenceImplementationVerification) {
    // tanh(0) = 0
    EXPECT_NEAR(std::tanh(0.0), 0.0, 1e-15) << "tanh(0) should be 0";

    // tanh(large) → 1
    EXPECT_NEAR(std::tanh(100.0), 1.0, 1e-6) << "tanh(100) should approach 1";

    // tanh(-large) → -1
    EXPECT_NEAR(std::tanh(-100.0), -1.0, 1e-6) << "tanh(-100) should approach -1";

    // tanh(-x) = -tanh(x)
    double t_pos = std::tanh(2.0);
    double t_neg = std::tanh(-2.0);
    EXPECT_NEAR(t_pos, -t_neg, 1e-15) << "tanh must be an odd function";

    // tanh(1) ≈ 0.7615941...
    EXPECT_NEAR(std::tanh(1.0), 0.7615941, 1e-6) << "tanh(1) ≈ 0.7616";

    log_debug(tt::LogTest, "tanh(0) = {}", std::tanh(0.0));
    log_debug(tt::LogTest, "tanh(1) = {} (expected ~0.7616)", std::tanh(1.0));
    log_debug(tt::LogTest, "tanh(100) = {} (expected ~1.0)", std::tanh(100.0));
    log_debug(tt::LogTest, "tanh(-100) = {} (expected ~-1.0)", std::tanh(-100.0));
}

// =============================================================================
// ULP calculator verification: formula vs brute-force sorted table
// =============================================================================

// Validates bf16_value_order_index_daz() against a mechanically-constructed
// reference table. The reference enumerates all BF16 bit patterns, sorts by
// float value, and assigns order indices. Any formula bug will show as a
// mismatch. This test does NOT require a device.
TEST_F(TanhFwUlpTest, UlpCalculatorVerification) {
    auto ref_table = bf16_ulp_reference::build_order_index_table();

    int index_mismatches = 0;
    int distance_mismatches = 0;
    int total_patterns = 0;

    std::ostringstream oss;
    oss << "\n============================================================\n"
        << "ULP CALCULATOR: FORMULA vs REFERENCE TABLE\n"
        << "============================================================\n";

    // Part 1: Verify order indices preserve distances (constant offset is OK)
    // The formula and reference may use different origins, so we check that
    // (formula[a] - formula[b]) == (ref[a] - ref[b]) for all pairs.
    // Equivalently: (formula[x] - ref[x]) must be the same constant for all x.
    int32_t offset = 0;
    bool offset_set = false;
    for (uint32_t b = 0; b <= 0xFFFF; ++b) {
        uint16_t bits = static_cast<uint16_t>(b);
        uint16_t exp = (bits >> 7) & 0xFF;

        // Skip NaN and Inf — formula returns -1 for these
        if (exp == 0xFF) {
            continue;
        }

        total_patterns++;

        int32_t formula_idx = bf16_ulp_tanh_fw::bf16_value_order_index_daz(bits);
        uint16_t norm_bits = bf16_ulp_tanh_fw::bf16_daz_normalize(bits);
        auto it = ref_table.find(norm_bits);

        if (it == ref_table.end()) {
            index_mismatches++;
            if (index_mismatches <= 10) {
                oss << "  INDEX: bits=0x" << std::hex << bits << " norm=0x" << norm_bits << " formula=" << std::dec
                    << formula_idx << " ref=NOT_FOUND\n";
            }
            continue;
        }

        int32_t diff = formula_idx - it->second;
        if (!offset_set) {
            offset = diff;
            offset_set = true;
        } else if (diff != offset) {
            index_mismatches++;
            if (index_mismatches <= 10) {
                float val = bf16_ulp_tanh_fw::bf16_bits_to_float(norm_bits);
                oss << "  INDEX: bits=0x" << std::hex << bits << " val=" << std::scientific << val
                    << " formula=" << std::dec << formula_idx << " ref=" << it->second << " offset=" << diff
                    << " (expected " << offset << ")\n";
            }
        }
    }

    oss << "Index offset (formula - ref): " << offset << "\n";

    // Part 2: Verify ULP distance for strategic value pairs
    struct DistTestCase {
        float a;
        float b;
    };
    std::vector<DistTestCase> dist_cases = {
        {0.0f, 0.0f},
        {1.0f, 1.0f},
        {1.0f, 1.0078125f},    // adjacent BF16 values
        {-1.0f, -1.0078125f},  // adjacent negative
        {0.0f, 1.0f},
        {-1.0f, 1.0f},
        {-3.389e+38f, 3.389e+38f},  // near extremes
        {1.175494e-38f, 0.0f},      // FLT_MIN to zero (DAZ boundary)
    };

    for (auto& tc : dist_cases) {
        uint16_t a_bits = bf16_ulp_tanh_fw::bf16_daz_normalize(bf16_ulp_tanh_fw::float_to_bf16_bits(tc.a));
        uint16_t b_bits = bf16_ulp_tanh_fw::bf16_daz_normalize(bf16_ulp_tanh_fw::float_to_bf16_bits(tc.b));

        int32_t formula_dist = bf16_ulp_tanh_fw::ulp_distance_bf16_daz(tc.a, tc.b);
        int32_t ref_dist = bf16_ulp_reference::reference_ulp_distance(ref_table, a_bits, b_bits);

        if (formula_dist != ref_dist) {
            distance_mismatches++;
            oss << "  DISTANCE: a=" << tc.a << " b=" << tc.b << " formula=" << formula_dist << " ref=" << ref_dist
                << "\n";
        }
    }

    // Part 3: Exhaustive ULP distance check — every adjacent pair in sorted order
    // Collect all unique DAZ-normalized values and sort them
    std::vector<std::pair<float, uint16_t>> sorted_values;
    std::set<uint16_t> seen_norm;
    for (uint32_t b = 0; b <= 0xFFFF; ++b) {
        uint16_t bits = static_cast<uint16_t>(b);
        uint16_t exp = (bits >> 7) & 0xFF;
        if (exp == 0xFF) {
            continue;
        }
        uint16_t norm = bf16_ulp_tanh_fw::bf16_daz_normalize(bits);
        if (seen_norm.count(norm)) {
            continue;
        }
        seen_norm.insert(norm);
        sorted_values.push_back({bf16_ulp_tanh_fw::bf16_bits_to_float(norm), norm});
    }
    std::sort(sorted_values.begin(), sorted_values.end());

    int adjacent_mismatches = 0;
    for (size_t i = 0; i + 1 < sorted_values.size(); ++i) {
        float a = sorted_values[i].first;
        float b = sorted_values[i + 1].first;
        int32_t formula_dist = bf16_ulp_tanh_fw::ulp_distance_bf16_daz(a, b);

        // Adjacent unique values should be exactly 1 ULP apart
        if (formula_dist != 1) {
            adjacent_mismatches++;
            if (adjacent_mismatches <= 10) {
                oss << "  ADJACENT: a=" << std::scientific << a << " (0x" << std::hex << sorted_values[i].second
                    << ") b=" << std::scientific << b << " (0x" << sorted_values[i + 1].second << ") dist=" << std::dec
                    << formula_dist << " expected=1\n";
            }
        }
    }

    oss << "------------------------------------------------------------\n"
        << "Patterns tested: " << total_patterns << "\n"
        << "Index mismatches: " << index_mismatches << "\n"
        << "Distance mismatches (strategic): " << distance_mismatches << "\n"
        << "Adjacent distance mismatches: " << adjacent_mismatches << "\n"
        << "Unique DAZ values: " << sorted_values.size() << "\n"
        << "============================================================\n";
    log_debug(tt::LogTest, "{}", oss.str());

    EXPECT_EQ(index_mismatches, 0) << index_mismatches << " order index mismatches between formula and reference table";
    EXPECT_EQ(distance_mismatches, 0) << distance_mismatches << " ULP distance mismatches on strategic pairs";
    EXPECT_EQ(adjacent_mismatches, 0) << adjacent_mismatches << " adjacent value pairs with ULP distance != 1";
}

// =============================================================================
// Platinum reference verification: fp64 golden vs MPFR-256
// =============================================================================

// Validates that our fp64 golden reference (std::tanh) produces the same
// BF16 result as the MPFR-256 platinum reference across all valid BF16 values.
// Any disagreement means the golden reference has insufficient precision.
TEST_F(TanhFwUlpTest, PlatinumReferenceVerification) {
    auto all_values = bf16_ulp_tanh_fw::all_valid_bf16_values();

    int mismatches = 0;
    int total = 0;
    float worst_x = 0;
    float worst_golden = 0;
    float worst_platinum = 0;

    std::ostringstream oss;
    oss << "\n============================================================\n"
        << "GOLDEN (fp64) vs PLATINUM (MPFR-256) REFERENCE COMPARISON\n"
        << "============================================================\n";

    for (float x : all_values) {
        float golden = tanh_expected_bf16_daz(x);
        float platinum = tanh_mpfr256_bf16_daz(x);
        total++;

        if (golden != platinum) {
            mismatches++;
            if (mismatches <= 20) {
                int32_t ulp_diff = bf16_ulp_tanh_fw::ulp_distance_bf16_daz(golden, platinum);
                oss << "  MISMATCH at x=" << std::scientific << std::setprecision(6) << x << ": golden=" << golden
                    << " platinum=" << platinum << " ULP_diff=" << ulp_diff << "\n";
            }
            worst_x = x;
            worst_golden = golden;
            worst_platinum = platinum;
        }
    }

    oss << "------------------------------------------------------------\n"
        << "Total values: " << total << ", Mismatches: " << mismatches << "\n"
        << "============================================================\n";
    log_debug(tt::LogTest, "{}", oss.str());

    EXPECT_EQ(mismatches, 0) << mismatches << " of " << total
                             << " values differ between golden (fp64) and platinum (MPFR-256). "
                             << "Last mismatch at x=" << worst_x << ": golden=" << worst_golden
                             << " platinum=" << worst_platinum;
}

}  // namespace ttnn::test
