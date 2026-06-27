// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2026 Yugansh Tyagi
//
// SPDX-License-Identifier: Apache-2.0

/**
 * tanh fp32 ULP Precision Test — Full Range Sweep
 *
 * Tests ttnn::tanh fp32 accuracy across the full representable fp32 range via
 * a systematic stratified sweep: all 254 normal exponents × dense mantissa
 * samples, plus critical near-zero and saturation boundary regions.
 *
 * Addresses the gap identified in PR #48299 review: the existing test checks
 * only 1024 linearly-spaced values from -100 to +100, which cannot detect
 * worst-case ULP error across the full fp32 exponent range.
 *
 * Hardware model: DAZ+FTZ (denormals treated as zero, per Tensix spec).
 * NaN and Inf are excluded (per host-side filtering policy).
 *
 * Run: ./build_Debug/test/ttnn/unit_tests_ttnn --gtest_filter="*TanhFp32Ulp*"
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>
#include <iomanip>
#include <sstream>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::test {

namespace fp32_ulp_tanh {

// ---------------------------------------------------------------------------
// fp32 bit manipulation helpers
// ---------------------------------------------------------------------------

inline uint32_t f32_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

inline float bits_f32(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline bool is_nan(uint32_t u)     { return ((u >> 23) & 0xFF) == 0xFF && (u & 0x7FFFFF) != 0; }
inline bool is_inf(uint32_t u)     { return ((u >> 23) & 0xFF) == 0xFF && (u & 0x7FFFFF) == 0; }
inline bool is_denorm(uint32_t u)  { return ((u >> 23) & 0xFF) == 0 && (u & 0x7FFFFF) != 0; }

// DAZ: flush denormals to zero (Tensix hardware behaviour)
inline float daz(float f) {
    uint32_t u = f32_bits(f);
    if (is_denorm(u)) return 0.0f;
    return f;
}

// ULP distance between two normal (non-NaN, non-Inf) fp32 values under DAZ.
// Interprets fp32 bit patterns as signed-magnitude integers.
inline int64_t ulp_distance(float a, float b) {
    a = daz(a);
    b = daz(b);
    int64_t ai = static_cast<int64_t>(f32_bits(a));
    int64_t bi = static_cast<int64_t>(f32_bits(b));
    // Convert sign-magnitude to two's complement ordering
    if (ai < 0) ai = (1LL << 32) - ai;
    if (bi < 0) bi = (1LL << 32) - bi;
    return std::abs(ai - bi);
}

// ---------------------------------------------------------------------------
// Build a stratified set of fp32 test inputs covering the full range
//
// Strategy:
//   1. Dense near-zero: ±{2^k * m | k=-149..0, m=1..8} — polynomial accuracy
//   2. Normal exponents: for each of 254 biased exponents 1..254, 512 evenly
//      spaced mantissa bits.  Covers every decade of the fp32 range.
//   3. Saturation boundary: ±{2^k | k=4..7} with fine mantissa sweep — where
//      tanh transitions from rational to ±1.
//   4. Sign symmetry: all inputs duplicated with negation.
// ---------------------------------------------------------------------------
static std::vector<float> build_test_inputs() {
    std::vector<float> vals;
    vals.reserve(300000);

    // 1. Dense near-zero: all subnormals (treated as 0 by DAZ) and small normals
    for (int e = 1; e <= 20; ++e) {           // biased exponent 1..20 => 2^(e-127)
        for (int m = 0; m < 512; ++m) {
            uint32_t u = ((uint32_t)e << 23) | ((uint32_t)m << 14);
            float f = bits_f32(u);
            vals.push_back(f);
            vals.push_back(-f);
        }
    }

    // 2. Full normal exponent sweep
    for (int e = 1; e <= 254; ++e) {
        for (int m = 0; m < 512; ++m) {
            uint32_t u = ((uint32_t)e << 23) | ((uint32_t)m << 14);
            float f = bits_f32(u);
            vals.push_back(f);
            vals.push_back(-f);
        }
    }

    // 3. Fine sweep around saturation boundary |x| in [4, 128]
    //    (tanh saturates to ±1 in fp32 around |x| ~ 9-10)
    for (int e = 129; e <= 133; ++e) {        // biased exp 129..133 => |x| in [4,32]
        for (int m = 0; m < 8192; ++m) {
            uint32_t u = ((uint32_t)e << 23) | ((uint32_t)m << 10);
            float f = bits_f32(u);
            vals.push_back(f);
            vals.push_back(-f);
        }
    }

    // Dedup and remove NaN/Inf (should be none, but defensive)
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
    vals.erase(
        std::remove_if(vals.begin(), vals.end(), [](float f) {
            uint32_t u = f32_bits(f);
            return is_nan(u) || is_inf(u);
        }),
        vals.end());

    return vals;
}

struct UlpStats {
    int64_t max_ulp = 0;
    float   worst_input = 0.0f;
    float   worst_device = 0.0f;
    float   worst_ref = 0.0f;
    int64_t over3 = 0;   // inputs exceeding 3 ULP
    int64_t over2 = 0;
    int64_t total = 0;
};

}  // namespace fp32_ulp_tanh

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------
class TanhFp32UlpTest : public TTNNFixtureWithDevice {};

// ---------------------------------------------------------------------------
// Main test: stratified full-range ULP sweep
// ---------------------------------------------------------------------------
TEST_F(TanhFp32UlpTest, FullRangeSweep) {
    using namespace fp32_ulp_tanh;

    auto inputs = build_test_inputs();
    fp32_ulp_tanh::UlpStats stats;
    stats.total = static_cast<int64_t>(inputs.size());

    // Process in chunks to avoid allocating a single massive tensor
    constexpr int CHUNK = 8192;
    for (size_t offset = 0; offset < inputs.size(); offset += CHUNK) {
        size_t end = std::min(offset + CHUNK, inputs.size());
        size_t n   = end - offset;

        // Pad to multiple of 32 (tile requirement)
        size_t padded = ((n + 31) / 32) * 32;
        std::vector<float> chunk(padded, 0.0f);
        std::copy(inputs.begin() + offset, inputs.begin() + end, chunk.begin());

        // Build device tensor (fp32)
        auto shape = ttnn::Shape{1, 1, 1, static_cast<uint32_t>(padded)};
        auto host_tensor = ttnn::Tensor(
            ttnn::OwnedStorage{tt::tt_metal::owned_buffer::create(chunk)},
            shape,
            tt::tt_metal::DataType::FLOAT32,
            tt::tt_metal::Layout::ROW_MAJOR);
        auto device_input  = ttnn::to_device(host_tensor, &device_, ttnn::DRAM_MEMORY_CONFIG);
        auto device_output = ttnn::tanh(device_input);
        auto host_output   = ttnn::from_device(device_output);
        auto* out_data     = host_output.get_data_ptr<float>();

        for (size_t i = 0; i < n; ++i) {
            float xi  = daz(inputs[offset + i]);
            float got = daz(out_data[i]);
            double ref = std::tanh(static_cast<double>(xi));
            float ref_f = static_cast<float>(ref);

            int64_t ulp = ulp_distance(got, ref_f);
            if (ulp > stats.max_ulp) {
                stats.max_ulp      = ulp;
                stats.worst_input  = xi;
                stats.worst_device = got;
                stats.worst_ref    = ref_f;
            }
            if (ulp > 3) stats.over3++;
            if (ulp > 2) stats.over2++;
        }
    }

    std::ostringstream oss;
    oss << "\n==========================================================\n"
        << "tanh fp32 ULP sweep — " << stats.total << " inputs\n"
        << "==========================================================\n"
        << "  Max ULP error  : " << stats.max_ulp << "\n"
        << "  Worst input    : " << std::scientific << std::setprecision(8) << stats.worst_input << "\n"
        << "  Device output  : " << stats.worst_device << "\n"
        << "  Reference      : " << stats.worst_ref << "\n"
        << "  > 2 ULP        : " << stats.over2 << " / " << stats.total << "\n"
        << "  > 3 ULP        : " << stats.over3 << " / " << stats.total << "\n"
        << "==========================================================\n";
    log_info(tt::LogTest, "{}", oss.str());

    EXPECT_EQ(stats.over3, 0)
        << stats.over3 << " input(s) exceed 3 ULP. Worst: input=" << stats.worst_input
        << " device=" << stats.worst_device << " ref=" << stats.worst_ref
        << " ulp=" << stats.max_ulp;
    EXPECT_LE(stats.max_ulp, 3)
        << "Max ULP " << stats.max_ulp << " exceeds target of 3";
}

// ---------------------------------------------------------------------------
// Spot-check: exact values that are known to be tricky
// ---------------------------------------------------------------------------
TEST_F(TanhFp32UlpTest, TrickyValues) {
    using namespace fp32_ulp_tanh;

    // Values where range-reduction boundary effects or polynomial
    // end-point behaviour are most likely to appear.
    std::vector<float> tricky = {
        // Near zero — tanh(x) must equal x to <1 ULP
        1e-38f, 1e-10f, 1e-7f, 1e-4f,
        -1e-38f, -1e-10f, -1e-7f, -1e-4f,
        // ln(2)/2 ~ 0.3466 — range-reduction boundary
        0.3465735903f, -0.3465735903f,
        // ln(2) ~ 0.6931 — next boundary
        0.6931471806f, -0.6931471806f,
        // Saturation onset ~ 9.01
        9.0f, 9.01f, 9.5f, 10.0f,
        -9.0f, -9.01f, -9.5f, -10.0f,
        // Deep saturation
        20.0f, 50.0f, 88.0f, 100.0f,
        -20.0f, -50.0f, -88.0f, -100.0f,
        // ±0, ±1
        0.0f, -0.0f, 1.0f, -1.0f,
    };

    size_t padded = ((tricky.size() + 31) / 32) * 32;
    std::vector<float> buf(padded, 0.0f);
    std::copy(tricky.begin(), tricky.end(), buf.begin());

    auto shape = ttnn::Shape{1, 1, 1, static_cast<uint32_t>(padded)};
    auto host_tensor = ttnn::Tensor(
        ttnn::OwnedStorage{tt::tt_metal::owned_buffer::create(buf)},
        shape,
        tt::tt_metal::DataType::FLOAT32,
        tt::tt_metal::Layout::ROW_MAJOR);
    auto device_input  = ttnn::to_device(host_tensor, &device_, ttnn::DRAM_MEMORY_CONFIG);
    auto device_output = ttnn::tanh(device_input);
    auto host_output   = ttnn::from_device(device_output);
    auto* out_data     = host_output.get_data_ptr<float>();

    std::ostringstream oss;
    oss << "\n==========================================================\n"
        << "tanh fp32 tricky-value spot check\n"
        << "==========================================================\n"
        << std::left << std::setw(16) << "input"
        << std::setw(16) << "device"
        << std::setw(16) << "reference"
        << "ULP\n";

    int64_t max_ulp = 0;
    for (size_t i = 0; i < tricky.size(); ++i) {
        float xi  = daz(tricky[i]);
        float got = daz(out_data[i]);
        double ref = std::tanh(static_cast<double>(xi));
        float ref_f = static_cast<float>(ref);
        int64_t ulp = ulp_distance(got, ref_f);
        max_ulp = std::max(max_ulp, ulp);

        oss << std::scientific << std::setprecision(6)
            << std::setw(16) << xi
            << std::setw(16) << got
            << std::setw(16) << ref_f
            << ulp << "\n";
    }
    oss << "==========================================================\n"
        << "Max ULP: " << max_ulp << "\n";
    log_info(tt::LogTest, "{}", oss.str());

    EXPECT_LE(max_ulp, 3) << "Tricky-value max ULP " << max_ulp << " exceeds 3";
}

}  // namespace ttnn::test
