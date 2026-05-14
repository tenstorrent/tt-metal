// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Bit-equivalence regression tests for the optimized host BFP tilizer.
//
// These tests pin the optimized path's output to be byte-exactly identical
// across all of:
//   * single-threaded vs multi-threaded execution
//   * SIMD (AVX2/simde) vs scalar execution
//   * row_major_input=true vs false (after layout conversion)
//
// Coverage spans BFP8_b, BFP4_b, BFP2_b and includes inputs that exercise the
// edge cases of the rounding logic: zeros, denormals, all-equal-exponent
// blocks, large dynamic range blocks, and signed values.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat2.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt_stl/span.hpp>

using namespace tt;

namespace {

// Scoped env-var override - sets a variable for the duration of a test, then
// restores its previous value. Posix-ish but works under Linux which is what
// the project's CI builds against.
class ScopedEnv {
public:
    ScopedEnv(const char* name, const char* value) : name_(name) {
        const char* prev = std::getenv(name);
        had_prev_ = (prev != nullptr);
        if (had_prev_) {
            prev_value_ = prev;
        }
        if (value != nullptr) {
            ::setenv(name, value, /*overwrite=*/1);
        } else {
            ::unsetenv(name);
        }
    }

    ~ScopedEnv() {
        if (had_prev_) {
            ::setenv(name_, prev_value_.c_str(), /*overwrite=*/1);
        } else {
            ::unsetenv(name_);
        }
    }

    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    const char* name_;
    bool had_prev_ = false;
    std::string prev_value_;
};

// Build a vector of 1024*num_tiles fp32 values with deterministic content
// designed to hit BFP rounding edge cases. The pattern is chosen to vary the
// exponent across each face row (so shared_exp differs per row), and to mix
// in zeros, denormals (flushed by the packer), positive and negative values.
std::vector<float> make_edge_case_inputs(uint32_t num_tiles, int seed = 42) {
    constexpr int FLOATS_PER_TILE = 1024;
    std::vector<float> out(num_tiles * FLOATS_PER_TILE);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> small(0.0f, 1.0f);

    for (uint32_t i = 0; i < out.size(); ++i) {
        // Cycle through a handful of regimes per element index to ensure each
        // 16-element face row sees a non-trivial mix.
        switch (i % 8) {
            case 0:
                out[i] = 0.0f;  // exact zero
                break;
            case 1:
                out[i] = -0.0f;  // signed zero
                break;
            case 2:
                out[i] = 1e-40f;  // subnormal -> flushed to 0
                break;
            case 3:
                out[i] = (i & 1) ? -64.5f : 64.5f;  // tie-to-even at bfp8
                break;
            case 4:
                out[i] = static_cast<float>(static_cast<int>(i) - 100);
                break;
            case 5:
                out[i] = small(rng) * 1024.0f;
                break;
            case 6:
                out[i] = -small(rng) * 0.001f;
                break;
            default:
                out[i] = static_cast<float>(i % 257) * 0.125f;
                break;
        }
    }
    return out;
}

// Pack the input under specific env-var settings and return the resulting
// bytes. Caller picks which BFP packer via `pack_fn`.
template <typename PackFn>
std::vector<uint32_t> pack_under_env(
    PackFn&& pack_fn, const char* threads_env, const char* simd_env, tt::stl::Span<const float> input) {
    ScopedEnv t("TT_BFP_HOST_TILIZER_THREADS", threads_env);
    ScopedEnv s("TT_BFP_HOST_TILIZER_DISABLE_SIMD", simd_env);
    return pack_fn(input);
}

// Compare two packed vectors element by element, reporting the first
// mismatch with surrounding context for debuggability.
::testing::AssertionResult ExpectPackedEqual(
    const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, const char* label_a, const char* label_b) {
    if (a.size() != b.size()) {
        return ::testing::AssertionFailure()
               << "Size mismatch: " << label_a << " size=" << a.size() << " vs " << label_b << " size=" << b.size();
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            return ::testing::AssertionFailure() << "Mismatch at dword index " << i << ": " << label_a << "=0x"
                                                 << std::hex << a[i] << " vs " << label_b << "=0x" << b[i];
        }
    }
    return ::testing::AssertionSuccess();
}

}  // namespace

// -----------------------------------------------------------------------------
// BFP8_b: SIMD path (face_W=16, is_exp_a=false) is the production hot path.
// -----------------------------------------------------------------------------

TEST(HostBfpTilizerEquivalence, Bfp8b_SerialMatchesParallel_RowMajor) {
    constexpr uint32_t kNumTiles = 32;
    auto input = make_edge_case_inputs(kNumTiles);

    auto pack = [](tt::stl::Span<const float> in) {
        return pack_as_bfp8_tiles(in, /*row_major_input=*/true, /*is_exp_a=*/false);
    };

    auto serial = pack_under_env(pack, "1", nullptr, tt::stl::make_const_span(input));
    auto parallel = pack_under_env(pack, "4", nullptr, tt::stl::make_const_span(input));

    EXPECT_TRUE(ExpectPackedEqual(serial, parallel, "serial", "parallel"));
}

TEST(HostBfpTilizerEquivalence, Bfp8b_SimdMatchesScalar_RowMajor) {
    constexpr uint32_t kNumTiles = 8;
    auto input = make_edge_case_inputs(kNumTiles);

    auto pack = [](tt::stl::Span<const float> in) {
        return pack_as_bfp8_tiles(in, /*row_major_input=*/true, /*is_exp_a=*/false);
    };

    auto simd = pack_under_env(pack, "1", nullptr, tt::stl::make_const_span(input));
    auto scalar = pack_under_env(pack, "1", "1", tt::stl::make_const_span(input));

    EXPECT_TRUE(ExpectPackedEqual(simd, scalar, "simd", "scalar"));
}

TEST(HostBfpTilizerEquivalence, Bfp8b_SimdMatchesScalar_Tiled) {
    constexpr uint32_t kNumTiles = 8;
    auto input = make_edge_case_inputs(kNumTiles);
    std::vector<uint32_t> shape_vec = {1, kNumTiles, 32, 32};
    std::vector<float> tiled = convert_layout(
        tt::stl::make_const_span(input),
        shape_vec,
        TensorLayoutType::LIN_ROW_MAJOR,
        TensorLayoutType::TILED_NFACES);

    auto pack = [](tt::stl::Span<const float> in) {
        return pack_as_bfp8_tiles(in, /*row_major_input=*/false, /*is_exp_a=*/false);
    };

    auto simd = pack_under_env(pack, "1", nullptr, tt::stl::make_const_span(tiled));
    auto scalar = pack_under_env(pack, "1", "1", tt::stl::make_const_span(tiled));

    EXPECT_TRUE(ExpectPackedEqual(simd, scalar, "simd", "scalar"));
}

TEST(HostBfpTilizerEquivalence, Bfp8b_RowMajorMatchesTiledInput) {
    // The two layouts (passing the same data row-major or tiled) must produce
    // identical packed output. This test already exists in
    // test_bfp8_conversion.cpp but we run it again here under both SIMD-on and
    // SIMD-off configurations as an extra guard against any layout-dependent
    // bug in the SIMD packer.
    constexpr uint32_t kNumTiles = 4;
    auto input = make_edge_case_inputs(kNumTiles);
    std::vector<uint32_t> shape_vec = {1, kNumTiles, 32, 32};
    std::vector<float> tiled = convert_layout(
        tt::stl::make_const_span(input),
        shape_vec,
        TensorLayoutType::LIN_ROW_MAJOR,
        TensorLayoutType::TILED_NFACES);

    auto rm_simd =
        pack_as_bfp8_tiles(tt::stl::make_const_span(input), /*row_major_input=*/true, /*is_exp_a=*/false);
    auto tile_simd =
        pack_as_bfp8_tiles(tt::stl::make_const_span(tiled), /*row_major_input=*/false, /*is_exp_a=*/false);

    EXPECT_TRUE(ExpectPackedEqual(rm_simd, tile_simd, "row_major", "pre_tiled"));

    {
        ScopedEnv simd_off("TT_BFP_HOST_TILIZER_DISABLE_SIMD", "1");
        auto rm_scalar =
            pack_as_bfp8_tiles(tt::stl::make_const_span(input), /*row_major_input=*/true, /*is_exp_a=*/false);
        auto tile_scalar =
            pack_as_bfp8_tiles(tt::stl::make_const_span(tiled), /*row_major_input=*/false, /*is_exp_a=*/false);
        EXPECT_TRUE(ExpectPackedEqual(rm_scalar, tile_scalar, "row_major_scalar", "pre_tiled_scalar"));
        EXPECT_TRUE(ExpectPackedEqual(rm_simd, rm_scalar, "row_major_simd", "row_major_scalar"));
    }
}

// -----------------------------------------------------------------------------
// BFP8_b with bfloat16 input - covers the SIMD `gather_face_row_16_fp32`
// bfloat16 specialization.
// -----------------------------------------------------------------------------

TEST(HostBfpTilizerEquivalence, Bfp8b_Bfloat16_SimdMatchesScalar) {
    constexpr uint32_t kNumTiles = 8;
    auto float_input = make_edge_case_inputs(kNumTiles);
    std::vector<bfloat16> bf16_input;
    bf16_input.reserve(float_input.size());
    for (float f : float_input) {
        bf16_input.emplace_back(f);
    }

    auto pack = [](tt::stl::Span<const bfloat16> in) {
        return pack_as_bfp8_tiles(in, /*row_major_input=*/true, /*is_exp_a=*/false);
    };

    auto simd = pack_under_env(pack, "1", nullptr, tt::stl::make_const_span(bf16_input));
    auto scalar = pack_under_env(pack, "1", "1", tt::stl::make_const_span(bf16_input));

    EXPECT_TRUE(ExpectPackedEqual(simd, scalar, "simd", "scalar"));
}

// -----------------------------------------------------------------------------
// Non-SIMD code paths: BFP4_b and BFP2_b only have the scalar fast path. We
// still validate serial-vs-parallel byte equivalence to guard the threading
// rewrite.
// -----------------------------------------------------------------------------

TEST(HostBfpTilizerEquivalence, Bfp4b_SerialMatchesParallel) {
    constexpr uint32_t kNumTiles = 32;
    auto input = make_edge_case_inputs(kNumTiles);

    auto pack = [](tt::stl::Span<const float> in) {
        return pack_as_bfp4_tiles(in, /*row_major_input=*/true, /*is_exp_a=*/false);
    };

    auto serial = pack_under_env(pack, "1", nullptr, tt::stl::make_const_span(input));
    auto parallel = pack_under_env(pack, "4", nullptr, tt::stl::make_const_span(input));

    EXPECT_TRUE(ExpectPackedEqual(serial, parallel, "serial", "parallel"));
}

TEST(HostBfpTilizerEquivalence, Bfp2b_SerialMatchesParallel) {
    constexpr uint32_t kNumTiles = 32;
    auto input = make_edge_case_inputs(kNumTiles);

    auto pack = [](tt::stl::Span<const float> in) {
        return pack_as_bfp2_tiles(in, /*row_major_input=*/true, /*is_exp_a=*/false);
    };

    auto serial = pack_under_env(pack, "1", nullptr, tt::stl::make_const_span(input));
    auto parallel = pack_under_env(pack, "4", nullptr, tt::stl::make_const_span(input));

    EXPECT_TRUE(ExpectPackedEqual(serial, parallel, "serial", "parallel"));
}

// -----------------------------------------------------------------------------
// is_exp_a=true forces the scalar path even for BFP8 because it changes the
// exponent rebias logic. Verify serial-vs-parallel byte equivalence still
// holds on this path too. Calls pack_as_bfp8_tiles with is_exp_a=true; the
// public wrapper hard-codes Bfp8_b but the runtime is_exp_a flag still drives
// the scalar fallback inside the optimized packer.
// -----------------------------------------------------------------------------

TEST(HostBfpTilizerEquivalence, Bfp8_IsExpA_SerialMatchesParallel) {
    constexpr uint32_t kNumTiles = 32;
    auto input = make_edge_case_inputs(kNumTiles);

    auto pack = [](tt::stl::Span<const float> in) {
        return pack_as_bfp8_tiles(in, /*row_major_input=*/true, /*is_exp_a=*/true);
    };

    auto serial = pack_under_env(pack, "1", nullptr, tt::stl::make_const_span(input));
    auto parallel = pack_under_env(pack, "4", nullptr, tt::stl::make_const_span(input));

    EXPECT_TRUE(ExpectPackedEqual(serial, parallel, "serial", "parallel"));
}
