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
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat2.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt_stl/span.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/rtoptions.hpp"

using namespace tt;

namespace {

// Scoped override of the host BFP tilizer RunTimeOptions knobs
// (TT_METAL_BFP_HOST_TILIZER_THREADS / TT_METAL_BFP_HOST_TILIZER_DISABLE_SIMD).
// The env vars are parsed once into RunTimeOptions at context creation, so
// tests toggle the cached values via setters and restore them on scope exit.
// std::nullopt leaves the corresponding setting unchanged.
class ScopedTilizerConfig {
public:
    ScopedTilizerConfig(std::optional<int> threads, std::optional<bool> disable_simd) :
        rtoptions_(tt::tt_metal::MetalContext::instance().rtoptions()),
        prev_threads_(rtoptions_.get_bfp_host_tilizer_threads()),
        prev_disable_simd_(rtoptions_.get_bfp_host_tilizer_disable_simd()) {
        if (threads.has_value()) {
            rtoptions_.set_bfp_host_tilizer_threads(*threads);
        }
        if (disable_simd.has_value()) {
            rtoptions_.set_bfp_host_tilizer_disable_simd(*disable_simd);
        }
    }

    ~ScopedTilizerConfig() {
        rtoptions_.set_bfp_host_tilizer_threads(prev_threads_);
        rtoptions_.set_bfp_host_tilizer_disable_simd(prev_disable_simd_);
    }

    ScopedTilizerConfig(const ScopedTilizerConfig&) = delete;
    ScopedTilizerConfig& operator=(const ScopedTilizerConfig&) = delete;

private:
    tt::llrt::RunTimeOptions& rtoptions_;
    int prev_threads_;
    bool prev_disable_simd_;
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

// Pack the input under specific tilizer settings and return the resulting
// bytes. Caller picks which BFP packer via `pack_fn`. Templated on the input
// element type so we can drive both `Span<const float>` and
// `Span<const bfloat16>` paths with the same helper.
template <typename PackFn, typename T>
std::vector<uint32_t> pack_under_config(
    PackFn&& pack_fn, int num_threads, bool disable_simd, tt::stl::Span<const T> input) {
    ScopedTilizerConfig config(num_threads, disable_simd);
    return std::forward<PackFn>(pack_fn)(input);
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

    auto serial = pack_under_config(pack, 1, false, tt::stl::make_const_span(input));
    auto parallel = pack_under_config(pack, 4, false, tt::stl::make_const_span(input));

    EXPECT_TRUE(ExpectPackedEqual(serial, parallel, "serial", "parallel"));
}

TEST(HostBfpTilizerEquivalence, Bfp8b_SimdMatchesScalar_RowMajor) {
    constexpr uint32_t kNumTiles = 8;
    auto input = make_edge_case_inputs(kNumTiles);

    auto pack = [](tt::stl::Span<const float> in) {
        return pack_as_bfp8_tiles(in, /*row_major_input=*/true, /*is_exp_a=*/false);
    };

    auto simd = pack_under_config(pack, 1, false, tt::stl::make_const_span(input));
    auto scalar = pack_under_config(pack, 1, true, tt::stl::make_const_span(input));

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

    auto simd = pack_under_config(pack, 1, false, tt::stl::make_const_span(tiled));
    auto scalar = pack_under_config(pack, 1, true, tt::stl::make_const_span(tiled));

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
        ScopedTilizerConfig simd_off(/*threads=*/std::nullopt, /*disable_simd=*/true);
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

    auto simd = pack_under_config(pack, 1, false, tt::stl::make_const_span(bf16_input));
    auto scalar = pack_under_config(pack, 1, true, tt::stl::make_const_span(bf16_input));

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

    auto serial = pack_under_config(pack, 1, false, tt::stl::make_const_span(input));
    auto parallel = pack_under_config(pack, 4, false, tt::stl::make_const_span(input));

    EXPECT_TRUE(ExpectPackedEqual(serial, parallel, "serial", "parallel"));
}

TEST(HostBfpTilizerEquivalence, Bfp2b_SerialMatchesParallel) {
    constexpr uint32_t kNumTiles = 32;
    auto input = make_edge_case_inputs(kNumTiles);

    auto pack = [](tt::stl::Span<const float> in) {
        // Fully qualify to disambiguate from the `tt::tt_metal` overload that
        // becomes visible via `using namespace tt::tt_metal;` in other TUs
        // included in the same unity build.
        return ::pack_as_bfp2_tiles(in, /*row_major_input=*/true, /*is_exp_a=*/false);
    };

    auto serial = pack_under_config(pack, 1, false, tt::stl::make_const_span(input));
    auto parallel = pack_under_config(pack, 4, false, tt::stl::make_const_span(input));

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

    auto serial = pack_under_config(pack, 1, false, tt::stl::make_const_span(input));
    auto parallel = pack_under_config(pack, 4, false, tt::stl::make_const_span(input));

    EXPECT_TRUE(ExpectPackedEqual(serial, parallel, "serial", "parallel"));
}
