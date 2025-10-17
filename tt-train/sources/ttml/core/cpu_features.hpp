// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <string>

#if defined(_MSC_VER)
#include <intrin.h>  // Provides __cpuid() and __cpuidex()
#endif

namespace ttml::core {

// ============================================================================
// CPU Feature Detection (Modern C++20)
// ============================================================================

class CpuFeatures {
public:
    struct Features {
        bool sse4_2{false};
        bool aes{false};
        bool avx2{false};

        constexpr auto operator<=>(const Features&) const = default;
    };

    // Get CPU features (cached, thread-safe)
    static const Features& get() {
        static Features features = detect();
        return features;
    }

    // Check if we can use AVX2-based RNG
    static bool has_avx2_support() {
        const auto& f = get();
        return f.avx2 && f.aes && f.sse4_2;
    }

    // Check if we can use SSE-based RNG
    static bool has_sse_support() {
        const auto& f = get();
        return f.sse4_2 && f.aes;
    }

    // Get a human-readable feature string
    static std::string to_string() {
        const auto& f = get();
        std::string result = "CPU Features:";

        if (f.sse4_2)
            result += " SSE4.2";
        if (f.aes)
            result += " AES-NI";
        if (f.avx2)
            result += " AVX2";

        return result;
    }

    // Get recommended RNG implementation
    enum class RngImpl {
        MT19937,  // Scalar, no SIMD
        SSE,      // 128-bit SIMD (4x float)
        AVX2      // 256-bit SIMD (8x float)
    };

    static RngImpl get_recommended_rng() {
        const auto& f = get();

        // Check for AVX2
        if (f.avx2 && f.aes && f.sse4_2)
            return RngImpl::AVX2;

        // Check for SSE
        if (f.sse4_2 && f.aes)
            return RngImpl::SSE;

        // Fallback to scalar
        return RngImpl::MT19937;
    }

    static const char* rng_impl_name(RngImpl impl) {
        switch (impl) {
            case RngImpl::MT19937: return "MT19937 (scalar)";
            case RngImpl::SSE: return "AES-NI + SSE (4x float)";
            case RngImpl::AVX2: return "AES-NI + AVX2 (8x float)";
            default: return "Unknown";
        }
    }

private:
    static Features detect() noexcept {
        Features features;

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#if defined(__GNUC__) || defined(__clang__)
        // Modern approach using GCC/Clang built-ins (available since GCC 4.8 / Clang 3.8)
        __builtin_cpu_init();

        features.sse4_2 = __builtin_cpu_supports("sse4.2");
        features.aes = __builtin_cpu_supports("aes");
        features.avx2 = __builtin_cpu_supports("avx2");

#elif defined(_MSC_VER)
        // MSVC: use CPUID intrinsics
        std::array<int, 4> cpu_info;

        // Get max CPUID level
        __cpuid(cpu_info.data(), 0);
        const int max_level = cpu_info[0];

        // CPUID leaf 1: SSE4.2 and AES
        if (max_level >= 1) {
            __cpuid(cpu_info.data(), 1);
            features.sse4_2 = (cpu_info[2] & (1 << 20)) != 0;
            features.aes = (cpu_info[2] & (1 << 25)) != 0;
        }

        // CPUID leaf 7: AVX2
        if (max_level >= 7) {
            __cpuidex(cpu_info.data(), 7, 0);
            features.avx2 = (cpu_info[1] & (1 << 5)) != 0;
        }
#endif

#endif  // x86/x64 architecture

        return features;
    }
};

}  // namespace ttml::core
