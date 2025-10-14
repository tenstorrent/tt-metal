// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <string>
#include <string_view>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define TTML_X86_ARCH 1
#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>  // Provides __get_cpuid() and __get_cpuid_count()
#elif defined(_MSC_VER)
#include <intrin.h>  // Provides __cpuid() and __cpuidex()
#endif
#else
#define TTML_X86_ARCH 0
#endif

namespace ttml::core {

// ============================================================================
// CPU Feature Detection (Modernized C++20)
// ============================================================================

class CpuFeatures {
public:
    struct Features {
        bool sse{false};
        bool sse2{false};
        bool sse3{false};
        bool ssse3{false};
        bool sse4_1{false};
        bool sse4_2{false};
        bool aes{false};
        bool avx{false};
        bool avx2{false};
        bool avx512f{false};
        bool avx512_vaes{false};  // Vector AES for AVX-512
        bool fma{false};
        bool bmi1{false};
        bool bmi2{false};

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
        using namespace std::literals;
        const auto& f = get();

        std::string result = "CPU Features:"s;

        constexpr std::array features = {
            std::pair{"SSE"sv, &Features::sse},
            std::pair{"SSE2"sv, &Features::sse2},
            std::pair{"SSE3"sv, &Features::sse3},
            std::pair{"SSSE3"sv, &Features::ssse3},
            std::pair{"SSE4.1"sv, &Features::sse4_1},
            std::pair{"SSE4.2"sv, &Features::sse4_2},
            std::pair{"AES-NI"sv, &Features::aes},
            std::pair{"AVX"sv, &Features::avx},
            std::pair{"AVX2"sv, &Features::avx2},
            std::pair{"AVX-512F"sv, &Features::avx512f},
            std::pair{"AVX-512-VAES"sv, &Features::avx512_vaes},
            std::pair{"FMA"sv, &Features::fma},
            std::pair{"BMI1"sv, &Features::bmi1},
            std::pair{"BMI2"sv, &Features::bmi2}};

        for (const auto& [name, ptr] : features) {
            if (f.*ptr) {
                result += " ";
                result += name;
            }
        }

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
    // CPUID bit positions (CPUID leaf 1, EDX)
    static constexpr uint32_t CPUID_1_EDX_SSE = 1 << 25;
    static constexpr uint32_t CPUID_1_EDX_SSE2 = 1 << 26;

    // CPUID bit positions (CPUID leaf 1, ECX)
    static constexpr uint32_t CPUID_1_ECX_SSE3 = 1 << 0;
    static constexpr uint32_t CPUID_1_ECX_SSSE3 = 1 << 9;
    static constexpr uint32_t CPUID_1_ECX_FMA = 1 << 12;
    static constexpr uint32_t CPUID_1_ECX_SSE4_1 = 1 << 19;
    static constexpr uint32_t CPUID_1_ECX_SSE4_2 = 1 << 20;
    static constexpr uint32_t CPUID_1_ECX_AES = 1 << 25;
    static constexpr uint32_t CPUID_1_ECX_AVX = 1 << 28;

    // CPUID bit positions (CPUID leaf 7, EBX)
    static constexpr uint32_t CPUID_7_EBX_BMI1 = 1 << 3;
    static constexpr uint32_t CPUID_7_EBX_AVX2 = 1 << 5;
    static constexpr uint32_t CPUID_7_EBX_BMI2 = 1 << 8;
    static constexpr uint32_t CPUID_7_EBX_AVX512F = 1 << 16;

    // CPUID bit positions (CPUID leaf 7, ECX)
    static constexpr uint32_t CPUID_7_ECX_AVX512_VAES = 1 << 9;

    // Helper to check if a bit is set
    static constexpr bool has_bit(uint32_t value, uint32_t mask) noexcept {
        return (value & mask) != 0;
    }

    static Features detect() noexcept {
        Features features;

#if TTML_X86_ARCH
        uint32_t eax, ebx, ecx, edx;

// Get max supported CPUID level
#if defined(__GNUC__) || defined(__clang__)
        __get_cpuid(0, &eax, &ebx, &ecx, &edx);
        const uint32_t max_level = eax;

        // CPUID leaf 1: Feature flags
        if (max_level >= 1 && __get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
            features.sse = has_bit(edx, CPUID_1_EDX_SSE);
            features.sse2 = has_bit(edx, CPUID_1_EDX_SSE2);
            features.sse3 = has_bit(ecx, CPUID_1_ECX_SSE3);
            features.ssse3 = has_bit(ecx, CPUID_1_ECX_SSSE3);
            features.sse4_1 = has_bit(ecx, CPUID_1_ECX_SSE4_1);
            features.sse4_2 = has_bit(ecx, CPUID_1_ECX_SSE4_2);
            features.aes = has_bit(ecx, CPUID_1_ECX_AES);
            features.avx = has_bit(ecx, CPUID_1_ECX_AVX);
            features.fma = has_bit(ecx, CPUID_1_ECX_FMA);
        }

        // CPUID leaf 7: Extended features
        if (max_level >= 7 && __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
            features.bmi1 = has_bit(ebx, CPUID_7_EBX_BMI1);
            features.avx2 = has_bit(ebx, CPUID_7_EBX_AVX2);
            features.bmi2 = has_bit(ebx, CPUID_7_EBX_BMI2);
            features.avx512f = has_bit(ebx, CPUID_7_EBX_AVX512F);
            features.avx512_vaes = has_bit(ecx, CPUID_7_ECX_AVX512_VAES);
        }

#elif defined(_MSC_VER)
        std::array<int, 4> cpu_info;

        // Get max CPUID level
        __cpuid(cpu_info.data(), 0);
        const int max_level = cpu_info[0];

        // CPUID leaf 1
        if (max_level >= 1) {
            __cpuid(cpu_info.data(), 1);
            features.sse = has_bit(cpu_info[3], CPUID_1_EDX_SSE);
            features.sse2 = has_bit(cpu_info[3], CPUID_1_EDX_SSE2);
            features.sse3 = has_bit(cpu_info[2], CPUID_1_ECX_SSE3);
            features.ssse3 = has_bit(cpu_info[2], CPUID_1_ECX_SSSE3);
            features.sse4_1 = has_bit(cpu_info[2], CPUID_1_ECX_SSE4_1);
            features.sse4_2 = has_bit(cpu_info[2], CPUID_1_ECX_SSE4_2);
            features.aes = has_bit(cpu_info[2], CPUID_1_ECX_AES);
            features.avx = has_bit(cpu_info[2], CPUID_1_ECX_AVX);
            features.fma = has_bit(cpu_info[2], CPUID_1_ECX_FMA);
        }

        // CPUID leaf 7
        if (max_level >= 7) {
            __cpuidex(cpu_info.data(), 7, 0);
            features.bmi1 = has_bit(cpu_info[1], CPUID_7_EBX_BMI1);
            features.avx2 = has_bit(cpu_info[1], CPUID_7_EBX_AVX2);
            features.bmi2 = has_bit(cpu_info[1], CPUID_7_EBX_BMI2);
            features.avx512f = has_bit(cpu_info[1], CPUID_7_EBX_AVX512F);
            features.avx512_vaes = has_bit(cpu_info[2], CPUID_7_ECX_AVX512_VAES);
        }
#endif

#endif  // TTML_X86_ARCH

        return features;
    }
};

}  // namespace ttml::core
