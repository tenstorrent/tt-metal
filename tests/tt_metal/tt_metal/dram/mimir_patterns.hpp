// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace tt::tt_metal::mimir_test {

enum class MimirPattern { AllZeros, AllOnes, Walking, Alt00FF, AltAA55, AddressDerived, Prng };

inline constexpr std::array<MimirPattern, 7> kAllMimirPatterns = {
    MimirPattern::AllZeros,
    MimirPattern::AllOnes,
    MimirPattern::Walking,
    MimirPattern::Alt00FF,
    MimirPattern::AltAA55,
    MimirPattern::AddressDerived,
    MimirPattern::Prng,
};
// Constant patterns can match unwritten memory or a misrouted read, so read legs use only these.
inline constexpr std::array<MimirPattern, 2> kReadLegPatterns = {MimirPattern::AddressDerived, MimirPattern::Prng};

inline std::string pattern_name(MimirPattern pattern) {
    switch (pattern) {
        case MimirPattern::AllZeros: return "AllZeros";
        case MimirPattern::AllOnes: return "AllOnes";
        case MimirPattern::Walking: return "Walking";
        case MimirPattern::Alt00FF: return "Alt00FF";
        case MimirPattern::AltAA55: return "AltAA55";
        case MimirPattern::AddressDerived: return "AddressDerived";
        case MimirPattern::Prng: return "Prng";
    }
    return "Unknown";
}

inline std::vector<uint32_t> make_pattern(
    MimirPattern pattern, size_t num_words, uint64_t base_address, uint32_t seed) {
    std::vector<uint32_t> words(num_words);
    std::mt19937 generator(seed ^ static_cast<uint32_t>(base_address));
    for (size_t i = 0; i < num_words; ++i) {
        const uint32_t bit = 1u << static_cast<uint32_t>((i / 2) % 32);
        switch (pattern) {
            case MimirPattern::AllZeros: words[i] = 0x00000000u; break;
            case MimirPattern::AllOnes: words[i] = 0xFFFFFFFFu; break;
            case MimirPattern::Walking: words[i] = (i % 2 == 0) ? bit : ~bit; break;
            case MimirPattern::Alt00FF: words[i] = 0x00FF00FFu; break;
            case MimirPattern::AltAA55: words[i] = 0xAA55AA55u; break;
            case MimirPattern::AddressDerived:
                words[i] = static_cast<uint32_t>(base_address + 4ull * i) ^ 0x5A5A0000u;
                break;
            case MimirPattern::Prng: words[i] = static_cast<uint32_t>(generator()); break;
        }
    }
    return words;
}

inline uint32_t mimir_test_seed() {
    static const uint32_t seed = [] {
        uint32_t value = static_cast<uint32_t>(::testing::UnitTest::GetInstance()->random_seed());
        log_info(tt::LogTest, "mimir_tests random seed: {}", value);
        return value;
    }();
    return seed;
}

inline constexpr uint32_t kGuardBytes = 64;
inline constexpr uint32_t kGuardWords = kGuardBytes / 4;

inline std::vector<uint32_t> make_guard(uint32_t tag, size_t num_words = kGuardWords) {
    TT_FATAL(tag <= 0xFFFu, "guard tag 0x{:x} exceeds the 12-bit tag field", tag);
    std::vector<uint32_t> words(num_words);
    for (size_t i = 0; i < num_words; ++i) {
        words[i] = 0x6A000000u | ((tag & 0xFFFu) << 12) | (static_cast<uint32_t>(i) & 0xFFFu);
    }
    return words;
}

inline std::vector<uint32_t> make_seed_pattern(uint32_t tag, size_t num_words) {
    TT_FATAL(num_words <= 65536, "seed pattern of {} words exceeds the 16-bit word index", num_words);
    std::vector<uint32_t> words(num_words);
    for (size_t i = 0; i < num_words; ++i) {
        words[i] = 0x5EED0000u ^ ((tag & 0xFFFFu) << 16) ^ (static_cast<uint32_t>(i) & 0xFFFFu);
    }
    return words;
}

inline ::testing::AssertionResult compare_words(
    std::string_view what,
    const std::vector<uint32_t>& expected,
    const std::vector<uint32_t>& actual,
    size_t window = 8) {
    if (expected.size() != actual.size()) {
        return ::testing::AssertionFailure() << fmt::format(
                   "{} size mismatch: expected {} words, actual {} words", what, expected.size(), actual.size());
    }
    const size_t common_size = std::min(expected.size(), actual.size());
    size_t mismatch_count = 0;
    size_t first_mismatch = 0;
    for (size_t i = 0; i < common_size; ++i) {
        if (expected[i] != actual[i]) {
            if (mismatch_count == 0) {
                first_mismatch = i;
            }
            ++mismatch_count;
        }
    }

    if (mismatch_count == 0) {
        return ::testing::AssertionSuccess();
    }

    const size_t window_start = first_mismatch > window / 2 ? first_mismatch - window / 2 : 0;
    const size_t window_end = std::min(common_size, window_start + window);
    std::string message = fmt::format(
        "{} mismatch: {} of {} words differ; first mismatch at index {}\n",
        what,
        mismatch_count,
        common_size,
        first_mismatch);
    for (size_t i = window_start; i < window_end; ++i) {
        message += fmt::format("  {}: expected 0x{:08x} actual 0x{:08x}\n", i, expected[i], actual[i]);
    }
    return ::testing::AssertionFailure() << message;
}

inline uint32_t xor_of(const std::vector<uint32_t>& words) {
    uint32_t signature = 0;
    for (const uint32_t word : words) {
        signature ^= word;
    }
    return signature;
}

}  // namespace tt::tt_metal::mimir_test
