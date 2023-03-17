#pragma once

#include "common/assert.hpp"

#include <cstdint>

inline std::uint32_t divup(std::uint32_t a, std::uint32_t b) {
    TT_ASSERT(b > 0);
    return static_cast<std::uint32_t>((a+b-1)/b);
}

inline std::uint32_t roundup(std::uint32_t a, std::uint32_t b) {
    return b*divup(a, b);
}
