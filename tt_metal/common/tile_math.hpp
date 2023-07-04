#pragma once

#include "common/assert.hpp"

#include <cstdint>

namespace tt{
inline std::uint32_t div_up(std::uint32_t a, std::uint32_t b) {
    TT_ASSERT(b > 0);
    return static_cast<std::uint32_t>((a+b-1)/b);
}

inline std::uint32_t round_up(std::uint32_t a, std::uint32_t b) {
    return b*div_up(a, b);
}

}

inline std::uint32_t rounddown(std::uint32_t a, std::uint32_t b) {
    return  a / b * b;
}
