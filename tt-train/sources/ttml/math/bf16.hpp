// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>

#include <array>
#include <cstdint>
#include <ostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xeval.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xnoalias.hpp>

namespace ttml::math {

class bfloat16 {
private:
    uint16_t m_raw_value = 0;

public:
    bfloat16() = default;

    constexpr inline explicit bfloat16(int v) noexcept {
        m_raw_value = float_to_bfloat16(static_cast<float>(v));
    }

    constexpr inline explicit bfloat16(float f) noexcept {
        m_raw_value = float_to_bfloat16(f);
    }

    constexpr inline explicit bfloat16(double d) noexcept {
        m_raw_value = float_to_bfloat16(static_cast<float>(d));
    }

    constexpr inline operator float() const noexcept {
        return bfloat16_to_float(m_raw_value);
    }

    constexpr inline operator double() const noexcept {
        return static_cast<double>(bfloat16_to_float(m_raw_value));
    }

    constexpr inline bfloat16 operator+(const bfloat16 &rhs) const noexcept {
        float lhs_f = static_cast<float>(*this);
        float rhs_f = static_cast<float>(rhs);
        return bfloat16(lhs_f + rhs_f);
    }

    constexpr inline bfloat16 operator-(const bfloat16 &rhs) const noexcept {
        float lhs_f = static_cast<float>(*this);
        float rhs_f = static_cast<float>(rhs);
        return bfloat16(lhs_f - rhs_f);
    }

    constexpr inline bfloat16 operator*(const bfloat16 &rhs) const noexcept {
        float lhs_f = static_cast<float>(*this);
        float rhs_f = static_cast<float>(rhs);
        return bfloat16(lhs_f * rhs_f);
    }

    constexpr inline bfloat16 operator/(const bfloat16 &rhs) const noexcept {
        float lhs_f = static_cast<float>(*this);
        float rhs_f = static_cast<float>(rhs);
        return bfloat16(lhs_f / rhs_f);
    }

    constexpr inline bfloat16 &operator+=(const bfloat16 &rhs) noexcept {
        *this = *this + rhs;
        return *this;
    }
    constexpr inline bfloat16 &operator-=(const bfloat16 &rhs) noexcept {
        *this = *this - rhs;
        return *this;
    }
    constexpr inline bfloat16 &operator*=(const bfloat16 &rhs) noexcept {
        *this = *this * rhs;
        return *this;
    }
    constexpr inline bfloat16 &operator/=(const bfloat16 &rhs) noexcept {
        *this = *this / rhs;
        return *this;
    }

    constexpr inline bool operator==(const bfloat16 &rhs) const noexcept {
        return static_cast<float>(*this) == static_cast<float>(rhs);
    }
    constexpr inline bool operator!=(const bfloat16 &rhs) const noexcept {
        return !(*this == rhs);
    }
    constexpr inline bool operator<(const bfloat16 &rhs) const noexcept {
        return static_cast<float>(*this) < static_cast<float>(rhs);
    }
    constexpr inline bool operator>(const bfloat16 &rhs) const noexcept {
        return rhs < *this;
    }
    constexpr inline bool operator<=(const bfloat16 &rhs) const noexcept {
        return !(*this > rhs);
    }
    constexpr inline bool operator>=(const bfloat16 &rhs) const noexcept {
        return !(*this < rhs);
    }
    constexpr inline uint16_t get_raw_value() const noexcept {
        return m_raw_value;
    }
    constexpr bfloat16 operator-() const noexcept {
        return bfloat16(-static_cast<float>(*this));
    }

private:
    constexpr static uint16_t float_to_bfloat16(float f) noexcept {
        std::array<uint16_t, 2> raw_arr = std::bit_cast<std::array<uint16_t, 2>>(f);
        uint16_t raw_res = 0;

        switch (std::fpclassify(f)) {
            case FP_SUBNORMAL:
            case FP_ZERO:
                raw_res = raw_arr[1];
                raw_res &= 0x8000;
                break;
            case FP_INFINITE: raw_res = raw_arr[1]; break;
            case FP_NAN:
                raw_res = raw_arr[1];
                raw_res |= 1 << 6;
                break;
            case FP_NORMAL:
                const uint32_t rounding_bias = 0x00007FFF + (raw_arr[1] & 0x1);
                const uint32_t int_raw = std::bit_cast<uint32_t>(f) + rounding_bias;
                raw_arr = std::bit_cast<std::array<uint16_t, 2>>(int_raw);
                raw_res = raw_arr[1];
                break;
        }
        return raw_res;
    }

    constexpr static float bfloat16_to_float(uint16_t v) noexcept {
        std::array<uint16_t, 2> raw_arr = {{0, v}};
        return bit_cast<float>(raw_arr);
    }
};

// ----------------------------------------------------------------------
// Non-member overloads for arithmetic operators with arithmetic types
// These allow mixing bfloat16 with types like double, float, int, etc.
// ----------------------------------------------------------------------
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
constexpr bfloat16 operator+(const bfloat16 &lhs, T rhs) noexcept {
    return bfloat16(static_cast<float>(lhs) + static_cast<float>(rhs));
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
constexpr bfloat16 operator+(T lhs, const bfloat16 &rhs) noexcept {
    return bfloat16(static_cast<float>(lhs) + static_cast<float>(rhs));
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
constexpr bfloat16 operator-(const bfloat16 &lhs, T rhs) noexcept {
    return bfloat16(static_cast<float>(lhs) - static_cast<float>(rhs));
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
constexpr bfloat16 operator-(T lhs, const bfloat16 &rhs) noexcept {
    return bfloat16(static_cast<float>(lhs) - static_cast<float>(rhs));
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
constexpr bfloat16 operator*(const bfloat16 &lhs, T rhs) noexcept {
    return bfloat16(static_cast<float>(lhs) * static_cast<float>(rhs));
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
constexpr bfloat16 operator*(T lhs, const bfloat16 &rhs) noexcept {
    return bfloat16(static_cast<float>(lhs) * static_cast<float>(rhs));
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
constexpr bfloat16 operator/(const bfloat16 &lhs, T rhs) noexcept {
    return bfloat16(static_cast<float>(lhs) / static_cast<float>(rhs));
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
constexpr bfloat16 operator/(T lhs, const bfloat16 &rhs) noexcept {
    return bfloat16(static_cast<float>(lhs) / static_cast<float>(rhs));
}
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
constexpr bool operator==(const bfloat16 &lhs, T rhs) noexcept {
    return static_cast<float>(lhs) == static_cast<float>(rhs);
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
constexpr bool operator==(T lhs, const bfloat16 &rhs) noexcept {
    return static_cast<float>(lhs) == static_cast<float>(rhs);
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
constexpr bool operator!=(const bfloat16 &lhs, T rhs) noexcept {
    return !(lhs == rhs);
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
constexpr bool operator!=(T lhs, const bfloat16 &rhs) noexcept {
    return !(lhs == rhs);
}
}  // namespace ttml::math

// ============================================================
// STL integration

namespace ttml {
namespace math {

constexpr bool isnan(bfloat16 x) noexcept {
    // Convert to float and then use std::isnan
    return std::isnan(static_cast<float>(x));
}

inline std::ostream &operator<<(std::ostream &os, const bfloat16 &bf) {
    os << static_cast<float>(bf);
    return os;
}
}  // namespace math
}  // namespace ttml

namespace std {
template <>
class numeric_limits<ttml::math::bfloat16> {
public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr int radix = 2;
    // bfloat16 has 7 explicit fraction bits plus 1 implicit bit.
    static constexpr int digits = 8;
    static constexpr int digits10 = 2;
    static constexpr int max_digits10 = 4;

    static ttml::math::bfloat16 min() noexcept {
        // Smallest positive normalized value in bfloat16.
        // (Exponent = 1, Fraction = 0 → 2^(1-127))
        return ttml::math::bfloat16(1.17549435e-38f);
    }

    static ttml::math::bfloat16 max() noexcept {
        constexpr uint16_t raw = 0x7F7F;
        constexpr uint32_t bits = static_cast<uint32_t>(raw) << 16;
        constexpr float f = std::bit_cast<float>(bits);
        return ttml::math::bfloat16(f);
    }

    // The lowest (most negative) finite bfloat16 value is represented by:
    // sign = 1, exponent = 0xFE, fraction = 0x7F.
    // That is: 1 1111110 1111111 in binary, which as a 16-bit integer is 0xFF7F.
    static ttml::math::bfloat16 lowest() noexcept {
        constexpr uint16_t raw = 0xFF7F;
        constexpr uint32_t bits = static_cast<uint32_t>(raw) << 16;
        constexpr float f = std::bit_cast<float>(bits);
        return ttml::math::bfloat16(f);
    }
    static ttml::math::bfloat16 epsilon() noexcept {
        // The difference between 1.0 and the next representable value.
        // For bfloat16 this is 2^-7 = 0.0078125.
        return ttml::math::bfloat16(0.0078125f);
    }
    static ttml::math::bfloat16 round_error() noexcept {
        return ttml::math::bfloat16(0.5f);
    }

    static constexpr int min_exponent = -126;
    static constexpr int max_exponent = 127;
    static constexpr int min_exponent10 = -37;
    static constexpr int max_exponent10 = 38;

    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = false;
    static constexpr std::float_denorm_style has_denorm = std::denorm_present;
    static constexpr bool is_iec559 = true;
    static constexpr bool is_bounded = true;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;
    static constexpr std::float_round_style round_style = std::round_to_nearest;
};

template <>
struct hash<ttml::math::bfloat16> {
    size_t operator()(const ttml::math::bfloat16 &bf) const noexcept {
        return std::hash<uint16_t>()(bf.get_raw_value());
    }
};

template <>
struct common_type<ttml::math::bfloat16, float> {
    using type = float;
};
template <>
struct common_type<float, ttml::math::bfloat16> {
    using type = float;
};
template <>
struct common_type<ttml::math::bfloat16, double> {
    using type = double;
};
template <>
struct common_type<double, ttml::math::bfloat16> {
    using type = double;
};

}  // namespace std

namespace xt {
template <>
struct xscalar<ttml::math::bfloat16> : std::true_type {};
}  // namespace xt
