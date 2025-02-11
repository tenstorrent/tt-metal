// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>

#include <array>
#include <cstdint>

namespace ttml::math {

class bfloat16 {
public:
    uint16_t m_raw_value = 0;

    bfloat16() = default;

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

}  // namespace ttml::math
