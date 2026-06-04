#pragma once

#include <cstddef>
#include <type_traits>

namespace llk::util
{

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr size_t get_magnitude(const T number)
{
    if constexpr (std::is_unsigned_v<T>)
    {
        return static_cast<size_t>(number);
    }

    if (number >= 0)
    {
        return static_cast<size_t>(number);
    }

    // signed type, so expression can't overflow.
    return static_cast<size_t>(-(number + 1) + 1);
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr bool is_negative(const T number)
{
    if constexpr (std::is_unsigned_v<T>)
    {
        return false;
    }

    if (number >= 0)
    {
        return false;
    }

    return true;
}

constexpr size_t count_digits(size_t magnitude, size_t radix = 10)
{
    if (magnitude == 0)
    {
        return 1;
    }

    size_t digits = 0;
    while (magnitude > 0)
    {
        magnitude /= radix;
        ++digits;
    }
    return digits;
}

} // namespace llk::util
