// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <limits>

namespace ll_api {

template <class Integer>
constexpr Integer round_to_power_of_2(Integer x) {
    static_assert(std::numeric_limits<Integer>::is_integer, "Parameter must be an integer.");
    assert(x >= Integer(0));

    if (std::numeric_limits<Integer>::digits != 0)  // fixed size
    {
        assert(std::numeric_limits<Integer>::radix == 2);

        Integer power_of_2(1);
        for (int i = 0; i < std::numeric_limits<Integer>::digits; i++) {
            if (power_of_2 >= x) {
                return power_of_2;
            }
            power_of_2 <<= 1;
        }

        assert(false);
    } else  // variable size
    {
        Integer power_of_2(1);
        while (power_of_2 < x) {
            power_of_2 <<= 1;
        }
        return power_of_2;
    }
}

template <class T, class U>
constexpr T round_up_to(T x, U multiple) {
    static_assert(std::numeric_limits<T>::is_integer, "T (first parameter) must be an integer type.");
    static_assert(std::numeric_limits<U>::is_integer, "U (second parameter) must be an integer type.");
    assert(multiple > 0);

    T remainder(x % multiple);

    if (remainder != 0) {
        return x - remainder + multiple;
    } else {
        return x;
    }
}

template <class T, class U>
constexpr T round_up_div(T dividend, U divisor) {
    static_assert(std::numeric_limits<T>::is_integer, "T (first parameter) must be an integer type.");
    static_assert(std::numeric_limits<U>::is_integer, "U (second parameter) must be an integer type.");
    assert(divisor > 0);

    T up((dividend % divisor != 0) ? 1 : 0);

    return dividend / divisor + up;
}

template <class Integer>
static constexpr Integer log2_const(Integer x) {
    static_assert(std::numeric_limits<Integer>::is_integer, "Parameter must be an integer type.");
    static_assert(std::numeric_limits<Integer>::digits != 0, "Parameter must be a fixed-size type.");
    static_assert(std::numeric_limits<Integer>::radix == 2, "Parameter must be a radix-2 type.");
    // static_assert((x & (x-1)) == 0, "log2_const argument must be a power of two.");
    // assert(x >= Integer(0));

    for (int i = 0; i < std::numeric_limits<Integer>::digits; i++) {
        if (x <= Integer(1) << i) {
            return i;
        }
    }

    return std::numeric_limits<Integer>::digits;
}

template <class Integer>
static constexpr Integer log2_flr_const(Integer x) {
    static_assert(std::numeric_limits<Integer>::is_integer, "Parameter must be an integer type.");
    static_assert(std::numeric_limits<Integer>::digits != 0, "Parameter must be a fixed-size type.");
    static_assert(std::numeric_limits<Integer>::radix == 2, "Parameter must be a radix-2 type.");
    // static_assert((x & (x-1)) == 0, "log2_const argument must be a power of two.");
    // assert(x >= Integer(0));

    for (int i = 0; i < std::numeric_limits<Integer>::digits; i++) {
        if (x >> i == Integer(1)) {
            return i;
        }
    }

    return std::numeric_limits<Integer>::digits;
}

}  // namespace ll_api
