#include <math.h>

#include "common/assert.hpp"

namespace tt {

std::uint32_t positive_pow_of_2(std::uint32_t exponent) {
    TT_ASSERT(exponent >= 0 && exponent < 32);
    std::uint32_t result = 1;
    for (std::uint32_t current_exp = 0; current_exp < exponent; current_exp++) {
        result *= 2;
    }

    return result;
}

}  // namespace tt
