#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <tt-metalium/bfloat16.hpp>

namespace tt {
namespace tt_metal {

inline std::vector<bfloat16> generate_exhaustive_bf16_inputs(float test_min, float test_max) {
    std::vector<bfloat16> bf16_values;
    bf16_values.reserve(65536);

    for (uint32_t bits = 0; bits < 65536; bits++) {
        uint8_t exponent = (bits >> 7) & 0xFF;
        uint8_t mantissa = bits & 0x7F;
        bool is_subnormal = (exponent == 0x00) && (mantissa != 0x00);
        if (is_subnormal) {
            continue;
        }

        uint32_t fp32_bits = static_cast<uint32_t>(bits) << 16;
        float val;
        std::memcpy(&val, &fp32_bits, sizeof(float));
        if (std::isfinite(val) && val >= test_min && val <= test_max) {
            uint16_t bits_u16 = static_cast<uint16_t>(bits);
            bfloat16 bf16_val;
            std::memcpy(&bf16_val, &bits_u16, sizeof(uint16_t));
            bf16_values.push_back(bf16_val);
        }
    }

    return bf16_values;
}

inline size_t fill_buffer_with_exhaustive_bf16(
    bfloat16* output_ptr, size_t num_elements, float test_min, float test_max) {
    auto bf16_values = generate_exhaustive_bf16_inputs(test_min, test_max);
    const size_t pattern_size = bf16_values.size();
    if (pattern_size == 0) {
        return 0;
    }

    const size_t pattern_bytes = pattern_size * sizeof(bfloat16);
    size_t offset = 0;
    while (offset + pattern_size <= num_elements) {
        std::memcpy(output_ptr + offset, bf16_values.data(), pattern_bytes);
        offset += pattern_size;
    }
    if (offset < num_elements) {
        size_t remaining = num_elements - offset;
        std::memcpy(output_ptr + offset, bf16_values.data(), remaining * sizeof(bfloat16));
    }
    return pattern_size;
}

}  // namespace tt_metal
}  // namespace tt
