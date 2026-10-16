// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "experimental/cce_gddr.h"

namespace cce_test {

inline __attribute__((always_inline)) void copy_words(
    volatile uint32_t* dst, const volatile uint32_t* src, uint32_t num_words, uint32_t& xor_accumulator) {
    for (uint32_t i = 0; i < num_words; i++) {
        const uint32_t value = src[i];
        dst[i] = value;
        xor_accumulator ^= value;
    }
}

}  // namespace cce_test
