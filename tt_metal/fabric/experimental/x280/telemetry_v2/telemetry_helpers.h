// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Freestanding helpers for the X280 telemetry firmware. There is no libc (-nostdlib, -ffreestanding),
// and GCC may turn memset/memcpy-shaped code into calls that fail to link, so these are plain loops.
// Written as static inline so the header can be included from any C file without duplicate symbols.

#pragma once 

#include <stddef.h>
#include <stdint.h>

static inline void clear_memory(volatile void* ptr, size_t size) {
    volatile uint8_t* curr = (volatile uint8_t*)ptr;
    for (size_t i = 0; i < size; ++i) {
        *curr = 0;
        ++curr;
    }
}

static inline void copy_memory(volatile void* dest_ptr, volatile const void* src_ptr, size_t num_bytes_to_copy) {
    const volatile uint8_t* src = (volatile const uint8_t*)src_ptr;
    volatile uint8_t* dest = (volatile uint8_t*)dest_ptr;
    for (size_t i = 0; i < num_bytes_to_copy; ++i) {
        *dest = *src;
        ++dest;
        ++src;
    }
}

// If we're going over NOC, copy word at a time.
static inline void copy_u32_memory(
    volatile uint32_t* dest_ptr, volatile const uint32_t* src_ptr, size_t num_words_to_copy) {
    for (size_t i = 0; i < num_words_to_copy; ++i) {
        *dest_ptr = *src_ptr;
        ++dest_ptr;
        ++src_ptr;
    }
}

static inline size_t str_len(const char* str) {
    if (str == NULL) {
        return 0;
    }

    size_t len = 0;
    while (*str) {
        ++len;
        ++str;
    }

    return len;
}

