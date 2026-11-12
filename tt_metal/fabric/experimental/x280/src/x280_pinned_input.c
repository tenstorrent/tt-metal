// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rt/x280_idle.h"
#include "rt/x280_pinned_input.h"

#include <stdint.h>

#if defined(X280_PINNED_INPUT_BUILD_PREPARE) && X280_PINNED_INPUT_BUILD_PREPARE != 1
#error "X280_PINNED_INPUT_BUILD_PREPARE must be 1 when defined"
#endif

#if defined(X280_PINNED_INPUT_BUILD_LIBRARY) && X280_PINNED_INPUT_BUILD_LIBRARY != 1
#error "X280_PINNED_INPUT_BUILD_LIBRARY must be 1 when defined"
#endif

#define PI_BUILD_COUNT ((defined(X280_PINNED_INPUT_BUILD_PREPARE) ? 1 : 0) + (defined(X280_PINNED_INPUT_BUILD_LIBRARY) ? 1 : 0))

#if PI_BUILD_COUNT != 1
#error "define exactly one pinned-input image operation"
#endif

static inline volatile uint64_t* x280_pinned_waymask(uint32_t master) {
    return (volatile uint64_t*)(uintptr_t)(X280_PINNED_INPUT_WAYMASK_BASE_ADDR + (uint64_t)master * sizeof(uint64_t));
}

static uint64_t x280_pinned_mask_error(uint32_t master) { return X280_PINNED_INPUT_RESULT_BAD_MASK_BASE + master; }

static uint64_t x280_pinned_readback_error(uint32_t master) { return X280_PINNED_INPUT_RESULT_READBACK + master; }

static uint64_t x280_pinned_set_active_bits(
    uint64_t hartid, uint64_t run_value, uint64_t other_masters, uint64_t population_master) {
    for (uint32_t master = 0; master < X280_PINNED_INPUT_MASTER_COUNT; ++master) {
        if (x280_hart_stop_requested(hartid, run_value)) {
            return X280_PINNED_INPUT_RESULT_STOPPED;
        }
        volatile uint64_t* waymask = x280_pinned_waymask(master);
        const uint64_t original = *waymask;
        const uint64_t active = master == X280_PINNED_INPUT_POPULATION_MASTER ? population_master : other_masters;
        const uint64_t expected = (original & ~X280_PINNED_INPUT_ACTIVE_BITS) | active;
        *waymask = expected;
        x280_fence_iorw();
        const uint64_t observed = *x280_pinned_waymask(master);
        x280_fence_iorw();
        if (observed != expected) {
            return x280_pinned_readback_error(master);
        }
    }
    return 0;
}

#if defined(X280_PINNED_INPUT_BUILD_PREPARE) || defined(X280_PINNED_INPUT_BUILD_LIBRARY)

static uint64_t x280_pinned_check_active_bits(uint64_t expected) {
    const volatile uint32_t* wayenable = (volatile const uint32_t*)(uintptr_t)X280_PINNED_INPUT_WAYENABLE_ADDR;
    x280_fence_iorw();
    const uint32_t observed_wayenable = *wayenable;
    x280_fence_iorw();
    if (observed_wayenable == 0xFFFFFFFFu || (observed_wayenable & UINT32_C(0xF)) != X280_PINNED_INPUT_WAYENABLE) {
        return X280_PINNED_INPUT_RESULT_BAD_SHAPE;
    }

    for (uint32_t master = 0; master < X280_PINNED_INPUT_MASTER_COUNT; ++master) {
        const uint64_t observed = *x280_pinned_waymask(master);
        x280_fence_iorw();
        if (observed == UINT64_C(0xFFFFFFFFFFFFFFFF) || (observed & X280_PINNED_INPUT_ACTIVE_BITS) != expected) {
            return x280_pinned_mask_error(master);
        }
    }
    return 0;
}

static uint64_t x280_pinned_pattern(uint32_t line, uint32_t word) {
    return UINT64_C(0x9E3779B97F4A7C15) ^ ((uint64_t)line << 32) ^ (uint64_t)word;
}

static uint64_t x280_pinned_populate(uint64_t hartid, uint64_t run_value) {
    volatile uint64_t* region = (volatile uint64_t*)(uintptr_t)X280_PINNED_INPUT_BASE;
    const uint32_t words_per_line = (uint32_t)(X280_PINNED_INPUT_CACHE_LINE_BYTES / sizeof(uint64_t));

    for (uint32_t line = 0; line < X280_PINNED_INPUT_CACHE_LINES; ++line) {
        if (x280_hart_stop_requested(hartid, run_value)) {
            return X280_PINNED_INPUT_RESULT_STOPPED;
        }
        for (uint32_t word = 0; word < words_per_line; ++word) {
            region[(uint64_t)line * words_per_line + word] = x280_pinned_pattern(line, word);
        }
    }
    x280_fence_iorw();
    return 0;
}

static uint64_t x280_pinned_verify_data(uint64_t hartid, uint64_t run_value) {
    const volatile uint64_t* region = (volatile const uint64_t*)(uintptr_t)X280_PINNED_INPUT_BASE;
    const uint32_t words_per_line = (uint32_t)(X280_PINNED_INPUT_CACHE_LINE_BYTES / sizeof(uint64_t));

    for (uint32_t line = 0; line < X280_PINNED_INPUT_CACHE_LINES; ++line) {
        if (x280_hart_stop_requested(hartid, run_value)) {
            return X280_PINNED_INPUT_RESULT_STOPPED;
        }
        for (uint32_t word = 0; word < words_per_line; ++word) {
            if (region[(uint64_t)line * words_per_line + word] != x280_pinned_pattern(line, word)) {
                return X280_PINNED_INPUT_RESULT_DATA + line;
            }
        }
    }
    x280_fence_iorw();
    return 0;
}

uint64_t x280_pinned_input_prepare(uint64_t hartid, uint64_t run_value, const x280_pinned_input_region_t* region) {
    if (!x280_pinned_input_validate_region(region)) {
        return X280_PINNED_INPUT_RESULT_BAD_SHAPE;
    }

    uint64_t result = x280_pinned_check_active_bits(X280_PINNED_INPUT_BASELINE_BITS);
    if (result != 0) {
        return result;
    }
    if (x280_hart_stop_requested(hartid, run_value)) {
        return X280_PINNED_INPUT_RESULT_STOPPED;
    }

    /* Before populating a writable span, remove stale cached lines via L3 CCACHE0_FLUSH64 */
    if (region->kind == (uint32_t)X280_PINNED_INPUT_WRITABLE_SAMPLES) {
        const uint64_t end_addr = region->base + region->bytes;
        for (uint64_t addr = region->base; addr < end_addr; addr += 64u) {
            if (x280_hart_stop_requested(hartid, run_value)) {
                return X280_PINNED_INPUT_RESULT_STOPPED;
            }
            volatile uint64_t* flush_reg = (volatile uint64_t*)(uintptr_t)X280_CCACHE0_FLUSH64_ADDR;
            *flush_reg = addr;
            x280_fence_iorw();
        }
    }

    result = x280_pinned_set_active_bits(
        hartid, run_value, UINT64_C(1) << X280_PINNED_INPUT_GENERAL_WAY, UINT64_C(1) << X280_PINNED_INPUT_RESERVED_WAY);
    if (result != 0) {
        return result;
    }

    if (region->kind == (uint32_t)X280_PINNED_INPUT_ROUTER_FIXED) {
        result = x280_pinned_populate(hartid, run_value);
        if (result != 0) {
            return result;
        }
    } else {
        /* Writable samples: zero-populate all lines via volatile cached writes */
        volatile uint64_t* words = (volatile uint64_t*)(uintptr_t)region->base;
        const uint64_t word_count = region->bytes / sizeof(uint64_t);
        const uint64_t words_per_line = 64u / sizeof(uint64_t);
        for (uint64_t w = 0; w < word_count; ++w) {
            if ((w % words_per_line) == 0 && x280_hart_stop_requested(hartid, run_value)) {
                return X280_PINNED_INPUT_RESULT_STOPPED;
            }
            words[w] = 0u;
        }
        x280_fence_iorw();
    }

    result = x280_pinned_set_active_bits(
        hartid, run_value, X280_PINNED_INPUT_PREPARED_BITS, X280_PINNED_INPUT_PREPARED_BITS);
    if (result != 0) {
        return result;
    }

    const x280_pinned_input_admission_t admission = x280_pinned_input_check_policy(region, X280_PINNED_INPUT_PREPARED);
    if (admission.code == X280_PINNED_INPUT_WRONG_WAYENABLE || admission.code == X280_PINNED_INPUT_WRONG_SHAPE) {
        return X280_PINNED_INPUT_RESULT_BAD_SHAPE;
    }
    if (admission.code == X280_PINNED_INPUT_WRONG_WAYMASK) {
        return x280_pinned_mask_error(admission.master_id);
    }

    if (region->kind == (uint32_t)X280_PINNED_INPUT_ROUTER_FIXED) {
        result = x280_pinned_verify_data(hartid, run_value);
        if (result != 0) {
            return result;
        }
    }

    return X280_PINNED_INPUT_RESULT_PREPARED;
}

#endif

#if defined(X280_PINNED_INPUT_BUILD_LIBRARY)

uint64_t x280_pinned_input_teardown(uint64_t hartid, uint64_t run_value, const x280_pinned_input_region_t* region) {
    const x280_pinned_input_admission_t admission = x280_pinned_input_check_policy(region, X280_PINNED_INPUT_PREPARED);
    if (admission.code == X280_PINNED_INPUT_WRONG_WAYENABLE || admission.code == X280_PINNED_INPUT_WRONG_SHAPE) {
        return X280_PINNED_INPUT_RESULT_BAD_SHAPE;
    }
    if (admission.code == X280_PINNED_INPUT_WRONG_WAYMASK) {
        return x280_pinned_mask_error(admission.master_id);
    }
    if (x280_hart_stop_requested(hartid, run_value)) {
        return X280_PINNED_INPUT_RESULT_STOPPED;
    }

    const uint64_t result = x280_pinned_set_active_bits(
        hartid, run_value, X280_PINNED_INPUT_BASELINE_BITS, X280_PINNED_INPUT_BASELINE_BITS);
    return result == 0 ? X280_PINNED_INPUT_RESULT_TORN_DOWN : result;
}

#endif

#if !defined(X280_PINNED_INPUT_BUILD_LIBRARY)

int main(uint64_t hartid) {
    if (hartid != 0) {
        x280_hart_set_result(hartid, X280_PINNED_INPUT_RESULT_NOOP_HART);
        return 0;
    }

    const uint64_t run_value = x280_hart_control(hartid)->run_value;
    const uint64_t result = x280_pinned_input_prepare(hartid, run_value, &X280_PINNED_INPUT_ROUTER_REGION);
    x280_hart_set_result(hartid, result);
    return 0;
}

#endif
