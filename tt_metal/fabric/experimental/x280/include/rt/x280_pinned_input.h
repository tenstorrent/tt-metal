// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* Fixed cached Zero Device input. Policy validation is RV64-only because the
 * way masks require native 64-bit MMIO reads. */
#ifndef X280_RT_PINNED_INPUT_H
#define X280_RT_PINNED_INPUT_H

#include <stdint.h>
#include <stddef.h>

#if defined(__riscv) && __riscv_xlen == 64
#include "rt/x280_hw.h"
#endif

#define X280_PINNED_INPUT_BASE UINT64_C(0x0A000000)
#define X280_PINNED_INPUT_SPAN UINT64_C(0x000010C0)
#define X280_PINNED_INPUT_END (X280_PINNED_INPUT_BASE + X280_PINNED_INPUT_SPAN)
#define X280_PINNED_INPUT_CACHE_LINE_BYTES UINT64_C(64)
#define X280_PINNED_INPUT_CACHE_LINES UINT32_C(67)

#define X280_PINNED_INPUT_ROUTING_OFFSET UINT64_C(0)
#define X280_PINNED_INPUT_ROUTING_BYTES UINT64_C(2704)
#define X280_PINNED_INPUT_HEADERS_OFFSET UINT64_C(0x0AC0)
#define X280_PINNED_INPUT_HEADER_COUNT UINT32_C(16)
#define X280_PINNED_INPUT_HEADER_BYTES UINT64_C(96)
#define X280_PINNED_INPUT_FIRST_HEADER_BASE (X280_PINNED_INPUT_BASE + X280_PINNED_INPUT_HEADERS_OFFSET)
#define X280_PINNED_INPUT_HEADERS_BYTES (X280_PINNED_INPUT_HEADER_COUNT * X280_PINNED_INPUT_HEADER_BYTES)

#define X280_PINNED_INPUT_WAYENABLE UINT32_C(1)
#define X280_PINNED_INPUT_GENERAL_WAY UINT32_C(0)
#define X280_PINNED_INPUT_RESERVED_WAY UINT32_C(1)
#define X280_PINNED_INPUT_ACTIVE_BITS UINT64_C(0x3)
#define X280_PINNED_INPUT_BASELINE_BITS UINT64_C(0x3)
#define X280_PINNED_INPUT_PREPARED_BITS UINT64_C(0x1)
#define X280_PINNED_INPUT_MASTER_COUNT UINT32_C(38)
#define X280_PINNED_INPUT_POPULATION_MASTER UINT32_C(34)

#define X280_PINNED_INPUT_CCACHE_BASE UINT64_C(0x02010000)
#define X280_PINNED_INPUT_WAYENABLE_ADDR (X280_PINNED_INPUT_CCACHE_BASE + UINT64_C(0x0008))
#define X280_PINNED_INPUT_WAYMASK_BASE_ADDR (X280_PINNED_INPUT_CCACHE_BASE + UINT64_C(0x0800))
#define X280_CCACHE0_FLUSH64_ADDR (X280_PINNED_INPUT_CCACHE_BASE + UINT64_C(0x0200))

#define X280_PINNED_INPUT_RESULT_PREPARED UINT64_C(0x50494E4E45440001)
#define X280_PINNED_INPUT_RESULT_TORN_DOWN UINT64_C(0x50494E4E45440002)
#define X280_PINNED_INPUT_RESULT_NOOP_HART UINT64_C(0x50494E4E45440003)
#define X280_PINNED_INPUT_RESULT_STOPPED UINT64_C(0x50494E4E4544E003)
#define X280_PINNED_INPUT_RESULT_BAD_SHAPE UINT64_C(0x50494E4E4544E004)
#define X280_PINNED_INPUT_RESULT_BAD_MASK_BASE UINT64_C(0x50494E4E4544E100)
#define X280_PINNED_INPUT_RESULT_READBACK UINT64_C(0x50494E4E4544E200)
#define X280_PINNED_INPUT_RESULT_DATA UINT64_C(0x50494E4E4544E300)

#if defined(__cplusplus)
#define X280_PINNED_ASSERT static_assert
#define X280_PINNED_ALIGNAS(n) alignas(n)
#define X280_PINNED_ALIGNOF(t) alignof(t)
#else
#define X280_PINNED_ASSERT _Static_assert
#define X280_PINNED_ALIGNAS(n) _Alignas(n)
#define X280_PINNED_ALIGNOF(t) _Alignof(t)
#endif

X280_PINNED_ASSERT(
    X280_PINNED_INPUT_ROUTING_OFFSET + X280_PINNED_INPUT_ROUTING_BYTES <= X280_PINNED_INPUT_HEADERS_OFFSET,
    "routing image fits before headers");
X280_PINNED_ASSERT(
    X280_PINNED_INPUT_HEADERS_OFFSET + X280_PINNED_INPUT_HEADERS_BYTES == X280_PINNED_INPUT_SPAN,
    "headers end at fixed region end");
X280_PINNED_ASSERT(
    X280_PINNED_INPUT_CACHE_LINES* X280_PINNED_INPUT_CACHE_LINE_BYTES == X280_PINNED_INPUT_SPAN,
    "fixed region is an integral number of cache lines");
X280_PINNED_ASSERT(
    X280_PINNED_INPUT_FIRST_HEADER_BASE == UINT64_C(0x0A000AC0), "first prepared header retains its fixed address");
X280_PINNED_ASSERT(
    X280_PINNED_INPUT_FIRST_HEADER_BASE % X280_PINNED_INPUT_CACHE_LINE_BYTES == 0u,
    "first prepared header is cache-line aligned");
X280_PINNED_ASSERT(
    X280_PINNED_INPUT_HEADER_BYTES <= X280_PINNED_INPUT_END - X280_PINNED_INPUT_FIRST_HEADER_BASE,
    "first prepared header fits inside the fixed region");

enum x280_pinned_input_kind { X280_PINNED_INPUT_ROUTER_FIXED = 0, X280_PINNED_INPUT_WRITABLE_SAMPLES = 1 };

typedef struct {
    uint64_t base;
    uint64_t bytes;
    uint32_t kind;
    uint32_t reserved;
    uint64_t reserved_tail;
} x280_pinned_input_region_t;

X280_PINNED_ASSERT(sizeof(x280_pinned_input_region_t) == 32u, "region descriptor is 32 bytes");
X280_PINNED_ASSERT(X280_PINNED_ALIGNOF(x280_pinned_input_region_t) == 8u, "region descriptor is 8-aligned");
X280_PINNED_ASSERT(offsetof(x280_pinned_input_region_t, base) == 0u, "offset base");
X280_PINNED_ASSERT(offsetof(x280_pinned_input_region_t, bytes) == 8u, "offset bytes");
X280_PINNED_ASSERT(offsetof(x280_pinned_input_region_t, kind) == 16u, "offset kind");
X280_PINNED_ASSERT(offsetof(x280_pinned_input_region_t, reserved) == 20u, "offset reserved");
X280_PINNED_ASSERT(offsetof(x280_pinned_input_region_t, reserved_tail) == 24u, "offset reserved_tail");

#if defined(__cplusplus)
inline constexpr x280_pinned_input_region_t X280_PINNED_INPUT_ROUTER_REGION = {
    X280_PINNED_INPUT_BASE, X280_PINNED_INPUT_SPAN, static_cast<uint32_t>(X280_PINNED_INPUT_ROUTER_FIXED), 0u, 0u};
#else
static const x280_pinned_input_region_t X280_PINNED_INPUT_ROUTER_REGION = {
    X280_PINNED_INPUT_BASE, X280_PINNED_INPUT_SPAN, (uint32_t)X280_PINNED_INPUT_ROUTER_FIXED, 0u, 0u};
#endif

enum x280_pinned_input_admission_code {
    X280_PINNED_INPUT_ADMITTED = 0,
    X280_PINNED_INPUT_WRONG_WAYENABLE = 1,
    X280_PINNED_INPUT_WRONG_WAYMASK = 2,
    X280_PINNED_INPUT_WRONG_SHAPE = 3,
};

enum x280_pinned_input_policy {
    X280_PINNED_INPUT_BASELINE = 0,
    X280_PINNED_INPUT_PREPARED = 1,
};

typedef struct {
    uint32_t code;
    uint32_t master_id;
    uint64_t observed;
} x280_pinned_input_admission_t;

/*
 * Returns nonzero exactly when [address, address + bytes) is inside the given
 * region. Overflow is rejected using subtraction.
 */
static inline int x280_pinned_input_contains(
    const x280_pinned_input_region_t* region, uint64_t address, uint64_t bytes) {
    if (!region) {
        return 0;
    }
    const uint64_t end = region->base + region->bytes;
    return address >= region->base && address <= end && bytes <= end - address;
}

static inline int x280_pinned_input_validate_region(const x280_pinned_input_region_t* region) {
    if (!region) {
        return 0;
    }
    if (region->reserved != 0u || region->reserved_tail != 0u) {
        return 0;
    }
    if (region->kind == (uint32_t)X280_PINNED_INPUT_ROUTER_FIXED) {
        return region->base == X280_PINNED_INPUT_BASE && region->bytes == X280_PINNED_INPUT_SPAN;
    }
    if (region->kind == (uint32_t)X280_PINNED_INPUT_WRITABLE_SAMPLES) {
        return region->base == X280_PINNED_INPUT_BASE && region->bytes > 0 &&
               (region->bytes % X280_PINNED_INPUT_CACHE_LINE_BYTES == 0u) && region->bytes <= 131072u;
    }
    return 0;
}

#if defined(__riscv) && __riscv_xlen == 64

X280_PINNED_ASSERT(
    UINTPTR_MAX >= X280_PINNED_INPUT_END &&
        UINTPTR_MAX >= X280_PINNED_INPUT_WAYMASK_BASE_ADDR + X280_PINNED_INPUT_MASTER_COUNT * sizeof(uint64_t),
    "pinned input firmware requires native addresses for region and masks");

/*
 * Reads, but never repairs, the policy. `master_id` is UINT32_MAX for
 * a WAYENABLE/shape mismatch and the failing 0..37 ID for a mask mismatch.
 * `observed` is the complete native-width register value.
 */
static inline x280_pinned_input_admission_t x280_pinned_input_check_policy(
    const x280_pinned_input_region_t* region, enum x280_pinned_input_policy policy) {
    x280_pinned_input_admission_t result = {
        X280_PINNED_INPUT_ADMITTED,
        UINT32_MAX,
        0,
    };

    if (policy != X280_PINNED_INPUT_BASELINE && policy != X280_PINNED_INPUT_PREPARED) {
        result.code = X280_PINNED_INPUT_WRONG_SHAPE;
        return result;
    }

    if (policy == X280_PINNED_INPUT_PREPARED) {
        if (!x280_pinned_input_validate_region(region)) {
            result.code = X280_PINNED_INPUT_WRONG_SHAPE;
            return result;
        }
    } else {
        if (region != (const x280_pinned_input_region_t*)0 && !x280_pinned_input_validate_region(region)) {
            result.code = X280_PINNED_INPUT_WRONG_SHAPE;
            return result;
        }
    }

    const volatile uint32_t* wayenable = (volatile const uint32_t*)(uintptr_t)X280_PINNED_INPUT_WAYENABLE_ADDR;

    x280_fence_iorw();
    const uint32_t observed_wayenable = *wayenable;
    x280_fence_iorw();
    if (observed_wayenable == 0xFFFFFFFFu || (observed_wayenable & UINT32_C(0xF)) != X280_PINNED_INPUT_WAYENABLE) {
        result.code = X280_PINNED_INPUT_WRONG_WAYENABLE;
        result.observed = observed_wayenable;
        return result;
    }

    const uint64_t expected_bits =
        (policy == X280_PINNED_INPUT_PREPARED) ? X280_PINNED_INPUT_PREPARED_BITS : X280_PINNED_INPUT_BASELINE_BITS;

    for (uint32_t master = 0; master < X280_PINNED_INPUT_MASTER_COUNT; ++master) {
        const volatile uint64_t* waymask = (volatile const uint64_t*)(uintptr_t)(X280_PINNED_INPUT_WAYMASK_BASE_ADDR +
                                                                                 (uint64_t)master * sizeof(uint64_t));
        const uint64_t observed = *waymask;
        x280_fence_iorw();
        if (observed == UINT64_C(0xFFFFFFFFFFFFFFFF)) {
            result.code = X280_PINNED_INPUT_WRONG_WAYMASK;
            result.master_id = master;
            result.observed = observed;
            return result;
        }
        if ((observed & X280_PINNED_INPUT_ACTIVE_BITS) != expected_bits) {
            result.code = X280_PINNED_INPUT_WRONG_WAYMASK;
            result.master_id = master;
            result.observed = observed;
            return result;
        }
    }
    return result;
}

#ifdef __cplusplus
extern "C" {
#endif

uint64_t x280_pinned_input_prepare(uint64_t hartid, uint64_t run_value, const x280_pinned_input_region_t* region);

uint64_t x280_pinned_input_teardown(uint64_t hartid, uint64_t run_value, const x280_pinned_input_region_t* region);

#ifdef __cplusplus
}
#endif

#endif

#undef X280_PINNED_ASSERT

#endif
