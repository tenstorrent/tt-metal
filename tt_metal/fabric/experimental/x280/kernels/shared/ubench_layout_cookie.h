// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * The convention a Metal runtime-argument schema follows when its slots are
 * named rather than a cookie-led section: slot 0 carries a layout cookie, a
 * three-letter schema tag and a layout number, which the kernel checks before
 * reading any other slot. A schema changes its layout number with any change
 * to its slots.
 */
#ifndef X280_UBENCH_LAYOUT_COOKIE_H
#define X280_UBENCH_LAYOUT_COOKIE_H

#include <stdbool.h>
#include <stdint.h>

#define UBENCH_LAYOUT_COOKIE_SLOT 0

#define UBENCH_LAYOUT_COOKIE(a, b, c, layout) \
    (((uint32_t)(a) << 24) | ((uint32_t)(b) << 16) | ((uint32_t)(c) << 8) | (uint32_t)(layout))

/** Whether `observed`, read from slot 0, is the schema's `current` cookie. */
static inline bool ubench_layout_is_current(uint32_t observed, uint32_t current) { return observed == current; }

#endif /* X280_UBENCH_LAYOUT_COOKIE_H */
