// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Per-section argument cookies. Each suite supplies a distinct magic.
 * See docs/benchmarks/design.md, "Kernel entry point".
 */
#ifndef X280_UBENCH_COOKIES_H
#define X280_UBENCH_COOKIES_H

#include <stdint.h>

/* Section identities. Values are part of the wire format: a rename or reorder
 * that changes a section's identity changes this enum, never a suite magic. */
typedef enum { UBENCH_SECTION_COMMON = 0u, UBENCH_SECTION_BACKEND = 1u } ubench_section_t;

#if defined(__cplusplus)
constexpr uint32_t ubench_cookie_for(uint32_t suite_magic, ubench_section_t s) { return suite_magic ^ (uint32_t)s; }
#else
static inline uint32_t ubench_cookie_for(uint32_t suite_magic, ubench_section_t s) { return suite_magic ^ (uint32_t)s; }
#endif

#endif /* X280_UBENCH_COOKIES_H */
