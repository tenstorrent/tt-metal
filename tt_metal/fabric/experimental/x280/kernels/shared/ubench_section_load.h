// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Shared section loader and layout macros for microbenchmarks.
 * Writable as words and readable as flat POD structs through an anonymous
 * union lvalue, preserving GCC's documented type-punning guarantee without
 * emitting libc memcpy calls in freestanding images.
 */
#ifndef X280_UBENCH_SECTION_LOAD_H
#define X280_UBENCH_SECTION_LOAD_H

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
#define UBENCH_LAYOUT_ASSERT static_assert
#else
#define UBENCH_LAYOUT_ASSERT _Static_assert
#endif

#define UBENCH_SECTION_STORAGE(Section, name)     \
    union {                                       \
        Section name;                             \
        uint32_t as_uint32[sizeof(Section) / 4u]; \
    }

#if defined(__cplusplus)

/*
 * Copies section `index` of a block of sizeof(Section)-byte sections, out of
 * volatile device memory src, into as_uint32 word storage. Checks extent and
 * alignment (src aligned to alignof(Section)) before any read: with
 * exact_bytes zero, src_bytes must cover sections 0..index; otherwise
 * src_bytes must equal exact_bytes, which must cover them too. Returns true on
 * success; returns false leaving as_uint32 untouched if src is null, the
 * extent is refused, or src is misaligned.
 */
template <typename Section, size_t N = sizeof(Section) / 4u>
inline bool load_section(
    const volatile void* src,
    uint64_t src_bytes,
    uint32_t (&as_uint32)[N],
    uint64_t index = 0u,
    uint64_t exact_bytes = 0u) {
    static_assert(__is_trivially_copyable(Section));
    static_assert(sizeof(Section) == N * 4u);

    if (src == nullptr || index >= src_bytes / sizeof(Section) || (exact_bytes != 0u && src_bytes != exact_bytes)) {
        return false;
    }
    if ((uintptr_t)src % alignof(Section) != 0u) {
        return false;
    }

#if defined(__riscv)
    __asm__ volatile("fence ir, ir" ::: "memory");
#else
    __asm__ volatile("" ::: "memory");
#endif

    const volatile uint32_t* in = static_cast<const volatile uint32_t*>(src) + index * N;
    for (uint32_t i = 0u; i < N; ++i) {
        as_uint32[i] = in[i];
    }
    return true;
}

#endif  // defined(__cplusplus)

#endif  // X280_UBENCH_SECTION_LOAD_H
