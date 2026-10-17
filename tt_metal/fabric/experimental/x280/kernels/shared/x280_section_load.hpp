// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* The argument prologue of an X280 microbenchmark image. */
#ifndef X280_SECTION_LOAD_HPP
#define X280_SECTION_LOAD_HPP

#include <stddef.h>
#include <stdint.h>

#include "rt/x280_idle.h"
#include "ubench_section_load.h"

/**
 * Copies this hart's section of the argument block the host published for the
 * run into as_uint32. Harts 0..participants-1 each own one sizeof(Section)
 * section, hart h the h-th; the block's extent and alignment are checked as by
 * load_section, with exact_bytes as there (zero: the extent covers this hart's
 * section; nonzero: the extent is exactly exact_bytes).
 *
 * Returns false, leaving as_uint32 untouched, for a hart outside the
 * participants, when no block was published, or when the extent or alignment
 * is refused. Publishes nothing: on a refusal the image has no result address.
 */
template <typename Section, size_t N>
inline bool x280_load_section(
    uint64_t hartid, uint64_t participants, uint32_t (&as_uint32)[N], uint64_t exact_bytes = 0u) {
    if (hartid >= participants) {
        return false;
    }
    return load_section<Section>(
        reinterpret_cast<const volatile void*>(x280_hart_args(hartid)),
        x280_hart_args_bytes(),
        as_uint32,
        hartid,
        exact_bytes);
}

#endif /* X280_SECTION_LOAD_HPP */
