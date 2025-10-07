// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _RISC_ATTRIBS_H_
#define _RISC_ATTRIBS_H_

#include <stdint.h>

union tt_uint64_t {
    uint64_t v;
    struct {
        uint32_t hi;
        uint32_t lo;
    };
};

#define tt_l1_ptr __attribute__((rvtt_l1_ptr))
#define tt_reg_ptr __attribute__((rvtt_reg_ptr))

// This enum is used to specify the dest location type for inline writes.
// It is needed because inline writes use all 4 memory ports and may hang on Blackhole when there is back-pressure.
// This hang only manifests when the inline writes are issued to a L1 location. The workaround on BH is for inline
// writes to L1 to use noc async writes.
enum class InlineWriteDst : uint8_t { DEFAULT = 0, L1 = 1, REG = 2 };

inline __attribute__((always_inline)) uint64_t tt_l1_load(tt_uint64_t tt_l1_ptr* p) {
    tt_uint64_t v;

    v.hi = p->hi;
    v.lo = p->lo;
    return v.v;
}

inline __attribute__((always_inline)) uint64_t tt_l1_load(volatile tt_uint64_t* tt_l1_ptr p) {
    tt_uint64_t v;

    v.hi = p->hi;
    v.lo = p->lo;
    return v.v;
}

// In certain cases enabling watcher can cause the code size to be too large. Give an option to
// disable inlining if we need to.
#if defined(WATCHER_ENABLED) && defined(WATCHER_NOINLINE)
#define FORCE_INLINE
#else
#define FORCE_INLINE inline __attribute__((always_inline))
#endif

#endif  // _RISC_ATTRIBS_H_
