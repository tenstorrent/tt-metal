// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Bare-metal shim for risc_attribs.h: drop sfpi-only rvtt_* attributes.

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

#define tt_l1_ptr
#define tt_reg_ptr

enum class InlineWriteDst : uint8_t { DEFAULT = 0, L1 = 1, REG = 2 };

inline __attribute__((always_inline)) uint64_t tt_l1_load(tt_uint64_t* p) { return p->v; }

inline __attribute__((always_inline)) uint64_t tt_l1_load(volatile tt_uint64_t* p) {
    tt_uint64_t v;
    v.hi = p->hi;
    v.lo = p->lo;
    return v.v;
}

#define FORCE_INLINE inline __attribute__((always_inline))

#endif  // _RISC_ATTRIBS_H_
