// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Bare-metal port shim for tt_metal/hw/inc/internal/risc_attribs.h
//
// The in-tree header decorates pointers with `__attribute__((rvtt_l1_ptr))` and
// `__attribute__((rvtt_reg_ptr))`. Those attributes exist only in Tenstorrent's
// GCC fork (sfpi). An upstream riscv64-unknown-elf-gcc -- which is what
// freedom-e-sdk drives -- does not know them and warns under -Wattributes.
//
// This shim keeps the same names and semantics but drops the TT-specific
// attributes, so the fabric headers compile under a stock RISC-V toolchain.

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

// On a real Tensix/Quasar target these mark pointers into device L1 / register
// space so the compiler can pick the right load/store forms. A generic core has
// no such distinction, so they degrade to nothing.
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
