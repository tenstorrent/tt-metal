// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Hammers the 64-bit wall clock and records every read that is not consistent with its predecessor. A read torn
// across a low-word wrap (low half from before the wrap, high half from after) shows as a forward jump near
// 2^32 followed by a backward step. GAP_NOPS widens the window between the low and high reads; 0 is the real
// get_timestamp() sequence.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "internal/tt-1xx/risc_common.h"

namespace {

constexpr uint32_t kGapNops = get_compile_time_arg_val(0);
// 8 = RISCV_DEBUG_REG_WALL_CLOCK_H (the latched half per the ISA doc), 4 = the live high half.
constexpr uint32_t kHiOff = get_compile_time_arg_val(1);
// Cycles to spin between read pairs; 0 = back-to-back. ~9450 mimics the relay's idle sweep.
constexpr uint32_t kSpinCycles = get_compile_time_arg_val(2);

enum Out : uint32_t {
    kDone = 0,
    kStop = 1,
    kItersLo = 2,
    kItersHi = 3,
    kWraps = 4,
    kFwdJumps = 5,
    kBackSteps = 6,
    kMaxFwdLo = 7,
    kMaxFwdHi = 8,
    kFirstPrevLo = 9,
    kFirstPrevHi = 10,
    kFirstCurLo = 11,
    kFirstCurHi = 12,
    kHiOnly = 13,
    kWords = 16,
};

constexpr uint32_t kDoneMagic = 0xD0E0'0001u;

// The profiler's read_wall_clock form: plain locals, so the two register loads issue back to back.
inline __attribute__((always_inline)) uint64_t read_clock() {
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    const uint32_t lo = p_reg[0];
    if constexpr (kGapNops != 0) {
        for (uint32_t i = 0; i < kGapNops; i++) {
            asm volatile("nop");
        }
    }
    const uint32_t hi = p_reg[kHiOff / 4];
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

}  // namespace

void kernel_main() {
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    for (uint32_t i = 0; i < kWords; i++) {
        out[i] = 0;
    }

    uint64_t prev = read_clock();
    uint64_t iters = 0;
    uint64_t max_fwd = 0;
    uint32_t wraps = 0, fwd = 0, back = 0, hi_only = 0;
    while (true) {
        const uint64_t cur = read_clock();
        const bool low_wrapped = static_cast<uint32_t>(cur) < static_cast<uint32_t>(prev);
        const bool hi_changed = static_cast<uint32_t>(cur >> 32) != static_cast<uint32_t>(prev >> 32);
        wraps += low_wrapped ? 1u : 0u;
        if (cur < prev) {
            back++;
        } else if (cur - prev > (1ull << 30)) {
            fwd++;
            if (cur - prev > max_fwd) {
                max_fwd = cur - prev;
            }
            if (fwd == 1) {
                out[kFirstPrevLo] = static_cast<uint32_t>(prev);
                out[kFirstPrevHi] = static_cast<uint32_t>(prev >> 32);
                out[kFirstCurLo] = static_cast<uint32_t>(cur);
                out[kFirstCurHi] = static_cast<uint32_t>(cur >> 32);
            }
        }
        // The tear signature proper: the high half moved without the low half wrapping.
        hi_only += (hi_changed && !low_wrapped) ? 1u : 0u;
        prev = cur;
        iters++;
        if constexpr (kSpinCycles != 0) {
            const uint32_t t0 = get_timestamp_32b();
            while (get_timestamp_32b() - t0 < kSpinCycles) {
            }
        }
        if ((iters & 0xFFFFu) == 0) {
            out[kItersLo] = static_cast<uint32_t>(iters);
            out[kItersHi] = static_cast<uint32_t>(iters >> 32);
            out[kWraps] = wraps;
            out[kFwdJumps] = fwd;
            out[kBackSteps] = back;
            out[kHiOnly] = hi_only;
            invalidate_l1_cache();
            if (out[kStop] != 0) {
                break;
            }
        }
    }
    out[kItersLo] = static_cast<uint32_t>(iters);
    out[kItersHi] = static_cast<uint32_t>(iters >> 32);
    out[kWraps] = wraps;
    out[kFwdJumps] = fwd;
    out[kBackSteps] = back;
    out[kMaxFwdLo] = static_cast<uint32_t>(max_fwd);
    out[kMaxFwdHi] = static_cast<uint32_t>(max_fwd >> 32);
    out[kHiOnly] = hi_only;
    out[kDone] = kDoneMagic;
}
