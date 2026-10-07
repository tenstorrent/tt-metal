// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Repro for the address-generator register-read fence (AddrgenFenceRepro): which sequences hang when the reads of a
// walk's position are followed by only one fence, instead of a fence after each read. It replays the spill in
// AddrgenLoopProbe's BankInner2OuterSpill case one step at a time, and stops after `stage`:
//   stage 0  program addrgen_1's source side (2 banks, bank inner, inner + outer loops), pop 3 times, save the position
//            (3 register reads), reset addrgen_1
//   stage 1  ... then reset addrgen_0 and restore the walk (program + saved position) into addrgen_0's source side
//   stage 2  ... then pop addrgen_0 3 more times (the walk continues where it stopped)
//   stage 3  stage 2, plus a snapshot of 11 registers of the walk's side right before the save and right after the
//            restore (each snapshot read fenced), exactly as AddrgenLoopProbe does
// The save's 3 reads use `fence_mode`:
//   0  a fence after each read
//   1  the 3 reads back to back (3 outstanding), then one fence
//   2  each read's result used (a register move, which waits for it) before the next read; no fence
//   3  as 2, then one fence
// Finding (emu-quasar-2x3): mode 1 hangs at stage 0, and so does any sequence with two reads in flight; what matters
// is that a read's response arrives before the next read issues, not where the fence is.
// No NoC transaction is issued. A hang shows up as the kernel never writing its report (the test times out).
//
// Compile-time args: stage, fence_mode.
// Runtime args: report_addr -- 8 64-bit words (low word, then high word): the saved BANK_CURRENT, INNER_ADDRESS,
// OUTER_ADDRESS, the 3 pops after the restore (stage >= 2, else 0), the snapshot's INNER_ADDRESS after the restore
// (stage 3, else 0), then a done marker.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"
#include "internal/tt-2xx/quasar/overlay/addrgen_state.hpp"

namespace {

constexpr overlay::Side kSide = overlay::Side::Src;

#define REPRO_AG_REG(name) TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_##name##_REG_OFFSET

// The 11 registers AddrgenLoopProbe's snapshot reads, each followed by a fence. Returns INNER_ADDRESS.
template <overlay::AddrGen G>
uint64_t snapshot() {
    uint64_t sum = 0;
    sum += overlay::read_reg_fenced<G>(REPRO_AG_REG(BANK_CURRENT));
    sum += overlay::read_reg_fenced<G>(REPRO_AG_REG(BANK_BASE));
    sum += overlay::read_reg_fenced<G>(REPRO_AG_REG(BANK_SIZE));
    sum += overlay::read_reg_fenced<G>(REPRO_AG_REG(BANK_SKIP));
    sum += overlay::read_reg_fenced<G>(REPRO_AG_REG(INNER_STRIDE));
    sum += overlay::read_reg_fenced<G>(REPRO_AG_REG(INNER_END));
    const uint64_t inner = overlay::read_reg_fenced<G>(REPRO_AG_REG(INNER_ADDRESS));
    sum += overlay::read_reg_fenced<G>(REPRO_AG_REG(OUTER_STRIDE));
    sum += overlay::read_reg_fenced<G>(REPRO_AG_REG(OUTER_END));
    sum += overlay::read_reg_fenced<G>(REPRO_AG_REG(OUTER_ADDRESS));
    sum += overlay::read_reg_fenced<G>(TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_MISC_REG_OFFSET);
    asm volatile("" : : "r"(sum));
    return inner;
}

}  // namespace

void kernel_main() {
    constexpr uint32_t stage = get_arg(args::stage);
    constexpr uint32_t fence_mode = get_arg(args::fence_mode);
    const uint32_t report_addr = get_arg(args::report_addr);

    const overlay::AddrgenProgram program{
        .banking =
            {.endpoint_id_shift = 26, .size = 2, .skip = 1, .base = 0, .current = 0, .bank_order = overlay::BANK_INNER},
        .inner_stride = 0x40,
        .inner_end = 0x100,
        .outer_stride = 0x1000,
        .outer_end = 0x10000,
    };
    overlay::reset_addrgen<overlay::ADDRGEN_1>();
    overlay::setup_banking_addrgen<overlay::ADDRGEN_1, kSide>(program.banking);
    overlay::setup_inner_loop_addrgen<overlay::ADDRGEN_1, kSide>(program.inner_stride, program.inner_end, 0);
    overlay::setup_outer_loop_addrgen<overlay::ADDRGEN_1, kSide>(program.outer_stride, program.outer_end, 0);
    for (uint32_t i = 0; i < 3; ++i) {
        (void)overlay::pop_addrgen<overlay::ADDRGEN_1, kSide>(1);
    }

    if constexpr (stage >= 3) {
        (void)snapshot<overlay::ADDRGEN_1>();
    }
    overlay::AddrgenPosition saved{};
    if constexpr (fence_mode == 0) {
        overlay::save_position_addrgen<overlay::ADDRGEN_1, kSide>(saved);  // a fence after each read
    } else if constexpr (fence_mode == 1) {
        saved.bank_current = overlay::read_reg_addrgen<overlay::ADDRGEN_1>(REPRO_AG_REG(BANK_CURRENT));
        saved.inner_address = overlay::read_reg_addrgen<overlay::ADDRGEN_1>(REPRO_AG_REG(INNER_ADDRESS));
        saved.outer_address = overlay::read_reg_addrgen<overlay::ADDRGEN_1>(REPRO_AG_REG(OUTER_ADDRESS));
        overlay::fence_reg_reads_addrgen();
    } else {
        // "mv rd, rd" reads the result register, so the core waits for the read's response before going on.
        uint64_t v = overlay::read_reg_addrgen<overlay::ADDRGEN_1>(REPRO_AG_REG(BANK_CURRENT));
        asm volatile("mv %0, %0" : "+r"(v));
        saved.bank_current = v;
        v = overlay::read_reg_addrgen<overlay::ADDRGEN_1>(REPRO_AG_REG(INNER_ADDRESS));
        asm volatile("mv %0, %0" : "+r"(v));
        saved.inner_address = v;
        v = overlay::read_reg_addrgen<overlay::ADDRGEN_1>(REPRO_AG_REG(OUTER_ADDRESS));
        asm volatile("mv %0, %0" : "+r"(v));
        saved.outer_address = v;
        if constexpr (fence_mode == 3) {
            overlay::fence_reg_reads_addrgen();
        }
    }
    overlay::reset_addrgen<overlay::ADDRGEN_1>();

    uint64_t pops[3] = {0, 0, 0};
    uint64_t inner_after = 0;
    if constexpr (stage >= 1) {
        overlay::reset_addrgen<overlay::ADDRGEN_0>();
        overlay::restore_addrgen<overlay::ADDRGEN_0, kSide>(program, saved);
        if constexpr (stage >= 3) {
            inner_after = snapshot<overlay::ADDRGEN_0>();
        }
        if constexpr (stage >= 2) {
            for (uint32_t i = 0; i < 3; ++i) {
                pops[i] = overlay::pop_addrgen<overlay::ADDRGEN_0, kSide>(1);
            }
        }
        overlay::reset_addrgen<overlay::ADDRGEN_0>();
    }

    volatile tt_l1_ptr uint32_t* report =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr + MEM_L1_UNCACHED_BASE);
    const uint64_t words[7] = {
        saved.bank_current, saved.inner_address, saved.outer_address, pops[0], pops[1], pops[2], inner_after};
    for (uint32_t i = 0; i < 7; ++i) {
        report[2 * i] = static_cast<uint32_t>(words[i]);
        report[2 * i + 1] = static_cast<uint32_t>(words[i] >> 32);
    }
    report[14] = 0x600DD00Du;
    report[15] = 0;
}
