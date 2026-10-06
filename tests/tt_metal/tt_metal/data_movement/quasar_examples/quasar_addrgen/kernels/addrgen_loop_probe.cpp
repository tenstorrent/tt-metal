// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Records the address sequence addrgen_1's source (or destination) side pops for a small loop-nest configuration, so
// the host can check the loop semantics the TensorAccessor walkers rely on (tensor_accessor_addrgen.h): what the inner
// loop wraps to when the walk starts mid-row, when the bank advances under each bank order, and how the outer loop
// steps. Addresses only -- no NoC transaction is issued.
//
// Compile-time args:
//   bank_order    - overlay::bank_order_e for the banking loop
//   num_banks     - BankingConfig.size (skip 1)
//   bank_base     - BankingConfig.base: first bank's endpoint id
//   bank_start    - BankingConfig.current: starting bank, relative to base
//   inner_start   - inner-loop starting address (X_ADDRESS)
//   outer_start   - outer-loop starting address (Y_ADDRESS)
//   use_outer     - 1 = program the outer loop, 0 = leave it at its reset value
//   pop_amount    - addresses each pop advances by (the hardware skip; 1 = next address)
//   plain_pop     - 1 = use the count-less pop_src_addrgen() instead (must behave as pop_amount 1)
//   spill_after   - 0 = off; else after this many pops, save the walk's position (save_position_addrgen), reset
//                   addrgen_1, restore the walk (its program + that position) into addrgen_0 and continue there: the
//                   sequence must not notice. Requires use_outer (the restore writes the outer loop).
//   dest_side     - 1 = program and pop the destination side instead of the source side (same registers, same model)
//   spill_dirty   - 1 = before the restore, run a different walk on addrgen_0 (no reset in between), as when a
//                   parked walk is restored into an address generator another walk was just using
// Runtime args:
//   report_addr   - L1 address for kNumPops 64-bit addresses (low word, then high word), then, when spill_after != 0,
//                   two register snapshots of the walk's side (kNumRegWords 64-bit words each, RegSnapshot order):
//                   addrgen_1 just before the save, and addrgen_0 just after the restore. They must be equal: a
//                   restore that leaves any register stale shows here even if this walk's pops never depend on it.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"
#include "internal/tt-2xx/quasar/overlay/addrgen_state.hpp"

// Must match b1::kLoopProbe* in test_tensor_accessor_addrgen.cpp.
constexpr uint32_t kNumPops = 24;
constexpr uint32_t kBankShift = 26;
constexpr uint64_t kInnerStride = 0x40;
constexpr uint64_t kInnerEnd = 0x100;
constexpr uint64_t kOuterStride = 0x1000;
constexpr uint64_t kOuterEnd = 0x10000;
constexpr uint32_t kNumRegWords = 12;

// Every register of one side the walk depends on, read straight from hardware (fenced: see read_reg_fenced) into
// `out` as 64-bit words (low word, then high word). Written straight to the report: the DM stack is small.
#define PROBE_AG_REG(SIDE, name)                                                                       \
    ((SIDE) == ::overlay::Side::Src ? TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_##name##_REG_OFFSET \
                                    : TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_##name##_REG_OFFSET)
template <overlay::AddrGen G, overlay::Side S>
void snapshot_regs(volatile tt_l1_ptr uint32_t* out) {
    uint64_t r[kNumRegWords];
    r[0] = overlay::read_reg_fenced<G>(PROBE_AG_REG(S, BANK_CURRENT));
    r[1] = overlay::read_reg_fenced<G>(PROBE_AG_REG(S, BANK_BASE));
    r[2] = overlay::read_reg_fenced<G>(PROBE_AG_REG(S, BANK_SIZE));
    r[3] = overlay::read_reg_fenced<G>(PROBE_AG_REG(S, BANK_SKIP));
    r[4] = overlay::read_reg_fenced<G>(PROBE_AG_REG(S, INNER_STRIDE));
    r[5] = overlay::read_reg_fenced<G>(PROBE_AG_REG(S, INNER_END));
    r[6] = overlay::read_reg_fenced<G>(PROBE_AG_REG(S, INNER_ADDRESS));
    r[7] = overlay::read_reg_fenced<G>(PROBE_AG_REG(S, OUTER_STRIDE));
    r[8] = overlay::read_reg_fenced<G>(PROBE_AG_REG(S, OUTER_END));
    r[9] = overlay::read_reg_fenced<G>(PROBE_AG_REG(S, OUTER_ADDRESS));
    TT_ROCC_ADDRESS_GEN_MISC_reg_u misc;
    misc.val = overlay::read_reg_fenced<G>(TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_MISC_REG_OFFSET);
    r[10] = misc.f.bank_offset;
    r[11] = S == overlay::Side::Src ? misc.f.src_bank_order : misc.f.dst_bank_order;
    for (uint32_t i = 0; i < kNumRegWords; ++i) {
        out[2 * i] = static_cast<uint32_t>(r[i]);
        out[2 * i + 1] = static_cast<uint32_t>(r[i] >> 32);
    }
}
#undef PROBE_AG_REG

void kernel_main() {
    constexpr uint32_t bank_order = get_arg(args::bank_order);
    constexpr uint32_t num_banks = get_arg(args::num_banks);
    constexpr uint32_t bank_base = get_arg(args::bank_base);
    constexpr uint32_t bank_start = get_arg(args::bank_start);
    constexpr uint32_t inner_start = get_arg(args::inner_start);
    constexpr uint32_t outer_start = get_arg(args::outer_start);
    constexpr uint32_t use_outer = get_arg(args::use_outer);
    constexpr uint32_t pop_amount = get_arg(args::pop_amount);
    constexpr uint32_t plain_pop = get_arg(args::plain_pop);
    constexpr uint32_t spill_after = get_arg(args::spill_after);
    constexpr uint32_t spill_dirty = get_arg(args::spill_dirty);
    constexpr overlay::Side side = get_arg(args::dest_side) ? overlay::Side::Dest : overlay::Side::Src;
    const uint32_t report_addr = get_arg(args::report_addr);
    static_assert(spill_after == 0 || use_outer, "a spill restores the outer loop, so it must be programmed");

    // The walk's program, as the walkers keep it: all a restore needs besides the saved position.
    const overlay::AddrgenProgram program{
        .banking =
            {
                .endpoint_id_shift = kBankShift,
                .size = num_banks,
                .skip = 1,
                .base = bank_base,
                .current = bank_start,
                .bank_order = static_cast<overlay::bank_order_e>(bank_order),
            },
        .inner_stride = kInnerStride,
        .inner_end = kInnerEnd,
        .outer_stride = kOuterStride,
        .outer_end = kOuterEnd,
    };

    overlay::reset_addrgen<overlay::ADDRGEN_1>();
    overlay::setup_banking_addrgen<overlay::ADDRGEN_1, side>(program.banking);
    overlay::setup_inner_loop_addrgen<overlay::ADDRGEN_1, side>(kInnerStride, kInnerEnd, inner_start);
    if constexpr (use_outer) {
        overlay::setup_outer_loop_addrgen<overlay::ADDRGEN_1, side>(kOuterStride, kOuterEnd, outer_start);
    }

    // Quasar DM stores go through the data cache; report through the uncached alias so the host sees it.
    volatile tt_l1_ptr uint32_t* report =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr + MEM_L1_UNCACHED_BASE);
    volatile tt_l1_ptr uint32_t* before = report + 2 * kNumPops;
    volatile tt_l1_ptr uint32_t* after = before + 2 * kNumRegWords;

    uint64_t pops[kNumPops];
    bool moved = false;  // walk now lives in addrgen_0
    for (uint32_t i = 0; i < kNumPops; ++i) {
        if (spill_after != 0 && i == spill_after) {
            snapshot_regs<overlay::ADDRGEN_1, side>(before);
            overlay::AddrgenPosition saved{};
            overlay::save_position_addrgen<overlay::ADDRGEN_1, side>(saved);
            overlay::reset_addrgen<overlay::ADDRGEN_1>();
            overlay::reset_addrgen<overlay::ADDRGEN_0>();
            if constexpr (spill_dirty) {
                // Another walk: different banks, loops, and position; popped a few times.
                overlay::setup_banking_addrgen<overlay::ADDRGEN_0, side>(overlay::BankingConfig{
                    .endpoint_id_shift = kBankShift,
                    .size = 3,
                    .skip = 1,
                    .base = 5,
                    .current = 2,
                    .bank_order = overlay::BANK_OUTER});
                overlay::setup_inner_loop_addrgen<overlay::ADDRGEN_0, side>(0x20, 0x80, 0x60);
                overlay::setup_outer_loop_addrgen<overlay::ADDRGEN_0, side>(0x400, 0x4000, 0x800);
                for (uint32_t n = 0; n < 7; ++n) {
                    (void)overlay::pop_addrgen<overlay::ADDRGEN_0, side>(1);
                }
            }
            overlay::restore_addrgen<overlay::ADDRGEN_0, side>(program, saved);
            snapshot_regs<overlay::ADDRGEN_0, side>(after);
            moved = true;
        }
        if (moved) {
            pops[i] = overlay::pop_addrgen<overlay::ADDRGEN_0, side>(pop_amount);
        } else if constexpr (plain_pop) {
            pops[i] = side == overlay::Side::Src ? overlay::pop_src_addrgen<overlay::ADDRGEN_1>()
                                                 : overlay::pop_dest_addrgen<overlay::ADDRGEN_1>();
        } else {
            pops[i] = overlay::pop_addrgen<overlay::ADDRGEN_1, side>(pop_amount);
        }
    }
    overlay::reset_addrgen<overlay::ADDRGEN_1>();
    overlay::reset_addrgen<overlay::ADDRGEN_0>();

    for (uint32_t i = 0; i < kNumPops; ++i) {
        report[2 * i] = static_cast<uint32_t>(pops[i]);
        report[2 * i + 1] = static_cast<uint32_t>(pops[i] >> 32);
    }
}
