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
//   spill_after   - 0 = off; else after this many pops, save the walk (save_src_state_addrgen), reset addrgen_1,
//                   restore the walk into addrgen_0 and continue there: the sequence must not notice
//   dest_side     - 1 = program and pop the destination side instead of the source side (same registers, same model)
//   spill_dirty   - 1 = before the restore, run a different walk on addrgen_0 (no reset in between), as when a
//                   parked walk is restored into an address generator another walk was just using
// Runtime args:
//   report_addr   - L1 address for kNumPops 64-bit addresses (low word, then high word), then the saved state
//                   (kNumStateWords 64-bit words, AddrgenSrcState field order) when spill_after != 0

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

    overlay::reset_addrgen<overlay::ADDRGEN_1>();
    overlay::setup_banking_addrgen<overlay::ADDRGEN_1, side>(overlay::BankingConfig{
        .endpoint_id_shift = kBankShift,
        .size = num_banks,
        .skip = 1,
        .base = bank_base,
        .current = bank_start,
        .bank_order = static_cast<overlay::bank_order_e>(bank_order),
    });
    overlay::setup_inner_loop_addrgen<overlay::ADDRGEN_1, side>(kInnerStride, kInnerEnd, inner_start);
    if constexpr (use_outer) {
        overlay::setup_outer_loop_addrgen<overlay::ADDRGEN_1, side>(kOuterStride, kOuterEnd, outer_start);
    }

    uint64_t pops[kNumPops];
    overlay::AddrgenState saved{};
    bool moved = false;  // walk now lives in addrgen_0
    for (uint32_t i = 0; i < kNumPops; ++i) {
        if (spill_after != 0 && i == spill_after) {
            overlay::save_state_addrgen<overlay::ADDRGEN_1, side>(saved);
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
            overlay::restore_state_addrgen<overlay::ADDRGEN_0, side>(saved);
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

    // Quasar DM stores go through the data cache; report through the uncached alias so the host sees it.
    volatile tt_l1_ptr uint32_t* report =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr + MEM_L1_UNCACHED_BASE);
    for (uint32_t i = 0; i < kNumPops; ++i) {
        report[2 * i] = static_cast<uint32_t>(pops[i]);
        report[2 * i + 1] = static_cast<uint32_t>(pops[i] >> 32);
    }
    const uint64_t state[] = {
        saved.bank_current(),
        saved.bank_base(),
        saved.bank_size(),
        saved.bank_skip(),
        saved.inner_stride,
        saved.inner_end,
        saved.inner_address,
        saved.outer_stride,
        saved.outer_end,
        saved.outer_address,
        saved.bank_offset(),
        saved.bank_order()};
    for (uint32_t i = 0; i < sizeof(state) / sizeof(state[0]); ++i) {
        report[2 * (kNumPops + i)] = static_cast<uint32_t>(state[i]);
        report[2 * (kNumPops + i) + 1] = static_cast<uint32_t>(state[i] >> 32);
    }
}
