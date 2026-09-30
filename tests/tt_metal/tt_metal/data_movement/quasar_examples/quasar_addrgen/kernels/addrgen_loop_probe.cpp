// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Records the address sequence addrgen_1's source side pops for a small loop-nest configuration, so the host can
// check the loop semantics the TensorAccessor walkers rely on (tensor_accessor_addrgen.h): what the inner loop
// wraps to when the walk starts mid-row, when the bank advances under each bank order, and how the outer loop steps.
// Addresses only -- no NoC transaction is issued.
//
// Compile-time args:
//   bank_order    - overlay::bank_order_e for the banking loop
//   num_banks     - BankingConfig.size (skip 1, base kBankBase)
//   bank_start    - BankingConfig.offset: starting bank, relative to base
//   inner_start   - inner-loop starting address (X_ADDRESS)
//   outer_start   - outer-loop starting address (Y_ADDRESS)
//   use_outer     - 1 = program the outer loop, 0 = leave it at its reset value
//   pop_amount    - addresses each pop advances by (the hardware skip; 1 = next address)
// Runtime args:
//   report_addr   - L1 address for kNumPops 64-bit addresses (low word, then high word)

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"

// Must match b1::kLoopProbe* in test_tensor_accessor_addrgen.cpp.
constexpr uint32_t kNumPops = 24;
constexpr uint32_t kBankShift = 26;
constexpr uint32_t kBankBase = 0;
constexpr uint64_t kInnerStride = 0x40;
constexpr uint64_t kInnerEnd = 0x100;
constexpr uint64_t kOuterStride = 0x1000;
constexpr uint64_t kOuterEnd = 0x10000;

void kernel_main() {
    constexpr uint32_t bank_order = get_arg(args::bank_order);
    constexpr uint32_t num_banks = get_arg(args::num_banks);
    constexpr uint32_t bank_start = get_arg(args::bank_start);
    constexpr uint32_t inner_start = get_arg(args::inner_start);
    constexpr uint32_t outer_start = get_arg(args::outer_start);
    constexpr uint32_t use_outer = get_arg(args::use_outer);
    constexpr uint32_t pop_amount = get_arg(args::pop_amount);
    const uint32_t report_addr = get_arg(args::report_addr);

    overlay::reset_addrgen_1();
    overlay::setup_src_banking_addrgen_1(overlay::BankingConfig{
        .endpoint_id_shift = kBankShift,
        .size = num_banks,
        .skip = 1,
        .base = kBankBase,
        .offset = bank_start,
        .bank_order = static_cast<overlay::bank_order_e>(bank_order),
    });
    overlay::setup_src_inner_loop_addrgen_1(kInnerStride, kInnerEnd, inner_start);
    if constexpr (use_outer) {
        overlay::setup_src_outer_loop_addrgen_1(kOuterStride, kOuterEnd, outer_start);
    }

    uint64_t pops[kNumPops];
    for (uint32_t i = 0; i < kNumPops; ++i) {
        pops[i] = overlay::pop_src_addrgen_1(pop_amount);
    }
    overlay::reset_addrgen_1();

    // Quasar DM stores go through the data cache; report through the uncached alias so the host sees it.
    volatile tt_l1_ptr uint32_t* report =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr + MEM_L1_UNCACHED_BASE);
    for (uint32_t i = 0; i < kNumPops; ++i) {
        report[2 * i] = static_cast<uint32_t>(pops[i]);
        report[2 * i + 1] = static_cast<uint32_t>(pops[i] >> 32);
    }
}
