// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Minimal repro for the address-generator register-read hazard (AddrgenRdRegHazard): reading an address generator's
// position registers with back-to-back rd_reg instructions hangs on emu-quasar-2x3, while the same reads with a fence
// after each one return the right values.
//
// The kernel programs addrgen_1's source side with a two-bank walk (bank inner, an inner loop of 4 addresses and an
// outer loop), pops it `num_pops` times, then reads back the three registers a context save needs -- BANK_CURRENT,
// INNER_ADDRESS, OUTER_ADDRESS -- in one of these ways:
//   mode 0  fence after each read (the workaround; control)
//   mode 1  three reads back to back, no fence
//   mode 2  three reads back to back, one fence after the last
//   mode 3  one fence, then three reads back to back
// No NoC transaction is issued. A hang shows up as the kernel never writing its report (the test times out).
//
// Compile-time args: mode, num_pops.
// Runtime args: report_addr -- 4 64-bit words (low word, then high word): BANK_CURRENT, INNER_ADDRESS, OUTER_ADDRESS,
// then a done marker.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"

namespace {

constexpr overlay::AddrGen G = overlay::ADDRGEN_1;
constexpr uint32_t kBankCurrent = TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_BANK_CURRENT_REG_OFFSET / 8;
constexpr uint32_t kInnerAddress = TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_INNER_ADDRESS_REG_OFFSET / 8;
constexpr uint32_t kOuterAddress = TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_OUTER_ADDRESS_REG_OFFSET / 8;

inline uint64_t rd(uint32_t reg) { return __builtin_riscv_ttrocc_addrgen_rd_reg(G, reg); }
inline void fence() { asm volatile("fence" ::: "memory"); }

}  // namespace

void kernel_main() {
    constexpr uint32_t mode = get_arg(args::mode);
    constexpr uint32_t num_pops = get_arg(args::num_pops);
    const uint32_t report_addr = get_arg(args::report_addr);

    overlay::reset_addrgen<G>();
    overlay::setup_src_banking_addrgen<G>(overlay::BankingConfig{
        .endpoint_id_shift = 26, .size = 2, .skip = 1, .base = 0, .current = 0, .bank_order = overlay::BANK_INNER});
    overlay::setup_src_inner_loop_addrgen<G>(0x40, 0x100, 0);
    overlay::setup_src_outer_loop_addrgen<G>(0x1000, 0x10000, 0);
    for (uint32_t i = 0; i < num_pops; ++i) {
        (void)overlay::pop_src_addrgen<G>(1);
    }

    uint64_t r[3];
    if constexpr (mode == 0) {
        r[0] = rd(kBankCurrent);
        fence();
        r[1] = rd(kInnerAddress);
        fence();
        r[2] = rd(kOuterAddress);
        fence();
    } else if constexpr (mode == 1) {
        r[0] = rd(kBankCurrent);
        r[1] = rd(kInnerAddress);
        r[2] = rd(kOuterAddress);
    } else if constexpr (mode == 2) {
        r[0] = rd(kBankCurrent);
        r[1] = rd(kInnerAddress);
        r[2] = rd(kOuterAddress);
        fence();
    } else {
        fence();
        r[0] = rd(kBankCurrent);
        r[1] = rd(kInnerAddress);
        r[2] = rd(kOuterAddress);
    }
    overlay::reset_addrgen<G>();

    volatile tt_l1_ptr uint32_t* report =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr + MEM_L1_UNCACHED_BASE);
    for (uint32_t i = 0; i < 3; ++i) {
        report[2 * i] = static_cast<uint32_t>(r[i]);
        report[2 * i + 1] = static_cast<uint32_t>(r[i] >> 32);
    }
    report[6] = 0x600DD00Du;
    report[7] = 0;
}
