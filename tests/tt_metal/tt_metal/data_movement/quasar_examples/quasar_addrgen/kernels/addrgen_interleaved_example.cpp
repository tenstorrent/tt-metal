// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Demonstrates bank-interleaved address generation combined with a 1D inner loop.
// For each inner-loop address step, the hardware cycles through all 4 banks before
// advancing to the next address. Destination advances linearly by transfer_size.
//
// Hardware loop structure (src):
//   for (inner = 0; inner < inner_end; inner += transfer_size) {  // inner loop
//     for (bank = first_bank; bank < first_bank + num_banks; bank++) {  // bank loop (BANK_INNER, banks 20–23)
//       yield base + inner + (bank << bank_offset)
//     }
//   }
//
// Compile-time args:
//   0: (unused) reserved for harness compatibility
//   1: (unused) reserved for harness compatibility
//   2: num_of_addresses - total addresses to generate; must equal num_banks * num_inner_steps
// Banking LOOP can be moved to be inside inner loop, this way we would reade all addresses for one bank first, then
// move to next bank. This can be configured by changing bank_order in BankingConfig to BANK_MIDDLE or BANK_OUTER and
//  adjusting the loop structure in the comment above accordingly.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"
#include <cstdint>

constexpr uint32_t num_banks = 4;
constexpr uint32_t first_bank = 20;   // use banks 20–23
constexpr uint32_t bank_offset = 36;  // banks are 1MB (2^20 B) apart in address space
constexpr uint64_t transfer_size = 2048;

constexpr uint32_t src_base = 0x10000;
constexpr uint32_t dst_base = 0x200000;

void kernel_main() {
    constexpr uint32_t num_of_addresses = get_compile_time_arg_val(2);
    constexpr uint32_t num_inner_steps = num_of_addresses / num_banks;

    reset_addrgen_0();

    // Src: 4-bank interleaving (BANK_INNER) with 1D inner loop
    // Examples assumes DRAMs banks are 20, 21, 22, 23 entry in ATT
    constexpr BankingConfig src_banking = {
        .endpoint_id_shift = bank_offset,
        .size = num_banks,
        .skip = 1,
        .base = first_bank,
        .offset = 0,
        .bank_order = BANK_INNER,
    };
    setup_src_base_start_addrgen_0(src_base);
    setup_src_banking_addrgen_0(src_banking);
    setup_src_inner_loop_addrgen_0(transfer_size, num_inner_steps * transfer_size);

    // Dst: simple 1D stride, one new address per transaction
    setup_dest_base_start_addrgen_0(dst_base);
    setup_dest_inner_loop_addrgen_0(transfer_size, num_of_addresses * transfer_size);

    /* For real NOC transfers peek/pop are not needed — replace this loop body with:
     *   push_both_addrgen_0();
     *   issue_transaction_cmdbuf_0; */
    for (uint32_t i = 0; i < num_of_addresses; ++i) {
        uint64_t src_addr = peek_src_addrgen_0();
        uint64_t dest_addr = peek_dest_addrgen_0();
        pop_src_addrgen_0();
        pop_dest_addrgen_0();
        DPRINT << "  Source address: " << HEX() << (uint64_t)src_addr << " Destination address: " << HEX()
               << (uint64_t)dest_addr << ENDL();
    }
}
