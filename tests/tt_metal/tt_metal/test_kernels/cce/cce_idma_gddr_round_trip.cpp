// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// dram_gddr_round_trip.cpp through iDMA: for each slot, GDDR -> own TL1 staging -> GDDR, every operand a compile-time
// arg. Each copy waits for its ack before the next is issued. On a timeout the kernel returns and the host's data
// checks fail.

#include "api/compile_time_args.h"
#include "cce_idma_blocking_copy.h"
#include "cce_idma_copy_abi.h"
#include "experimental/cce_gddr.h"

namespace abi = cce_idma_copy_abi;

void kernel_main() {
    constexpr uint32_t src_buffer_address = get_compile_time_arg_val(0);
    constexpr uint32_t dst_buffer_address = get_compile_time_arg_val(1);
    constexpr uint32_t mimir_index = get_compile_time_arg_val(2);
    constexpr uint32_t staging_l1_address = get_compile_time_arg_val(3);
    constexpr uint32_t num_words = get_compile_time_arg_val(4);
    constexpr uint32_t num_slots = get_compile_time_arg_val(5);
    constexpr uint32_t slot_stride = get_compile_time_arg_val(6);
    constexpr uint32_t slot_base = get_compile_time_arg_val(7);
    constexpr uint32_t cce_index = get_compile_time_arg_val(8);
    constexpr uint32_t write_backend_vc = get_compile_time_arg_val(9);
    constexpr uint32_t len_bytes = num_words * sizeof(uint32_t);
    static_assert(len_bytes <= abi::kMaxBytesInPacket, "one iDMA command per copy");

    const uint64_t staging_address = experimental::cce_sram_address(cce_index, staging_l1_address);
    const cce_test::WaitDeadline deadline(cce_test::kIdmaWaitBudgetCycles);
    for (uint32_t slot = 0; slot < num_slots; slot++) {
        const uint64_t offset = static_cast<uint64_t>(slot_base + slot) * slot_stride;
        if (!cce_test::idma_copy_and_wait(
                experimental::cce_gddr_address(mimir_index, src_buffer_address + offset),
                staging_address,
                len_bytes,
                abi::kBackendFromRemote,
                abi::kTrid,
                deadline)) {
            return;
        }
        if (!cce_test::idma_copy_and_wait(
                staging_address,
                experimental::cce_gddr_address(mimir_index, dst_buffer_address + offset),
                len_bytes,
                write_backend_vc,
                abi::kTrid,
                deadline)) {
            return;
        }
    }
}
