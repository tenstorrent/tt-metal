// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tensix GDDR read bandwidth benchmark.
//
// Compile args:
//   0: src_dram_bank_id
//   1: src_dram_addr
//   2: l1_dst_addr
//   3: bytes_per_iter  (must be multiple of 16; for PIPELINED must also be multiple of 32)
//   4: iters
//
// Timing (uint64_t cycles) is written at l1_dst_addr + bytes_per_iter.
//
// Two variants selected at compile time:
//   default   — serial: one read of bytes_per_iter per iteration, full barrier after each.
//   PIPELINED — double-buffer: two in-flight half-buffer reads (TxnIdMode A/B), overlapped.

#include "api/compile_time_args.h"
#include "experimental/core_local_mem.h"
#include "experimental/endpoints.h"
#include "experimental/noc.h"
#include "risc_common.h"

void kernel_main() {
    constexpr uint32_t src_dram_bank_id = get_compile_time_arg_val(0);
    constexpr uint32_t src_dram_addr = get_compile_time_arg_val(1);
    constexpr uint32_t l1_dst_addr = get_compile_time_arg_val(2);
    constexpr uint32_t bytes_per_iter = get_compile_time_arg_val(3);
    constexpr uint32_t iters = get_compile_time_arg_val(4);

    experimental::Noc noc;
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> bank;

#ifdef PIPELINED
    constexpr uint32_t half_bytes = bytes_per_iter / 2;
    constexpr uint32_t trid_A = 2;
    constexpr uint32_t trid_B = 3;

    experimental::CoreLocalMem<uint8_t> buf_a(l1_dst_addr);
    experimental::CoreLocalMem<uint8_t> buf_b(l1_dst_addr + half_bytes);

    uint64_t start = get_timestamp();
    noc.async_read<experimental::Noc::TxnIdMode::ENABLED>(
        bank,
        buf_a,
        half_bytes,
        {.bank_id = src_dram_bank_id, .addr = src_dram_addr},
        {},
        NOC_UNICAST_WRITE_VC,
        trid_A);
    noc.async_read<experimental::Noc::TxnIdMode::ENABLED>(
        bank,
        buf_b,
        half_bytes,
        {.bank_id = src_dram_bank_id, .addr = src_dram_addr},
        {},
        NOC_UNICAST_WRITE_VC,
        trid_B);
    for (uint32_t i = 0; i < iters - 2; i++) {
        noc.async_read_barrier<experimental::Noc::BarrierMode::TXN_ID>(trid_A);
        noc.async_read<experimental::Noc::TxnIdMode::ENABLED>(
            bank,
            buf_a,
            half_bytes,
            {.bank_id = src_dram_bank_id, .addr = src_dram_addr},
            {},
            NOC_UNICAST_WRITE_VC,
            trid_A);
        noc.async_read_barrier<experimental::Noc::BarrierMode::TXN_ID>(trid_B);
        noc.async_read<experimental::Noc::TxnIdMode::ENABLED>(
            bank,
            buf_b,
            half_bytes,
            {.bank_id = src_dram_bank_id, .addr = src_dram_addr},
            {},
            NOC_UNICAST_WRITE_VC,
            trid_B);
    }
    noc.async_read_barrier<experimental::Noc::BarrierMode::TXN_ID>(trid_A);
    noc.async_read_barrier<experimental::Noc::BarrierMode::TXN_ID>(trid_B);
    uint64_t elapsed = get_timestamp() - start;

#else  // serial
    experimental::CoreLocalMem<uint8_t> dst(l1_dst_addr);

    uint64_t start = get_timestamp();
    for (uint32_t i = 0; i < iters; i++) {
        noc.async_read(bank, dst, bytes_per_iter, {.bank_id = src_dram_bank_id, .addr = src_dram_addr}, {});
        noc.async_read_barrier();
    }
    uint64_t elapsed = get_timestamp() - start;
#endif
    experimental::CoreLocalMem<uint64_t> total_time_res(l1_dst_addr);
    uint32_t offset = (bytes_per_iter) / sizeof(uint64_t);
    total_time_res[offset] = elapsed;
}
