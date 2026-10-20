// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "internal/ethernet/tt_eth_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_phase_profile_types.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_phase_profiler.hpp"

#include <cstdint>

// Experimental primitive-cost microbenchmark (TT_FABRIC_PHASE_PROFILE=3). Runs once at router start, before any
// stateful NoC command-buffer setup, and stores the average core cycles per operation (x100) for each
// FabricMicrobench entry at RESULT_ADDR. NoC writes target a scratch region in this core's own L1.
template <size_t RESULT_ADDR, size_t SCRATCH_ADDR, uint8_t NOC, uint8_t CMD_BUF, uint8_t TXQ>
void run_fabric_erisc_microbench() {
    constexpr uint32_t N = 64;
    constexpr uint32_t TRID = 0xF;
    volatile tt_l1_ptr uint32_t* results = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(RESULT_ADDR);
    auto record = [&](FabricMicrobench entry, uint32_t start, uint32_t end, uint32_t ops) {
        results[entry] = ((end - start) * 100) / ops;
    };
    auto wait_writes = [&]() {
        while (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(NOC, TRID)) {
        }
        while (!noc_cmd_buf_ready(NOC, CMD_BUF)) {
        }
    };

    // Timer read cost.
    uint32_t start = fabric_core_cycles();
    uint32_t sink = 0;
    for (uint32_t i = 0; i < N; i++) {
        sink += fabric_core_cycles();
    }
    record(MB_CYCLE_CSR_READ, start, fabric_core_cycles(), N);

    // Dependent L1 load chain (pointer chase through one self-referencing word).
    volatile tt_l1_ptr uint32_t* chase = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SCRATCH_ADDR);
    chase[0] = SCRATCH_ADDR;
    uint32_t addr = SCRATCH_ADDR;
    start = fabric_core_cycles();
    for (uint32_t i = 0; i < N; i++) {
        addr = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    }
    record(MB_L1_LOAD_DEPENDENT, start, fabric_core_cycles(), N);
    sink += addr;

    // Stream register read (the receiver's packets-received poll).
    start = fabric_core_cycles();
    for (uint32_t i = 0; i < N; i++) {
        sink += NOC_STREAM_READ_REG(0, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
    }
    record(MB_STREAM_REG_READ, start, fabric_core_cycles(), N);

    // Ethernet txq busy poll and wall-clock register read.
    start = fabric_core_cycles();
    for (uint32_t i = 0; i < N; i++) {
        sink += internal_::eth_txq_is_busy(TXQ);
    }
    record(MB_ETH_TXQ_BUSY_READ, start, fabric_core_cycles(), N);
    start = fabric_core_cycles();
    for (uint32_t i = 0; i < N; i++) {
        sink += eth_risc_reg_read(ETH_RISC_WALL_CLOCK_0);
    }
    record(MB_WALL_CLOCK_READ, start, fabric_core_cycles(), N);

    // NoC command-buffer ready poll.
    start = fabric_core_cycles();
    for (uint32_t i = 0; i < N; i++) {
        sink += noc_cmd_buf_ready(NOC, CMD_BUF);
    }
    record(MB_NOC_CMD_BUF_READY_READ, start, fabric_core_cycles(), N);

    // Stateless one-packet NoC writes to this core's L1, as the receiver's local delivery issues them.
    const uint32_t src = SCRATCH_ADDR + 64;
    const uint64_t dst = get_noc_addr(my_x[NOC], my_y[NOC], SCRATCH_ADDR + 4096, NOC);
    for (const uint32_t size : {64u, 2048u}) {
        wait_writes();
        start = fabric_core_cycles();
        for (uint32_t i = 0; i < N; i++) {
            noc_async_write_one_packet_with_trid<false, false>(src, dst, size, TRID, CMD_BUF, NOC);
        }
        record(size == 64 ? MB_NOC_WRITE_64B : MB_NOC_WRITE_2KB, start, fabric_core_cycles(), N);
    }

    // The same writes through the stateful API: destination core, VC and size preset once.
    wait_writes();
    noc_async_write_one_packet_with_trid_set_state<false>(dst, CMD_BUF, NOC);
    start = fabric_core_cycles();
    for (uint32_t i = 0; i < N; i++) {
        noc_async_write_one_packet_with_trid_with_state<false>(src, SCRATCH_ADDR + 4096, 64, TRID, CMD_BUF, NOC);
    }
    record(MB_NOC_WRITE_STATEFUL_64B, start, fabric_core_cycles(), N);

    // Inline 4B write to L1 (emulated on Blackhole as a local store plus a NoC write).
    wait_writes();
    start = fabric_core_cycles();
    for (uint32_t i = 0; i < N; i++) {
        noc_inline_dw_write<InlineWriteDst::L1, false, false>(
            dst, i, 0xF, NOC, NOC_UNICAST_WRITE_VC, SCRATCH_ADDR + 32);
    }
    record(MB_NOC_INLINE_L1_WRITE, start, fabric_core_cycles(), N);
    wait_writes();
    while (!ncrisc_noc_nonposted_writes_flushed(NOC)) {
    }

    // Instruction fetch: build 2047 NOPs + ret in the scratch region (never executed before, so no stale
    // instruction-cache lines), then run the last n of them, once to warm, then timed.
    constexpr uint32_t MAX_NOPS = PHASE_PROFILE_MICROBENCH_SCRATCH_SIZE / 4 - 1;
    volatile tt_l1_ptr uint32_t* code = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SCRATCH_ADDR);
    for (uint32_t i = 0; i < MAX_NOPS; i++) {
        code[i] = 0x00000013;  // addi x0, x0, 0
    }
    code[MAX_NOPS] = 0x00008067;  // jalr x0, 0(ra)
    sink += code[MAX_NOPS];       // read back so the stores have landed before the first fetch
    asm volatile("fence" ::: "memory");
    constexpr uint32_t REPS = 8;
    uint32_t entry = MB_ICACHE_64;
    for (uint32_t n = 64; n <= MAX_NOPS + 1; n *= 2, entry++) {
        const uint32_t nops = n > MAX_NOPS ? MAX_NOPS : n;
        auto* fn = reinterpret_cast<void (*)()>(SCRATCH_ADDR + (MAX_NOPS - nops) * 4);
        fn();
        start = fabric_core_cycles();
        for (uint32_t r = 0; r < REPS; r++) {
            fn();
        }
        record(static_cast<FabricMicrobench>(entry), start, fabric_core_cycles(), REPS * nops);
    }

    results[MB_SINK] = sink;
    results[MB_DONE] = 1;
}
