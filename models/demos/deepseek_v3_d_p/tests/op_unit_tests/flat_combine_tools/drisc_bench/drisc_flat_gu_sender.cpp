// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DRISC flat-expert gate/up sender (test_drisc_flat_bench.cpp): the job of flat_routed_expert's Tensix weight reader
// + forwarder (se_reader.cpp + se10_fwd.cpp) on a DRAM core. Streams N_BLOCKS blocks of BLK_BYTES from its own bank
// over the GDDR DMA (block i at gddr + i * BLK_BYTES) through SLOTS L1 slots, keeping SLOTS reads in flight, and
// writes block i to receiver i % R at landing slot (i / R) % LAND_SLOTS. A slot is refilled once its write has left
// L1 (flushed, by transaction id). No receiver credits: delivery bandwidth only. Writes the start / end wall clock
// (2 x 64 bit) at TIMING_ADDR.
//
// Start barrier (slow dispatch launches cores ~1 ms apart): write 1 into 32 B slot IDX of the N-slot arrival array on
// the barrier core (one NoC-aligned line per kernel), then poll the array until every slot is set; the clock starts after that.
//
// RT: 0 gddr address, 1 DRISC L1 slot base, 2 BLK_BYTES, 3 N_BLOCKS, 4 SLOTS (<= 8), 5 R (<= 8), 6 landing address,
//     7 LAND_SLOTS, 8 timing address (32 B aligned; scratch follows at +32), 9 barrier noc_x, 10 noc_y, 11 array address, 12 IDX,
//     13 N, 14.. R x (noc_x, noc_y) of the receivers

#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "experimental/gddr_dma.h"
#include "risc_common.h"

void kernel_main() {
    const uint32_t gddr = get_arg_val<uint32_t>(0);
    const uint32_t l1 = get_arg_val<uint32_t>(1);
    const uint32_t blk = get_arg_val<uint32_t>(2);
    const uint32_t n = get_arg_val<uint32_t>(3);
    const uint32_t slots = get_arg_val<uint32_t>(4);
    const uint32_t R = get_arg_val<uint32_t>(5);
    const uint32_t land = get_arg_val<uint32_t>(6);
    const uint32_t land_slots = get_arg_val<uint32_t>(7);
    const uint32_t timing = get_arg_val<uint32_t>(8);
    uint32_t rx[8], ry[8];
    for (uint32_t j = 0; j < R; ++j) {
        rx[j] = get_arg_val<uint32_t>(14 + 2 * j);
        ry[j] = get_arg_val<uint32_t>(15 + 2 * j);
    }
    experimental::dma_set_burst_size(255);
    Noc noc;
    UnicastEndpoint src, dst;
    constexpr uint8_t stream = 0;
    {
        const uint32_t bx = get_arg_val<uint32_t>(9), by = get_arg_val<uint32_t>(10), arr = get_arg_val<uint32_t>(11);
        const uint32_t idx = get_arg_val<uint32_t>(12), nall = get_arg_val<uint32_t>(13), scratch = timing + 32;
        volatile uint32_t* one = reinterpret_cast<volatile uint32_t*>(scratch);
        one[0] = 1;
        noc.async_write(src, dst, sizeof(uint32_t), {.addr = scratch}, {.noc_x = bx, .noc_y = by, .addr = arr + 32 * idx});
        noc.async_write_barrier();
        volatile uint32_t* seen = reinterpret_cast<volatile uint32_t*>(scratch + 32);
        CoreLocalMem<uint32_t> seen_mem(scratch + 32);
        for (bool all = false; !all;) {
            noc.async_read(src, seen_mem, 32 * nall, {.noc_x = bx, .noc_y = by, .addr = arr}, {});
            noc.async_read_barrier();
            all = true;
            for (uint32_t i = 0; i < nall && all; ++i) {
                all = seen[8 * i] != 0;
            }
        }
    }
    const uint64_t t0 = get_timestamp();
    // prime: up to SLOTS reads in flight (FIFO completion on one stream)
    const uint32_t prime = n < slots ? n : slots;
    for (uint32_t i = 0; i < prime; ++i) {
        experimental::dma_async_read(stream, gddr + i * blk, l1 + i * blk, blk);
    }
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t s = i % slots;
        const uint32_t ahead = n - i - 1 < slots - 1 ? n - i - 1 : slots - 1;  // reads still in flight after i's
        experimental::dma_async_read_wait_n(stream, static_cast<uint8_t>(ahead));
        const uint32_t j = i % R;
        noc.async_write<NocOptions::TXN_ID>(
            src,
            dst,
            blk,
            {.addr = l1 + s * blk},
            {.noc_x = rx[j], .noc_y = ry[j], .addr = land + ((i / R) % land_slots) * blk},
            NocOptVals{.trid = static_cast<uint8_t>(1 + s)});
        if (i + slots < n) {
            noc.async_writes_flushed<NocOptions::TXN_ID>({.trid = static_cast<uint8_t>(1 + s)});
            experimental::dma_async_read(stream, gddr + (i + slots) * blk, l1 + s * blk, blk);
        }
    }
    for (uint32_t s = 0; s < prime; ++s) {
        noc.async_write_barrier<NocOptions::TXN_ID>({.trid = static_cast<uint8_t>(1 + s)});
    }
    const uint64_t t1 = get_timestamp();
    CoreLocalMem<uint64_t> out(timing);
    out[0] = t0;
    out[1] = t1;
}
