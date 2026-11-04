// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Traffic-only NoC kernel for the Blackhole NoC wedge reproducers (tests/ttnn/unit_tests/operations/debug/noc_wedge).
//
// Every role issues a fixed number of NoC transactions with fixed addresses and then drains its own counters. There
// are no semaphore waits, no circular-buffer handshakes, no compute and no data dependencies between cores: every
// write lands in scratch nobody reads for its value. The only waits are this RISC's own barriers (reads returned,
// writes acknowledged / sent) and noc_cmd_buf_ready. A dispatch timeout therefore means a transaction never
// completed in the NoC (or DRAM controller), not a protocol deadlock.
//
// Compile-time arg 0 selects the role; args 1-3 are role flags (see each role; always 4 CT args, unused ones 0).
// Runtime args are per core.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t ROLE_DRAM_READ = 0;   // stream DRAM reads over the banks
constexpr uint32_t ROLE_DRAM_WRITE = 1;  // units of DRAM writes over the banks (+ own reads in flight)
constexpr uint32_t ROLE_UNICAST = 2;     // round-robin unicast writes (+ atomic increments) to a list of cores
constexpr uint32_t ROLE_MCAST = 3;       // multicast chains into a rectangle (+ another command buffer mid-chain)

// DRAM write modes
constexpr uint32_t W_ACKED = 0;         // non-posted writes, acks collected at the next flush / barrier
constexpr uint32_t W_POSTED = 1;        // posted writes (no acks)
constexpr uint32_t W_POSTED_FENCE = 2;  // posted writes + one non-posted 4 B write per bank and a barrier per unit

// Mid-chain traffic for ROLE_MCAST
constexpr uint32_t MID_NONE = 0;
constexpr uint32_t MID_DRAM_READ = 1;   // a DRAM read (read command buffer) while a linked chain is open
constexpr uint32_t MID_ATOMIC = 2;      // an atomic increment (atomic command buffer) while a linked chain is open
constexpr uint32_t MID_READ_AFTER = 3;  // the same DRAM read, but issued only after the chain closed (control)

FORCE_INLINE uint64_t xy_addr(uint32_t xy, uint32_t addr) { return get_noc_addr(xy >> 16, xy & 0xFFFF, addr); }

}  // namespace

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    uint32_t a = 0;

    if constexpr (role == ROLE_DRAM_READ) {
        // CT: [1] reads in flight before a barrier
        constexpr uint32_t in_flight = get_compile_time_arg_val(1);
        const uint32_t iters = get_arg_val<uint32_t>(a++);
        const uint32_t dram = get_arg_val<uint32_t>(a++);    // bank-local base of the DRAM scratch
        const uint32_t region = get_arg_val<uint32_t>(a++);  // bytes of DRAM scratch per bank to cycle through
        const uint32_t bytes = get_arg_val<uint32_t>(a++);
        const uint32_t first_bank = get_arg_val<uint32_t>(a++);
        const uint32_t nbanks = get_arg_val<uint32_t>(a++);
        const uint32_t l1 = get_arg_val<uint32_t>(a++);      // landing (in_flight x bytes)
        const uint32_t pinned = get_arg_val<uint32_t>(a++);  // 1: always read first_bank
        const uint32_t slots = region / bytes;
        for (uint32_t it = 0; it < iters; ++it) {
            const uint32_t bank = pinned ? first_bank : (first_bank + it) % nbanks;
            const uint32_t off = ((it / nbanks) % slots) * bytes;
            noc_async_read(get_noc_addr_from_bank_id<true>(bank, dram + off), l1 + (it % in_flight) * bytes, bytes);
            if (it % in_flight == in_flight - 1) {
                noc_async_read_barrier();
            }
        }
    } else if constexpr (role == ROLE_DRAM_WRITE) {
        // CT: [1] write mode (W_*), [2] write barrier after every unit (1) or only at the end (0)
        constexpr uint32_t mode = get_compile_time_arg_val(1);
        constexpr uint32_t barrier_each_unit = get_compile_time_arg_val(2);
        const uint32_t units = get_arg_val<uint32_t>(a++);
        const uint32_t dram = get_arg_val<uint32_t>(a++);
        const uint32_t region = get_arg_val<uint32_t>(a++);  // bytes of DRAM scratch per bank this core may write
        const uint32_t writes = get_arg_val<uint32_t>(a++);  // writes per unit
        const uint32_t bytes = get_arg_val<uint32_t>(a++);   // bytes per write
        const uint32_t first_bank = get_arg_val<uint32_t>(a++);
        const uint32_t nbanks = get_arg_val<uint32_t>(a++);
        const uint32_t src = get_arg_val<uint32_t>(a++);
        const uint32_t own_reads = get_arg_val<uint32_t>(a++);  // DRAM reads issued before each unit's writes
        const uint32_t read_bytes = get_arg_val<uint32_t>(a++);
        const uint32_t read_dram = get_arg_val<uint32_t>(a++);  // where the own reads come from (bank-local)
        const uint32_t read_region = get_arg_val<uint32_t>(a++);
        const uint32_t rd_l1 = get_arg_val<uint32_t>(a++);
        const uint32_t fence_src = get_arg_val<uint32_t>(a++);   // 4 B the fence writes
        const uint32_t fence_dram = get_arg_val<uint32_t>(a++);  // the fence word (bank-local), per core
        const uint32_t slots = region / bytes;
        const uint32_t read_slots = read_region / (read_bytes ? read_bytes : 1);
        uint32_t n = 0, r = 0;
        for (uint32_t u = 0; u < units; ++u) {
            for (uint32_t q = 0; q < own_reads; ++q, ++r) {
                const uint32_t bank = (first_bank + r) % nbanks;
                noc_async_read(
                    get_noc_addr_from_bank_id<true>(bank, read_dram + ((r / nbanks) % read_slots) * read_bytes),
                    rd_l1 + q * read_bytes,
                    read_bytes);
            }
            for (uint32_t j = 0; j < writes; ++j, ++n) {
                const uint32_t bank = (first_bank + j + u) % nbanks;
                const uint64_t dst = get_noc_addr_from_bank_id<true>(bank, dram + (n % slots) * bytes);
                if constexpr (mode == W_ACKED) {
                    noc_async_write(src, dst, bytes);
                } else {
                    noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(src, dst, bytes);
                }
            }
            if constexpr (mode == W_ACKED) {
                if constexpr (barrier_each_unit) {
                    noc_async_write_barrier();
                } else {
                    noc_async_writes_flushed();
                }
            } else {
                noc_async_posted_writes_flushed();
                if constexpr (mode == W_POSTED_FENCE) {
                    for (uint32_t b = 0; b < nbanks; ++b) {
                        noc_async_write(fence_src, get_noc_addr_from_bank_id<true>(b, fence_dram), 4);
                    }
                    noc_async_write_barrier();
                }
            }
            if (own_reads) {
                noc_async_read_barrier();
            }
        }
    } else if constexpr (role == ROLE_UNICAST) {
        // CT: [1] write VC
        constexpr uint32_t vc = get_compile_time_arg_val(1);
        const uint32_t iters = get_arg_val<uint32_t>(a++);
        const uint32_t src = get_arg_val<uint32_t>(a++);
        const uint32_t bytes = get_arg_val<uint32_t>(a++);  // 0: atomics only
        const uint32_t land = get_arg_val<uint32_t>(a++);
        const uint32_t land_slots = get_arg_val<uint32_t>(a++);
        const uint32_t atomic_every = get_arg_val<uint32_t>(a++);  // 0: no atomics
        const uint32_t word = get_arg_val<uint32_t>(a++);          // atomic target address
        const uint32_t atomic_xy = get_arg_val<uint32_t>(a++);     // atomic target core (0: the write's target)
        const uint32_t pace = get_arg_val<uint32_t>(a++);          // riscv_wait cycles after each write (0: none)
        const uint32_t n_targets = get_arg_val<uint32_t>(a++);
        const uint32_t t0 = a;
        for (uint32_t it = 0; it < iters; ++it) {
            const uint32_t xy = get_arg_val<uint32_t>(t0 + it % n_targets);
            if (bytes) {
                noc_async_write(src, xy_addr(xy, land + (it % land_slots) * bytes), bytes, noc_index, vc);
            }
            if (atomic_every && it % atomic_every == 0) {
                noc_semaphore_inc(xy_addr(atomic_xy ? atomic_xy : xy, word), 1);
            }
            if (it % 8 == 7) {
                noc_async_writes_flushed();
            }
            if (pace) {
                riscv_wait(pace);
            }
        }
    } else if constexpr (role == ROLE_MCAST) {
        // CT: [1] linked chains, [2] semaphore-set multicast after each chain, [3] mid-chain traffic (MID_*)
        constexpr uint32_t linked_chains = get_compile_time_arg_val(1);
        constexpr uint32_t sem_mcast = get_compile_time_arg_val(2);
        constexpr uint32_t mid = get_compile_time_arg_val(3);
        const uint32_t iters = get_arg_val<uint32_t>(a++);
        const uint32_t src = get_arg_val<uint32_t>(a++);
        const uint32_t start_xy = get_arg_val<uint32_t>(a++);  // the NoC's multicast start corner (NOC1: far corner)
        const uint32_t end_xy = get_arg_val<uint32_t>(a++);
        const uint32_t ndests = get_arg_val<uint32_t>(a++);
        const uint32_t piece = get_arg_val<uint32_t>(a++);
        const uint32_t chain = get_arg_val<uint32_t>(a++);
        const uint32_t land = get_arg_val<uint32_t>(a++);
        const uint32_t word = get_arg_val<uint32_t>(a++);    // semaphore-set target / local source word
        const uint32_t mid_at = get_arg_val<uint32_t>(a++);  // chain piece after which the mid-chain traffic goes
        const uint32_t dram = get_arg_val<uint32_t>(a++);    // mid-chain DRAM read source (bank-local)
        const uint32_t nbanks = get_arg_val<uint32_t>(a++);
        const uint32_t rd_bytes = get_arg_val<uint32_t>(a++);
        const uint32_t rd_l1 = get_arg_val<uint32_t>(a++);
        const uint32_t atomic_xy = get_arg_val<uint32_t>(a++);
        const uint32_t pace = get_arg_val<uint32_t>(a++);  // riscv_wait cycles after each chain (0: none)
        const uint64_t rect =
            get_noc_multicast_addr(start_xy >> 16, start_xy & 0xFFFF, end_xy >> 16, end_xy & 0xFFFF, 0);
        for (uint32_t it = 0; it < iters; ++it) {
            for (uint32_t c = 0; c < chain; ++c) {
                const bool linked = linked_chains && c + 1 < chain;
                noc_async_write_multicast(src, rect | (land + (c % 3) * piece), piece, ndests, linked);
                if constexpr (mid == MID_DRAM_READ) {
                    if (c == mid_at) {
                        noc_async_read(get_noc_addr_from_bank_id<true>(it % nbanks, dram), rd_l1, rd_bytes);
                    }
                } else if constexpr (mid == MID_ATOMIC) {
                    if (c == mid_at) {
                        noc_semaphore_inc(xy_addr(atomic_xy, word), 1);
                    }
                }
            }
            if constexpr (mid == MID_READ_AFTER) {
                noc_async_read(get_noc_addr_from_bank_id<true>(it % nbanks, dram), rd_l1, rd_bytes);
            }
            if constexpr (sem_mcast) {
                noc_semaphore_set_multicast(word, rect | (word + 16), ndests);
            }
            noc_async_writes_flushed();
            if constexpr (mid == MID_DRAM_READ || mid == MID_READ_AFTER) {
                noc_async_read_barrier();
            }
            if (pace) {
                riscv_wait(pace);
            }
        }
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();
    noc_async_read_barrier();
}
