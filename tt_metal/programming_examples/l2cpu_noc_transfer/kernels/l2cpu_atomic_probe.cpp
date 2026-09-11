// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// NOC atomic-increment probe against a Blackhole L2CPU tile.
//
// For each target word (up to two: one in LIM, one in the uncached GDDR alias):
//   1. seed the word with a known value via noc_inline_dw_write (plain write),
//   2. snapshot the NIU atomic counters,
//   3. issue ONE noc_semaphore_inc (NOC atomic add) against the word,
//   4. bounded-spin watching NIU_MST_ATOMIC_RESP_RECEIVED instead of calling the
//      blocking noc_async_atomic_barrier(),
//   5. read the word back with a plain NOC read.
//
// This separates "did the increment land in L2CPU memory" from "did the L2CPU
// return an atomic response", which a barrier-based test cannot distinguish.
//
// Result record per target (8 x u32): {addr, seed, readback, resp_before,
// resp_after, spins_waited, started_before, started_after}.
void kernel_main() {
    uint32_t l1_out = get_arg_val<uint32_t>(0);
    uint32_t dram_out = get_arg_val<uint32_t>(1);
    uint32_t out_size = get_arg_val<uint32_t>(2);
    uint32_t l2cpu_x = get_arg_val<uint32_t>(3);
    uint32_t l2cpu_y = get_arg_val<uint32_t>(4);
    uint32_t addr_a = get_arg_val<uint32_t>(5);
    uint32_t addr_b = get_arg_val<uint32_t>(6);  // 0 = skip second target
    uint32_t spin_cap = get_arg_val<uint32_t>(7);
    uint32_t incr = get_arg_val<uint32_t>(8);

    constexpr auto out0_args = TensorAccessorArgs<0>();
    const auto out0 = TensorAccessor(out0_args, dram_out, out_size);

    volatile tt_l1_ptr uint32_t* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_out);
    for (uint32_t i = 0; i < 32; i++) {
        rec[i] = 0;
    }
    // 16 B-aligned landing zone for the read-back (past the 128 B record area).
    const uint32_t l1_rb = l1_out + 0x80;

    const uint32_t targets[2] = {addr_a, addr_b};
    for (uint32_t k = 0; k < 2; k++) {
        const uint32_t addr = targets[k];
        if (addr == 0) {
            continue;
        }
        volatile tt_l1_ptr uint32_t* r = rec + k * 8;
        const uint64_t noc_addr = get_noc_addr(l2cpu_x, l2cpu_y, addr);
        const uint32_t seed = 0x1000u * (k + 1);

        // 1. Seed with a plain inline write (known-good primitive).
        noc_inline_dw_write(noc_addr, seed);
        noc_async_write_barrier();

        // 2. Counter snapshot.
        const uint32_t resp_before = NOC_STATUS_READ_REG(noc_index, NIU_MST_ATOMIC_RESP_RECEIVED);
        const uint32_t started_before = NOC_STATUS_READ_REG(noc_index, NIU_MST_NONPOSTED_ATOMIC_STARTED);

        // 3. One NOC atomic increment.
        noc_semaphore_inc(noc_addr, incr);

        // 4. Bounded wait for the response counter to move.
        uint32_t waited = 0;
        uint32_t resp_after = resp_before;
        while (waited < spin_cap) {
            resp_after = NOC_STATUS_READ_REG(noc_index, NIU_MST_ATOMIC_RESP_RECEIVED);
            if (resp_after != resp_before) {
                break;
            }
            waited++;
        }

        // 5. Plain read-back of the 16 B line holding the word.
        noc_async_read(get_noc_addr(l2cpu_x, l2cpu_y, addr & ~0xFu), l1_rb, 16);
        noc_async_read_barrier();
        invalidate_l1_cache();
        const uint32_t readback = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_rb)[(addr & 0xF) >> 2];

        r[0] = addr;
        r[1] = seed;
        r[2] = readback;
        r[3] = resp_before;
        r[4] = resp_after;
        r[5] = waited;
        r[6] = started_before;
        r[7] = NOC_STATUS_READ_REG(noc_index, NIU_MST_NONPOSTED_ATOMIC_STARTED);
    }

    // Resynchronise the software ack counter with the hardware so that nothing
    // downstream (dispatch, a later barrier) waits on a response that never came.
    noc_nonposted_atomics_acked[noc_index] = NOC_STATUS_READ_REG(noc_index, NIU_MST_ATOMIC_RESP_RECEIVED);

    noc_async_write_page(0, out0, l1_out);
    noc_async_write_barrier();
}
