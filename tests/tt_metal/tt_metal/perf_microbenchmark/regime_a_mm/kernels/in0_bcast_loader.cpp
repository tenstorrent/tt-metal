// SPDX-License-Identifier: Apache-2.0
// Regime-A in0 broadcast loader (Phase 2). One dedicated core reads the FULL in0 [M_block, Kt] from
// interleaved DRAM up-front (deep pipeline), then streams it out block-by-block: multicast cb0[k] into
// every compute core's cb0[k] + inc a monotonic "valid" so compute starts after block 0 (overlaps the
// broadcast). Reading up-front removes per-block read barriers from the mcast loop; the mcast targets the
// compute-core bounding box (one/two contiguous worker bands), not the whole grid. Uses cb0 as read target
// AND mcast source (uniform L1 base with the receivers). Kills the 8P× redundant per-core in0 read (Exp6/7).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t M_block = get_compile_time_arg_val(0);
    constexpr uint32_t kb = get_compile_time_arg_val(1);
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t Kt = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t cb0 = get_compile_time_arg_val(5);
    constexpr uint32_t num_banks = get_compile_time_arg_val(7);
    constexpr uint32_t nbands = get_compile_time_arg_val(8);
    constexpr uint32_t chunk = get_compile_time_arg_val(9);  // K-blocks per mcast (amortize per-mcast cost)

    uint32_t ai = 0;
    const uint32_t in0_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t valid_sem_id = get_arg_val<uint32_t>(ai++);  // this loader's valid semaphore
    const uint32_t kstart = get_arg_val<uint32_t>(ai++);        // first K-block this loader owns
    const uint32_t kcount = get_arg_val<uint32_t>(ai++);        // # K-blocks this loader owns
    const uint32_t rt_base = ai;                                // then nbands * {ndest, x0, y0, x1, y1}

    constexpr uint32_t in0_blk = M_block * kb;
    constexpr uint32_t blk_bytes = in0_blk * tile_bytes;
    const uint32_t base = get_write_ptr(cb0);
    const uint32_t valid_addr = get_semaphore(valid_sem_id);
    const uint32_t kend = kstart + kcount;

    // Per-chunk read THEN mcast over this loader's K-range [kstart,kend). `valid` counts blocks delivered
    // WITHIN this range (0..kcount) so the feeder waits per-loader. Reads overlap compute (only chunk-0 read
    // is a small prologue). L=2 loaders split the contended in0 read + mcast.
    for (uint32_t k = kstart; k < kend; k += chunk) {
        uint32_t n = (kend - k < chunk) ? (kend - k) : chunk;
        uint32_t s = base + k * blk_bytes;
        uint32_t bytes = n * blk_bytes;
        // read n blocks (n*in0_blk tiles) of in0 into cb0[k..] from interleaved DRAM
        uint32_t w = s;
        for (uint32_t kk = 0; kk < n; ++kk) {
            uint32_t kblk = k + kk;
            for (uint32_t m = 0; m < M_block; ++m) {
                for (uint32_t c = 0; c < kb; ++c) {
                    uint32_t tid = m * Kt + (kblk * kb + c);
                    uint64_t src =
                        get_noc_addr_from_bank_id<true>(tid % num_banks, in0_addr + (tid / num_banks) * tile_bytes);
                    noc_async_read(src, w, tile_bytes);
                    w += tile_bytes;
                }
            }
        }
        noc_async_read_barrier();
        // mcast chunk + inc valid by n (data then inc are NoC-ordered per dest => valid lands after data)
        ai = rt_base;
        for (uint32_t b = 0; b < nbands; ++b) {
            uint32_t nd = get_arg_val<uint32_t>(ai++);
            uint32_t x0 = get_arg_val<uint32_t>(ai++), y0 = get_arg_val<uint32_t>(ai++);
            uint32_t x1 = get_arg_val<uint32_t>(ai++), y1 = get_arg_val<uint32_t>(ai++);
            uint64_t maddr = get_noc_multicast_addr(x0, y0, x1, y1, s);
            noc_async_write_multicast_loopback_src(s, maddr, bytes, nd);
            uint64_t vaddr = get_noc_multicast_addr(x0, y0, x1, y1, valid_addr);
            noc_semaphore_inc_multicast(vaddr, n, nd, noc_index);
        }
    }
    noc_async_write_barrier();
}
