// SPDX-License-Identifier: Apache-2.0
// Regime-A in0 STREAMING broadcast loader (pipelined). cb0 = D-deep ring. Reads are issued D-ahead of the
// mcast (per-slot TRID) so block k+D's read overlaps block k's mcast; the mcast into every receiver's
// cb0[k%D] (uniform L1 base) is credit-gated (ready >= (k+1)*num_recv). Kills the 8P x redundant per-core in0
// read with bounded L1. `contig` = diagnostic that reads each block as ONE burst from a contiguous DRAM
// region (constant-input only) to isolate per-tile read-issue overhead vs mcast/sync cost.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t M_block = get_compile_time_arg_val(0);
    constexpr uint32_t kb = get_compile_time_arg_val(1);
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t Kt = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t cb0 = get_compile_time_arg_val(5);
    constexpr uint32_t num_recv = get_compile_time_arg_val(6);
    constexpr uint32_t D = get_compile_time_arg_val(7);
    constexpr uint32_t num_banks = get_compile_time_arg_val(8);
    constexpr uint32_t nbands = get_compile_time_arg_val(9);
    constexpr uint32_t valid_sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t ready_sem_id = get_compile_time_arg_val(11);
    constexpr uint32_t contig = get_compile_time_arg_val(12);
    constexpr uint32_t L = get_compile_time_arg_val(13);     // # loaders (K-blocks interleaved k%L across them)
    constexpr uint32_t lidx = get_compile_time_arg_val(14);  // this loader's index

    uint32_t ai = 0;
    const uint32_t in0_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t rt_base = ai;

    constexpr uint32_t in0_blk = M_block * kb;
    constexpr uint32_t blk_bytes = in0_blk * tile_bytes;
    const uint32_t base = get_write_ptr(cb0);
    const uint32_t valid_addr = get_semaphore(valid_sem_id);
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_sem_id));

    auto issue_read = [&](uint32_t k, uint32_t slot) {
        noc_async_read_set_trid((k % D) + 1);
        if constexpr (contig) {  // one burst from a contiguous per-bank region (constant-input timing diagnostic)
            uint64_t src =
                get_noc_addr_from_bank_id<true>(k % num_banks, in0_addr + (uint32_t)(k / num_banks) * blk_bytes);
            noc_async_read(src, slot, blk_bytes);
        } else {
            uint32_t w = slot;
            for (uint32_t m = 0; m < M_block; ++m) {
                for (uint32_t c = 0; c < kb; ++c) {
                    uint32_t tid = m * Kt + (k * kb + c);
                    uint64_t src =
                        get_noc_addr_from_bank_id<true>(tid % num_banks, in0_addr + (tid / num_banks) * tile_bytes);
                    noc_async_read(src, w, tile_bytes);
                    w += tile_bytes;
                }
            }
        }
    };

    // This loader owns the interleaved K-blocks k = lidx, lidx+L, ... (blocks stay in global K-order in the
    // shared cb0 ring so compute consumes them in order). cnt = this loader's own block index (its valid/ready
    // counter). Reads pipelined `depth` ahead. TRID = (k%D)+1. Multiple loaders inject mcasts in parallel.
    uint32_t depth = D / L;
    if (depth == 0) {
        depth = 1;
    }
    uint32_t cnt = 0;
    for (uint32_t k = lidx; k < K_num_blocks && cnt < depth; k += L, ++cnt) {
        issue_read(k, base + (k % D) * blk_bytes);
    }
    cnt = 0;
    for (uint32_t k = lidx; k < K_num_blocks; k += L, ++cnt) {
        uint32_t slot = base + (k % D) * blk_bytes;
        noc_async_read_barrier_with_trid((k % D) + 1);
        noc_semaphore_wait_min(ready, (cnt + 1) * num_recv);  // all receivers freed this loader's block `cnt`
        ai = rt_base;
        for (uint32_t b = 0; b < nbands; ++b) {
            uint32_t nd = get_arg_val<uint32_t>(ai++);
            uint32_t x0 = get_arg_val<uint32_t>(ai++), y0 = get_arg_val<uint32_t>(ai++);
            uint32_t x1 = get_arg_val<uint32_t>(ai++), y1 = get_arg_val<uint32_t>(ai++);
            uint64_t maddr = get_noc_multicast_addr(x0, y0, x1, y1, slot);
            noc_async_write_multicast_loopback_src(slot, maddr, blk_bytes, nd);
            uint64_t vaddr = get_noc_multicast_addr(x0, y0, x1, y1, valid_addr);
            noc_semaphore_inc_multicast(vaddr, 1, nd, noc_index);
        }
        uint32_t nk = k + depth * L;  // refill this slot with our block `cnt+depth`
        if (nk < K_num_blocks) {
            noc_async_writes_flushed();
            issue_read(nk, base + (nk % D) * blk_bytes);
        }
    }
    noc_async_write_barrier();
}
