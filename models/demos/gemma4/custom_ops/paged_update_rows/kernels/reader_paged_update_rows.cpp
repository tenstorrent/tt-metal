// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Reader of PagedUpdateRows: once the P index sticks; per unit (b, h) the page-table row b, the unit's job table
// (every position's cache tile row through the ring wrap and the page table, grouped into at most J tile rows),
// the two source tile rows of head h that hold rows b*P .. b*P+P-1 (the second one stale past the source's end),
// then per job the cache tile row (a null job pushes a stale block). Every L1 landing address is a CB page start.
// runtime args: [src_addr, pt_addr, cache_addr, idx_addr_0 .. idx_addr_{P-1}, n_units, (b, h) x n_units]
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/assert.h"
#include "paged_update_rows_common.hpp"

void kernel_main() {
    using namespace pur;
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t pt_addr = get_arg_val<uint32_t>(1);
    const uint32_t cache_addr = get_arg_val<uint32_t>(2);
    const uint32_t n_units = get_arg_val<uint32_t>(3 + P);
    constexpr auto src_args = TensorAccessorArgs<CT_END>();
    constexpr auto idx_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr auto cache_args = TensorAccessorArgs<pt_args.next_compile_time_args_offset()>();
    const auto s_src = TensorAccessor(src_args, src_addr, SRC_TILE);
    const auto s_pt = TensorAccessor(pt_args, pt_addr, PT_STICK);
    const auto s_cache = TensorAccessor(cache_args, cache_addr, CACHE_TILE);

    CircularBuffer cb_src(CB_SRC), cb_cache(CB_CACHE), cb_jobs(CB_JOBS);

    // The P index sticks (the same tensors for every unit of this core): reader-private scratch, read once. All
    // P tensors share one accessor layout (the wrapper asserts it); only the base address differs.
    const uint32_t idx_base = get_write_ptr(CB_IDX);
    for (uint32_t p = 0; p < P; ++p) {
        const auto s_idx = TensorAccessor(idx_args, get_arg_val<uint32_t>(3 + p), IDX_STICK);
        noc_async_read(s_idx.get_noc_addr(0), idx_base + p * IDX_STICK, IDX_STICK);
    }
    const uint32_t pt_l1 = get_write_ptr(CB_PT);
    volatile tt_l1_ptr uint32_t* pt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pt_l1);
    noc_async_read_barrier();

    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t b = get_arg_val<uint32_t>(4 + P + 2 * u);
        const uint32_t h = get_arg_val<uint32_t>(5 + P + 2 * u);
        noc_async_read(s_pt.get_noc_addr(b), pt_l1, PT_STICK);
        noc_async_read_barrier();

        // ── the job table: position p -> (job, row in tile, row in the untilized source block) ──
        const uint32_t src_tile0 = (b * P) / 32;  // the first source tile row of the unit's rows
        uint32_t job_valid[J], job_tile[J];
        for (uint32_t j = 0; j < J; ++j) {
            job_valid[j] = 0;
            job_tile[j] = 0;
        }
        cb_jobs.reserve_back(1);
        volatile tt_l1_ptr uint32_t* jobs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_jobs.get_write_ptr());
        uint32_t n_jobs = 0;
        for (uint32_t p = 0; p < P; ++p) {
            volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_base + p * IDX_STICK);
            volatile tt_l1_ptr uint32_t* e = jobs + 2 * J + 3 * p;
            const uint32_t raw = idx[b];
            e[0] = NO_JOB;
            e[1] = 0;
            e[2] = 0;
            if (raw == NO_JOB) {
                continue;  // the stock skip: update_idx == -1
            }
            // the stock paged_update_cache address: wrap into the ring, then the page table (cache_position_modulo
            // is a multiple of the block size, so the intra-block offset is preserved)
            const uint32_t q = MODULO > 0 ? raw % MODULO : raw;
            const uint32_t vblock = q / BLOCK;
            ASSERT(vblock < PT_ENTRIES);
            const uint32_t pblock = pt[vblock];
            const uint32_t tile_id = pblock * NH * St * Wt + h * St * Wt + ((q % BLOCK) / 32) * Wt;
            uint32_t j = 0;
            while (j < n_jobs && job_tile[j] != tile_id) {
                ++j;
            }
            if (j == n_jobs) {
                // a third tile row cannot come from P <= 32 consecutive positions: the position is dropped
                ASSERT(n_jobs < J);
                if (n_jobs == J) {
                    continue;
                }
                job_valid[j] = 1;
                job_tile[j] = tile_id;
                ++n_jobs;
            }
            e[0] = j;
            e[1] = q % 32;
            e[2] = b * P + p - src_tile0 * 32;
        }
        for (uint32_t j = 0; j < J; ++j) {
            jobs[2 * j] = job_valid[j];
            jobs[2 * j + 1] = job_tile[j];
        }
        cb_jobs.push_back(1);

        // ── the source tile rows: rows b*P .. b*P+P-1 of head h lie in tile rows src_tile0 and src_tile0 + 1 ──
        cb_src.reserve_back(N_SRC * Wt);
        const uint32_t src_l1 = cb_src.get_write_ptr();
        for (uint32_t s = 0; s < N_SRC; ++s) {
            if (src_tile0 + s >= SRC_ST) {
                break;  // past the source: the block stays stale and no row of it is referenced
            }
            const uint32_t tile0 = h * SRC_ST * Wt + (src_tile0 + s) * Wt;
            for (uint32_t w = 0; w < Wt; ++w) {
                noc_async_read(s_src.get_noc_addr(tile0 + w), src_l1 + (s * Wt + w) * SRC_TILE, SRC_TILE);
            }
        }
        noc_async_read_barrier();
        cb_src.push_back(N_SRC * Wt);

        // ── per job the cache tile row (the read-modify-write's read) ──
        for (uint32_t j = 0; j < J; ++j) {
            cb_cache.reserve_back(Wt);
            if (job_valid[j]) {
                const uint32_t l1 = cb_cache.get_write_ptr();
                for (uint32_t w = 0; w < Wt; ++w) {
                    noc_async_read(s_cache.get_noc_addr(job_tile[j] + w), l1 + w * CACHE_TILE, CACHE_TILE);
                }
                noc_async_read_barrier();
            }
            cb_cache.push_back(Wt);
        }
    }
    noc_async_read_barrier();
}
