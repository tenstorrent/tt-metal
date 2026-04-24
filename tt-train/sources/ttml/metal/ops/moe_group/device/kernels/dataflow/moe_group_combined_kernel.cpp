// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Combined scan + worker-reader kernel (NCRISC), runs on EVERY worker core.
//
// Phases:
//   1. Local count of metadata slice [my_slice_start, my_slice_end).
//   2. Lead core (0,0) reduces counts, computes offsets, computes per-core
//      starts, pre-fills plan with SENTINEL, signals all cores.
//   3. Each core scatters plan entries for its slice, using its per-core start.
//   4. Lead core signals plan_ready_sem on all cores.
//   5. Each core does worker-reader work: gather rows from `dispatched` into
//      cb_src0 for its assigned tile-rows.
//
// Compile-time args (in order):
//   0: h
//   1: num_chunks
//   2: hidden_chunk_bytes
//   3: tiles_per_chunk
//   4: last_chunk_bytes
//   5: total_rows         (D*B*S)
//   6: k                  (topk)
//   7: e_local
//   8: t_cap
//   9: num_total_cores    (number of cores running this kernel)
//   10: shared_slot_u32   (per-core shared-table slot size in uint32s;
//                          host picks round_up(e_local, L1_ALIGN_U32))
//   11: l1_align_u32           (arch NOC L1 write alignment in uint32s)
//   12,13: lead_core_x, lead_core_y
//   14,15,16: scan_phase{1,2,3}_sem_id
//   17: plan_ready_sem_id
//   18: shared_tables_offset   (offset within lead's cb_scan to shared tables)
//   19-23: mcast rectangle     (sx, sy, ex, ey, num_dests_incl_self) used for
//                              phase2/plan_ready broadcast via
//                              mcast_sender_signal_receivers_loopback
//   24+: TensorAccessorArgs for plan, dispatched, metadata, counts, offsets, leids
//
// Runtime args (13 total, same layout on every core — all globally-constant
// values moved to CT):
//   0: plan_addr             1: dispatched_addr
//   2: my_worker_start       3: my_worker_count
//   4: metadata_addr         5: counts_addr
//   6: offsets_addr          7: leids_addr
//   8: my_core_idx           9: my_slice_start      10: my_slice_end
//   11: next_core_x         12: next_core_y  (tail-flush chain NOC XY;
//                                              (0,0) for the last core)

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

constexpr uint32_t cb_src0 = tt::CBIndex::c_0;
constexpr uint32_t cb_plan = tt::CBIndex::c_4;
constexpr uint32_t cb_scan = tt::CBIndex::c_3;

constexpr uint32_t h = get_compile_time_arg_val(0);
constexpr uint32_t num_chunks = get_compile_time_arg_val(1);
constexpr uint32_t hidden_chunk_bytes = get_compile_time_arg_val(2);
constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(3);
constexpr uint32_t last_chunk_bytes = get_compile_time_arg_val(4);
constexpr uint32_t total_rows = get_compile_time_arg_val(5);
constexpr uint32_t k = get_compile_time_arg_val(6);
constexpr uint32_t e_local = get_compile_time_arg_val(7);
constexpr uint32_t t_cap = get_compile_time_arg_val(8);
constexpr uint32_t num_total_cores = get_compile_time_arg_val(9);
// Per-core shared-table slot size in uint32s: host picks
// round_up_to_align(e_local) so each slot is a multiple of the arch's L1
// alignment and holds exactly e_local uint32s with minimum padding.
constexpr uint32_t SHARED_SLOT_U32 = get_compile_time_arg_val(10);
// Arch-specific NOC L1 write alignment expressed in uint32s
// (= tt::tt_metal::hal::get_l1_alignment() / 4). 4 on WH/BH today, 2 or 8
// on other parts. Used for per-core plan write padding so boundaries land
// on L1-aligned addresses.
constexpr uint32_t L1_ALIGN_U32 = get_compile_time_arg_val(11);
constexpr uint32_t L1_ALIGN_U32_MASK = L1_ALIGN_U32 - 1U;
// Globally-constant values previously passed as RT args, now baked in as CT.
constexpr uint32_t lead_core_x = get_compile_time_arg_val(12);
constexpr uint32_t lead_core_y = get_compile_time_arg_val(13);
constexpr uint32_t scan_phase1_sem_id = get_compile_time_arg_val(14);
constexpr uint32_t scan_phase2_sem_id = get_compile_time_arg_val(15);
constexpr uint32_t scan_phase3_sem_id = get_compile_time_arg_val(16);
constexpr uint32_t plan_ready_sem_id = get_compile_time_arg_val(17);
constexpr uint32_t shared_tables_offset = get_compile_time_arg_val(18);
constexpr uint32_t mcast_sx = get_compile_time_arg_val(19);
constexpr uint32_t mcast_sy = get_compile_time_arg_val(20);
constexpr uint32_t mcast_ex = get_compile_time_arg_val(21);
constexpr uint32_t mcast_ey = get_compile_time_arg_val(22);
constexpr uint32_t mcast_num_dests_incl_self = get_compile_time_arg_val(23);

constexpr auto plan_args = TensorAccessorArgs<24>();
constexpr auto dispatched_args = TensorAccessorArgs<plan_args.next_compile_time_args_offset()>();
constexpr auto metadata_args = TensorAccessorArgs<dispatched_args.next_compile_time_args_offset()>();
constexpr auto counts_args = TensorAccessorArgs<metadata_args.next_compile_time_args_offset()>();
constexpr auto offsets_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
constexpr auto leids_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();

constexpr uint32_t md_aligned_page = decltype(metadata_args)::AlignedPageSize;
constexpr uint32_t leids_aligned_page = decltype(leids_args)::AlignedPageSize;
constexpr uint32_t cnt_page_bytes = decltype(counts_args)::AlignedPageSize;
constexpr uint32_t off_page_bytes = decltype(offsets_args)::AlignedPageSize;

constexpr uint32_t TILE_H = 32U;
constexpr uint32_t SENTINEL = 0xFFFFFFFFU;
constexpr uint32_t PLAN_CHUNK = 32U;
constexpr uint32_t MD_ROW_STRIDE_U16 = md_aligned_page / sizeof(uint16_t);
// SHARED_SLOT_U32 is defined above from CT arg 10. Each shared-table slot is
// SHARED_SLOT_U32 uint32s (multiple of 16B) to keep adjacent cores' writes
// from overlapping and to meet the NOC L1 write address alignment.

inline uint32_t round_up_32(uint32_t x) {
    return ((x + 31U) >> 5U) << 5U;
}

void kernel_main() {
    // ---- Runtime args (only per-core + buffer addrs; everything globally
    //      constant has been moved to CT args above) ----
    const uint32_t plan_addr = get_arg_val<uint32_t>(0);
    const uint32_t dispatched_addr = get_arg_val<uint32_t>(1);
    const uint32_t my_worker_start = get_arg_val<uint32_t>(2);
    const uint32_t my_worker_count = get_arg_val<uint32_t>(3);
    const uint32_t metadata_addr = get_arg_val<uint32_t>(4);
    const uint32_t counts_addr = get_arg_val<uint32_t>(5);
    const uint32_t offsets_addr = get_arg_val<uint32_t>(6);
    const uint32_t leids_addr = get_arg_val<uint32_t>(7);
    const uint32_t my_core_idx = get_arg_val<uint32_t>(8);
    const uint32_t my_slice_start = get_arg_val<uint32_t>(9);
    const uint32_t my_slice_end = get_arg_val<uint32_t>(10);
    // Chain args (every core): next core's NOC XY for sequencing tail flushes.
    // For last core (my_core_idx == num_total_cores-1) these are 0,0 (unused).
    const uint32_t next_core_x = get_arg_val<uint32_t>(11);
    const uint32_t next_core_y = get_arg_val<uint32_t>(12);
    constexpr uint32_t worker_stride = num_total_cores;  // was RT arg, always == num_total_cores

    // ---- Address generators ----
    const auto plan_addrgen = TensorAccessor(plan_args, plan_addr);
    const auto dispatched_addrgen = TensorAccessor(dispatched_args, dispatched_addr, h * 2U);
    const auto md_addrgen = TensorAccessor(metadata_args, metadata_addr);
    const auto cnt_addrgen = TensorAccessor(counts_args, counts_addr);
    const auto off_addrgen = TensorAccessor(offsets_args, offsets_addr);
    const auto leids_addrgen = TensorAccessor(leids_args, leids_addr);

    // ---- L1 scratch layout in cb_scan ----
    // [stage(32B)] [leids_buf(32B)] [counts(e_local)] [offsets(e_local+1)] [cursors(e_local)]
    // [shared_local_counts_table(num_total_cores * e_local)] [shared_per_core_start_table(...)]
    // [md_block (32B aligned)] [plan_stage(e_local * PLAN_CHUNK)] [fill(e_local)]
    uint32_t scratch = get_write_ptr(cb_scan);
    // stage must be large enough to hold (e_local+1) uint32s for counts/offsets writes.
    // Use 32 uint32s = 128 bytes to support e_local up to 31.
    volatile tt_l1_ptr uint32_t* stage = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    volatile tt_l1_ptr uint32_t* leids_buf = stage + 32U;  // +128 B
    volatile tt_l1_ptr uint32_t* counts = leids_buf + 8U;
    volatile tt_l1_ptr uint32_t* offsets = counts + e_local;
    volatile tt_l1_ptr uint32_t* cursors = offsets + (e_local + 1U);
    // Shared tables — identical CB allocation on every core, so local address
    // is the same across cores. Cross-core NOC uses (lead_core_x/y, local_addr).
    volatile tt_l1_ptr uint32_t* shared_local_counts =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + shared_tables_offset);
    volatile tt_l1_ptr uint32_t* shared_per_core_start = shared_local_counts + num_total_cores * SHARED_SLOT_U32;
    // md_block 32B aligned, after shared tables
    uint32_t md_block_addr_raw = (uint32_t)(shared_per_core_start + num_total_cores * e_local);
    uint32_t md_block_addr = (md_block_addr_raw + 31U) & ~31U;
    volatile tt_l1_ptr uint16_t* md_block = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(md_block_addr);
    // BLOCK_ROWS for streaming metadata. Use slice size if it fits, else 1024.
    uint32_t my_slice_size = my_slice_end - my_slice_start;
    uint32_t block_rows = my_slice_size < 1024U ? my_slice_size : 1024U;
    uint32_t md_block_bytes = block_rows * md_aligned_page;
    uint32_t plan_stage_addr = (md_block_addr + md_block_bytes + 31U) & ~31U;
    volatile tt_l1_ptr uint32_t* plan_stage = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(plan_stage_addr);
    uint32_t plan_stage_bytes = e_local * PLAN_CHUNK * sizeof(uint32_t);
    uint32_t fill_addr = (plan_stage_addr + plan_stage_bytes + 31U) & ~31U;
    volatile tt_l1_ptr uint32_t* fill = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fill_addr);

    // ---- Load leids (uint16) into leids_buf ----
    noc_async_read(get_noc_addr(0, leids_addrgen), (uint32_t)leids_buf, leids_aligned_page);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* leids_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(leids_buf);

    // ===========================================================
    // PHASE 1: count my slice
    // ===========================================================
    for (uint32_t e = 0; e < e_local; ++e) counts[e] = 0;

    if (my_slice_size > 0U) {
        for (uint32_t block_start = my_slice_start; block_start < my_slice_end; block_start += block_rows) {
            uint32_t block_end = block_start + block_rows;
            if (block_end > my_slice_end)
                block_end = my_slice_end;
            // Batched reads
            for (uint32_t row = block_start; row < block_end; ++row) {
                uint32_t local_row = row - block_start;
                noc_async_read(
                    get_noc_addr(row, md_addrgen), md_block_addr + local_row * md_aligned_page, md_aligned_page);
            }
            noc_async_read_barrier();
            // Count
            for (uint32_t row = block_start; row < block_end; ++row) {
                uint32_t off = (row - block_start) * MD_ROW_STRIDE_U16;
                for (uint32_t e = 0; e < e_local; ++e) {
                    for (uint32_t ki = 0; ki < k; ++ki) {
                        if ((uint32_t)md_block[off + ki] == (uint32_t)leids_u16[e]) {
                            counts[e]++;
                            break;
                        }
                    }
                }
            }
        }
    }

    // ---- Publish my counts to lead's shared_local_counts table ----
    // Lead core writes to its own L1 directly; others NOC-write.
    if (my_core_idx == 0) {
        for (uint32_t e = 0; e < e_local; ++e) {
            shared_local_counts[my_core_idx * SHARED_SLOT_U32 + e] = counts[e];
        }
    } else {
        // Stage counts in plan_stage (large, 32B aligned, unused in phase 1).
        // Write SHARED_SLOT_U32 uint32s (64B) so adjacent cores don't overlap.
        for (uint32_t e = 0; e < e_local; ++e) plan_stage[e] = counts[e];
        for (uint32_t e = e_local; e < SHARED_SLOT_U32; ++e) plan_stage[e] = 0;
        uint64_t dest_noc =
            get_noc_addr(lead_core_x, lead_core_y, (uint32_t)(shared_local_counts + my_core_idx * SHARED_SLOT_U32));
        noc_async_write((uint32_t)plan_stage, dest_noc, SHARED_SLOT_U32 * sizeof(uint32_t));
        noc_async_write_barrier();
        // Increment lead's phase1 semaphore.
        uint64_t sem_noc = get_noc_addr(lead_core_x, lead_core_y, get_semaphore(scan_phase1_sem_id));
        noc_semaphore_inc(sem_noc, 1U);
    }

    // ===========================================================
    // PHASE 2: lead core reduces, computes offsets, signals others
    // ===========================================================
    if (my_core_idx == 0) {
        // Wait for all other cores to publish their counts.
        if (num_total_cores > 1U) {
            volatile tt_l1_ptr uint32_t* phase1_sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(scan_phase1_sem_id));
            noc_semaphore_wait(phase1_sem, num_total_cores - 1U);
        }

        // Reduce: counts[e] = sum over cores of shared_local_counts[c][e]
        for (uint32_t e = 0; e < e_local; ++e) {
            uint32_t total = 0;
            for (uint32_t c = 0; c < num_total_cores; ++c) {
                total += shared_local_counts[c * SHARED_SLOT_U32 + e];
            }
            counts[e] = total;
        }

        // per_core_start[c][e] = offsets[e] + sum_{c' < c} round_up(local_counts[c'][e], L1_ALIGN_U32)
        // Padding local_counts up to an L1_ALIGN_U32-multiple ensures each
        // core's cursor is aligned → byte offset (cursor*4) lands on the
        // arch's L1 write-alignment boundary (16 B / 4 uint32s on WH/BH).
        // Also compute padded expert total for offsets.
        offsets[0] = 0;
        for (uint32_t e = 0; e < e_local; ++e) {
            uint32_t running = offsets[e];
            for (uint32_t c = 0; c < num_total_cores; ++c) {
                shared_per_core_start[c * SHARED_SLOT_U32 + e] = running;
                uint32_t padded =
                    (shared_local_counts[c * SHARED_SLOT_U32 + e] + L1_ALIGN_U32_MASK) & ~L1_ALIGN_U32_MASK;
                running += padded;
            }
            // offsets[e+1] rounded up to 32-row (tile alignment for grouped)
            offsets[e + 1U] = (running + 31U) & ~31U;
        }

        // Pre-fill plan DRAM with SENTINEL in 32-entry bursts.
        for (uint32_t i = 0; i < PLAN_CHUNK; ++i) plan_stage[i] = SENTINEL;
        uint64_t plan_base_noc = get_noc_addr(0, plan_addrgen);
        for (uint32_t base = 0; base < t_cap; base += PLAN_CHUNK) {
            uint32_t n = (base + PLAN_CHUNK <= t_cap) ? PLAN_CHUNK : (t_cap - base);
            noc_async_write((uint32_t)plan_stage, plan_base_noc + base * sizeof(uint32_t), n * sizeof(uint32_t));
        }

        // Write counts and offsets to DRAM (use stage as staging)
        for (uint32_t e = 0; e < e_local; ++e) stage[e] = counts[e];
        noc_async_write((uint32_t)stage, get_noc_addr(0, cnt_addrgen), cnt_page_bytes);
        for (uint32_t e = 0; e <= e_local; ++e) stage[e] = offsets[e];
        noc_async_write((uint32_t)stage, get_noc_addr(0, off_addrgen), off_page_bytes);
        noc_async_write_barrier();

        // Signal phase 2 done on every core (incl. lead) via ONE NOC multicast
        // over the full worker grid. mcast_{sx,sy,ex,ey,num_dests_incl_self}
        // are CT constants (see ttml::mcast_sender_signal_receivers_loopback
        // in metal/common/dataflow_utils.hpp).
        uint32_t phase2_sem_addr = get_semaphore(scan_phase2_sem_id);
        volatile tt_l1_ptr uint32_t* phase2_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(phase2_sem_addr);
        mcast_sender_signal_receivers_loopback(
            phase2_sem_ptr, phase2_sem_addr, mcast_sx, mcast_sy, mcast_ex, mcast_ey, mcast_num_dests_incl_self);
    } else {
        // Non-lead cores wait for phase 2 signal.
        volatile tt_l1_ptr uint32_t* phase2_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(scan_phase2_sem_id));
        noc_semaphore_wait(phase2_sem, 1U);
    }

    // ===========================================================
    // PHASE 3: each core scatters its slice's plan entries
    // ===========================================================

    // Read my per_core_start[my_core_idx] into local cursors.
    if (my_core_idx == 0) {
        for (uint32_t e = 0; e < e_local; ++e) {
            cursors[e] = shared_per_core_start[e];
        }
    } else {
        // NOC-read from lead core's L1. Use plan_stage as 32B-aligned scratch.
        uint64_t src_noc =
            get_noc_addr(lead_core_x, lead_core_y, (uint32_t)(shared_per_core_start + my_core_idx * SHARED_SLOT_U32));
        noc_async_read(src_noc, (uint32_t)plan_stage, SHARED_SLOT_U32 * sizeof(uint32_t));
        noc_async_read_barrier();
        for (uint32_t e = 0; e < e_local; ++e) cursors[e] = plan_stage[e];
    }

    for (uint32_t e = 0; e < e_local; ++e) fill[e] = 0;

    uint64_t plan_base_noc = get_noc_addr(0, plan_addrgen);
    if (my_slice_size > 0U) {
        for (uint32_t block_start = my_slice_start; block_start < my_slice_end; block_start += block_rows) {
            uint32_t block_end = block_start + block_rows;
            if (block_end > my_slice_end)
                block_end = my_slice_end;
            // Read metadata block
            for (uint32_t row = block_start; row < block_end; ++row) {
                uint32_t local_row = row - block_start;
                noc_async_read(
                    get_noc_addr(row, md_addrgen), md_block_addr + local_row * md_aligned_page, md_aligned_page);
            }
            noc_async_read_barrier();
            // Scatter
            for (uint32_t row = block_start; row < block_end; ++row) {
                uint32_t off = (row - block_start) * MD_ROW_STRIDE_U16;
                for (uint32_t e = 0; e < e_local; ++e) {
                    for (uint32_t ki = 0; ki < k; ++ki) {
                        if ((uint32_t)md_block[off + ki] == (uint32_t)leids_u16[e]) {
                            plan_stage[e * PLAN_CHUNK + fill[e]] = row;
                            fill[e]++;
                            if (fill[e] == PLAN_CHUNK) {
                                noc_async_write(
                                    (uint32_t)(plan_stage + e * PLAN_CHUNK),
                                    plan_base_noc + cursors[e] * sizeof(uint32_t),
                                    PLAN_CHUNK * sizeof(uint32_t));
                                noc_async_write_barrier();
                                cursors[e] += PLAN_CHUNK;
                                fill[e] = 0;
                            }
                            break;
                        }
                    }
                }
            }
            noc_async_write_barrier();
        }
    }

    // Tail flush + chain signal — must run on EVERY core, including ones
    // with my_slice_size == 0, so the chained sem propagates through to
    // the lead. (fill[e]==0 → nothing to flush, just the chain hop.)
    // Core 0 flushes first (no wait), then signals core 1. Core 1..N-2
    // waits, flushes, signals next. Core N-1 waits, flushes, signals lead.
    if (my_core_idx > 0) {
        volatile tt_l1_ptr uint32_t* phase3_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(scan_phase3_sem_id));
        noc_semaphore_wait(phase3_sem, 1U);
    }
    for (uint32_t e = 0; e < e_local; ++e) {
        if (fill[e] > 0) {
            uint32_t n = fill[e];
            // Round up to L1_ALIGN_U32 uint32s — arch-specific NOC L1 write
            // alignment (16 B / 4 uint32s on WH/BH).
            uint32_t n_aligned = (n + L1_ALIGN_U32_MASK) & ~L1_ALIGN_U32_MASK;
            for (uint32_t i = n; i < n_aligned; ++i) plan_stage[e * PLAN_CHUNK + i] = SENTINEL;
            noc_async_write(
                (uint32_t)(plan_stage + e * PLAN_CHUNK),
                plan_base_noc + cursors[e] * sizeof(uint32_t),
                n_aligned * sizeof(uint32_t));
            // Advance cursor by n_aligned so it stays 4-aligned and matches
            // the per-core padded layout that offsets[e+1] was computed with.
            cursors[e] += n_aligned;
            fill[e] = 0;
        }
    }
    noc_async_write_barrier();
    // Chain: signal the next core (or lead if we're last).
    if (my_core_idx + 1U < num_total_cores) {
        uint64_t sem_noc = get_noc_addr(next_core_x, next_core_y, get_semaphore(scan_phase3_sem_id));
        noc_semaphore_inc(sem_noc, 1U);
    } else if (num_total_cores > 1U) {
        // Last core signals lead.
        uint64_t sem_noc = get_noc_addr(lead_core_x, lead_core_y, get_semaphore(scan_phase3_sem_id));
        noc_semaphore_inc(sem_noc, 1U);
    }

    // ===========================================================
    // BARRIER C: lead waits for all scatter, then signals workers
    // ===========================================================
    if (my_core_idx == 0) {
        // Wait for the last core to finish its tail flush (chained signal).
        if (num_total_cores > 1U) {
            volatile tt_l1_ptr uint32_t* phase3_sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(scan_phase3_sem_id));
            noc_semaphore_wait(phase3_sem, 1U);
        }
        // Signal plan_ready_sem on every core (incl. lead) via ONE multicast.
        uint32_t plan_ready_l1_addr = get_semaphore(plan_ready_sem_id);
        volatile tt_l1_ptr uint32_t* plan_ready_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(plan_ready_l1_addr);
        mcast_sender_signal_receivers_loopback(
            plan_ready_ptr, plan_ready_l1_addr, mcast_sx, mcast_sy, mcast_ex, mcast_ey, mcast_num_dests_incl_self);
    }

    // ===========================================================
    // WORKER PHASE: wait for plan_ready, then gather rows
    // ===========================================================
    volatile tt_l1_ptr uint32_t* plan_ready_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(plan_ready_sem_id));
    noc_semaphore_wait(plan_ready_sem, 1U);

    // Pull offsets[e_local] from DRAM so we can short-circuit reads for
    // tail tiles past the last active expert slice.
    noc_async_read(get_noc_addr(0, off_addrgen), (uint32_t)stage, (e_local + 1U) * sizeof(uint32_t));
    noc_async_read_barrier();
    const uint32_t max_active_tiles = stage[e_local] / TILE_H;

    // Existing worker-reader logic.
    cb_reserve_back(cb_plan, 1U);
    uint32_t plan_l1_addr = get_write_ptr(cb_plan);
    volatile tt_l1_ptr uint32_t* plan_l1_buf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(plan_l1_addr);

    uint32_t tile_row = my_worker_start;
    for (uint32_t step = 0; step < my_worker_count; ++step, tile_row += worker_stride) {
        const bool tile_active = tile_row < max_active_tiles;

        if (tile_active) {
            uint64_t plan_noc = get_noc_addr(0, plan_addrgen) + tile_row * TILE_H * sizeof(uint32_t);
            noc_async_read(plan_noc, plan_l1_addr, TILE_H * sizeof(uint32_t));
            noc_async_read_barrier();
        }

        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            cb_reserve_back(cb_src0, tiles_per_chunk);
            if (!tile_active) {
                cb_push_back(cb_src0, tiles_per_chunk);
                continue;
            }
            uint32_t dst = get_write_ptr(cb_src0);
            uint32_t chunk_off_bytes = chunk * hidden_chunk_bytes;
            bool is_last_chunk = (chunk == num_chunks - 1U);
            uint32_t read_bytes = is_last_chunk ? last_chunk_bytes : hidden_chunk_bytes;
            uint32_t pad_bytes = hidden_chunk_bytes - read_bytes;

            for (uint32_t r = 0; r < TILE_H; ++r) {
                uint32_t src = plan_l1_buf[r];
                uint32_t row_dst = dst + r * hidden_chunk_bytes;
                if (src == SENTINEL) {
                    volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(row_dst);
                    for (uint32_t i = 0; i < hidden_chunk_bytes / 2U; ++i) p[i] = 0U;
                } else {
                    uint64_t row_noc = get_noc_addr(src, dispatched_addrgen) + chunk_off_bytes;
                    noc_async_read(row_noc, row_dst, read_bytes);
                    if (pad_bytes > 0U) {
                        volatile tt_l1_ptr uint16_t* p =
                            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(row_dst + read_bytes);
                        for (uint32_t i = 0; i < pad_bytes / 2U; ++i) p[i] = 0U;
                    }
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_src0, tiles_per_chunk);
        }
    }
}
