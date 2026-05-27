// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer (BRISC) for moe_ungroup. Per core:
//   1. Pre-zero my range of `ungrouped` DRAM rows. With pre-zero, every
//      expert iteration can do the same RMW path — the first expert to
//      touch a row reads zero and effectively writes.
//   2. Local handshake with NCRISC reader: signal "brisc_done", wait
//      "brisc_release". Reader does the cross-core mcast barrier on NOC_0
//      (matching moe_group's pattern of putting mcast in the NCRISC kernel).
//   3. For e in 0..E_local:
//        - Process this core's tile-row slice of expert e: read existing
//          ungrouped rows, hand them to compute which adds the scaled
//          expert contribution, write back.
//        - If e+1 < E_local: another local handshake with NCRISC.
//
// With moe_group emitting `grouped_scores[T_cap]` (= scores[plan[i], k_slot])
// directly, the writer no longer scans metadata to find k_slot or look up
// the scalar. Per tile-row, it reads a 32-entry bf16 slice of grouped_scores
// straight into w_buf — one DMA, zero scalar work.

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/moe_ungroup/device/kernels/moe_ungroup_utils.hpp"

constexpr uint32_t cb_out0 = tt::CBIndex::c_2;
constexpr uint32_t cb_scratch = tt::CBIndex::c_4;      // BRISC scratch (zero buf, plan slice, gs slice, rmw_buf)
constexpr uint32_t cb_w = tt::CBIndex::c_5;            // 32×32 broadcast weight tile pushed to compute
constexpr uint32_t cb_existing_rm = tt::CBIndex::c_6;  // row-major existing rows from ungrouped

constexpr uint32_t h = get_compile_time_arg_val(0);
constexpr uint32_t num_chunks = get_compile_time_arg_val(1);
constexpr uint32_t hidden_chunk_bytes = get_compile_time_arg_val(2);
constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(3);
constexpr uint32_t last_chunk_bytes = get_compile_time_arg_val(4);
constexpr uint32_t total_rows = get_compile_time_arg_val(5);  // D*B*S
constexpr uint32_t e_local = get_compile_time_arg_val(6);
constexpr uint32_t num_total_cores = get_compile_time_arg_val(7);
constexpr uint32_t brisc_done_sem_id = get_compile_time_arg_val(8);
constexpr uint32_t brisc_release_sem_id = get_compile_time_arg_val(9);
constexpr uint32_t l1_align = get_compile_time_arg_val(10);

constexpr auto ungrouped_args = TensorAccessorArgs<11>();
constexpr auto plan_args = TensorAccessorArgs<ungrouped_args.next_compile_time_args_offset()>();
constexpr auto offsets_args = TensorAccessorArgs<plan_args.next_compile_time_args_offset()>();
constexpr auto gs_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();

constexpr uint32_t off_page_bytes = decltype(offsets_args)::AlignedPageSize;
constexpr uint32_t ungrouped_aligned_page = decltype(ungrouped_args)::AlignedPageSize;

constexpr uint32_t SENTINEL = 0xFFFFFFFFU;

// Local BRISC<->NCRISC handshake on the same core (no NOC; both RISCs share L1).
// Pattern: BRISC sets brisc_done = 1, polls brisc_release until 1, resets it.
// NCRISC observes brisc_done == 1, runs the cross-core mcast barrier, sets
// brisc_release = 1 (and resets brisc_done).
inline void brisc_signal_done_wait_release() {
    volatile tt_l1_ptr uint32_t* brisc_done = get_sem_ptr(brisc_done_sem_id);
    volatile tt_l1_ptr uint32_t* brisc_release = get_sem_ptr(brisc_release_sem_id);
    *brisc_done = 1U;
    do {
        invalidate_l1_cache();
    } while ((*brisc_release) == 0U);
    *brisc_release = 0U;
}

void kernel_main() {
    const uint32_t ungrouped_addr = get_arg_val<uint32_t>(0);
    const uint32_t plan_addr = get_arg_val<uint32_t>(1);
    const uint32_t offsets_addr = get_arg_val<uint32_t>(2);
    const uint32_t gs_addr = get_arg_val<uint32_t>(3);
    const uint32_t my_core_idx = get_arg_val<uint32_t>(4);

    const auto ungrouped_addrgen = TensorAccessor(ungrouped_args, ungrouped_addr, ungrouped_aligned_page);
    const auto plan_addrgen = TensorAccessor(plan_args, plan_addr);
    const auto offsets_addrgen = TensorAccessor(offsets_args, offsets_addr);
    const auto gs_addrgen = TensorAccessor(gs_args, gs_addr);

    // ---------------------------------------------------------------
    // L1 scratch layout in cb_scratch (BRISC):
    //   [zero_buf    (h*2 bytes)]
    //   [offsets_buf (offset TensorAccessor page bytes)]
    //   [plan_buf    (32*4 bytes)]
    //   [w_buf       (32*2 bytes — bf16, read directly from grouped_scores)]
    // ---------------------------------------------------------------
    cb_reserve_back(cb_scratch, 1U);
    uint32_t scratch = get_write_ptr(cb_scratch);
    uint32_t off = 0U;

    uint32_t zero_buf_addr = scratch + off;
    // offsets_buf is read with a TensorAccessor page. BH requires the L1
    // destination start to be aligned to that page size (64B for small rows).
    off += round_up(h * 2U, off_page_bytes);

    uint32_t offsets_buf_addr = scratch + off;
    // The DMA below reads a full TensorAccessor page. On BH, row-major uint32
    // DRAM pages are 64B-aligned; reserving only L1/32B here corrupts plan/w_buf.
    off += off_page_bytes;
    volatile tt_l1_ptr uint32_t* offsets_buf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(offsets_buf_addr);

    uint32_t plan_buf_addr = scratch + off;
    off += round_up(32U * 4U, l1_align);
    volatile tt_l1_ptr uint32_t* plan_buf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(plan_buf_addr);

    uint32_t w_buf_addr = scratch + off;
    off += round_up(32U * sizeof(uint16_t), l1_align);
    volatile tt_l1_ptr uint16_t* w_buf = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(w_buf_addr);

    cb_push_back(cb_scratch, 1U);

    // Pre-zero: tokens whose top-K is entirely non-local never get touched by
    // any expert iteration, so without pre-zero those rows would read uninit
    // DRAM. With pre-zero, every expert iteration can do the same RMW path
    // (the first to touch a row reads zero and effectively writes), and tokens
    // with no local experts stay zero.
    fill_zeros_async(zero_buf_addr, h * 2U);
    noc_async_read_barrier();
    auto zero_slice = ttml::metal::moe_ungroup::slice_for_core(total_rows, num_total_cores, my_core_idx);
    for (uint32_t row = zero_slice.start; row < zero_slice.start + zero_slice.count; ++row) {
        uint64_t dst_noc = get_noc_addr(row, ungrouped_addrgen);
        noc_async_write(zero_buf_addr, dst_noc, h * 2U);
    }
    noc_async_write_barrier();
    brisc_signal_done_wait_release();

    // ---------------------------------------------------------------
    // Read offsets into L1 (one-shot). leids no longer needed —
    // grouped_scores already encodes the right scalar per active row.
    // ---------------------------------------------------------------
    noc_async_read(get_noc_addr(0, offsets_addrgen), offsets_buf_addr, off_page_bytes);
    noc_async_read_barrier();

    // ---------------------------------------------------------------
    // Per-expert loop.
    // ---------------------------------------------------------------
    for (uint32_t e = 0; e < e_local; ++e) {
        auto slice = ttml::metal::moe_ungroup::expert_slice_for_core(
            offsets_buf, e, tt::constants::TILE_HEIGHT, num_total_cores, my_core_idx);
        uint32_t my_real_count = slice.my_count;

        for (uint32_t step = 0; step < my_real_count; ++step) {
            uint32_t tr_global = slice.my_start_tr_global + step;

            // Pre-fetch plan slice + grouped_scores slice. Both are
            // TILE_HEIGHT-entry contiguous slices of [T_cap]-sized
            // ROW_MAJOR DRAM tensors, indexed by tr_global * TILE_HEIGHT.
            //
            // grouped_scores[tr_global*32 .. tr_global*32 + 32) goes
            // straight into w_buf — moe_group already pre-baked
            // scores[plan[i], k_slot] per row, so no metadata scan, no
            // k_slot lookup. SENTINEL plan rows have grouped_scores = 0,
            // which produces a no-op multiply downstream.
            //
            // Pre-zero of the output buffer makes the first-touch RMW read 0
            // for every row, so we don't need a pure-write fast path — every
            // expert just does scaled += and the first one effectively writes.
            {
                uint64_t plan_noc =
                    get_noc_addr(0, plan_addrgen) + tr_global * tt::constants::TILE_HEIGHT * sizeof(uint32_t);
                noc_async_read(plan_noc, plan_buf_addr, tt::constants::TILE_HEIGHT * sizeof(uint32_t));
                uint64_t gs_noc =
                    get_noc_addr(0, gs_addrgen) + tr_global * tt::constants::TILE_HEIGHT * sizeof(uint16_t);
                noc_async_read(gs_noc, w_buf_addr, tt::constants::TILE_HEIGHT * sizeof(uint16_t));
                noc_async_read_barrier();
            }

            // Per chunk:
            //   1. Build w_tile (32×32 broadcast w[r]) → cb_w
            //   2. Read existing rows from ungrouped[plan[r]] into cb_existing_rm
            //   3. cb_wait_front(cb_out0) — compute has already done mul + add + untilize
            //   4. NOC-write cb_out0 rows back to ungrouped[plan[r]]
            // BRISC does ZERO scalar arithmetic; all math is on FPU/SFPU.
            for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                bool is_last_chunk = (chunk == num_chunks - 1U);
                uint32_t write_bytes = is_last_chunk ? last_chunk_bytes : hidden_chunk_bytes;

                // (1) Build w_tile and push to cb_w.
                cb_reserve_back(cb_w, 1U);
                {
                    volatile tt_l1_ptr uint16_t* w_tile =
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_w));
                    for (uint32_t face = 0; face < 4U; ++face) {
                        uint32_t face_r_offset = (face / 2U) * 16U;
                        uint32_t face_base = face * 256U;
                        for (uint32_t in_r = 0; in_r < 16U; ++in_r) {
                            uint32_t r = face_r_offset + in_r;
                            uint16_t w_bf = w_buf[r];
                            uint32_t row_off = face_base + in_r * 16U;
                            for (uint32_t in_c = 0; in_c < 16U; ++in_c) {
                                w_tile[row_off + in_c] = w_bf;
                            }
                        }
                    }
                }
                cb_push_back(cb_w, 1U);

                // (2) Read existing rows from ungrouped DRAM into cb_existing_rm.
                // CB has 32 pages per chunk (asymmetric, one page per row).
                // For the last chunk, write_bytes < hidden_chunk_bytes when h
                // isn't tile-aligned — zero-pad the L1 tail so tilize sees
                // zeros in the partial last tile column (otherwise it'd read
                // uninit bytes that round-trip into the writer's RMW).
                cb_reserve_back(cb_existing_rm, tt::constants::TILE_HEIGHT);
                uint32_t existing_l1 = get_write_ptr(cb_existing_rm);
                uint32_t pad_bytes = hidden_chunk_bytes - write_bytes;
                for (uint32_t r = 0; r < tt::constants::TILE_HEIGHT; ++r) {
                    uint32_t flat = plan_buf[r];
                    uint32_t row_buf = existing_l1 + r * hidden_chunk_bytes;
                    if (flat == SENTINEL) {
                        // Skipped row — fill via NOC DMA so tilize sees zeros.
                        fill_zeros_async(row_buf, hidden_chunk_bytes);
                        continue;
                    }
                    uint64_t dst_noc = get_noc_addr(flat, ungrouped_addrgen) + chunk * hidden_chunk_bytes;
                    noc_async_read(dst_noc, row_buf, write_bytes);
                }
                noc_async_read_barrier();
                // Zero the partial-last-tile tail AFTER the reads complete:
                // doing it before the barrier would race with NOC writes that
                // can land slightly after their request size due to packet
                // alignment, overwriting the pad with neighbour-row bytes.
                if (pad_bytes > 0U) {
                    for (uint32_t r = 0; r < tt::constants::TILE_HEIGHT; ++r) {
                        uint32_t flat = plan_buf[r];
                        if (flat == SENTINEL) {
                            continue;
                        }
                        volatile tt_l1_ptr uint16_t* tail = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                            existing_l1 + r * hidden_chunk_bytes + write_bytes);
                        for (uint32_t i = 0; i < pad_bytes / sizeof(uint16_t); ++i) {
                            tail[i] = 0U;
                        }
                    }
                }
                cb_push_back(cb_existing_rm, tt::constants::TILE_HEIGHT);

                // (3) Wait for compute's combined+untilized output.
                cb_wait_front(cb_out0, tiles_per_chunk);

                // (4) NOC-write cb_out0 rows back to ungrouped DRAM. Pure data
                // movement — no scalar arithmetic.
                {
                    uint32_t src_l1 = get_read_ptr(cb_out0);
                    for (uint32_t r = 0; r < tt::constants::TILE_HEIGHT; ++r) {
                        uint32_t flat = plan_buf[r];
                        if (flat == SENTINEL) {
                            continue;
                        }
                        uint64_t dst_noc = get_noc_addr(flat, ungrouped_addrgen) + chunk * hidden_chunk_bytes;
                        noc_async_write(src_l1 + r * hidden_chunk_bytes, dst_noc, write_bytes);
                    }
                    noc_async_write_barrier();
                }
                cb_pop_front(cb_out0, tiles_per_chunk);
            }
        }

        // Inter-expert barrier (skip after last expert): handshake with NCRISC,
        // which runs the cross-core mcast on NOC_0.
        if (e + 1U < e_local) {
            brisc_signal_done_wait_release();
        }
    }
}
