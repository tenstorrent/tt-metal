// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Writer (BRISC) for moe_ungroup. Per core:
//   1. Pre-zero my range of `ungrouped` DRAM rows.
//   2. Local handshake with NCRISC reader: signal "brisc_done", wait
//      "brisc_release". Reader does the cross-core mcast barrier on NOC_0
//      (matching moe_group's pattern of putting mcast in the NCRISC kernel).
//   3. For e in 0..E_local:
//        - Process this core's tile-row slice of expert e (chunks, scale-write
//          for e==0, RMW for e>0).
//        - If e+1 < E_local: another local handshake with NCRISC.
//
// NOTE on performance: the per-element bf16 RMW happens in BRISC scalar
// ops. For wide H, this is the bottleneck. Future optimization: fp32
// accumulator across experts to remove the K-1 intermediate bf16 round-trips.

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/moe_ungroup/device/kernels/moe_ungroup_utils.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/matmul_dataflow_common.hpp"

constexpr uint32_t cb_out0 = tt::CBIndex::c_2;
constexpr uint32_t cb_scratch = tt::CBIndex::c_4;      // BRISC scratch (zero buf, plan slice, md, sc, rmw_buf)
constexpr uint32_t cb_w = tt::CBIndex::c_5;            // 32×32 broadcast weight tile pushed to compute
constexpr uint32_t cb_existing_rm = tt::CBIndex::c_6;  // row-major existing rows from ungrouped

constexpr uint32_t h = get_compile_time_arg_val(0);
constexpr uint32_t num_chunks = get_compile_time_arg_val(1);
constexpr uint32_t hidden_chunk_bytes = get_compile_time_arg_val(2);
constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(3);
constexpr uint32_t last_chunk_bytes = get_compile_time_arg_val(4);
constexpr uint32_t total_rows = get_compile_time_arg_val(5);  // D*B*S
constexpr uint32_t k = get_compile_time_arg_val(6);
constexpr uint32_t e_local = get_compile_time_arg_val(7);
constexpr uint32_t num_total_cores = get_compile_time_arg_val(8);
constexpr uint32_t brisc_done_sem_id = get_compile_time_arg_val(9);
constexpr uint32_t brisc_release_sem_id = get_compile_time_arg_val(10);

constexpr auto ungrouped_args = TensorAccessorArgs<11>();
constexpr auto plan_args = TensorAccessorArgs<ungrouped_args.next_compile_time_args_offset()>();
constexpr auto offsets_args = TensorAccessorArgs<plan_args.next_compile_time_args_offset()>();
constexpr auto counts_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
constexpr auto metadata_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
constexpr auto scores_args = TensorAccessorArgs<metadata_args.next_compile_time_args_offset()>();
constexpr auto leids_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();

constexpr uint32_t md_aligned_page = decltype(metadata_args)::AlignedPageSize;
constexpr uint32_t sc_aligned_page = decltype(scores_args)::AlignedPageSize;
constexpr uint32_t leids_aligned_page = decltype(leids_args)::AlignedPageSize;
constexpr uint32_t off_page_bytes = decltype(offsets_args)::AlignedPageSize;
constexpr uint32_t cnt_page_bytes = decltype(counts_args)::AlignedPageSize;

constexpr uint32_t TILE_H = 32U;
constexpr uint32_t TILE_W = 32U;
constexpr uint32_t SENTINEL = 0xFFFFFFFFU;
constexpr uint32_t MD_ROW_STRIDE_U16 = md_aligned_page / sizeof(uint16_t);
constexpr uint32_t SC_ROW_STRIDE_U16 = sc_aligned_page / sizeof(uint16_t);

// Local BRISC<->NCRISC handshake on the same core (no NOC; both RISCs share L1).
// Pattern: BRISC sets brisc_done = 1, polls brisc_release until 1, resets it.
// NCRISC observes brisc_done == 1, runs the cross-core mcast barrier, sets
// brisc_release = 1 (and resets brisc_done).
inline void brisc_signal_done_wait_release() {
    volatile tt_l1_ptr uint32_t* brisc_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(brisc_done_sem_id));
    volatile tt_l1_ptr uint32_t* brisc_release =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(brisc_release_sem_id));
    *brisc_done = 1U;
    while (*brisc_release == 0U) {
    }
    *brisc_release = 0U;
}

void kernel_main() {
    const uint32_t ungrouped_addr = get_arg_val<uint32_t>(0);
    const uint32_t plan_addr = get_arg_val<uint32_t>(1);
    const uint32_t offsets_addr = get_arg_val<uint32_t>(2);
    const uint32_t counts_addr = get_arg_val<uint32_t>(3);
    const uint32_t metadata_addr = get_arg_val<uint32_t>(4);
    const uint32_t scores_addr = get_arg_val<uint32_t>(5);
    const uint32_t leids_addr = get_arg_val<uint32_t>(6);
    const uint32_t my_core_idx = get_arg_val<uint32_t>(7);

    const auto ungrouped_addrgen = TensorAccessor(ungrouped_args, ungrouped_addr, h * 2U);
    const auto plan_addrgen = TensorAccessor(plan_args, plan_addr);
    const auto offsets_addrgen = TensorAccessor(offsets_args, offsets_addr);
    const auto counts_addrgen = TensorAccessor(counts_args, counts_addr);
    const auto md_addrgen = TensorAccessor(metadata_args, metadata_addr);
    const auto sc_addrgen = TensorAccessor(scores_args, scores_addr);
    const auto leids_addrgen = TensorAccessor(leids_args, leids_addr);

    // ---------------------------------------------------------------
    // L1 scratch layout in cb_scratch (BRISC):
    //   [zero_buf (h*2 bytes)]
    //   [offsets_buf ((e_local+1)*4 bytes, aligned 32)]
    //   [counts_buf  (e_local*4 bytes,     aligned 32)]
    //   [leids_buf   (e_local*2 bytes,     aligned 32)]
    //   [plan_buf    (32*4 bytes)]
    //   [md_buf      (32*K*2 bytes per row, batched as needed)]
    //   [sc_buf      (32*K*2 bytes)]
    //   [w_buf       (32*2 bytes — bf16 weights, copied straight from sc_buf)]
    //   [rmw_buf     (max(hidden_chunk_bytes) bytes per row staging)]
    // ---------------------------------------------------------------
    cb_reserve_back(cb_scratch, 1U);
    uint32_t scratch = get_write_ptr(cb_scratch);
    uint32_t off = 0U;

    uint32_t zero_buf_addr = scratch + off;
    off += ((h * 2U) + 31U) & ~31U;

    uint32_t offsets_buf_addr = scratch + off;
    off += (((e_local + 1U) * 4U) + 31U) & ~31U;
    volatile tt_l1_ptr uint32_t* offsets_buf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(offsets_buf_addr);

    uint32_t counts_buf_addr = scratch + off;
    off += ((e_local * 4U) + 31U) & ~31U;
    volatile tt_l1_ptr uint32_t* counts_buf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_buf_addr);

    uint32_t leids_buf_addr = scratch + off;
    off += ((e_local * 2U) + 31U) & ~31U;
    volatile tt_l1_ptr uint16_t* leids_buf = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(leids_buf_addr);

    uint32_t plan_buf_addr = scratch + off;
    off += (32U * 4U + 31U) & ~31U;
    volatile tt_l1_ptr uint32_t* plan_buf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(plan_buf_addr);

    uint32_t md_buf_addr = scratch + off;
    off += (32U * md_aligned_page + 31U) & ~31U;
    volatile tt_l1_ptr uint16_t* md_buf = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(md_buf_addr);

    uint32_t sc_buf_addr = scratch + off;
    off += (32U * sc_aligned_page + 31U) & ~31U;
    volatile tt_l1_ptr uint16_t* sc_buf = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sc_buf_addr);

    uint32_t w_buf_addr = scratch + off;
    off += (32U * sizeof(uint16_t) + 31U) & ~31U;
    volatile tt_l1_ptr uint16_t* w_buf = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(w_buf_addr);

    // OPT 1: 32 contiguous slots of hidden_chunk_bytes each — one per row of
    // the tile. Lets us issue all 32 row reads/writes async and barrier once.
    uint32_t stage_buf_addr = scratch + off;
    // (off is unused after this point; sizing handled by program_factory.)
    cb_push_back(cb_scratch, 1U);

    // Pre-zero: tokens whose top-K is entirely non-local never get touched by
    // any expert iteration, so without pre-zero those rows would read uninit
    // DRAM. With pre-zero plus the per-row first-local detection below, every
    // output row is correct: tokens with no local experts stay zero; tokens
    // with K>=1 local experts get a pure scaled write from their first local
    // expert and += from the rest.
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
    // Read offsets, counts, leids into L1 (one-shot).
    // ---------------------------------------------------------------
    noc_async_read(get_noc_addr(0, offsets_addrgen), offsets_buf_addr, off_page_bytes);
    noc_async_read(get_noc_addr(0, counts_addrgen), counts_buf_addr, cnt_page_bytes);
    noc_async_read(get_noc_addr(0, leids_addrgen), leids_buf_addr, leids_aligned_page);
    noc_async_read_barrier();

    // ---------------------------------------------------------------
    // Per-expert loop.
    // ---------------------------------------------------------------
    for (uint32_t e = 0; e < e_local; ++e) {
        uint32_t expert_start_tr = offsets_buf[e] / TILE_H;
        uint32_t expert_total_tr = (offsets_buf[e + 1U] - offsets_buf[e]) / TILE_H;
        auto slice = ttml::metal::moe_ungroup::slice_for_core(expert_total_tr, num_total_cores, my_core_idx);
        uint32_t my_start_in_e = slice.start;
        uint32_t my_real_count = slice.count;

        uint32_t leid_e = leids_buf[e];

        // is_first[r]: true if expert e is the FIRST local expert in
        // metadata[plan[r], :K] (smallest leids index match). Drives the
        // pure-write vs RMW choice without needing pre-zero — the first
        // expert to touch a row writes, the rest accumulate.
        bool is_first[TILE_H];

        for (uint32_t step = 0; step < my_real_count; ++step) {
            uint32_t tr_global = expert_start_tr + my_start_in_e + step;

            // Pre-fetch plan slice + metadata + scores.
            {
                uint64_t plan_noc = get_noc_addr(0, plan_addrgen) + tr_global * TILE_H * sizeof(uint32_t);
                noc_async_read(plan_noc, plan_buf_addr, TILE_H * sizeof(uint32_t));
                noc_async_read_barrier();

                for (uint32_t r = 0; r < TILE_H; ++r) {
                    uint32_t flat = plan_buf[r];
                    uint32_t row_idx = tr_global * TILE_H + r;
                    if (flat == SENTINEL) {
                        continue;
                    }
                    noc_async_read(get_noc_addr(flat, md_addrgen), md_buf_addr + r * md_aligned_page, md_aligned_page);
                    noc_async_read(get_noc_addr(flat, sc_addrgen), sc_buf_addr + r * sc_aligned_page, sc_aligned_page);
                }
                noc_async_read_barrier();

                // Compute per-row weight w[r] AND is_first[r].
                for (uint32_t r = 0; r < TILE_H; ++r) {
                    uint32_t flat = plan_buf[r];
                    uint32_t row_idx = tr_global * TILE_H + r;
                    if (flat == SENTINEL) {
                        w_buf[r] = 0U;
                        is_first[r] = false;
                        continue;
                    }
                    uint32_t md_off = r * MD_ROW_STRIDE_U16;
                    uint32_t sc_off = r * SC_ROW_STRIDE_U16;
                    uint32_t k_slot = 0U;
                    bool found = false;
                    for (uint32_t ki = 0; ki < k; ++ki) {
                        if (static_cast<uint32_t>(md_buf[md_off + ki]) == leid_e) {
                            k_slot = ki;
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        w_buf[r] = 0U;
                        is_first[r] = false;
                        continue;
                    }
                    w_buf[r] = sc_buf[sc_off + k_slot];
                    // First-local check: is this the first leid index `e` whose
                    // value appears in metadata[flat, :K]? Scan other metadata
                    // entries for any earlier local expert (leids[0..e-1]).
                    bool has_earlier = false;
                    for (uint32_t ki = 0; ki < k; ++ki) {
                        if (ki == k_slot)
                            continue;
                        uint16_t md_val = md_buf[md_off + ki];
                        for (uint32_t eprime = 0; eprime < e; ++eprime) {
                            if (md_val == leids_buf[eprime]) {
                                has_earlier = true;
                                break;
                            }
                        }
                        if (has_earlier)
                            break;
                    }
                    is_first[r] = !has_earlier;
                }
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
                cb_reserve_back(cb_existing_rm, TILE_H);
                uint32_t existing_l1 = get_write_ptr(cb_existing_rm);
                for (uint32_t r = 0; r < TILE_H; ++r) {
                    uint32_t flat = plan_buf[r];
                    uint32_t row_idx = tr_global * TILE_H + r;
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
                cb_push_back(cb_existing_rm, TILE_H);

                // (3) Wait for compute's combined+untilized output.
                cb_wait_front(cb_out0, tiles_per_chunk);

                // (4) NOC-write cb_out0 rows back to ungrouped DRAM. Pure data
                // movement — no scalar arithmetic.
                {
                    uint32_t src_l1 = get_read_ptr(cb_out0);
                    for (uint32_t r = 0; r < TILE_H; ++r) {
                        uint32_t flat = plan_buf[r];
                        uint32_t row_idx = tr_global * TILE_H + r;
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
