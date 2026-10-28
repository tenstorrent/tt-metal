// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader of the decode step's flash-decode SDPA (tt/v41/flash_sdpa.py; fsdpa_common.hpp has the layout). Phase 0:
// the live page ([1, 1, 1, S] uint32: a row is live unless its word is LIVE_SENTINEL; the c1 step's ``c1put`` page)
// -> the live rows, handed to the compute and the writer as the plan page.
// Then for each of this core's units (row, j): the row's H query rows, the keys [j * KPU, (j + 1) * KPU) of its
// index row (the valid count up to the first sentinel; a unit with none is skipped by all three kernels), the K rows
// gathered as sparse_sdpa's reader gathers them (the packed scaled-FP8 slab with its scale rows scattered into the
// fp32 broadcast tiles, or the bf16 rows; the rows past the valid count zero) and the partial-boundary mask tile.
// Then for each of this core's merge units (row, v): the row's index row -> its valid count -> the number of
// partials to expect, the wait on the merge semaphore the units' writers increment once their partials are in
// this core's merge CBs, and the push of those pages to the compute.
//
// compile_time_args = fsdpa::ct, then the accessors of q, kv, idx, live
// common_runtime_args = [q_addr, kv_addr, idx_addr, live_addr]; runtime_args = [core]

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/sparse_sdpa_gather.hpp"
#include "models/demos/deepseek_v3_d_p/tt/v41/kernels/fsdpa_common.hpp"

using namespace fsdpa;

FORCE_INLINE uint32_t rows_in_slab(uint32_t valid, uint32_t row_base) {
    if (valid <= row_base) {
        return 0;
    }
    const uint32_t remaining = valid - row_base;
    return remaining < tt::constants::TILE_HEIGHT ? remaining : tt::constants::TILE_HEIGHT;
}

// One control page: word 0 (a unit's valid count, a merge's expected partials).
FORCE_INLINE void push_word(experimental::CB& c, uint32_t value) {
    c.reserve_back(1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(c.get_write_ptr())[0] = value;
    c.push_back(1);
}

// The plan page: the live row count and the live rows.
FORCE_INLINE void push_plan(experimental::CB& c, uint32_t n_live, volatile tt_l1_ptr uint32_t* live) {
    c.reserve_back(1);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(c.get_write_ptr());
    p[PLAN_N_LIVE] = n_live;
    for (uint32_t i = 0; i < n_live; ++i) {
        p[PLAN_LIVE + i] = live[i];
    }
    c.push_back(1);
}

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(ct::H);
    constexpr uint32_t S = get_compile_time_arg_val(ct::S);
    constexpr uint32_t W = get_compile_time_arg_val(ct::W);
    constexpr uint32_t D = get_compile_time_arg_val(ct::D);
    constexpr uint32_t V = get_compile_time_arg_val(ct::V);
    constexpr uint32_t KPU = get_compile_time_arg_val(ct::KPU);
    constexpr uint32_t UMAX = get_compile_time_arg_val(ct::UMAX);
    constexpr uint32_t N = get_compile_time_arg_val(ct::N);
    constexpr bool scaled_kv = get_compile_time_arg_val(ct::SCALED_KV) != 0;
    constexpr uint32_t packed_row_bytes = get_compile_time_arg_val(ct::PACKED_ROW_BYTES);
    constexpr uint32_t q_elem_bytes = get_compile_time_arg_val(ct::Q_ELEM_BYTES);
    constexpr uint32_t VT = get_compile_time_arg_val(ct::VT);
    constexpr uint32_t scale_blocks = get_compile_time_arg_val(ct::SCALE_BLOCKS);
    constexpr uint32_t k_row_bytes = get_compile_time_arg_val(ct::K_ROW_BYTES);
    constexpr uint32_t live_sentinel = get_compile_time_arg_val(ct::LIVE_SENTINEL);
    constexpr uint32_t HQ = get_compile_time_arg_val(ct::HQ);
    static_assert(HQ <= H, "the heads read are the tile's");
    constexpr uint32_t Skt = KPU / tt::constants::TILE_HEIGHT;
    constexpr uint32_t q_row_bytes = D * q_elem_bytes;
    constexpr uint32_t idx_row_bytes = W * 4;
    constexpr uint32_t idx_row_bytes_aligned = (idx_row_bytes + 63) / 64 * 64;
    constexpr uint32_t live_page_bytes_aligned = (S * 4 + 63) / 64 * 64;
    constexpr uint32_t latent_row_bytes = V;
    constexpr uint32_t plan_words = 16 + S;

    constexpr auto q_args = TensorAccessorArgs<ct::END, 0>();
    constexpr auto kv_args = TensorAccessorArgs<q_args.next_compile_time_args_offset(), 0>();
    constexpr auto idx_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset(), 0>();
    constexpr auto live_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset(), 0>();
    const uint32_t q_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t kv_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t live_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t core = get_arg_val<uint32_t>(0);
    const auto q = TensorAccessor(q_args, q_addr);
    const auto kv = TensorAccessor(kv_args, kv_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);
    const auto live_page = TensorAccessor(live_args, live_addr);

    Noc noc;
    const UnicastEndpoint local_l1;
    experimental::CB q_cb(cb_q_rm), k_cb(cb_k_rm), mask_part_cb(cb_mask_part), ctrl_c(cb_ctrl_c), ctrl_w(cb_ctrl_w);
    experimental::CB scale_bcast_cb(cb_k_scale_bcast), scratch_cb(cb_rscratch);
    experimental::CB pm_cb(cb_pm), ps_cb(cb_ps), pa_cb(cb_pa);

    // scratch: the live page (64-aligned), then one index row (64-aligned), then the live list
    scratch_cb.reserve_back(1);
    const uint32_t scratch = scratch_cb.get_write_ptr();
    const uint32_t page_l1 = scratch;
    const uint32_t row_l1 = scratch + live_page_bytes_aligned;
    volatile tt_l1_ptr uint32_t* page_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_l1);
    volatile tt_l1_ptr uint32_t* row_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row_l1);
    volatile tt_l1_ptr uint32_t* live = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row_l1 + idx_row_bytes_aligned);

    // ---- phase 0: the live rows ----
    noc.async_read(live_page, local_l1, S * 4, {.page_id = 0}, {.addr = page_l1});
    noc.async_read_barrier();
    constexpr uint32_t NSEM = get_compile_time_arg_val(ct::NSEM);
    constexpr uint32_t max_live = NSEM * N / VT;  // the merge slots a core bound the live rows (flash_sdpa.py)
    uint32_t n_live = 0;
    for (uint32_t s = 0; s < S && n_live < max_live; ++s) {
        if (page_words[s] != live_sentinel) {
            live[n_live++] = s;
        }
    }
    push_plan(ctrl_c, n_live, live);
    push_plan(ctrl_w, n_live, live);

    // ---- the units ----
    const uint32_t total_units = n_live * UMAX;
    for (uint32_t u = core; u < total_units; u += N) {
        const uint32_t i = u / UMAX, j = u % UMAX, r = live[i];
        // the unit's index slice -> row_ptr[0..KPU)
        noc.async_read(idx, local_l1, KPU * 4, {.page_id = r, .offset_bytes = j * KPU * 4}, {.addr = row_l1});
        noc.async_read_barrier();
        uint32_t valid = 0;
        while (valid < KPU && row_ptr[valid] != SENTINEL) {
            ++valid;
        }
        push_word(ctrl_c, valid);
        push_word(ctrl_w, valid);
        if (valid == 0) {
            continue;
        }
        // Q: the row's HQ real head rows (page h * S + r); the padded heads' rows zero
        q_cb.reserve_back(H);
        for (uint32_t h = 0; h < HQ; ++h) {
            noc.async_read(q, q_cb, q_row_bytes, {.page_id = h * S + r}, {.offset_bytes = h * q_row_bytes});
        }
        if constexpr (HQ < H) {
            noc.async_write_zeros(q_cb, (H - HQ) * q_row_bytes, {.offset_bytes = HQ * q_row_bytes});
            noc.write_zeros_l1_barrier();
        }
        noc.async_read_barrier();
        q_cb.push_back(H);
        if constexpr (scaled_kv) {
            for (uint32_t row_tile = 0; row_tile < Skt; ++row_tile) {
                const uint32_t row_base = row_tile * tt::constants::TILE_HEIGHT;
                const uint32_t rows = rows_in_slab(valid, row_base);
                k_cb.reserve_back(tt::constants::TILE_HEIGHT);
                scale_bcast_cb.reserve_back(scale_blocks);
                const uint32_t k_write_ptr = k_cb.get_write_ptr();
                const uint32_t scale_write_ptr = scale_bcast_cb.get_write_ptr();
                if (rows > 0) {
                    sparse_sdpa::trid_ring_gather<false, 1, 1, 0, 0, sparse_sdpa::SCALED_K_TRID_RING>(
                        noc, kv, k_write_ptr, row_ptr, row_base, 0, rows, packed_row_bytes, 0);
                    sparse_sdpa::scatter_packed_scales<scale_blocks>(
                        k_write_ptr, scale_write_ptr, 0, rows, packed_row_bytes, latent_row_bytes);
                }
                if (rows < tt::constants::TILE_HEIGHT) {
                    noc.async_write_zeros(
                        k_cb,
                        (tt::constants::TILE_HEIGHT - rows) * packed_row_bytes,
                        {.offset_bytes = rows * packed_row_bytes});
                    noc.write_zeros_l1_barrier();
                }
                sparse_sdpa::clear_scale_rows<scale_blocks>(scale_write_ptr, rows, tt::constants::TILE_HEIGHT);
                k_cb.push_back(tt::constants::TILE_HEIGHT);
                scale_bcast_cb.push_back(scale_blocks);
            }
        } else {
            k_cb.reserve_back(KPU);
            const uint32_t k_write_ptr = k_cb.get_write_ptr();
            sparse_sdpa::trid_ring_gather<false, 1, 1, 0, 0, sparse_sdpa::K_TRID_RING>(
                noc, kv, k_write_ptr, row_ptr, 0, 0, valid, k_row_bytes, 0);
            if (valid < KPU) {
                noc.async_write_zeros(k_cb, (KPU - valid) * k_row_bytes, {.offset_bytes = valid * k_row_bytes});
                noc.write_zeros_l1_barrier();
            }
            k_cb.push_back(KPU);
        }
        // the partial-boundary mask tile: row 0, -inf at the columns past valid % 32 (sparse_sdpa's reader)
        if (valid % tt::constants::TILE_WIDTH != 0) {
            mask_part_cb.reserve_back(1);
            const uint32_t first_masked_col = valid % tt::constants::TILE_WIDTH;
            volatile tt_l1_ptr uint16_t* mask_tile =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mask_part_cb.get_write_ptr());
            for (uint32_t col = 0; col < tt::constants::TILE_WIDTH; ++col) {
                const uint32_t elem_off = (col < tt::constants::FACE_WIDTH)
                                              ? col
                                              : (tt::constants::FACE_HW + col - tt::constants::FACE_WIDTH);
                mask_tile[elem_off] = (col >= first_masked_col) ? NEG_INF_BF16 : (uint16_t)0;
            }
            mask_part_cb.push_back(1);
        }
    }

    // ---- the merge units ----
    const uint32_t total_merges = n_live * VT;
    for (uint32_t mu = N - 1 - core; mu < total_merges; mu += N) {
        const uint32_t i = mu / VT, v = mu % VT, r = live[i];
        const uint32_t sem_id = merge_sem(mu, N);
        // the row's valid count -> the partials to expect
        noc.async_read(idx, local_l1, idx_row_bytes, {.page_id = r}, {.addr = row_l1});
        noc.async_read_barrier();
        uint32_t lo = 0, hi = W;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            if (row_ptr[mid] == SENTINEL) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        const uint32_t nv = lo == 0 ? 1 : lo;
        const uint32_t expected = ceil_div(nv, KPU);
        push_word(ctrl_c, expected);
        push_word(ctrl_w, expected);
        // The partials land in this core's cb_pm / cb_ps / cb_pa by the units' writers (pages
        // [sem_id * UMAX, (sem_id + 1) * UMAX): the CBs hold NSEM * UMAX pages and every merge takes UMAX, so the
        // k-th merge of a core is slot k and the pages never wrap inside one reserve). The semaphore says they
        // are all there; the reserve and the push only move this RISC's pointers for the compute.
        pm_cb.reserve_back(UMAX);
        ps_cb.reserve_back(UMAX);
        pa_cb.reserve_back(UMAX);
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
        noc_semaphore_wait_min(sem, expected);
        pm_cb.push_back(UMAX);
        ps_cb.push_back(UMAX);
        pa_cb.push_back(UMAX);
    }
}
