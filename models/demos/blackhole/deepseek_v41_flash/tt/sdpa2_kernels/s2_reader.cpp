// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// sdpa2 reader (NCRISC): decode key-split sparse_sdpa, one core = (query row, split g). Differences to
// ssplit_reader.cpp:
//  * the index row is read FIRST (nv, chunk share, ctrl message out in ~1 us); Q is read AFTER the first K slabs are
//  issued, and only the
//    two top faces (rows 0..15) of each of the 16 Q tile pages (the 9 live rows are in rows 0..8): 1 KB instead of 2 KB
//    per page
//  * the K gather is pushed per 32-row slab (compute tilizes slab by slab while the next slab is still in flight); two
//  slabs are in
//    flight (alternating NoC transaction ids), the other half of every slab is gathered by the writer (BRISC, other
//    NoC)
#include <stdint.h>
#include "api/tile_constants.h"
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "../sparse_split_kernels/ssplit_common.hpp"

constexpr uint32_t sentinel = 0xFFFFFFFFu;
// slab request words (reader -> writer)
constexpr uint32_t RQ_BASE = 0, RQ_SPLIT = 1, RQ_LAST = 2, RQ_DST = 3, RQ_ROWS = 4;

void kernel_main() {
    constexpr uint32_t topk = get_compile_time_arg_val(1);
    constexpr uint32_t k_chunk = get_compile_time_arg_val(2);
    constexpr uint32_t k_dim = get_compile_time_arg_val(3);
    constexpr uint32_t cb_k_rm = get_compile_time_arg_val(4);
    constexpr uint32_t cb_mask_part = get_compile_time_arg_val(5);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(6);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(7);
    constexpr uint32_t kv_elem_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(9);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(10);
    constexpr uint32_t NSPLIT = get_compile_time_arg_val(11);
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(12);
    constexpr uint32_t Q_FACE_BYTES =
        get_compile_time_arg_val(13);  // bytes of each Q tile page that are read (1024: faces 0, 1)
    constexpr auto q_args = TensorAccessorArgs<14, 3>();
    constexpr auto kv_args =
        TensorAccessorArgs<q_args.next_compile_time_args_offset(), q_args.next_common_runtime_args_offset()>();
    constexpr auto idx_args =
        TensorAccessorArgs<kv_args.next_compile_time_args_offset(), kv_args.next_common_runtime_args_offset()>();
    const uint32_t q_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t kv_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t tok = get_arg_val<uint32_t>(0);
    const uint32_t split_g = get_arg_val<uint32_t>(1);

    constexpr uint32_t k_row_bytes = k_dim * kv_elem_bytes;
    constexpr uint32_t idx_row_bytes = topk * 4;
    constexpr uint32_t Skt = k_chunk / 32;

    Noc noc;
    experimental::CB k_cb(cb_k_rm), mask_part_cb(cb_mask_part), idx_cb(cb_idx), ctrl_cb(cb_ctrl), kreq_cb(cb_kreq),
        kack_cb(cb_kack), q_in_cb(cb_q_in);
    const auto q = TensorAccessor(q_args, q_addr);
    const auto kv = TensorAccessor(kv_args, kv_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);
    const UnicastEndpoint local_l1;

    idx_cb.reserve_back(1);
    const uint32_t idx_l1 = idx_cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
    {
        DeviceZoneScopedN("S2_RIDX");
        noc.async_read(idx, idx_cb, idx_row_bytes, {.page_id = tok}, {.offset_bytes = 0});
        noc.async_read_barrier();
    }
    // first sentinel -> nv (valid-key count, sentinels are a contiguous tail)
    uint32_t nv;
    {
        uint32_t lo = 0, hi = topk;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            if (idx_ptr[mid] == sentinel) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        nv = lo == 0 ? 1 : lo;
    }
    const uint32_t n_active = (nv + k_chunk - 1) / k_chunk;
    const uint32_t c_base = n_active / NSPLIT, c_extra = n_active % NSPLIT;
    const uint32_t my_cnt = c_base + (split_g < c_extra ? 1u : 0u);
    const uint32_t c0 = split_g * c_base + (split_g < c_extra ? split_g : c_extra);
    const uint32_t n_used = n_active < NSPLIT ? n_active : NSPLIT;
    const uint32_t n_remote = n_used - 1;
    ctrl_cb.reserve_back(1);
    {
        volatile tt_l1_ptr uint32_t* cp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
        cp[ssplit::control_message::ACTIVE_CHUNKS] = my_cnt;
        cp[ssplit::control_message::VALID_KEYS] = nv;
        cp[ssplit::control_message::TOTAL_CHUNKS] = n_active;
        cp[ssplit::control_message::C_START] = c0;
        cp[ssplit::control_message::N_REMOTE] = n_remote;
        cp[5] = n_used;
        cp[6] = split_g;
    }
    ctrl_cb.push_back(1);
    kreq_cb.reserve_back(1);
    {
        volatile tt_l1_ptr uint32_t* request = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_write_ptr());
        request[RQ_BASE] = my_cnt;
        request[RQ_SPLIT] = n_remote;
        request[RQ_LAST] = split_g;
        request[RQ_DST] = n_used;
    }
    kreq_cb.push_back(1);

    uint32_t q_state = 0;  // 0 not issued, 1 reads in flight, 2 pushed
    auto issue_q = [&]() {
        DeviceZoneScopedN("S2_R_QT");
        q_in_cb.reserve_back(16);
        experimental::set_read_trid(noc, 3);
        for (uint32_t c = 0; c < 16; ++c) {
            noc.async_read(q, q_in_cb, Q_FACE_BYTES, {.page_id = tok * 16 + c}, {.offset_bytes = c * 2048});
        }
        experimental::set_read_trid(noc, 0);
        q_state = 1;
    };
    auto push_q = [&]() {
        experimental::async_read_barrier_with_trid(noc, 3);
        q_in_cb.push_back(16);
        q_state = 2;
    };
    uint32_t sc = 0;  // running slab counter -> transaction id parity
    for (uint32_t chunk = c0; chunk < c0 + my_cnt; ++chunk) {
        DeviceZoneScopedN("S2_R_CHUNK");
        const uint32_t base = chunk * k_chunk;
        const uint32_t valid = (nv - base) < k_chunk ? (nv - base) : k_chunk;
        k_cb.reserve_back(k_chunk);
        const uint32_t k_wr = k_cb.get_write_ptr();
        const bool last_chunk = (chunk == c0 + my_cnt - 1);
        uint32_t prev_rows = 0, prev_trid = 0;
        for (uint32_t s = 0; s <= Skt; ++s) {
            // 1) issue slab s (reader half), 2) finalize slab s-1
            uint32_t rows = 0, trid = 0;
            if (s < Skt) {
                const uint32_t row_base = s * 32;
                rows = valid > row_base ? (valid - row_base < 32 ? valid - row_base : 32) : 0;
                const uint32_t split = rows >> 1;
                const uint32_t dst = k_wr + row_base * k_row_bytes;
                kreq_cb.reserve_back(1);
                {
                    volatile tt_l1_ptr uint32_t* request =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_write_ptr());
                    request[RQ_BASE] = base + row_base;
                    request[RQ_SPLIT] = split;
                    request[RQ_LAST] =
                        (s == Skt - 1 ? 2u : 0u) |
                        ((last_chunk && s == Skt - 1) ? 1u
                                                      : 0u);  // bit0: last slab of the row, bit1: last of its chunk
                    request[RQ_DST] = dst;
                    request[RQ_ROWS] = rows;
                }
                kreq_cb.push_back(1);
                trid = (sc & 1) + 1;
                experimental::set_read_trid(noc, trid);
                for (uint32_t i = split; i < rows; ++i) {
                    noc.async_read(
                        kv,
                        local_l1,
                        k_row_bytes,
                        {.page_id = idx_ptr[base + row_base + i]},
                        {.addr = dst + i * k_row_bytes});
                }
                experimental::set_read_trid(noc, 0);
                ++sc;
                if (q_state == 0 && s == 1) {  // Q behind the first two slabs (needed at the first QK only)
                    issue_q();
                }
            }
            if (s > 0) {
                const uint32_t ps = s - 1;
                experimental::async_read_barrier_with_trid(noc, prev_trid);
                kack_cb.wait_front(1);  // writer's half of slab ps landed
                kack_cb.pop_front(1);
                if (prev_rows < 32) {
                    noc.async_write_zeros(
                        k_cb,
                        (32 - prev_rows) * k_row_bytes,
                        {.offset_bytes = prev_rows * k_row_bytes});  // wr ptr = start of slab ps
                    noc.write_zeros_l1_barrier();
                }
                if (ps == Skt - 1 && q_state == 1) {
                    push_q();
                }
                k_cb.push_back(32);
            }
            prev_rows = rows;
            prev_trid = trid;
        }

        // Boundary mask tile (only the global last chunk, mid-tile boundary), as in ssplit_reader
        if (chunk == n_active - 1 && (valid % 32) != 0) {
            mask_part_cb.reserve_back(1);
            const uint32_t first_masked_col = valid % 32;
            volatile tt_l1_ptr uint16_t* mask_tile =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mask_part_cb.get_write_ptr());
            for (uint32_t col = 0; col < 32; ++col) {
                const uint32_t elem_off = (col < 16) ? col : (256 + col - 16);
                mask_tile[elem_off] = (col >= first_masked_col) ? 0xFF80u : (uint16_t)0;
            }
            mask_part_cb.push_back(1);
        }
    }
    if (q_state == 0) {  // a core without chunks still owns a Q page set (compute pops it)
        issue_q();
    }
    if (q_state == 1) {
        push_q();
    }
}
