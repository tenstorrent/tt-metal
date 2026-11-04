// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// sdpa2 writer (BRISC): other-NoC half of the per-slab K gather, constants setup (after the first slabs are issued),
// partial-state send / receive and the output write. Slab protocol: see s2_reader.cpp.
#include <stdint.h>
#include "api/tile_constants.h"
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "../sparse_split_kernels/ssplit_common.hpp"

constexpr uint32_t one_bf16_packed = 0x3F803F80u;
constexpr uint16_t neg_inf_bf16 = 0xFF80u;
constexpr uint32_t RQ_BASE = 0, RQ_SPLIT = 1, RQ_LAST = 2, RQ_DST = 3, RQ_ROWS = 4;

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);
    constexpr uint32_t vDHt = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(3);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(4);
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(5);
    constexpr uint32_t out_elem_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t kv_elem_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(8);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(9);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(10);
    constexpr uint32_t cb_attention_sink = get_compile_time_arg_val(11);
    constexpr uint32_t cb_sink_scratch = get_compile_time_arg_val(12);
    constexpr uint32_t NSPLIT = get_compile_time_arg_val(13);
    constexpr uint32_t cb_recv_out = get_compile_time_arg_val(14);
    constexpr uint32_t cb_recv_max = get_compile_time_arg_val(15);
    constexpr uint32_t cb_recv_sum = get_compile_time_arg_val(16);
    constexpr uint32_t cb_out_a = get_compile_time_arg_val(17);
    constexpr uint32_t cb_out_b = get_compile_time_arg_val(18);
    constexpr uint32_t cb_max_a = get_compile_time_arg_val(19);
    constexpr uint32_t cb_max_b = get_compile_time_arg_val(20);
    constexpr uint32_t cb_sum_a = get_compile_time_arg_val(21);
    constexpr uint32_t cb_sum_b = get_compile_time_arg_val(22);
    constexpr uint32_t H = get_compile_time_arg_val(23);
    constexpr uint32_t k_dim = get_compile_time_arg_val(24);
    constexpr uint32_t W = get_compile_time_arg_val(25);  // output tiles per merger
    constexpr uint32_t sem_id = 0;
    constexpr uint32_t OUT_ROWS =
        get_compile_time_arg_val(26);  // head rows written (9: row 0 kv + 8 heads, what rope_concat reads)
    constexpr bool STREAM = get_compile_time_arg_val(27) !=
                            0;  // PV slices are sent as they are packed; the merger takes the states per sender slot
    constexpr auto out_args = TensorAccessorArgs<28, 3>();
    constexpr auto kv_args =
        TensorAccessorArgs<out_args.next_compile_time_args_offset(), out_args.next_common_runtime_args_offset()>();
    constexpr auto sink_args =
        TensorAccessorArgs<kv_args.next_compile_time_args_offset(), kv_args.next_common_runtime_args_offset()>();
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t kv_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t sink_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t tok = get_arg_val<uint32_t>(0);
    const uint32_t leader_x = get_arg_val<uint32_t>(2);
    const uint32_t leader_y = get_arg_val<uint32_t>(3);

    constexpr uint32_t row_bytes = vDHt * 32 * out_elem_bytes;
    constexpr uint32_t block_tiles = (H / 32) * vDHt;
    constexpr uint32_t k_row_bytes = k_dim * kv_elem_bytes;

    Noc noc;
    experimental::CB out_cb(cb_out_rm);
    const auto out = TensorAccessor(out_args, out_addr);
    const auto kv = TensorAccessor(kv_args, kv_addr);
    const UnicastEndpoint local_l1;
    experimental::CB idx_cb(cb_idx), kreq_cb(cb_kreq), kack_cb(cb_kack);
    volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_cb.get_write_ptr());

    DeviceZoneScopedN("S2_W_ALL");
    auto setup = [&]() {
        DeviceZoneScopedN("S2_W_SETUP");
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            cb_scale,
            ckernel::PoolType::MAX,
            ckernel::ReduceDim::REDUCE_ROW,
            /*reduce_factor=*/1>();
        generate_bcast_col_scalar(CircularBuffer(cb_col_identity), one_bf16_packed);
        {
            experimental::CB neginf_cb(cb_neginf);
            neginf_cb.reserve_back(1);
            volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(neginf_cb.get_write_ptr());
            for (uint32_t c = 0; c < 16; ++c) {
                p[c] = neg_inf_bf16;
                p[256 + c] = neg_inf_bf16;
            }
            neginf_cb.push_back(1);
        }
        {
            const auto sink = TensorAccessor(sink_args, sink_addr);
            experimental::CB sink_cb(cb_attention_sink), scratch(cb_sink_scratch);
            constexpr uint32_t sink_bytes = H * sizeof(uint16_t);
            constexpr uint32_t tile_bytes = 2048;
            constexpr uint32_t sink_tiles = H / 32;
            sink_cb.reserve_back(sink_tiles);
            scratch.reserve_back(1);
            noc.async_write_zeros(sink_cb, sink_tiles * tile_bytes);
            experimental::set_read_trid(noc, 3);
            noc.async_read(sink, scratch, sink_bytes, {.page_id = 0}, {.offset_bytes = 0});
            experimental::set_read_trid(noc, 0);
            experimental::async_read_barrier_with_trid(noc, 3);
            noc.write_zeros_l1_barrier();
            auto* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sink_cb.get_write_ptr());
            auto* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch.get_write_ptr());
            for (uint32_t h = 0; h < H; ++h) {
                const uint32_t row = h % 32;
                const uint32_t offset = (h / 32) * 1024 + (row / 16) * 2 * 256 + (row % 16) * 16;
                dst[offset] = src[h];
            }
            sink_cb.push_back(sink_tiles);
        }
    };

    kreq_cb.wait_front(1);
    uint32_t my_cnt, n_remote, split_g, n_used;
    {
        volatile tt_l1_ptr uint32_t* hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_read_ptr());
        my_cnt = hdr[RQ_BASE];
        n_remote = hdr[RQ_SPLIT];
        split_g = hdr[RQ_LAST];
        n_used = hdr[RQ_DST];
    }
    kreq_cb.pop_front(1);

    // ---- K gather, other-NoC half of every slab; two slabs in flight (transaction ids 1, 2)
    bool setup_done = false;
    {
        uint32_t sc = 0, prev_trid = 0;
        bool have_prev = false, last = (my_cnt == 0), drain = false;
        while (!last) {
            kreq_cb.wait_front(1);
            uint32_t base, split, dst, rows;
            {
                volatile tt_l1_ptr uint32_t* request =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_read_ptr());
                base = request[RQ_BASE];
                split = request[RQ_SPLIT];
                last = (request[RQ_LAST] & 1) != 0;
                drain = request[RQ_LAST] != 0;
                dst = request[RQ_DST];
                rows = request[RQ_ROWS];
            }
            kreq_cb.pop_front(1);
            const uint32_t trid = (sc & 1) + 1;
            experimental::set_read_trid(noc, trid);
            for (uint32_t i = 0; i < split; ++i) {
                noc.async_read(
                    kv, local_l1, k_row_bytes, {.page_id = idx_ptr[base + i]}, {.addr = dst + i * k_row_bytes});
            }
            experimental::set_read_trid(noc, 0);
            ++sc;
            if (!setup_done) {
                setup();
                setup_done = true;
            }
            if (have_prev) {
                experimental::async_read_barrier_with_trid(noc, prev_trid);
                kack_cb.reserve_back(1);
                kack_cb.push_back(1);
            }
            prev_trid = trid;
            have_prev = true;
            // the slab count of a chunk is fixed (Skt), the final slab of the last chunk carries last = 1: drain it
            // right away
            if (drain) {
                experimental::async_read_barrier_with_trid(noc, prev_trid);
                kack_cb.reserve_back(1);
                kack_cb.push_back(1);
                have_prev = false;
            }
            (void)rows;
        }
    }
    if (!setup_done) {
        setup();
    }
    // ---- SLICED merge: M = vDHt / W mergers (cores g < M of the row), W output tiles each. Every core with chunks (g
    // < n_used) sends its max tile, sum tile and the out slice m to merger m (itself included: loopback write) into
    // slot g of that merger's receive CBs; a merger collects the n_used slots in order and writes its W-tile slice of
    // the output row.
    constexpr uint32_t TILE_B = 2048;
    constexpr uint32_t M = vDHt / W;
    constexpr uint32_t ST = H / 32;
    experimental::CB r_out(cb_recv_out), r_max(cb_recv_max), r_sum(cb_recv_sum);
    const uint32_t o_base = r_out.get_write_ptr();  // identical on every core (same CB layout)
    const uint32_t m_base = r_max.get_write_ptr();
    const uint32_t l_base = r_sum.get_write_ptr();
    const bool merger = split_g < M;
    auto mx = [&](uint32_t m) { return m == 0 ? leader_x : get_arg_val<uint32_t>(4 + 2 * (m - 1)); };
    auto my = [&](uint32_t m) { return m == 0 ? leader_y : get_arg_val<uint32_t>(5 + 2 * (m - 1)); };
    const bool in_b = (my_cnt & 1) != 0;
    experimental::CB s_out(in_b ? cb_out_b : cb_out_a), s_max(in_b ? cb_max_b : cb_max_a),
        s_sum(in_b ? cb_sum_b : cb_sum_a);
    if (my_cnt > 0) {  // the max tile goes out as soon as it exists (right after the QK row max)
        DeviceZoneScopedN("S2_W_SENDM");
        s_max.wait_front(ST);
        for (uint32_t m = 0; m < M; ++m) {
            noc_async_write(
                s_max.get_read_ptr(), get_noc_addr(mx(m), my(m), m_base + split_g * ST * TILE_B), ST * TILE_B);
        }
        noc_async_write_barrier();
        for (uint32_t m = 0; m < M; ++m) {
            noc_semaphore_inc(get_noc_addr(mx(m), my(m), get_semaphore(sem_id + 0)), 1);
        }
    }
    if (merger) {  // collect the max tiles now: compute forms the global max and the corrections the moment its own
                   // chunk is done
        DeviceZoneScopedN("S2_W_RECVM");
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id + 0));
        noc_semaphore_wait(sem, n_used);  // one counting semaphore: every sender incs once, the slots are by sender id
        noc_semaphore_set(sem, 0);
        r_max.reserve_back(n_used * ST);
        r_max.push_back(n_used * ST);
    }
    if (my_cnt > 0) {
        DeviceZoneScopedN("S2_W_SEND");
        if constexpr (STREAM) {
            for (uint32_t m = 0; m < M; ++m) {  // slice m leaves as soon as its W tiles are packed
                s_out.wait_front(W * (m + 1));
                noc_async_write(
                    s_out.get_read_ptr() + m * W * TILE_B,
                    get_noc_addr(mx(m), my(m), o_base + split_g * W * TILE_B),
                    W * TILE_B);
            }
            s_sum.wait_front(ST);
            for (uint32_t m = 0; m < M; ++m) {
                noc_async_write(
                    s_sum.get_read_ptr(), get_noc_addr(mx(m), my(m), l_base + split_g * ST * TILE_B), ST * TILE_B);
            }
            noc_async_write_barrier();
            for (uint32_t m = 0; m < M; ++m) {
                noc_semaphore_inc(get_noc_addr(mx(m), my(m), get_semaphore(sem_id + 1 + split_g)), 1);
            }
        } else {
            s_out.wait_front(block_tiles);
            s_sum.wait_front(ST);
            for (uint32_t m = 0; m < M; ++m) {
                noc_async_write(
                    s_sum.get_read_ptr(), get_noc_addr(mx(m), my(m), l_base + split_g * ST * TILE_B), ST * TILE_B);
                noc_async_write(
                    s_out.get_read_ptr() + m * W * TILE_B,
                    get_noc_addr(mx(m), my(m), o_base + split_g * W * TILE_B),
                    W * TILE_B);
            }
            noc_async_write_barrier();
            for (uint32_t m = 0; m < M; ++m) {
                noc_semaphore_inc(get_noc_addr(mx(m), my(m), get_semaphore(sem_id + 1)), 1);
            }
        }
        noc_async_atomic_barrier();
    }
    if (!merger) {
        return;
    }
    {
        DeviceZoneScopedN("S2_WRECV");
        if constexpr (STREAM) {
            for (uint32_t q = 0; q < n_used; ++q) {  // slot order: compute merges slot q as soon as it is pushed
                volatile tt_l1_ptr uint32_t* sem =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id + 1 + q));
                noc_semaphore_wait(sem, 1);
                noc_semaphore_set(sem, 0);
                r_out.reserve_back(W);
                r_sum.reserve_back(ST);
                r_out.push_back(W);
                r_sum.push_back(ST);
            }
        } else {
            volatile tt_l1_ptr uint32_t* sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id + 1));
            noc_semaphore_wait(sem, n_used);
            noc_semaphore_set(sem, 0);
            r_out.reserve_back(n_used * W);
            r_sum.reserve_back(n_used * ST);
            r_out.push_back(n_used * W);
            r_sum.push_back(n_used * ST);
        }
    }
    {
        DeviceZoneScopedN("S2_W_OUTWAIT");
        out_cb.wait_front(W);
    }
    constexpr uint32_t slice_bytes = W * 32 * out_elem_bytes;
    for (uint32_t h = 0; h < OUT_ROWS; ++h) {
        noc.async_write(
            out_cb,
            out,
            slice_bytes,
            {.offset_bytes = h * slice_bytes},
            {.page_id = h * S + tok, .offset_bytes = split_g * slice_bytes});
    }
    {
        DeviceZoneScopedN("S2_W_OUTWR");
        noc.async_write_barrier();
    }
    out_cb.pop_front(W);
}
