// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Router tail, fast sort-core form (QWEN38_ROUTER_TAIL_FAST=1), reader of core A (reader_rt_main.cpp + RTF_ROW0) (rows
// == 1, token 0 in row 0). Today's reader (../../router_tail/kernels/reader_router_tail.cpp) minus the index tiles (the
// sort runs on core B), plus the hand-off from B:
//   - the logits tiles and the three reduce scalers, as today;
//   - while the softmax runs: reserve the [token, k] tiles cb_vals / cb_pad_div and zero them (today's fill_implicit_
//     tile_padding: everything but row 0, columns < top_k, is zero), reserve cb_idx_t;
//   - wait for B's semaphore: B's NoC write put the sorted, transposed index tile into this core's cb_idx_t;
//   - wait for the softmax's cb_probs (16 fp32 tiles); for slot k < top_k the sorted index e = idx[k * 16] (the
//     [k, token] tile, token 0) selects the probability probs[e >> 5][face (e >> 4) & 1][row 0][e & 15], a plain fp32
//     word copy, i.e. the very bits today's sort carried into the [token, k] tile; they go to row 0, column k of
//     cb_vals and cb_pad_div;
//   - push cb_vals + its token, cb_pad_div and cb_idx_t (the writer reads it); the sums broadcast is today's.
// Runtime args: logits_addr, tile_row, rows_in_tile (unused: rows == 1).  Semaphore id: named compile-time arg sem_id.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "../../kernels/zones.h"

#ifndef RTF_IDXF
#define RTF_IDXF 0
#endif
#ifndef RTF_NOTOKEN
#define RTF_NOTOKEN 0
#endif
#ifndef RTF_ROW0
#define RTF_ROW0 0
#endif

void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_row = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_max_scaler = get_named_compile_time_arg_val("cb_max_scaler");
    constexpr uint32_t cb_sum_scaler = get_named_compile_time_arg_val("cb_sum_scaler");
    constexpr uint32_t cb_norm_scaler = get_named_compile_time_arg_val("cb_norm_scaler");
    constexpr uint32_t cb_probs = get_named_compile_time_arg_val("cb_probs");
    constexpr uint32_t cb_idx_t = get_named_compile_time_arg_val("cb_idx_t");
    constexpr uint32_t cb_vals = get_named_compile_time_arg_val("cb_vals");
    constexpr uint32_t cb_vals_ready = get_named_compile_time_arg_val("cb_vals_ready");
    constexpr uint32_t cb_pad_div = get_named_compile_time_arg_val("cb_pad_div");
    constexpr uint32_t cb_sums = get_named_compile_time_arg_val("cb_sums");
    constexpr uint32_t cb_sums_ready = get_named_compile_time_arg_val("cb_sums_ready");
    constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
    constexpr uint32_t top_k = get_named_compile_time_arg_val("top_k");
    constexpr uint32_t sem_id = get_named_compile_time_arg_val("sem_id");

    constexpr auto logits_args = TensorAccessorArgs<0, 0>();
    const auto logits = TensorAccessor(logits_args, logits_addr);

    Noc noc;
    DataflowBuffer in0(cb_in0);

    {
        FUSED_ZONE("fz_rt_r_reads");
        const uint32_t tile_bytes = in0.get_entry_size();
        in0.reserve_back(Wt);
        for (uint32_t w = 0; w < Wt; ++w) {
            noc.async_read(logits, in0, tile_bytes, {.page_id = tile_row * Wt + w}, {.offset_bytes = w * tile_bytes});
        }
        noc.async_read_barrier();
        in0.push_back(Wt);

        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            cb_max_scaler,
            ckernel::PoolType::MAX,
            ckernel::ReduceDim::REDUCE_ROW>();
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            cb_sum_scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW>();
        dataflow_kernel_lib::
            prepare_reduce_scaler<cb_norm_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(1.0f);
    }

    DataflowBuffer vals(cb_vals);
    DataflowBuffer vals_ready(cb_vals_ready);
    DataflowBuffer pad_div(cb_pad_div);
    DataflowBuffer idx_t(cb_idx_t);
    DataflowBuffer probs(cb_probs);
    volatile tt_l1_ptr uint32_t* vtile;
    volatile tt_l1_ptr uint32_t* dtile;
    {
        FUSED_ZONE("fz_rt_r_pad_fill");  // hidden under the softmax: the reader is idle while compute runs
        vals.reserve_back(1);
        pad_div.reserve_back(1);
        idx_t.reserve_back(1);
        vtile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(vals.get_write_ptr());
        dtile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pad_div.get_write_ptr());
        for (uint32_t i = 0; i < 1024; ++i) {
            vtile[i] = 0u;
            dtile[i] = 0u;
        }
    }

    {
        FUSED_ZONE("fz_rt_r_gather");
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
        noc_semaphore_wait_min(sem, 1);
        noc_semaphore_set(sem, 0);
        probs.wait_front(Wt);
        volatile tt_l1_ptr uint32_t* idx =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_t.get_write_ptr());  // [k, token] tile, token 0
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(probs.get_read_ptr());
        for (uint32_t k = 0; k < top_k; ++k) {
#if RTF_IDXF
            const uint32_t bits = idx[k * 16];  // fp32 integer 0..511
            const uint32_t e = bits == 0u ? 0u : (((bits & 0x7FFFFFu) | 0x800000u) >> (150u - (bits >> 23)));
#else
            const uint32_t e = idx[k * 16];
#endif
            const uint32_t word = p[(e >> 5) * 1024 + ((e >> 4) & 1u) * 256 + (e & 15u)];
            vtile[k] = word;  // row 0, column k of face 0
            dtile[k] = word;
        }
        probs.pop_front(Wt);
        vals.push_back(1);
#if !RTF_NOTOKEN
        vals_ready.reserve_back(1);
        vals_ready.push_back(1);
#endif
        pad_div.push_back(1);
        idx_t.push_back(1);
    }

    // binary_ng col-bcast reader, in place on the sums tile: as today's reader
    {
        FUSED_ZONE("fz_rt_r_sum_bcast");
        DataflowBuffer sums(cb_sums);
        DataflowBuffer sums_ready(cb_sums_ready);
        sums.wait_front(1);
        {
            volatile tt_l1_ptr uint32_t* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sums.get_read_ptr());
#if RTF_ROW0
            for (uint32_t row = 0; row < 1; ++row) {  // only row 0 (token 0) is ever written out
#else
            for (uint32_t row = 0; row < 32; ++row) {
#endif
                const uint32_t base = (row >> 4) * 2 * 256 + (row & 15) * 16;
                const uint32_t value = tile[base];
                for (uint32_t col = 1; col < 16; ++col) {
                    tile[base + col] = value;
                }
            }
        }
        sums_ready.reserve_back(1);
        sums_ready.push_back(1);
    }
}
