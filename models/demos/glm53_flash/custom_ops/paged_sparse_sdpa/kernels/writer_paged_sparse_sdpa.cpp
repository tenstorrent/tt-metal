// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// paged_sparse_sdpa writer (model-local generic op, custom_ops/paged_sparse_sdpa): tt-metal's sparse_sdpa writer
// without the dual-NoC K-half gather (the paged reader gathers the K chunk itself) and without the attention sink.
// Builds the three persistent compute-input tiles once (the reduce identity scaler, the col-identity, the all-(-inf)
// tile), then per token drains the untilized [H, V_DIM] row-major block from compute into the ROW_MAJOR
// [1, H, S, V_DIM] output (head-row h of token tok = page h*S + tok).
#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"  // generate_bcast_col_scalar
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

constexpr uint32_t one_bf16_packed = 0x3F803F80u;  // bf16(1.0) double-packed; generate_bcast_col_scalar uses >>16
constexpr uint16_t neg_inf_bf16 = 0xFF80u;

namespace wct {
enum : uint32_t { H = 0, S, V_DHT, CB_OUT_RM, CB_SCALE, CB_COL_IDENTITY, CB_NEGINF, OUT_ELEM_BYTES, END };
}

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(wct::H);
    constexpr uint32_t S = get_compile_time_arg_val(wct::S);
    constexpr uint32_t vDHt = get_compile_time_arg_val(wct::V_DHT);
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(wct::CB_OUT_RM);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(wct::CB_SCALE);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(wct::CB_COL_IDENTITY);
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(wct::CB_NEGINF);
    constexpr uint32_t out_elem_bytes = get_compile_time_arg_val(wct::OUT_ELEM_BYTES);
    constexpr auto out_args = TensorAccessorArgs<wct::END>();

    const uint32_t tok_start = get_arg_val<uint32_t>(0);
    const uint32_t tok_count = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t row_bytes = vDHt * tt::constants::TILE_WIDTH * out_elem_bytes;  // V_DIM bytes per head-row
    constexpr uint32_t block_tiles = (H / tt::constants::TILE_HEIGHT) * vDHt;          // Sqt*vDHt: the [H,V_DIM] block

    Noc noc;
    experimental::CB out_cb(cb_out_rm);
    const auto out = TensorAccessor(out_args, out_addr);

    // --- persistent compute-input tiles (built once) ---
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
        for (uint32_t c = 0; c < tt::constants::FACE_WIDTH; ++c) {
            p[c] = neg_inf_bf16;
            p[tt::constants::FACE_HW + c] = neg_inf_bf16;
        }
        neginf_cb.push_back(1);
    }

    for (uint32_t tok = tok_start; tok < tok_start + tok_count; ++tok) {
        out_cb.wait_front(block_tiles);  // one untilized [H, V_DIM] block
        for (uint32_t h = 0; h < H; ++h) {
            noc.async_write(out_cb, out, row_bytes, {.offset_bytes = h * row_bytes}, {.page_id = h * S + tok});
        }
        noc.async_write_barrier();
        out_cb.pop_front(block_tiles);
    }
}
