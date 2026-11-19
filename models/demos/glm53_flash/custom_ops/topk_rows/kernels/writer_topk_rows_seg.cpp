// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Segmented-mode writer (custom_ops/topk_rows): user-major candidate tables, FP32 values, stage-2 translation.
// Model-local copy (custom_ops/topk_rows) of the shared topk_large_indices kernel with per-row valid lengths:
// valid_length_tensor is [rows, 1] (one 4-byte page per input row), the reader derives and pushes the bounds per row.

#include "topk_large_indices_runtime_args.hpp"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {

template <uint32_t source_slices_per_row, uint32_t output_slices_per_row, uint32_t slice_bytes>
FORCE_INLINE void copy_row_to_scratch(CircularBuffer& src_cb, CircularBuffer& scratch_cb, const Noc& noc) {
    static_assert(source_slices_per_row == 64 || source_slices_per_row == 128);
    static_assert(output_slices_per_row >= 1 && output_slices_per_row <= source_slices_per_row);
    constexpr uint32_t transfer_bytes = slice_bytes;
    static_assert(transfer_bytes <= NOC_MAX_BURST_SIZE);

    const uint32_t src_base = src_cb.get_read_ptr();
    const uint32_t dst_base = scratch_cb.get_write_ptr();
    const uint32_t noc_id = noc.get_noc_id();
    const auto local_src = [noc_id](uint32_t addr) {
        return noc_traits_t<UnicastEndpoint>::src_args_type{
            .noc_x = static_cast<uint32_t>(my_x[noc_id]), .noc_y = static_cast<uint32_t>(my_y[noc_id]), .addr = addr};
    };

    noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
        UnicastEndpoint{}, transfer_bytes, local_src(src_base));

    for (uint32_t dst_slice = 0; dst_slice < output_slices_per_row; ++dst_slice) {
        // pack_untilize emits 16-element slices in face-pair order:
        // [top-left, bottom-left, top-right, bottom-right] for each tile column.
        const uint32_t tile_col = dst_slice >> 2;
        const uint32_t face_col = dst_slice & 0x1;
        const uint32_t face_row_offset = (dst_slice & 0x2) ? source_slices_per_row / 2 : 0;
        const uint32_t src_slice = (2 * tile_col) + face_col + face_row_offset;
        const uint32_t src_addr = src_base + src_slice * slice_bytes;
        const uint32_t dst_addr = dst_base + dst_slice * slice_bytes;
        noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            UnicastEndpoint{},
            CoreLocalMem<uint32_t>(dst_addr),
            transfer_bytes,
            local_src(src_addr),
            {.offset_bytes = 0});
    }
    noc.async_read_barrier();
}

template <
    uint32_t source_slices_per_row,
    uint32_t output_slices_per_row,
    uint32_t slice_bytes,
    typename TensorAccessorT>
FORCE_INLINE void issue_reordered_row_write(
    CircularBuffer& src_cb,
    CircularBuffer& scratch_cb,
    const Noc& noc,
    const TensorAccessorT& tensor,
    uint32_t row,
    uint32_t row_bytes) {
    src_cb.wait_front(1);
    scratch_cb.reserve_back(1);
    copy_row_to_scratch<source_slices_per_row, output_slices_per_row, slice_bytes>(src_cb, scratch_cb, noc);
    src_cb.pop_front(1);

    scratch_cb.push_back(1);
    scratch_cb.wait_front(1);
    noc.async_write(scratch_cb, tensor, row_bytes, {.offset_bytes = 0}, {.page_id = row, .offset_bytes = 0});
}

template <typename TensorAccessorT>
FORCE_INLINE void issue_contiguous_row_write(
    CircularBuffer& src_cb, const Noc& noc, const TensorAccessorT& tensor, uint32_t row, uint32_t row_bytes) {
    src_cb.wait_front(1);
    noc.async_write(src_cb, tensor, row_bytes, {.offset_bytes = 0}, {.page_id = row, .offset_bytes = 0});
}

}  // namespace

void kernel_main() {
    const uint32_t indices_addr = get_common_arg_val<uint32_t>(topk_common_args::output_address);
    const uint32_t start_row = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    // segmented two-stage mode args (compile-time, 0 = off):
    //   seg_rows_per_user: stage 1 writes row r's ids / values to page r / seg_rows_per_user at column offset
    //                      (r % seg_rows_per_user) * K elements (user-major candidate tables)
    //   values_cb / values_addr: stage 1 also writes the FP32 values
    //   translate (stage 2): candidate slot v -> seg_len * (v / K) + ids1[b][v] through the ids1 table
    constexpr uint32_t cb_indices = get_compile_time_arg_val(0);
    constexpr uint32_t cb_indices_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t indices_page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t source_slices_per_row = get_compile_time_arg_val(3);
    constexpr uint32_t output_slices_per_row = get_compile_time_arg_val(4);
    constexpr uint32_t indices_slice_bytes = get_compile_time_arg_val(5);
    constexpr auto indices_args = TensorAccessorArgs<6>();
    constexpr uint32_t seg_base = indices_args.next_compile_time_args_offset();
    constexpr uint32_t seg_rows_per_user = get_compile_time_arg_val(seg_base);  // 0 = plain rows
    constexpr uint32_t values_cb = get_compile_time_arg_val(seg_base + 1);      // 0 = no values
    constexpr uint32_t translate = get_compile_time_arg_val(seg_base + 2);      // 1 = stage-2 translation
    constexpr uint32_t seg_len = get_compile_time_arg_val(seg_base + 3);        // elements per segment (translate)
    constexpr uint32_t ids1_cb = get_compile_time_arg_val(seg_base + 4);        // L1 scratch for the ids1 row
    constexpr auto values_args = TensorAccessorArgs<seg_base + 5>();
    constexpr auto ids1_args = TensorAccessorArgs<values_args.next_compile_time_args_offset()>();
    const uint32_t values_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t ids1_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t user_page_bytes = get_common_arg_val<uint32_t>(3);  // page of the user-major tables
    constexpr uint32_t K = indices_page_bytes / 4;
    constexpr uint32_t SENT = 0xFFFFFFFFu;
    constexpr uint32_t seg_div = seg_rows_per_user ? seg_rows_per_user : 1;  // a constant divisor of 0 is a build error

    const auto indices =
        TensorAccessor(indices_args, indices_addr, seg_rows_per_user ? user_page_bytes : indices_page_bytes);
    CircularBuffer indices_cb(cb_indices);
    Noc noc;
    static_assert(source_slices_per_row == 32, "segmented writer: K = 512 only");
    CircularBuffer indices_scratch_cb(cb_indices_scratch);
    for (uint32_t local_row = 0; local_row < num_rows; ++local_row) {
        const uint32_t row = start_row + local_row;
        const uint32_t page = seg_rows_per_user ? row / seg_div : row;
        const uint32_t col_off = seg_rows_per_user ? (row % seg_div) * indices_page_bytes : 0;
        if constexpr (translate) {
            // the contiguous ranked row into scratch, translate in place, write
            indices_cb.wait_front(1);
            indices_scratch_cb.reserve_back(1);
            const uint32_t src = indices_cb.get_read_ptr();
            const uint32_t dst = indices_scratch_cb.get_write_ptr();
            // the user's ids1 table (page `page` of [B, 1, 1, S*K] uint32)
            CircularBuffer ids1_cb_obj(ids1_cb);
            const uint32_t t1 = ids1_cb_obj.get_write_ptr();
            const auto ids1 = TensorAccessor(ids1_args, ids1_addr, user_page_bytes);
            noc.async_read(ids1, CoreLocalMem<uint32_t>(t1), user_page_bytes, {.page_id = page}, {});
            noc.async_read_barrier();
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
            volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
            volatile tt_l1_ptr uint32_t* table = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(t1);
            for (uint32_t i = 0; i < K; ++i) {
                const uint32_t v = s[i];
                uint32_t g = SENT;
                if (v != SENT) {
                    const uint32_t local = table[v];
                    g = (local == SENT) ? SENT : (v / K) * seg_len + local;
                }
                d[i] = g;
            }
            indices_cb.pop_front(1);
            indices_scratch_cb.push_back(1);
            indices_scratch_cb.wait_front(1);
            noc.async_write(
                indices_scratch_cb,
                indices,
                indices_page_bytes,
                {.offset_bytes = 0},
                {.page_id = page, .offset_bytes = col_off});
            noc.async_writes_flushed();
            indices_scratch_cb.pop_front(1);
        } else {
            indices_cb.wait_front(1);
            noc.async_write(
                indices_cb,
                indices,
                indices_page_bytes,
                {.offset_bytes = 0},
                {.page_id = page, .offset_bytes = col_off});
            noc.async_writes_flushed();
            indices_cb.pop_front(1);
            if constexpr (values_cb != 0) {
                CircularBuffer values(values_cb);
                const auto vals = TensorAccessor(values_args, values_addr, user_page_bytes);
                values.wait_front(1);
                noc.async_write(
                    values, vals, indices_page_bytes, {.offset_bytes = 0}, {.page_id = page, .offset_bytes = col_off});
                noc.async_writes_flushed();
                values.pop_front(1);
            }
        }
    }
    noc.async_write_barrier();
}
