// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for gated_rmsnorm fw/bw. One work item = (tile-row r, group h); it streams the Gt tiles of
// x (and gate, and dy in BACKWARD) at pages r*Wt + h*Gt + j. gamma (Gt tiles), its row-broadcast copy
// and the all-ones reduce tile are produced once per core.

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/gated_rmsnorm/device/kernels/gated_rmsnorm_cbs.hpp"

namespace cb = gated_rmsnorm_cb;

constexpr uint32_t Gt = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t num_groups = get_compile_time_arg_val(2);

constexpr uint32_t face_elems = 256U;
constexpr uint32_t face_rows = 16U;
constexpr uint32_t face_row_u32 = 8U;  // 16 bf16 per face row = 8 uint32

// gamma_b[r][c] = gamma[0][c] for every row r. Tile = 4 faces (16x16) in order (r<16,c<16), (r<16,c>=16),
// (r>=16,c<16), (r>=16,c>=16); row 0 of column-face (f & 1) is the source for all rows of face f.
FORCE_INLINE void broadcast_gamma_rows(uint32_t src_tile_addr, uint32_t dst_tile_addr) {
    const auto* src = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(src_tile_addr);
    auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_tile_addr);
    for (uint32_t face = 0; face < 4U; ++face) {
        const uint32_t src_row = ((face & 1U) * face_elems) / 2U;  // in uint32 units
        const uint32_t dst_face = (face * face_elems) / 2U;
        for (uint32_t row = 0; row < face_rows; ++row) {
            const uint32_t dst_row = dst_face + row * face_row_u32;
            for (uint32_t w = 0; w < face_row_u32; ++w) {
                dst[dst_row + w] = src[src_row + w];
            }
        }
    }
}

template <typename Accessor>
FORCE_INLINE void read_group(uint32_t cb_id, const Accessor& acc, uint32_t first_page, uint32_t tile_bytes) {
    cb_reserve_back(cb_id, Gt);
    uint32_t l1 = get_write_ptr(cb_id);
    for (uint32_t j = 0; j < Gt; ++j) {
        noc_async_read_page(first_page + j, acc, l1);
        l1 += tile_bytes;
    }
}

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t gate_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t dy_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t work_start = get_arg_val<uint32_t>(arg++);
    const uint32_t work_count = get_arg_val<uint32_t>(arg++);

    constexpr auto x_args = TensorAccessorArgs<3>();
    constexpr auto gate_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<gate_args.next_compile_time_args_offset()>();
    const auto x_acc = TensorAccessor(x_args, x_addr);
    const auto gate_acc = TensorAccessor(gate_args, gate_addr);
    const auto gamma_acc = TensorAccessor(gamma_args, gamma_addr);
#ifdef BACKWARD
    constexpr auto dy_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    const auto dy_acc = TensorAccessor(dy_args, dy_addr);
#endif

    const uint32_t tile_bytes = get_tile_size(cb::x);

    // Constants: all-ones reduce tile, gamma, row-broadcast gamma.
    constexpr uint16_t bf16_one = 0x3F80;
    generate_tile_with_bfloat16_value(cb::ones, bf16_one);

    read_group(cb::gamma, gamma_acc, 0, tile_bytes);
    noc_async_read_barrier();
    cb_push_back(cb::gamma, Gt);
    cb_reserve_back(cb::gamma_b, Gt);
    {
        const uint32_t src = get_read_ptr(cb::gamma);
        const uint32_t dst = get_write_ptr(cb::gamma_b);
        for (uint32_t j = 0; j < Gt; ++j) {
            broadcast_gamma_rows(src + j * tile_bytes, dst + j * tile_bytes);
        }
    }
    cb_push_back(cb::gamma_b, Gt);

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        const uint32_t r = work / num_groups;
        const uint32_t h = work - r * num_groups;
        const uint32_t first_page = r * Wt + h * Gt;

        read_group(cb::x, x_acc, first_page, tile_bytes);
        read_group(cb::gate, gate_acc, first_page, tile_bytes);
#ifdef BACKWARD
        read_group(cb::dy, dy_acc, first_page, tile_bytes);
#endif
        noc_async_read_barrier();
        cb_push_back(cb::x, Gt);
        cb_push_back(cb::gate, Gt);
#ifdef BACKWARD
        cb_push_back(cb::dy, Gt);
#endif
    }
}
