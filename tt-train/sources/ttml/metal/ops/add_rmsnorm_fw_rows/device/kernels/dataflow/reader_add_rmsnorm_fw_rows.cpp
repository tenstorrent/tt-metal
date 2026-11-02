// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// add_rmsnorm_fw_rows reader (one item (row r, slice s)): constants, the resident gamma slice, the residual and
// delta slices streamed in groups of two tiles; then the row's S partial tiles once the row's cores have written
// them (semaphore == S).

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/add_rmsnorm_fw_rows/device/kernels/add_rmsnorm_fw_rows_cbs.hpp"

namespace cb = add_rmsnorm_fw_rows_cb;

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t S = get_compile_time_arg_val(1);
constexpr uint32_t St = get_compile_time_arg_val(2);
constexpr uint32_t block = get_compile_time_arg_val(3);
constexpr uint32_t group = 2U;  // tiles per pass-1 group (CBs hold two groups)
constexpr uint32_t exchange_sem_id = get_compile_time_arg_val(4);
constexpr uint32_t Ht = get_compile_time_arg_val(5);  // tile rows of the residual: delta^T's tile-row stride
constexpr auto residual_args = TensorAccessorArgs<6>();
constexpr auto delta_args = TensorAccessorArgs<residual_args.next_compile_time_args_offset()>();
constexpr auto gamma_args = TensorAccessorArgs<delta_args.next_compile_time_args_offset()>();

template <typename Accessor>
FORCE_INLINE void read_resident(
    uint32_t cb_id, const Accessor& acc, uint32_t first_page, uint32_t n, uint32_t tile_bytes) {
    cb_reserve_back(cb_id, St);
    uint32_t l1 = get_write_ptr(cb_id);
    for (uint32_t j = 0; j < n; ++j) {
        noc_async_read_page(first_page + j, acc, l1);
        l1 += tile_bytes;
    }
}

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t residual_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t delta_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t r = get_arg_val<uint32_t>(arg++);
    const uint32_t s = get_arg_val<uint32_t>(arg++);

    const uint32_t tile_bytes = get_tile_size(cb::residual);
    const auto residual_acc = TensorAccessor(residual_args, residual_addr);
    const auto delta_acc = TensorAccessor(delta_args, delta_addr);
    const auto gamma_acc = TensorAccessor(gamma_args, gamma_addr);

    generate_tile_with_bfloat16_value(cb::zero, 0x0);
    generate_tile_with_bfloat16_value(cb::ones, 0x3F80);

    const uint32_t col0 = s * St;
    const uint32_t ncols = (col0 + St <= Wt) ? St : (Wt - col0);
    const uint32_t page = r * Wt + col0;

    // Resident gamma slice (a full St reservation; a shorter last slice leaves the tail unused).
    read_resident(cb::gamma, gamma_acc, col0, ncols, tile_bytes);
    noc_async_read_barrier();
    cb_push_back(cb::gamma, St);

    // Residual and delta streamed in full groups (a shorter last group still pushes `group` tiles).
    for (uint32_t c = 0; c < ncols; c += group) {
        const uint32_t n = (c + group <= ncols) ? group : (ncols - c);
        cb_reserve_back(cb::residual, group);
        cb_reserve_back(cb::delta, group);
        uint32_t l1_r = get_write_ptr(cb::residual);
        uint32_t l1_d = get_write_ptr(cb::delta);
        for (uint32_t j = 0; j < n; ++j) {
            noc_async_read_page(page + c + j, residual_acc, l1_r);
#ifdef DELTA_TRANSPOSED
            // delta is [C, S]: the residual's tile (r, col) is delta's tile (col, r) (iteration 37)
            noc_async_read_page((col0 + c + j) * Ht + r, delta_acc, l1_d);
#else
            noc_async_read_page(page + c + j, delta_acc, l1_d);
#endif
            l1_r += tile_bytes;
            l1_d += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb::residual, group);
        cb_push_back(cb::delta, group);
    }

    // The row's partial tiles: the row's S writers deposit slot j of cb::partials on every core of the row and
    // bump the semaphore; publish them to the compute once all S have arrived.
    cb_reserve_back(cb::partials, S);
    auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(exchange_sem_id));
    noc_semaphore_wait(sem, S);
    noc_semaphore_set(sem, 0);
    cb_push_back(cb::partials, S);
}
