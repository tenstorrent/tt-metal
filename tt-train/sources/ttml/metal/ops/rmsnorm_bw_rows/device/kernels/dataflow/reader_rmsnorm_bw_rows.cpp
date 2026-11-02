// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// rmsnorm_bw_rows reader (one item (row r, slice s)): constants, the rms tile, the resident input / gamma / dL_dout
// slices pushed in groups of `block` tiles as they arrive (iteration 41: pass 1 runs under the reads); then the
// row's S partial tiles once the row's cores have written them (semaphore == S); then the residual gradient
// streamed in blocks.

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/rmsnorm_bw_rows/device/kernels/rmsnorm_bw_rows_cbs.hpp"

namespace cb = rmsnorm_bw_rows_cb;

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t S = get_compile_time_arg_val(1);
constexpr uint32_t St = get_compile_time_arg_val(2);
constexpr uint32_t block = get_compile_time_arg_val(3);
constexpr uint32_t exchange_sem_id = get_compile_time_arg_val(4);
constexpr auto a_args = TensorAccessorArgs<5>();
constexpr auto gamma_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
constexpr auto dy_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
constexpr auto rms_args = TensorAccessorArgs<dy_args.next_compile_time_args_offset()>();
#ifdef ADD_RESIDUAL_GRAD
constexpr auto dres_args = TensorAccessorArgs<rms_args.next_compile_time_args_offset()>();
#endif

FORCE_INLINE void generate_ones_row0_tile(uint32_t cb_id) {
    cb_reserve_back(cb_id, 1);
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id));
    for (uint32_t i = 0; i < 1024U; ++i) {
        ptr[i] = 0;
    }
    for (uint32_t c = 0; c < 16U; ++c) {
        ptr[c] = 0x3F80;
        ptr[256U + c] = 0x3F80;
    }
    cb_push_back(cb_id, 1);
}

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t a_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t dy_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t rms_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t dres_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t r = get_arg_val<uint32_t>(arg++);
    const uint32_t s = get_arg_val<uint32_t>(arg++);

    const uint32_t tile_bytes = get_tile_size(cb::a);
    const auto a_acc = TensorAccessor(a_args, a_addr);
    const auto gamma_acc = TensorAccessor(gamma_args, gamma_addr);
    const auto dy_acc = TensorAccessor(dy_args, dy_addr);
    const auto rms_acc = TensorAccessor(rms_args, rms_addr);
#ifdef ADD_RESIDUAL_GRAD
    const auto dres_acc = TensorAccessor(dres_args, dres_addr);
#endif

    generate_tile_with_bfloat16_value(cb::zero, 0x0);
    generate_tile_with_bfloat16_value(cb::ones, 0x3F80);
    generate_ones_row0_tile(cb::ones_row0);

    const uint32_t col0 = s * St;
    const uint32_t ncols = (col0 + St <= Wt) ? St : (Wt - col0);
    const uint32_t page = r * Wt + col0;

    // Resident slices (a full St reservation each, ncols tiles pushed in groups of `block`: the CBs never wrap, the
    // compute waits cumulatively and pops ncols at the end); the rms tile with the first group.
    cb_reserve_back(cb::a, St);
    cb_reserve_back(cb::gamma, St);
    cb_reserve_back(cb::dy, St);
    cb_reserve_back(cb::rms, 1);
    noc_async_read_page(r, rms_acc, get_write_ptr(cb::rms));
    uint32_t l1_a = get_write_ptr(cb::a);
    uint32_t l1_g = get_write_ptr(cb::gamma);
    uint32_t l1_d = get_write_ptr(cb::dy);
    for (uint32_t c = 0; c < ncols; c += block) {
        const uint32_t n = (c + block <= ncols) ? block : (ncols - c);
        for (uint32_t j = 0; j < n; ++j) {
            noc_async_read_page(page + c + j, a_acc, l1_a);
            noc_async_read_page(col0 + c + j, gamma_acc, l1_g);
            noc_async_read_page(page + c + j, dy_acc, l1_d);
            l1_a += tile_bytes;
            l1_g += tile_bytes;
            l1_d += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb::a, n);
        cb_push_back(cb::gamma, n);
        cb_push_back(cb::dy, n);
        if (c == 0) {
            cb_push_back(cb::rms, 1);
        }
    }

    // The row's partial tiles: the row's S writers deposit slot j of cb::partials on every core of the row and
    // bump the semaphore; publish them to the compute once all S have arrived.
    cb_reserve_back(cb::partials, S);
    auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(exchange_sem_id));
    noc_semaphore_wait(sem, S);
    noc_semaphore_set(sem, 0);
    cb_push_back(cb::partials, S);

#ifdef ADD_RESIDUAL_GRAD
    for (uint32_t c = 0; c < ncols; c += block) {
        const uint32_t n = (c + block <= ncols) ? block : (ncols - c);
        cb_reserve_back(cb::dres, block);
        uint32_t l1 = get_write_ptr(cb::dres);
        for (uint32_t j = 0; j < n; ++j) {
            noc_async_read_page(page + c + j, dres_acc, l1);
            l1 += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb::dres, block);
    }
#endif
}
