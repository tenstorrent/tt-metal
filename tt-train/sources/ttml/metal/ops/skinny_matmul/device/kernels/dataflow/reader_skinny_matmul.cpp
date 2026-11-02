// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for both skinny_matmul phases.
//   phase A: item (row group g, K slice s): the W slice (slice_tiles x Nt tiles, laid out k-major) once, then for
//            every row of the group its x tiles of the slice, `block` at a time
//   phase B (REDUCE): item (row r, n): the S partial tiles of (r, n)

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/skinny_matmul/device/kernels/skinny_matmul_cbs.hpp"

namespace cb = skinny_matmul_cb;

constexpr uint32_t rows = get_compile_time_arg_val(0);
constexpr uint32_t Kt = get_compile_time_arg_val(1);
constexpr uint32_t Nt = get_compile_time_arg_val(2);
constexpr uint32_t R = get_compile_time_arg_val(3);
constexpr uint32_t S = get_compile_time_arg_val(4);
constexpr uint32_t St = get_compile_time_arg_val(5);
constexpr uint32_t block = get_compile_time_arg_val(6);
constexpr uint32_t Kx = get_compile_time_arg_val(7);  // x row stride in tiles (>= Kt)

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t in0_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t in1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t work_start = get_arg_val<uint32_t>(arg++);
    const uint32_t work_count = get_arg_val<uint32_t>(arg++);

    constexpr auto in0_args = TensorAccessorArgs<8>();
    const auto in0_acc = TensorAccessor(in0_args, in0_addr);

#ifdef REDUCE
    const uint32_t tile_bytes = get_tile_size(cb::partials);
    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        const uint32_t r = work / Nt;
        const uint32_t n = work - r * Nt;
        cb_reserve_back(cb::partials, S);
        uint32_t l1 = get_write_ptr(cb::partials);
        for (uint32_t s = 0; s < S; ++s) {
            noc_async_read_page(r * S * Nt + s * Nt + n, in0_acc, l1);
            l1 += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb::partials, S);
    }
#else
    constexpr auto in1_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto w_acc = TensorAccessor(in1_args, in1_addr);
    const uint32_t tile_bytes = get_tile_size(cb::x);
    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        const uint32_t g = work / S;
        const uint32_t s = work - g * S;
        const uint32_t k0 = s * St;
        const uint32_t nk = (k0 + St <= Kt) ? St : (Kt - k0);
        const uint32_t r0 = g * R;
        const uint32_t nr = (r0 + R <= rows) ? R : (rows - r0);

        // W slice, k-major: CB index k*Nt + n <- w tile (n, k) [transpose_b, page n*Kt + k] or (k, n) [page k*Nt + n].
        // Always a full St*Nt reservation (a shorter last slice leaves the tail stale): every push must return the
        // CB pointer to the buffer start, since a block written past the end of the CB is not wrapped.
        cb_reserve_back(cb::w, St * Nt);
        {
            uint32_t l1 = get_write_ptr(cb::w);
            for (uint32_t k = 0; k < nk; ++k) {
                for (uint32_t n = 0; n < Nt; ++n) {
#ifdef TRANSPOSE_B
                    noc_async_read_page(n * Kt + k0 + k, w_acc, l1);
#else
                    noc_async_read_page((k0 + k) * Nt + n, w_acc, l1);
#endif
                    l1 += tile_bytes;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb::w, St * Nt);

        // x tiles in full `block` pushes (the tail of a slice reads fewer tiles into a full block) for the same reason.
        for (uint32_t r = r0; r < r0 + nr; ++r) {
            for (uint32_t k = 0; k < nk; k += block) {
                const uint32_t n = (k + block <= nk) ? block : (nk - k);
                cb_reserve_back(cb::x, block);
                uint32_t l1 = get_write_ptr(cb::x);
                for (uint32_t j = 0; j < n; ++j) {
#ifdef X_TRANSPOSED
                    noc_async_read_page((k0 + k + j) * Kx + r, in0_acc, l1);  // x^T [K', M]: tile (r, k) at k * Mt + r
#else
                    noc_async_read_page(r * Kx + k0 + k + j, in0_acc, l1);
#endif
                    l1 += tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb::x, block);
            }
        }
    }
#endif
}
