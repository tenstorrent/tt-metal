// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 batched prefill (tt/prefill_tp.py EpMoe, GLM53_PF_MOE_DISPATCH=rows): the EP dispatch gather bounded by the
// rows the routing plan USES. xd[d] = x[disp[d]] for d < used (= the end of the last local expert's tile-aligned
// region, word 0 of the plan's meta tensor); rows d >= used are never read by moe_fused_swiglu (it reads its experts'
// regions only) and are not written. ROW_MAJOR rows of ROW_BYTES; x = the chunk's rows + 32 zero rows (row C = the
// padding rows' source). Compile-time: ROW_BYTES, CAP (dispatch rows), DEPTH (rows in flight). Runtime per core: x,
// disp, meta, xd addresses, first dispatch row, row count of this core. CB DISP_CB: CAP x 4 (aligned) + 64 + DEPTH x
// ROW_BYTES.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(0);
    constexpr uint32_t CAP = get_compile_time_arg_val(1);
    constexpr uint32_t DEPTH = get_compile_time_arg_val(2);
    constexpr uint32_t IDX_BYTES = (CAP * 4 + 63) / 64 * 64;
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t disp_addr = get_arg_val<uint32_t>(1);
    const uint32_t meta_addr = get_arg_val<uint32_t>(2);
    const uint32_t xd_addr = get_arg_val<uint32_t>(3);
    const uint32_t r0 = get_arg_val<uint32_t>(4);
    const uint32_t nr = get_arg_val<uint32_t>(5);
    if (nr == 0) {
        return;
    }
    const uint32_t base = get_write_ptr(DISP_CB);
    const uint32_t meta_l1 = base + IDX_BYTES;
    const uint32_t buf = meta_l1 + 64;
    const auto meta_g = TensorAccessor(tensor_accessor::make_interleaved_dspec</*is_dram=*/true>(), meta_addr, 32 * 4);
    noc_async_read(meta_g.get_noc_addr(0), meta_l1, 32 * 4);
    noc_async_read_barrier();
    const uint32_t used = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_l1)[0];
    const uint32_t end = (r0 + nr < used) ? r0 + nr : used;
    if (r0 >= end) {
        return;  // every read above has completed (barrier): nothing in flight
    }
    const auto disp_g = TensorAccessor(tensor_accessor::make_interleaved_dspec</*is_dram=*/true>(), disp_addr, CAP * 4);
    noc_async_read(disp_g.get_noc_addr(0), base, CAP * 4);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* disp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    const auto xg = TensorAccessor(tensor_accessor::make_interleaved_dspec</*is_dram=*/true>(), x_addr, ROW_BYTES);
    const auto og = TensorAccessor(tensor_accessor::make_interleaved_dspec</*is_dram=*/true>(), xd_addr, ROW_BYTES);
    for (uint32_t d0 = r0; d0 < end; d0 += DEPTH) {
        const uint32_t k = (end - d0 < DEPTH) ? end - d0 : DEPTH;
        for (uint32_t i = 0; i < k; ++i) {
            noc_async_read(xg.get_noc_addr(disp[d0 + i]), buf + i * ROW_BYTES, ROW_BYTES);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < k; ++i) {
            noc_async_write(buf + i * ROW_BYTES, og.get_noc_addr(d0 + i), ROW_BYTES);
        }
        noc_async_write_barrier();
    }
}
