// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Fused verify-indexer score, writer: output tiles (rows r valid) -> row-major bf16 rows [R, NS]; entries >= nvalid[r]
// are set to NEG (the validity mask).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/metadata_scalar_read.hpp"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);
    constexpr uint32_t CH = get_compile_time_arg_val(3);  // key tile columns per staged chunk
    constexpr uint32_t NS = get_compile_time_arg_val(4);
    constexpr uint32_t NEG = get_compile_time_arg_val(5);  // bf16 bits of -1e30
    constexpr uint32_t CB_NV = get_compile_time_arg_val(6);
    constexpr uint32_t NCORES = get_compile_time_arg_val(7);
    constexpr uint32_t NCT_TOTAL = get_compile_time_arg_val(8);
    constexpr auto o_args = TensorAccessorArgs<9>();
    constexpr auto nv_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<nv_args.next_compile_time_args_offset()>();
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t nv_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t core_idx = get_arg_val<uint32_t>(0);

    Noc noc;
    const auto oa = TensorAccessor(o_args, o_addr, NS * 2);
    const auto nva = TensorAccessor(nv_args, nv_addr, 128);
    experimental::CB out(cb_out), stage(cb_stage), cbnv(CB_NV);

    cbnv.reserve_back(1);
    const uint32_t valid = trace_metadata::read_metadata_scalar_u32(noc, v_args, v_addr, cbnv.get_write_ptr() + 128);
    uint32_t live = (valid + 31) / 32;
    if (live > NCT_TOTAL) {
        live = NCT_TOTAL;
    }
    const uint32_t per = live / NCORES, extra = live % NCORES;
    const uint32_t ct0 = core_idx * per + (core_idx < extra ? core_idx : extra);
    const uint32_t nct = per + (core_idx < extra ? 1 : 0);
    noc.async_read(nva, cbnv, 128, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    volatile tt_l1_ptr uint32_t* nv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbnv.get_write_ptr());

    stage.reserve_back(1);
    volatile tt_l1_ptr uint16_t* st = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage.get_write_ptr());
    for (uint32_t c0 = 0; c0 < nct; c0 += CH) {
        const uint32_t m = (nct - c0 < CH) ? (nct - c0) : CH;
        for (uint32_t j = 0; j < m; ++j) {
            out.wait_front(1);
            volatile tt_l1_ptr uint16_t* t = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out.get_read_ptr());
            const uint32_t e0 = (ct0 + c0 + j) * 32;
            for (uint32_t r = 0; r < R; ++r) {
                const uint32_t f0 = ((r >> 4) << 1) * 256 + (r & 15) * 16;
                const uint32_t nvr = nv[r];
                for (uint32_t c = 0; c < 32; ++c) {
                    uint16_t v = t[f0 + (c >> 4) * 256 + (c & 15)];
                    st[r * (CH * 32) + j * 32 + c] = (e0 + c < nvr) ? v : (uint16_t)NEG;
                }
            }
            out.pop_front(1);
        }
        for (uint32_t r = 0; r < R; ++r) {
            noc.async_write(
                stage, oa, m * 64, {.offset_bytes = r * CH * 64}, {.page_id = r, .offset_bytes = (ct0 + c0) * 64});
        }
        noc.async_write_barrier();
    }
}
