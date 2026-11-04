// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// otherlayers: fused verify-indexer score, writer, PER-USER split (see score3/score_reader.cpp): the output tiles of
// this core (row rl = the scores of global row g = user * NROW + rl) -> row-major bf16 rows of [R, NS]; entries >=
// nvalid[g] are set to NEG (the validity mask). The core index (user, local index) comes from the logical coordinates
// (grid width = common arg 3).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/metadata_scalar_read.hpp"
#include "../ol_zones.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(1);
    constexpr uint32_t NROW = get_compile_time_arg_val(2);
    constexpr uint32_t CH = get_compile_time_arg_val(3);  // key tile columns per staged chunk
    constexpr uint32_t NS = get_compile_time_arg_val(4);
    constexpr uint32_t NEG = get_compile_time_arg_val(5);  // bf16 bits of -1e30
    constexpr uint32_t CB_NV = get_compile_time_arg_val(6);
    constexpr uint32_t CPU = get_compile_time_arg_val(7);
    constexpr uint32_t NCT_TOTAL = get_compile_time_arg_val(8);
    constexpr auto o_args = TensorAccessorArgs<9>();
    constexpr auto nv_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<nv_args.next_compile_time_args_offset()>();
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t nv_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t core_idx = ol_core_idx(get_common_arg_val<uint32_t>(3));
    const uint32_t user = core_idx / CPU, li = core_idx % CPU;

    Noc noc;
    const auto oa = TensorAccessor(o_args, o_addr, NS * 2);
    const auto nva = TensorAccessor(nv_args, nv_addr, 128);
    experimental::CB out(cb_out), stage(cb_stage), cbnv(CB_NV);

    OLZ("Z_WALL");
    cbnv.reserve_back(1);
    const uint32_t valid = trace_metadata::read_metadata_scalar_u32(noc, v_args, v_addr, cbnv.get_write_ptr() + 128);
    uint32_t live = (valid + 31) / 32;
    if (live > NCT_TOTAL) {
        live = NCT_TOTAL;
    }
    const uint32_t per = live / CPU, extra = live % CPU;
    const uint32_t ct0 = li * per + (li < extra ? li : extra);
    const uint32_t nct = per + (li < extra ? 1 : 0);
    noc.async_read(nva, cbnv, 128, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    volatile tt_l1_ptr uint32_t* nv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbnv.get_write_ptr());

    stage.reserve_back(1);
    volatile tt_l1_ptr uint16_t* st = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage.get_write_ptr());
    for (uint32_t c0 = 0; c0 < nct; c0 += CH) {
        const uint32_t m = (nct - c0 < CH) ? (nct - c0) : CH;
        for (uint32_t j = 0; j < m; ++j) {
            OLZ("Z_WCOL");
            out.wait_front(1);
            volatile tt_l1_ptr uint16_t* t = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out.get_read_ptr());
            const uint32_t e0 = (ct0 + c0 + j) * 32;
            for (uint32_t rl = 0; rl < NROW; ++rl) {
                const uint32_t f0 = ((rl >> 4) << 1) * 256 + (rl & 15) * 16;
                const uint32_t nvr = nv[user * NROW + rl];
                if (e0 + 32 <= nvr) {
                    // fully valid tile column (all but the last live one): the two face halves as 32-bit words (8 words
                    // each)
                    volatile tt_l1_ptr uint32_t* sw =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(st + rl * (CH * 32) + j * 32);
                    volatile tt_l1_ptr uint32_t* tw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(t + f0);
                    for (uint32_t q = 0; q < 8; ++q) {
                        sw[q] = tw[q];
                        sw[8 + q] = tw[128 + q];
                    }
                } else {
                    for (uint32_t c = 0; c < 32; ++c) {
                        uint16_t v = t[f0 + (c >> 4) * 256 + (c & 15)];
                        st[rl * (CH * 32) + j * 32 + c] = (e0 + c < nvr) ? v : (uint16_t)NEG;
                    }
                }
            }
            out.pop_front(1);
        }
        for (uint32_t rl = 0; rl < NROW; ++rl) {
            noc.async_write(
                stage,
                oa,
                m * 64,
                {.offset_bytes = rl * CH * 64},
                {.page_id = user * NROW + rl, .offset_bytes = (ct0 + c0) * 64});
        }
        noc.async_write_barrier();
    }
}
