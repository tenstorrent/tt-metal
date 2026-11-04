// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// otherlayers: reader of the indexer-q RoPE program, variant 2: the cos / sin table rows are NOT replicated over the 32
// rows by RISC copy loops; the raw table tiles (row 0 valid) are read whole and the compute kernel broadcasts row 0
// with the unpacker (unary_bcast ROW, pure data movement: the same replicated operand as qpost_reader.cpp). x gather
// and P construction as qpost_reader.cpp.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ol_zones.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_r = get_compile_time_arg_val(1);  // raw c tile
    constexpr bool DIRECT =
        get_compile_time_arg_val(2) != 0;  // qlin is L1 interleaved: 32 B row pieces read straight into the x tile
    constexpr uint32_t cb_xs = get_compile_time_arg_val(3);  // scratch, 4096 B: 64 x 64 B pieces of the qlin rows
    constexpr uint32_t cb_r2 = get_compile_time_arg_val(4);  // P
    constexpr uint32_t cb_ri = get_compile_time_arg_val(5);  // raw s tile
    constexpr auto q_args = TensorAccessorArgs<6>();
    constexpr auto c_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();

    const uint32_t t = ol_core_idx(get_common_arg_val<uint32_t>(3));  // tile index r * 4 + c
    const uint32_t q_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t c_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t s_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t r = t >> 2, c = t & 3;

    Noc noc;
    const auto q_acc = TensorAccessor(q_args, q_addr, 2048);
    const auto c_acc = TensorAccessor(c_args, c_addr, 2048);
    const auto s_acc = TensorAccessor(s_args, s_addr, 2048);
    OLZ("Z_RALL");
    experimental::CB cbx(cb_x), cbr(cb_r), cbxs(cb_xs), cbr2(cb_r2), cbri(cb_ri);
    cbx.reserve_back(1);
    cbr.reserve_back(1);
    cbxs.reserve_back(1);
    cbr2.reserve_back(1);
    cbri.reserve_back(1);

    if constexpr (DIRECT) {
        // qlin is L1 interleaved (16 B aligned reads): the 32 B row pieces go straight into the face rows of the x tile
        // (no scratch, no RISC copy loop)
        const uint32_t ro = (((r >> 4) << 1) * 512) + ((r & 15) * 32);
        for (uint32_t h = 0; h < 32; ++h) {
            const uint32_t d0 = (((h >> 4) << 1) * 512) + ((h & 15) * 32);
            noc.async_read(q_acc, cbx, 32, {.page_id = h * 4 + c, .offset_bytes = ro}, {.offset_bytes = d0});
            noc.async_read(
                q_acc, cbx, 32, {.page_id = h * 4 + c, .offset_bytes = ro + 512}, {.offset_bytes = d0 + 512});
        }
    } else {
        const uint32_t roff = (((r >> 4) << 1) * 512) + (((r & 15) & ~1u) * 32);
        for (uint32_t h = 0; h < 32; ++h) {
            noc.async_read(
                q_acc, cbxs, 64, {.page_id = h * 4 + c, .offset_bytes = roff}, {.offset_bytes = (h * 2) * 64});
            noc.async_read(
                q_acc,
                cbxs,
                64,
                {.page_id = h * 4 + c, .offset_bytes = roff + 512},
                {.offset_bytes = (h * 2 + 1) * 64});
        }
    }
    noc.async_read(c_acc, cbr, 2048, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read(s_acc, cbri, 2048, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = 0});
    if constexpr (!DIRECT) {
        noc.async_write_zeros(cbx, 2048, {.offset_bytes = 0});
    }
    noc.async_write_zeros(cbr2, 2048, {.offset_bytes = 0});
    {
        OLZ("Z_WAIT");
        noc.write_zeros_l1_barrier();
        noc.async_read_barrier();
    }
    {
        if constexpr (!DIRECT) {
            volatile tt_l1_ptr uint32_t* xs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbxs.get_write_ptr());
            volatile tt_l1_ptr uint32_t* xt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbx.get_write_ptr());
            const uint32_t po = (r & 1) * 8;  // words
            for (uint32_t h = 0; h < 32; ++h) {
                volatile tt_l1_ptr uint32_t* d = xt + ((h >> 4) << 8) + ((h & 15) << 3);
                volatile tt_l1_ptr uint32_t* s0 = xs + (h * 2) * 16 + po;
                volatile tt_l1_ptr uint32_t* s1 = xs + (h * 2 + 1) * 16 + po;
                for (uint32_t q = 0; q < 8; ++q) {
                    d[q] = s0[q];
                    d[128 + q] = s1[q];
                }
            }
        }
        if (c >= 2) {
            volatile tt_l1_ptr uint16_t* R2 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbr2.get_write_ptr());
#define EL(rr, qq) ((((((rr) >> 4) << 1) + ((qq) >> 4)) << 8) + (((rr) & 15) << 4) + ((qq) & 15))
            for (uint32_t e = 0; e < 32; e += 2) {
                R2[EL(e, e + 1)] = 0x3F80;
                R2[EL(e + 1, e)] = 0xBF80;
            }
#undef EL
        }
    }
    cbr2.push_back(1);
    cbri.push_back(1);
    cbr.push_back(1);
    cbx.push_back(1);
}
