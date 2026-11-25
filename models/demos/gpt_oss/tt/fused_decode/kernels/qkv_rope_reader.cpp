// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-32 DFlash verify: the QKV output's heads split and the Q / K RoPE in one program (dflash/batched.py
// _QKVRopeHeads), reader (NCRISC). One core per (32-row tile r, HPC heads hc HPC .. hc HPC + HPC - 1) of the QKV output
// [rows, (QH + KH + VH) 64] (head h = column tiles 2 h, 2 h + 1); core (x = hc, y = r). Pushes the rotation matrix
// tile, the row tile's cos / sin (2 tiles each, [rows, 64]) and the heads' QKV tiles.
// QR_HEADNORM (the drafter: per-head RMSNorm before the RoPE): also the reduce scaler (cb_sc) and per normed head its
// weights' 2 tiles (cb_g; row 0 = the 64 weights; heads < QR_NQ: gq, then gk).
// common runtime args: [qkv_addr, cos_addr, sin_addr, trans_addr, (gq_addr, gk_addr)]
// compile-time args: [cb_x, cb_cos, cb_sin, cb_t, HPC, NT (QKV column tiles), cb_g, cb_sc, NORM_HEADS, accessors (qkv,
// cos, sin, trans, (gq, gk))]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_cos = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sin = get_compile_time_arg_val(2);
    constexpr uint32_t cb_t = get_compile_time_arg_val(3);
    constexpr uint32_t HPC = get_compile_time_arg_val(4);
    constexpr uint32_t NT = get_compile_time_arg_val(5);
    constexpr uint32_t cb_g = get_compile_time_arg_val(6);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(7);
    constexpr uint32_t NORM_HEADS = get_compile_time_arg_val(8);
    constexpr auto qkv_args = TensorAccessorArgs<9>();
    constexpr auto cos_args = TensorAccessorArgs<qkv_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile = 2048;
    const auto s_qkv = TensorAccessor(qkv_args, get_common_arg_val<uint32_t>(0), tile);
    const auto s_cos = TensorAccessor(cos_args, get_common_arg_val<uint32_t>(1), tile);
    const auto s_sin = TensorAccessor(sin_args, get_common_arg_val<uint32_t>(2), tile);
    const auto s_t = TensorAccessor(t_args, get_common_arg_val<uint32_t>(3), tile);
    const uint32_t hc = get_absolute_logical_x(), r = get_absolute_logical_y();

    cb_reserve_back(cb_t, 1);
    noc_async_read(s_t.get_noc_addr(0), get_write_ptr(cb_t), tile);
#ifdef QR_HEADNORM
    {
        constexpr auto gq_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
        constexpr auto gk_args = TensorAccessorArgs<gq_args.next_compile_time_args_offset()>();
        const auto s_gq = TensorAccessor(gq_args, get_common_arg_val<uint32_t>(4), tile);
        const auto s_gk = TensorAccessor(gk_args, get_common_arg_val<uint32_t>(5), tile);
        cb_reserve_back(cb_g, 2 * HPC);
        for (uint32_t i = 0; i < 2 * HPC; ++i) {
            const uint32_t h = hc * HPC + i / 2, d = i % 2;
            if (h < NORM_HEADS) {
                const uint64_t src = (h < QR_NQ ? s_gq.get_noc_addr(d) : s_gk.get_noc_addr(d));
                noc_async_read(src, get_write_ptr(cb_g) + i * tile, tile);
            }
        }
        cb_reserve_back(cb_sc, 1);
        const uint32_t sc = get_write_ptr(cb_sc);
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t z = 0; z < tile; z += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, sc + z, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
        volatile tt_l1_ptr uint16_t* s16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sc);
        for (uint32_t f = 0; f < 4; ++f) {
            for (uint32_t c = 0; c < 16; ++c) {
                s16[f * 256 + c] = 0x3F80;
            }
        }
        cb_push_back(cb_sc, 1);
        cb_push_back(cb_g, 2 * HPC);
    }
#endif
    noc_async_read_barrier();
    cb_push_back(cb_t, 1);
#ifndef QR_MPC
#define QR_MPC 1
#endif
    // QR_MPC row tiles per core: row tile y QR_MPC + mi, one at a time
    for (uint32_t mi = 0; mi < QR_MPC; ++mi) {
        const uint32_t rt = r * QR_MPC + mi;
        cb_reserve_back(cb_cos, 2);
        cb_reserve_back(cb_sin, 2);
        for (uint32_t d = 0; d < 2; ++d) {
            noc_async_read(s_cos.get_noc_addr(rt * 2 + d), get_write_ptr(cb_cos) + d * tile, tile);
            noc_async_read(s_sin.get_noc_addr(rt * 2 + d), get_write_ptr(cb_sin) + d * tile, tile);
        }
        cb_reserve_back(cb_x, 2 * HPC);
        for (uint32_t i = 0; i < 2 * HPC; ++i) {
            noc_async_read(s_qkv.get_noc_addr(rt * NT + hc * 2 * HPC + i), get_write_ptr(cb_x) + i * tile, tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_cos, 2);
        cb_push_back(cb_sin, 2);
        cb_push_back(cb_x, 2 * HPC);
    }
}
