// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-32 DFlash verify: the QKV output's heads split and the Q / K RoPE in one program (dflash/batched.py
// _QKVRopeHeads), reader (NCRISC). One core per (32-row tile r, HPC heads hc HPC .. hc HPC + HPC - 1) of the QKV output
// [rows, (QH + KH + VH) 64] (head h = column tiles 2 h, 2 h + 1); core (x = hc, y = r). Pushes the rotation matrix
// tile, the row tile's cos / sin (2 tiles each, [rows, 64]) and the heads' QKV tiles.
// common runtime args: [qkv_addr, cos_addr, sin_addr, trans_addr]
// compile-time args: [cb_x, cb_cos, cb_sin, cb_t, HPC, NT (QKV column tiles), accessors (qkv, cos, sin, trans)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_cos = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sin = get_compile_time_arg_val(2);
    constexpr uint32_t cb_t = get_compile_time_arg_val(3);
    constexpr uint32_t HPC = get_compile_time_arg_val(4);
    constexpr uint32_t NT = get_compile_time_arg_val(5);
    constexpr auto qkv_args = TensorAccessorArgs<6>();
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
    cb_reserve_back(cb_cos, 2);
    cb_reserve_back(cb_sin, 2);
    for (uint32_t d = 0; d < 2; ++d) {
        noc_async_read(s_cos.get_noc_addr(r * 2 + d), get_write_ptr(cb_cos) + d * tile, tile);
        noc_async_read(s_sin.get_noc_addr(r * 2 + d), get_write_ptr(cb_sin) + d * tile, tile);
    }
    cb_reserve_back(cb_x, 2 * HPC);
    for (uint32_t i = 0; i < 2 * HPC; ++i) {
        noc_async_read(s_qkv.get_noc_addr(r * NT + hc * 2 * HPC + i), get_write_ptr(cb_x) + i * tile, tile);
    }
    noc_async_read_barrier();
    cb_push_back(cb_t, 1);
    cb_push_back(cb_cos, 2);
    cb_push_back(cb_sin, 2);
    cb_push_back(cb_x, 2 * HPC);
}
