// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B49 iHC prefill K2 (combine) reader: core r reads the 5 constant tiles (ones -> cb_cst; SCALE, BASE, MUL, ADD ->
// cb_cst2) and the NCH (M, Q) partial pairs of token tile-row r -> cb_part pages [M_0..M_{NCH-1}, Q_0..Q_{NCH-1}]. CT:
// 0 cb_cst, 1 cb_part, 2 NCH, 3 cb_cst2, accessors cst, part. Common: addr cst, addr part. RT: r.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_cst = get_compile_time_arg_val(0);
    constexpr uint32_t cb_part = get_compile_time_arg_val(1);
    constexpr uint32_t NCH = get_compile_time_arg_val(2);
    constexpr uint32_t cb_cst2 = get_compile_time_arg_val(3);
    constexpr auto c_args = TensorAccessorArgs<4>();
    constexpr auto p_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    const uint32_t r = get_arg_val<uint32_t>(0);
    const auto c_acc = TensorAccessor(c_args, get_common_arg_val<uint32_t>(0), 4096);
    const auto p_acc = TensorAccessor(p_args, get_common_arg_val<uint32_t>(1), 4096);
    Noc noc;
    experimental::CB cst(cb_cst), cst2(cb_cst2), part(cb_part);
    cst.reserve_back(1);
    cst2.reserve_back(4);
    part.reserve_back(2 * NCH);
    noc.async_read(c_acc, cst, 4096, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    for (uint32_t t = 1; t < 5; ++t) {
        noc.async_read(c_acc, cst2, 4096, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = (t - 1) * 4096});
    }
    for (uint32_t k = 0; k < NCH; ++k) {
        noc.async_read(
            p_acc, part, 4096, {.page_id = (r * NCH + k) * 2, .offset_bytes = 0}, {.offset_bytes = k * 4096});
        noc.async_read(
            p_acc,
            part,
            4096,
            {.page_id = (r * NCH + k) * 2 + 1, .offset_bytes = 0},
            {.offset_bytes = (NCH + k) * 4096});
    }
    noc.async_read_barrier();
    cst.push_back(1);
    cst2.push_back(4);
    part.push_back(2 * NCH);
}
