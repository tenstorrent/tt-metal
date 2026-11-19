// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t face(uint32_t r, uint32_t c) { return (r >> 4) * 512 + (c >> 4) * 256 + (r & 15u) * 16 + (c & 15u); }
void kernel_main() {
    constexpr auto oa = TensorAccessorArgs<0>();
    constexpr auto pa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(0), 4096);
    const auto positions = TensorAccessor(pa, get_arg_val<uint32_t>(1), 4096);
    const uint32_t first = get_arg_val<uint32_t>(2), stride = get_arg_val<uint32_t>(3);
    const uint32_t kt = get_arg_val<uint32_t>(4), rows = get_arg_val<uint32_t>(5);
    cb_reserve_back(5, 1);
    cb_reserve_back(6, 1);
    const uint32_t output_address = get_write_ptr(5), positions_address = get_write_ptr(6);
    noc_async_read(positions.get_noc_addr(0), positions_address, 4096);
    noc_async_read_barrier();
    const volatile tt_l1_ptr uint32_t* pos = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(positions_address);
    volatile tt_l1_ptr uint32_t* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(output_address);
    uint32_t live[32];
    for (uint32_t q = 0; q < rows; ++q) {
        const uint32_t bits = pos[face(q, 0)];
        const uint32_t exponent = (bits >> 23) & 255u;
        const uint32_t position = bits == 0 ? 0 : (((bits & 0x7fffffu) | 0x800000u) >> (150u - exponent));
        live[q] = (position + 1) / 4;
    }
    for (uint32_t job = first; job < kt * 4; job += stride) {
        const uint32_t key_tile = job % kt;
        for (uint32_t i = 0; i < 1024; ++i) {
            tile[i] = 0;
        }
        for (uint32_t q = 0; q < rows; ++q) {
            if (key_tile * 32 >= live[q]) {
                continue;
            }
            cb_wait_front(4, 1);
            const volatile tt_l1_ptr uint32_t* dots =
                reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(get_read_ptr(4));
            for (uint32_t k = 0; k < 32 && key_tile * 32 + k < live[q]; ++k) {
                tile[face(q, k)] = dots[face(k, 0)];
            }
            cb_pop_front(4, 1);
        }
        noc_async_write(output_address, output.get_noc_addr(job), 4096);
        noc_async_write_barrier();
    }
}
