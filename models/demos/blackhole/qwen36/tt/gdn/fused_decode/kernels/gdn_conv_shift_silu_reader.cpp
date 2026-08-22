// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// GDN decode conv reader: per tile-column wi, feed the compute kernel the four
// FIR inputs (old st1, st2, st3, current qkv from qkvzab) plus the four taps,
// and separately feed the writer the same four input tiles (cb_shift) so it can
// perform the shift-register writeback without re-touching DRAM state that the
// compute pipeline is still reading.

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_x = 0;      // conv inputs, 4 tiles per wi: [st1, st2, st3, qkv]
constexpr uint32_t cb_taps = 1;   // taps, 4 tiles per wi: [tap0, tap1, tap2, tap3]
constexpr uint32_t cb_shift = 2;  // writer copy of [st1, st2, st3, qkv]

void kernel_main() {
    constexpr bool qkvzab_is_dram = get_named_compile_time_arg_val("qkvzab_is_dram") == 1;
    constexpr bool st_is_dram = get_named_compile_time_arg_val("st_is_dram") == 1;
    constexpr bool tap_is_dram = get_named_compile_time_arg_val("tap_is_dram") == 1;

    const uint32_t qkvzab_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t st1_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t st2_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t st3_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t tap0_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t tap1_addr = get_common_arg_val<uint32_t>(5);
    const uint32_t tap2_addr = get_common_arg_val<uint32_t>(6);
    const uint32_t tap3_addr = get_common_arg_val<uint32_t>(7);

    const uint32_t wi_start = get_arg_val<uint32_t>(0);
    const uint32_t wi_count = get_arg_val<uint32_t>(1);

    const uint32_t tb = get_tile_size(cb_x);  // all operands bf16
    const auto qkvzab_acc = TensorAccessor(tensor_accessor::make_interleaved_dspec<qkvzab_is_dram>(), qkvzab_addr, tb);
    const auto st1_acc = TensorAccessor(tensor_accessor::make_interleaved_dspec<st_is_dram>(), st1_addr, tb);
    const auto st2_acc = TensorAccessor(tensor_accessor::make_interleaved_dspec<st_is_dram>(), st2_addr, tb);
    const auto st3_acc = TensorAccessor(tensor_accessor::make_interleaved_dspec<st_is_dram>(), st3_addr, tb);
    const auto tap0_acc = TensorAccessor(tensor_accessor::make_interleaved_dspec<tap_is_dram>(), tap0_addr, tb);
    const auto tap1_acc = TensorAccessor(tensor_accessor::make_interleaved_dspec<tap_is_dram>(), tap1_addr, tb);
    const auto tap2_acc = TensorAccessor(tensor_accessor::make_interleaved_dspec<tap_is_dram>(), tap2_addr, tb);
    const auto tap3_acc = TensorAccessor(tensor_accessor::make_interleaved_dspec<tap_is_dram>(), tap3_addr, tb);

    for (uint32_t wi = wi_start; wi < wi_start + wi_count; wi++) {
        cb_reserve_back(cb_x, 4);
        uint32_t l1 = get_write_ptr(cb_x);
        noc_async_read_page(wi, st1_acc, l1 + 0 * tb);
        noc_async_read_page(wi, st2_acc, l1 + 1 * tb);
        noc_async_read_page(wi, st3_acc, l1 + 2 * tb);
        noc_async_read_page(wi, qkvzab_acc, l1 + 3 * tb);  // qkv = leading columns of qkvzab
        noc_async_read_barrier();
        cb_push_back(cb_x, 4);

        cb_reserve_back(cb_taps, 4);
        l1 = get_write_ptr(cb_taps);
        noc_async_read_page(wi, tap0_acc, l1 + 0 * tb);
        noc_async_read_page(wi, tap1_acc, l1 + 1 * tb);
        noc_async_read_page(wi, tap2_acc, l1 + 2 * tb);
        noc_async_read_page(wi, tap3_acc, l1 + 3 * tb);
        noc_async_read_barrier();
        cb_push_back(cb_taps, 4);

        cb_reserve_back(cb_shift, 4);
        l1 = get_write_ptr(cb_shift);
        noc_async_read_page(wi, st1_acc, l1 + 0 * tb);
        noc_async_read_page(wi, st2_acc, l1 + 1 * tb);
        noc_async_read_page(wi, st3_acc, l1 + 2 * tb);
        noc_async_read_page(wi, qkvzab_acc, l1 + 3 * tb);
        noc_async_read_barrier();
        cb_push_back(cb_shift, 4);
    }
}
