// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// GDN decode conv writer: per tile-column, write the conv output tile, then perform
// the shift-register writeback st0<-st1, st1<-st2, st2<-st3, st3<-qkv from the
// reader's cb_shift copies. Waiting on cb_out first is the ordering guarantee: the
// compute kernel has consumed cb_x by then, so the reader's DRAM reads of the old
// state are complete before this kernel overwrites that state.

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_shift = 2;
constexpr uint32_t cb_out = 3;

void kernel_main() {
    constexpr bool conv_is_dram = get_named_compile_time_arg_val("conv_is_dram") == 1;
    constexpr bool st_is_dram = get_named_compile_time_arg_val("st_is_dram") == 1;

    const uint32_t conv_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t st0_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t st1_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t st2_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t st3_addr = get_common_arg_val<uint32_t>(4);

    const uint32_t wi_start = get_arg_val<uint32_t>(0);
    const uint32_t wi_count = get_arg_val<uint32_t>(1);

    const uint32_t tb = get_tile_size(cb_out);
    const auto conv_acc = TensorAccessor(tensor_accessor::make_interleaved_dspec<conv_is_dram>(), conv_addr, tb);
    const auto st0_acc = TensorAccessor(tensor_accessor::make_interleaved_dspec<st_is_dram>(), st0_addr, tb);
    const auto st1_acc = TensorAccessor(tensor_accessor::make_interleaved_dspec<st_is_dram>(), st1_addr, tb);
    const auto st2_acc = TensorAccessor(tensor_accessor::make_interleaved_dspec<st_is_dram>(), st2_addr, tb);
    const auto st3_acc = TensorAccessor(tensor_accessor::make_interleaved_dspec<st_is_dram>(), st3_addr, tb);

    for (uint32_t wi = wi_start; wi < wi_start + wi_count; wi++) {
        cb_wait_front(cb_out, 1);
        uint32_t l1 = get_read_ptr(cb_out);
        noc_async_write_page(wi, conv_acc, l1, tb, 0);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);

        cb_wait_front(cb_shift, 4);
        l1 = get_read_ptr(cb_shift);
        noc_async_write_page(wi, st0_acc, l1 + 0 * tb, tb, 0);
        noc_async_write_page(wi, st1_acc, l1 + 1 * tb, tb, 0);
        noc_async_write_page(wi, st2_acc, l1 + 2 * tb, tb, 0);
        noc_async_write_page(wi, st3_acc, l1 + 3 * tb, tb, 0);
        noc_async_write_barrier();
        cb_pop_front(cb_shift, 4);
    }
}
