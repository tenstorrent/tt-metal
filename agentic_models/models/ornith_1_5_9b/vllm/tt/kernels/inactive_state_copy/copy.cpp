// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(src_args, get_arg_val<uint32_t>(0), 4096);
    const auto dst = TensorAccessor(dst_args, get_arg_val<uint32_t>(1), 4096);
    const auto mask = TensorAccessor(mask_args, get_arg_val<uint32_t>(2), 128);
    const uint32_t batch = get_arg_val<uint32_t>(3);
    const uint32_t core = get_arg_val<uint32_t>(4);
    const uint32_t scratch = get_write_ptr(0);
    const uint32_t mask_l1 = get_write_ptr(1);
    noc_async_read(mask.get_noc_addr(0), mask_l1, 128);
    noc_async_read_barrier();
    const volatile tt_l1_ptr uint32_t* active = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mask_l1);
    // A slot owns 8 heads * 4 * 4 FP32 tiles. Distinct cores own disjoint tiles;
    // no arithmetic, overlap, or partial-tile writes can change preserved bits.
    for (uint32_t slot = 0; slot < batch; ++slot) {
        if (active[slot] != 0) {
            continue;
        }
        for (uint32_t tile = core; tile < 128; tile += 64) {
            const uint32_t index = slot * 128 + tile;
            noc_async_read(src.get_noc_addr(index), scratch, 4096);
            noc_async_read_barrier();
            noc_async_write(scratch, dst.get_noc_addr(index), 4096);
            noc_async_write_barrier();
        }
    }
}
