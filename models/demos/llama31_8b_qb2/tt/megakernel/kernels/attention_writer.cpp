// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#ifndef QB2_ENTRY
#define QB2_ENTRY kernel_main
#endif
#define kernel_main native_attention_writer
#include "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp"
#undef kernel_main
#include "tools/profiler/kernel_profiler.hpp"
#include "zero_l1.hpp"

void QB2_ENTRY() {
    native_attention_writer();
    if (get_arg_val<uint32_t>(3) == 0) { return; }  // Only the two KV-head reducers write output.
    // Native writer consumed its capacity-one position CB before returning;
    // the pointer has wrapped and the value remains until the layer boundary.
    const bool inactive = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(8)) == UINT32_MAX;
    const uint32_t cx = get_arg_val<uint32_t>(CONCAT_RT_OFFSET + 1);
    const uint32_t cy = get_arg_val<uint32_t>(CONCAT_RT_OFFSET + 2);
    noc_semaphore_inc(get_noc_addr(cx, cy, get_semaphore(3)), 1);
    noc_async_atomic_barrier();
    if (get_arg_val<uint32_t>(4) == 0) { return; }
    {
        DeviceZoneScopedN("ATTENTION-CONCAT-WAIT");
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(3)), 2);
    }
    DeviceZoneScopedN("ATTENTION-CONCAT");
    constexpr auto source_args = TensorAccessorArgs<28>();
    constexpr auto target_args = TensorAccessorArgs<CONCAT_CT_OFFSET>();
    const auto source = TensorAccessor(source_args, get_arg_val<uint32_t>(0), 2048);
    const auto target = TensorAccessor(target_args, get_arg_val<uint32_t>(CONCAT_RT_OFFSET), 2048);
    cb_reserve_back(32, 32);
    const uint32_t scratch = get_write_ptr(32);
    zero_l1<32 * 2048>(scratch);
    for (uint32_t head = 0; !inactive && head < 8; ++head) {
        for (uint32_t column = 0; column < 4; ++column) {
            const uint32_t destination = scratch + (head * 4 + column) * 2048;
            const uint64_t origin = source.get_noc_addr(column) + head * 32;
            noc_async_read(origin, destination, 32);
            noc_async_read(origin + 512, destination + 512, 32);
        }
    }
    noc_async_read_barrier();
    for (uint32_t tile = 0; tile < 32; ++tile) {
        noc_async_write_page(tile, target, scratch + tile * 2048);
    }
    noc_async_write_barrier();
    for (uint32_t core = 0; core < 8; ++core) {
        noc_semaphore_inc(
            get_noc_addr(get_arg_val<uint32_t>(CONCAT_RT_OFFSET + 3 + 2 * core),
                         get_arg_val<uint32_t>(CONCAT_RT_OFFSET + 4 + 2 * core), get_semaphore(14)), 1);
    }
    noc_async_atomic_barrier();
}
