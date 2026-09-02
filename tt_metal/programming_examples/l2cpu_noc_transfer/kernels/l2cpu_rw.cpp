// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// NOC transfers between this Tensix core and a Blackhole L2CPU (x280 cluster) tile.
//
// The L2CPU tile is a passive NOC target here: its inbound port passes NOC requests
// straight through to the x280 physical address space, so its 2 MiB L3-as-scratchpad
// (LIM, x280 physical 0x0800_0000) is readable/writable with the ordinary dataflow
// APIs even while the x280 harts are still held in reset. No kernel runs on the
// L2CPU side — there is nothing on that tile to run one.
void kernel_main() {
    uint32_t l1_base = get_arg_val<uint32_t>(0);
    uint32_t src_dram = get_arg_val<uint32_t>(1);
    uint32_t dst_dram = get_arg_val<uint32_t>(2);
    uint32_t size_bytes = get_arg_val<uint32_t>(3);
    uint32_t l2cpu_x = get_arg_val<uint32_t>(4);
    uint32_t l2cpu_y = get_arg_val<uint32_t>(5);
    uint32_t lim_addr = get_arg_val<uint32_t>(6);
    uint32_t patch_word_idx = get_arg_val<uint32_t>(7);
    uint32_t do_atomic = get_arg_val<uint32_t>(8);
    uint32_t atomic_word_idx = get_arg_val<uint32_t>(9);

    uint32_t l1_src = l1_base;
    uint32_t l1_dst = l1_base + size_bytes;

    constexpr auto in0_args = TensorAccessorArgs<0>();
    const auto in0 = TensorAccessor(in0_args, src_dram, size_bytes);
    constexpr auto out0_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto out0 = TensorAccessor(out0_args, dst_dram, size_bytes);

    // 1. Stage the pattern: DRAM -> local L1.
    noc_async_read_page(0, in0, l1_src);
    noc_async_read_barrier();

    // 2. Bulk NOC write: L1 -> L2CPU LIM. The L2CPU tile keeps its NOC0 coordinates
    //    under Blackhole's translation tables, so (l2cpu_x, l2cpu_y) is used as-is.
    uint64_t lim_noc_addr = get_noc_addr(l2cpu_x, l2cpu_y, lim_addr);
    noc_async_write(l1_src, lim_noc_addr, size_bytes);
    noc_async_write_barrier();

    // 3. Single-word inline write into the middle of the region (the primitive worker
    //    adapters use for credit/pointer updates).
    noc_inline_dw_write(get_noc_addr(l2cpu_x, l2cpu_y, lim_addr + patch_word_idx * 4), 0xC0FFEE55);
    noc_async_write_barrier();

    // 4. Optional: NOC atomic increment against LIM. Off by default: measured on a
    //    p100a (BH), the L2CPU bridge does NOT implement NOC atomics — the atomic
    //    response never arrives and the barrier below hangs (recoverable by rerunning,
    //    device init resets this core).
    if (do_atomic) {
        noc_semaphore_inc(get_noc_addr(l2cpu_x, l2cpu_y, lim_addr + atomic_word_idx * 4), 5);
        noc_async_atomic_barrier();
    }

    // 5. Bulk NOC read: L2CPU LIM -> a second L1 region.
    noc_async_read(lim_noc_addr, l1_dst, size_bytes);
    noc_async_read_barrier();

    // 6. Return the round-tripped data: L1 -> DRAM for host verification.
    noc_async_write_page(0, out0, l1_dst);
    noc_async_write_barrier();
}
