// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// One-time weight preparation: copy a DRAM-interleaved tile weight [kt rows][nt_src] into [kt rows][nt_dst]
// (nt_dst >= nt_src), the extra tile columns zero, one tile row at a time.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t kt = get_compile_time_arg_val(1);
    constexpr uint32_t nt_src = get_compile_time_arg_val(2);
    constexpr uint32_t nt_dst = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);

    const uint32_t buf = get_write_ptr(cb);
    const uint32_t zero_l1 = buf + nt_src * tile_bytes;
    volatile tt_l1_ptr uint32_t* zero = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_l1);
    for (uint32_t i = 0; i < tile_bytes / 4; i++) {
        zero[i] = 0;
    }

    for (uint32_t r = 0; r < kt; r++) {
        for (uint32_t c = 0; c < nt_src; c++) {
            const uint32_t p = r * nt_src + c;
            noc_async_read(
                get_noc_addr_from_bank_id<true>(p % NUM_DRAM_BANKS, src_addr + (p / NUM_DRAM_BANKS) * tile_bytes),
                buf + c * tile_bytes,
                tile_bytes);
        }
        noc_async_read_barrier();
        for (uint32_t c = 0; c < nt_dst; c++) {
            const uint32_t p = r * nt_dst + c;
            const uint32_t l1 = c < nt_src ? buf + c * tile_bytes : zero_l1;
            noc_async_write(
                l1,
                get_noc_addr_from_bank_id<true>(p % NUM_DRAM_BANKS, dst_addr + (p / NUM_DRAM_BANKS) * tile_bytes),
                tile_bytes);
        }
        noc_async_write_barrier();
    }
}
