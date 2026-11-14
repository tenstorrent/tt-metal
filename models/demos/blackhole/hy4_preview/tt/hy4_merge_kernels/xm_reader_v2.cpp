// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Fused exchange + merge (fuse-sparse-merge A2), merge-core reader, V2 (A2b) = the OWN slot is written by the leader core straight into the landing slot (no DRAM round trip; counts as one extra semaphore increment). Landing buffer CB_LAND = [8 source slots][TPC o tiles, ms, l] (aliases a persistent L1 tensor; the 7 remote sources
// write their slot by fabric with a fused global-semaphore increment per packet). This core (user s, tile columns g*TPC..) reads its OWN source slot from DRAM (x = the local split-kernel output,
// owner block c), the sink tile, builds the ones tile, waits for 7 x (TPC/2 + 1) packets, re-arms the semaphore and releases the CB to compute.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);
    constexpr uint32_t TPC = get_compile_time_arg_val(1);
    constexpr uint32_t CB_ST = get_compile_time_arg_val(2);
    constexpr uint32_t CB_LAND = get_compile_time_arg_val(3);
    constexpr uint32_t CB_ONE = get_compile_time_arg_val(4);
    constexpr uint32_t DBG = get_compile_time_arg_val(5);
    constexpr auto s_args = TensorAccessorArgs<6, 3>();
    constexpr uint32_t NC = 8, TILE_B = 2048, TPR = 18, LT = TPC + 2;
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t s_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t gsem = get_common_arg_val<uint32_t>(2);
    const uint32_t s = get_arg_val<uint32_t>(0);
    const uint32_t g = get_arg_val<uint32_t>(1);
    const uint32_t c = get_arg_val<uint32_t>(2);  // ring position of this chip
    const auto sa = TensorAccessor(s_args, s_addr, TILE_B);
    CircularBuffer st(CB_ST), land(CB_LAND), one(CB_ONE);
    land.reserve_back(NC * LT);
    const uint32_t p = land.get_write_ptr();
    st.reserve_back(1);
    noc_async_read(sa.get_noc_addr(s), st.get_write_ptr(), TILE_B);
    one.reserve_back(1);
    {
        volatile tt_l1_ptr uint32_t* po = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(one.get_write_ptr());
        for (uint32_t i = 0; i < TILE_B / 4; ++i) {
            po[i] = 0x3F803F80u;  // bf16 1.0 x 2
        }
    }
    one.push_back(1);
    noc_async_read_barrier();
    st.push_back(1);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gsem);
    if constexpr (DBG == 0) {
        noc_semaphore_wait_min(sem, 7 * (TPC / 2 + 1) + 1);
        noc_semaphore_set(sem, 0);
    }
    land.push_back(NC * LT);
}
