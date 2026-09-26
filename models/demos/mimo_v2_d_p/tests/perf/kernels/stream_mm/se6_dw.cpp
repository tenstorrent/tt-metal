// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Spatially pipelined expert: a down core's weight reader (NCRISC, NOC1). Streams this core's down weight slice of
// every expert ([KBLK_D x PCD] blocks, a contiguous region of one DRAM bank, one read per block) into its in1 ring,
// which holds two experts so the next expert loads while the current one is still in use. Up to BATCH blocks per read
// barrier.
//
// CT: 0 IN1_CB, 1 SLOT_TILES, 2 TILE_BYTES, 3 BLOCKS (all experts), 4 BATCH, 5 IS_BFP8, 6 NUM_E (SE_DYN)
// RT: 0 weight bank base address, 1 bank id, 2 byte offset of this core's region in the bank, 3.. se_dyn.hpp args
// (SE_DYN)
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#ifdef SE_DYN
#include "se_dyn.hpp"
#endif

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t slot = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t blocks = get_compile_time_arg_val(3);
    constexpr uint32_t batch = get_compile_time_arg_val(4);
    constexpr bool is_bfp8 = get_compile_time_arg_val(5) != 0;
#ifdef SE_DW_DELAY
    // Experiment: leave DRAM to the gate/up readers while they load the first expert.
    {
        auto clk = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
        const uint32_t t0 = clk();
        while (clk() - t0 < static_cast<uint32_t>(SE_DW_DELAY)) {
        }
    }
#endif
    const uint64_t src =
        get_noc_addr_from_bank_id<true>(get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(0) + get_arg_val<uint32_t>(2));
#ifdef SE_DYN
    // only the active experts' blocks (CT 6 NUM_E: BLOCKS / NUM_E per expert; RT 3.. se_dyn.hpp args, CB 7 scratch)
    constexpr uint32_t num_e = get_compile_time_arg_val(6);
    constexpr uint32_t per_e = blocks / num_e;
    SeDyn dyn;
    se_dyn_load<num_e>(dyn, 3, get_write_ptr(tt::CBIndex::c_7) + 2048, 1);  // NCRISC: upper half of CB 7
    const uint32_t total = dyn.n_act * per_e;
    auto blk = [&](uint32_t b) { return dyn.eid[b / per_e] * per_e + b % per_e; };
#else
    constexpr uint32_t total = blocks;
    auto blk = [](uint32_t b) { return b; };
#endif
    for (uint32_t b = 0; b < total; b += batch) {
        const uint32_t n = total - b < batch ? total - b : batch;
        uint32_t l1[batch];
        for (uint32_t i = 0; i < n; ++i) {
            cb_reserve_back(cb, (i + 1) * slot);
            l1[i] = get_write_ptr(cb) + i * slot * tile_bytes;
        }
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_read(src + blk(b + i) * slot * tile_bytes, l1[i], slot * tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb, n * slot);
    }
}
