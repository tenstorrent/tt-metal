// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stream matmul — weight reader (BRISC, NoC0): this core's weight stream (megakernel layout "mk-stream-v1": the core's
// K x NR tiles in K-row-major order, contiguous in ONE DRAM bank) block by block into the 3-deep ring CB_W; block j =
// KB K rows = KB * NR tiles = one contiguous byte range, read in TPT-tile transactions under one transaction id per
// block (2 blocks in flight). Same ring protocol as ds_mm/wreader.cpp (CB ids from ds_mm_common.hpp).
// Compile-time args: NB, KB, NR, TB (tile bytes), TPT. Runtime args: tensor bank base address, bank id, first page of
// the stream within the bank (m0), VC.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/ds_mm/ds_mm_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"

using namespace dsmm;

void kernel_main() {
    constexpr uint32_t NB = get_compile_time_arg_val(0);
    constexpr uint32_t KB = get_compile_time_arg_val(1);
    constexpr uint32_t NR = get_compile_time_arg_val(2);
    constexpr uint32_t TB = get_compile_time_arg_val(3);
    constexpr uint32_t TPT = get_compile_time_arg_val(4);
    constexpr uint32_t TXN_BYTES = TPT * TB;
    constexpr uint32_t BLOCK_TILES = KB * NR;
    constexpr uint32_t BLOCK_BYTES = BLOCK_TILES * TB;
    constexpr uint32_t TXNS_PER_BLOCK = BLOCK_TILES / TPT;
    constexpr uint32_t SLOTS = 3;
    static_assert(BLOCK_TILES % TPT == 0 && TXN_BYTES <= NOC_MAX_BURST_SIZE, "transaction size");

    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank = get_arg_val<uint32_t>(1);
    const uint32_t m0 = get_arg_val<uint32_t>(2);
    const uint32_t vc = get_arg_val<uint32_t>(3);
    MK_MARK();  // 0: start

    const uint64_t dram = get_noc_addr_from_bank_id<true>(bank, 0);
    noc_async_read_one_packet_set_state<true>(dram, TXN_BYTES, vc);
    const uint32_t ring = get_write_ptr(CB_W);
    uint32_t src = w_addr + m0 * TB;
    uint32_t slot = 0, trid_issue = 1, trid_wait = 1;
    bool in_flight = false;
    cb_reserve_back(CB_W, BLOCK_TILES);
    for (uint32_t blk = 0; blk < NB; ++blk) {
        noc_async_read_set_trid(trid_issue);
        uint32_t dst = ring + slot * BLOCK_BYTES;
        for (uint32_t t = 0; t < TXNS_PER_BLOCK; ++t) {
            noc_async_read_one_packet_with_state_with_trid<false, false>(dram, src, dst, trid_issue);
            src += TXN_BYTES;
            dst += TXN_BYTES;
        }
        slot = slot + 1 == SLOTS ? 0 : slot + 1;
        trid_issue = trid_issue == SLOTS ? 1 : trid_issue + 1;
        if (in_flight) {
            noc_async_read_barrier_with_trid(trid_wait);
            cb_push_back(CB_W, BLOCK_TILES);
            if (blk == 1) {
                MK_MARK();  // 1: first weight block landed
            }
            trid_wait = trid_wait == SLOTS ? 1 : trid_wait + 1;
            cb_reserve_back(CB_W, 2 * BLOCK_TILES);
        }
        in_flight = true;
    }
    noc_async_read_barrier_with_trid(trid_wait);
    cb_push_back(CB_W, BLOCK_TILES);
    MK_MARK();  // 2: last weight block landed
#if defined(MK_PROBE_ARG)
    MK_FLUSH(get_write_ptr(MK_PROBE_CB));
#endif
    noc_async_read_set_trid(0);
    noc_async_read_one_packet_set_state<true>(dram, TXN_BYTES, 1);  // firmware default VC (NOC_CTRL persists)
}
