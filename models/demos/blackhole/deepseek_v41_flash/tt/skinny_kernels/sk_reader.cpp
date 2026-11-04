// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// skinny-matmul chains, NCRISC: stream up to three weight shards from this core's DRAM bank into ONE ring cb_w with the
// blaze DRAMStreamingMatmul reader (reused as is). Phase p streams columns of per_core_n tiles, K in nsub blocks of sbk
// tiles each. A phase with pn == 0 is skipped. The ring is continuous across the phases (the next phase prefetches
// while the compute side is busy with the exchange between two matmuls). CT args: 0 cb_w, 1 bank_id, 2 vc, then per
// phase p (0..2) 9 values at 3 + 9 p: addr, page, npages, sbk, pn, blk, nsub, nbuf, (spare)

#include "models/demos/deepseek_v3_b1/unified_kernels/dram_streaming_matmul.hpp"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t bank_id = get_compile_time_arg_val(1);
    constexpr uint32_t vc = get_compile_time_arg_val(2);
    constexpr uint32_t cb_unused = 31;

#define SKP(p, k) get_compile_time_arg_val(3 + 9 * (p) + (k))
    constexpr uint32_t pn0 = SKP(0, 4);
    constexpr uint32_t pn1 = SKP(1, 4);
    constexpr uint32_t pn2 = SKP(2, 4);

    if constexpr (pn0 > 0) {
        using MM = deepseek_b1_ops::DRAMStreamingMatmul::ReaderCTArgs<
            cb_w,
            cb_unused,
            SKP(0, 0),
            SKP(0, 1),
            SKP(0, 2),
            SKP(0, 3),
            SKP(0, 4),
            SKP(0, 5),
            SKP(0, 4),
            SKP(0, 6),
            bank_id,
            vc>;
        deepseek_b1_ops::DRAMStreamingMatmul::Op<MM, true, true, false, 0, false, false, SKP(0, 7)> mm;
        DeviceZoneScopedN("N_P0");
        mm();
    }
    if constexpr (pn1 > 0) {
        using MM = deepseek_b1_ops::DRAMStreamingMatmul::ReaderCTArgs<
            cb_w,
            cb_unused,
            SKP(1, 0),
            SKP(1, 1),
            SKP(1, 2),
            SKP(1, 3),
            SKP(1, 4),
            SKP(1, 5),
            SKP(1, 4),
            SKP(1, 6),
            bank_id,
            vc>;
        deepseek_b1_ops::DRAMStreamingMatmul::Op<MM, true, true, false, 0, false, false, SKP(1, 7)> mm;
        DeviceZoneScopedN("N_P1");
        mm();
    }
    if constexpr (pn2 > 0) {
        using MM = deepseek_b1_ops::DRAMStreamingMatmul::ReaderCTArgs<
            cb_w,
            cb_unused,
            SKP(2, 0),
            SKP(2, 1),
            SKP(2, 2),
            SKP(2, 3),
            SKP(2, 4),
            SKP(2, 5),
            SKP(2, 4),
            SKP(2, 6),
            bank_id,
            vc>;
        deepseek_b1_ops::DRAMStreamingMatmul::Op<MM, true, true, false, 0, false, false, SKP(2, 7)> mm;
        DeviceZoneScopedN("N_P2");
        mm();
    }
}
