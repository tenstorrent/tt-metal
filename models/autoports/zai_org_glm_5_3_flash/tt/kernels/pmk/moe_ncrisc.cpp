// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK MoE phase set -- NCRISC (NoC1) composite: the expert FFN's activation mover (moe_xffn_writer: the x-row gather
// through the plan's disp rows into CB_XRM, the untilized output rows into out_rm) -> barrier SEM_FFN (every core's
// rows landed) -> the combine reader (moe_combine_rm_reader: the K rows of each token + its weights). Defines as
// moe_brisc.cpp (PMK_CT_FFN / COMB, PMK_RT_FFN / COMB, PMK_RT_CORES, PMK_P).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/moe_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/prefill/moe_xffn_probe.hpp"

using namespace pmk_moe;

namespace ffn {
constexpr uint32_t CT0 = PMK_CT_FFN;
constexpr uint32_t RT0 = PMK_RT_FFN;
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/ffn_writer.hpp"
}  // namespace ffn
namespace comb {
constexpr uint32_t CT0 = PMK_CT_COMB;
constexpr uint32_t RT0 = PMK_RT_COMB;
constexpr uint32_t HALF = 0;  // PMK_COMB_SPLIT: the even slots, y_s, the y write
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/comb_reader.hpp"
}  // namespace comb

void kernel_main() {
    PMK_B_PROLOGUE();  // GLM53_PMK_ARGS=table: this core's arg row -> rta_l1_base (moe_common.hpp)
#if defined(PMK_BW_STAGE) && PMK_BW_STAGE < 1  // the bisect's stage 0 (Stage 14 B2.3): the launch alone, no body
    return;
#endif
#if defined(PMK_XS_FILL)
    // v2 lever 2, STANDALONE form of H8 (the mock / the pair bench): holder h (this core, PMK_RT_XSF + 0 = h, or
    // 0xFFFFFFFF = not a holder) reads its 32-row block of x [C, H] bf16 TILE (page h H_T + k -> CB_XS tile k) and
    // every core barriers on SEM_XS (the family instead lands the block through the lanes and waits FS_ARRIVE 15)
    {
        const uint32_t h = get_arg_val<uint32_t>(PMK_RT_XSF + 0);
        if (h != 0xFFFFFFFFu) {
            const uint32_t x_addr = get_arg_val<uint32_t>(PMK_RT_XSF + 1);
            const uint32_t h_t = get_arg_val<uint32_t>(PMK_RT_XSF + 2);
            const uint32_t xs = get_write_ptr(CB_XS);
#ifdef PMK_XS_ROWS
            // Stage 14 B2.3: the holder's 32 token rows of x_ext [C + 32, H] bf16 ROW_MAJOR (one row per page) -> CB_XS
            // row t at t H_T 64 (the writer's row-store layout)
            const uint32_t row_b = h_t * 32 * 2;
            const InterleavedAddrGen<true> xg{.bank_base_address = x_addr, .page_size = row_b};
            for (uint32_t t = 0; t < 32; ++t) {
                noc_async_read(xg.get_noc_addr(h * 32 + t), xs + t * row_b, row_b);
            }
#else
            const InterleavedAddrGen<true> xg{.bank_base_address = x_addr, .page_size = 2048};
            for (uint32_t k = 0; k < h_t; ++k) {
                noc_async_read(xg.get_noc_addr(h * h_t + k), xs + k * 2048, 2048);
            }
#endif
            noc_async_read_barrier();
        }
        barrier_all(SEM_XS, PMK_RT_CORES, PMK_P);
    }
#endif
#if defined(PMK_BW_STAGE) && PMK_BW_STAGE < 2  // the M-B1 bisect: plan only (the BRISC's)
    return;
#endif
    ffn::phase_main();
    PMK_PP_MARK();                              // GLM53_PMK_PHASE_PROBE: B.ffnw (this core's out_rm rows written)
    barrier_all(SEM_FFN, PMK_RT_CORES, PMK_P);  // every core's expert output rows are in out_rm
    PMK_PP_MARK();  // GLM53_PMK_PHASE_PROBE: B.ffn_barrier (the lockstep wait for the slowest core)
#if defined(PMK_BW_STAGE) && PMK_BW_STAGE < 3  // the M-B1 bisect: no combine
    return;
#endif
    comb::phase_main();
    PMK_PP_MARK();  // GLM53_PMK_PHASE_PROBE: B.combr (this core's combine reads done)
}
