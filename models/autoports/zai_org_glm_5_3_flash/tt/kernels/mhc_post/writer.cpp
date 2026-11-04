// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused mHC post-mix — writer (NCRISC, NoC1): the 4 output stream tiles of each column tile -> the new residual in the
// flat page order (OUT_MODE 0: [1,1,T,16384] / [1,4,T,4096], page s * 128 + c) or the stacked [T,1,4,4096] layout
// (OUT_MODE 1: page t * 128 + c, row s = stream s's row t). Compile-time args: OUT_MODE, TensorAccessorArgs of out.
// Runtime args: out address, c0, T; megakernel (MK_POST_SIGNAL_SEM): then n and the NoC (x, y) of the n cores
// reading this core's output columns, signalled after the last write.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "mhc_post_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"

using namespace mpost;

namespace {

FORCE_INLINE void copy_row(uint32_t src_l1, uint32_t src_row, uint32_t dst_l1, uint32_t dst_row) {
    for (uint32_t h = 0; h < 2; ++h) {
        volatile tt_l1_ptr uint32_t* s =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_l1) + tile_off(src_row, 16 * h);
        volatile tt_l1_ptr uint32_t* d =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_l1) + tile_off(dst_row, 16 * h);
        for (uint32_t i = 0; i < 16; ++i) {
            d[i] = s[i];
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t OUT_MODE = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr auto out_args = TensorAccessorArgs<PH_CT0 + 1>();
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(ph_rt0 + 0));
    const uint32_t c0 = get_arg_val<uint32_t>(ph_rt0 + 1);
    const uint32_t T = get_arg_val<uint32_t>(ph_rt0 + 2);

    DeviceZoneScopedN("MP-WR");  // device-profiler zone
    for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
        cb_wait_front(CB_OUT, N_STREAMS);
        invalidate_l1_cache();  // packer-written tiles read by this RISC (stacked mode)
        const uint32_t l1 = get_read_ptr(CB_OUT);
        if constexpr (OUT_MODE == 0) {
            for (uint32_t s = 0; s < N_STREAMS; ++s) {
                noc_async_write_page(s * COL_TILES + c0 + j, out, l1 + s * FP32_TILE);
            }
            noc_async_write_barrier();
        } else {
            cb_reserve_back(CB_STK, 1);
            const uint32_t stk = get_write_ptr(CB_STK);
            {
                // rows 4..31 stay zero; rows 0..3 (the streams) are fully rewritten for every token
                volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stk);
                for (uint32_t i = 0; i < 1024; ++i) {
                    z[i] = 0;
                }
            }
            for (uint32_t t = 0; t < T; ++t) {
                for (uint32_t s = 0; s < N_STREAMS; ++s) {
                    copy_row(l1 + s * FP32_TILE, t, stk, s);
                }
                noc_async_write_page(t * COL_TILES + c0 + j, out, stk);
                noc_async_write_barrier();
            }
        }
        cb_pop_front(CB_OUT, N_STREAMS);
    }
#if defined(MK_POST_SIGNAL_SEM)
    mk::signal_handoff(MK_POST_SIGNAL_SEM, ph_rt0 + 4, get_arg_val<uint32_t>(ph_rt0 + 3));
#endif
}
