// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused mHC post-mix, PACKED decode mode (T <= 4) — writer (NCRISC, NoC1): the packed output tile's rows r = (s, j, t)
// into whole output pages assembled in two alternating zeroed scratch pages (CB_STK; every page reuses the same row
// positions, so the rest stays zero): flat (OUT_MODE 0: page s * 128 + c0 + j, row t) or stacked (OUT_MODE 1: page
// t * 128 + c0 + j, row s). Compile-time args: OUT_MODE, TensorAccessorArgs of out.
// Runtime args: out address, c0, T; megakernel (MK_POST_SIGNAL_SEM): then n and the NoC (x, y) of the n cores reading
// this core's output columns, signalled after the last write.

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

    cb_reserve_back(CB_STK, 2);
    const uint32_t stk = get_write_ptr(CB_STK);
    {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stk);
        for (uint32_t i = 0; i < 2 * FP32_TILE / 4; ++i) {
            z[i] = 0;
        }
    }
    cb_wait_front(CB_OUT, 1);
    DeviceZoneScopedN("MP-WR");
    invalidate_l1_cache();  // packer-written tile read by this RISC
    const uint32_t src = get_read_ptr(CB_OUT);
#if defined(MK_POST_ROWWRITE)
    // Stage 14 glue trim (flat output, decode rows): write only the T valid rows of each output page straight from the
    // packed tile (2 x 64-byte face-row writes per row) instead of whole zero-padded 4 KB pages (2 MB per call at
    // T = 1); the padding rows of `out` are never read for a result (every consumer is row-independent)
    if constexpr (OUT_MODE == 0) {
        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
                for (uint32_t t = 0; t < T; ++t) {
                    const uint32_t row = (s * COLS_PER_CORE + j) * T + t;
                    const uint32_t page = s * COL_TILES + c0 + j;
                    for (uint32_t h = 0; h < 2; ++h) {
                        noc_async_write(
                            src + tile_off(row, 16 * h) * 4, out.get_noc_addr(page, tile_off(t, 16 * h) * 4), 64);
                    }
                }
            }
        }
        noc_async_write_barrier();
        cb_pop_front(CB_OUT, 1);
#if defined(MK_POST_SIGNAL_SEM)
        mk::signal_handoff(MK_POST_SIGNAL_SEM, ph_rt0 + 4, get_arg_val<uint32_t>(ph_rt0 + 3));
#endif
        return;
    }
#endif
    uint32_t n = 0;
    // flat: one page per (s, j) with rows t; stacked: one page per (t, j) with rows s
    for (uint32_t a = 0; a < (OUT_MODE == 0 ? N_STREAMS : T); ++a) {
        for (uint32_t j = 0; j < COLS_PER_CORE; ++j, ++n) {
            const uint32_t buf = stk + (n & 1) * FP32_TILE;
            if (n >= 2) {
                noc_async_write_barrier();  // this scratch page's previous write landed
            }
            for (uint32_t b = 0; b < (OUT_MODE == 0 ? T : N_STREAMS); ++b) {
                const uint32_t s = OUT_MODE == 0 ? a : b, t = OUT_MODE == 0 ? b : a;
                copy_row(src, (s * COLS_PER_CORE + j) * T + t, buf, OUT_MODE == 0 ? t : s);
            }
            noc_async_write_page(a * COL_TILES + c0 + j, out, buf);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(CB_OUT, 1);
#if defined(MK_POST_SIGNAL_SEM)
    mk::signal_handoff(MK_POST_SIGNAL_SEM, ph_rt0 + 4, get_arg_val<uint32_t>(ph_rt0 + 3));
#endif
}
