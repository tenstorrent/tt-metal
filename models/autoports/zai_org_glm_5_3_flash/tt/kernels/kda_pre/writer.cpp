// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused KDA "pre" — writer (NCRISC, NoC1): g, gate_out, beta and the conv output into q | k | v.
// Compile-time args: ROWS, TensorAccessorArgs of q, k, v ([1,1,B,8192] fp32), g, gate_out, beta ([1,1,B,64]).
// Runtime args: the 6 addresses, core k, units (ROWS: user-row pairs; TILES: 12); megakernel (MK_KDA_SIGNAL_SEM):
// then n and the NoC (x, y) of the n head cores reading this core's outputs (kda_rec), signalled after the last write.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "kda_pre_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"

using namespace kdap;

void kernel_main() {
    constexpr uint32_t ROWS = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr auto q_args = TensorAccessorArgs<PH_CT0 + 1>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto go_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<go_args.next_compile_time_args_offset()>();
    uint32_t a = 0;
    const auto out_q = TensorAccessor(q_args, get_arg_val<uint32_t>(ph_rt0 + a++));
    const auto out_k = TensorAccessor(k_args, get_arg_val<uint32_t>(ph_rt0 + a++));
    const auto out_v = TensorAccessor(v_args, get_arg_val<uint32_t>(ph_rt0 + a++));
    const auto out_g = TensorAccessor(g_args, get_arg_val<uint32_t>(ph_rt0 + a++));
    const auto out_go = TensorAccessor(go_args, get_arg_val<uint32_t>(ph_rt0 + a++));
    const auto out_beta = TensorAccessor(beta_args, get_arg_val<uint32_t>(ph_rt0 + a++));
    const uint32_t kk = get_arg_val<uint32_t>(ph_rt0 + a++);
    const uint32_t units = get_arg_val<uint32_t>(ph_rt0 + a++);
    if (kk >= N_CORES * RSPLIT) {
        return;  // a core past the shard's heads: no kda_pre work, no hand-off
    }
    const uint32_t k = kk % N_CORES, rg = kk / N_CORES;
    const bool gate = rg == 0;  // row group 0: g / gate_out / beta
    uint32_t u0 = 0, u1 = units;
    if constexpr (ROWS) {
        rsplit_range(units, rg, u0, u1);
    }
    const uint32_t c0 = k * CONV_PER_CORE;
    const uint32_t n0 = k * G_PER_CORE;

    if (gate) {
        cb_wait_front(CB_G, G_PER_CORE);
        for (uint32_t j = 0; j < G_PER_CORE; ++j) {
            noc_async_write_page(n0 + j, out_g, get_read_ptr(CB_G) + j * FP32_TILE);
        }
        cb_wait_front(CB_GO, G_PER_CORE);
        for (uint32_t j = 0; j < G_PER_CORE; ++j) {
            noc_async_write_page(n0 + j, out_go, get_read_ptr(CB_GO) + j * FP32_TILE);
        }
        if (k < BETA_TILES) {
            cb_wait_front(CB_BETA, 1);
            noc_async_write_page(k, out_beta, get_read_ptr(CB_BETA));
        }
        noc_async_write_barrier();
        cb_pop_front(CB_G, G_PER_CORE);
        cb_pop_front(CB_GO, G_PER_CORE);
        if (k < BETA_TILES) {
            cb_pop_front(CB_BETA, 1);
        }
    }  // gate

    // ROWS: this core's pairs [u0, u1), UP of them per packed tile (KDA_PACK)
    const uint32_t n_units = ROWS ? (u1 - u0 + UP - 1) / UP : u1 - u0;
    for (uint32_t i = 0; i < n_units; ++i) {
        const uint32_t u = u0 + i;  // (TILES: the column tile)
        cb_wait_front(CB_CONV, 1);
        const uint32_t l1 = get_read_ptr(CB_CONV);
        if constexpr (ROWS) {
            // packed row pair UP r + pp -> rows (2 up, 2 up + 1) of column tile c0 + r, up = u0 + UP i + pp (fp32:
            // 2 x 128 B per face half)
            const uint32_t np = u1 - u0 - UP * i < UP ? u1 - u0 - UP * i : UP;
            for (uint32_t r = 0; r < CONV_PER_CORE; ++r) {
                const uint32_t c = c0 + r;
                const uint32_t region = c / W_TILES, page = c % W_TILES;
                for (uint32_t pp = 0; pp < np; ++pp) {
                    const uint32_t up = u0 + UP * i + pp, rp = UP * r + pp;
                    for (uint32_t h = 0; h < 2; ++h) {
                        const uint32_t src = l1 + tile_off(2 * rp, 16 * h) * 4;
                        const uint32_t off = tile_off(2 * up, 16 * h) * 4;
                        const uint64_t dst = region == 0   ? out_q.get_noc_addr(page, off)
                                             : region == 1 ? out_k.get_noc_addr(page, off)
                                                           : out_v.get_noc_addr(page, off);
                        noc_async_write(src, dst, 128);
                    }
                }
            }
        } else {
            const uint32_t c = c0 + u;
            const uint32_t region = c / W_TILES, page = c % W_TILES;
            if (region == 0) {
                noc_async_write_page(page, out_q, l1);
            } else if (region == 1) {
                noc_async_write_page(page, out_k, l1);
            } else {
                noc_async_write_page(page, out_v, l1);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(CB_CONV, 1);
    }
#if defined(MK_KDA_SIGNAL_SEM)
    mk::signal_handoff(MK_KDA_SIGNAL_SEM, ph_rt0 + a + 1, get_arg_val<uint32_t>(ph_rt0 + a));
#endif
}
