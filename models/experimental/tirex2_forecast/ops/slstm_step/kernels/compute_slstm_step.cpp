// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused sLSTM loop, all steps. Per step and work item (one d-tile j of a unit = 32 rows x one head):
//   pre[g, j] = Wx[g, j] + sum_k y_bf16[k] @ R[k, g, j] (Wx unpacked to fp32 DST slot g, matmul
//   accumulates on it), then gates in place in DST slots 0 i, 1 f, 2 z, 3 o, 4 m, 5 c, 6 n:
//      (one fused SFPI pass, slstm_gates_sfpu.h) lfm = logsig(f) + m; m' = max(i, lfm); ig = exp(i - m'); fg = exp(lfm - m');
//      c = fg c + ig tanh z; n = max(fg n + ig, 1); y = sigmoid(o) c / n
//      t = 0: m = i, c = tanh z, n = 1, y = sigmoid(o) tanh z.
// State c, n, m (fp32) live in ring CBs of one tile per item (pop, push the update). y (bf16) lives in a fixed
// buffer, tile (lu * 2 + parity) * 4 + j: step t reads parity (t - 1) % 2 and packs into parity t % 2 out of
// order; the reader's ready token says all 4 tiles of the unit (own and partner's) have landed.
//
// Compile-time args: neg80 (fp32 bits), one (fp32 bits)
// Runtime args: num_items, T, record, Bt, ablate, first_item, num_touched_units

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#include "slstm_gates_sfpu.h"
#include "api/debug/kernel_profiler.hpp"

// Timing-only compile-time ablation (wrong results): bit 0 no Wx copy, bit 1 no y packs, bit 2 no per-item
// reconfig/init calls (done once up front).
#ifndef SLSTM_CABL
#define SLSTM_CABL 0
#endif
constexpr uint32_t kCabl = SLSTM_CABL;

void kernel_main() {
    const uint32_t num_items = get_arg_val<uint32_t>(0);
    const uint32_t T = get_arg_val<uint32_t>(1);
    const bool record = get_arg_val<uint32_t>(2) != 0;
    // Timing ablation only (wrong results): bit 0 skips the matmul, bit 1 the gate chain.
    const uint32_t ablate = get_arg_val<uint32_t>(4);
    const uint32_t first_item = get_arg_val<uint32_t>(5);
    const uint32_t L = get_arg_val<uint32_t>(6);
    // bit 10 skips the c/n/m ring CB round trip (copy in, pack out) per item per step.
    const bool st = !(ablate & 1024);

    constexpr uint32_t wx_cb = tt::CBIndex::c_0;
    constexpr uint32_t r_cb = tt::CBIndex::c_1;
    constexpr uint32_t y_cb = tt::CBIndex::c_2;
    constexpr uint32_t c_cb = tt::CBIndex::c_3;
    constexpr uint32_t n_cb = tt::CBIndex::c_4;
    constexpr uint32_t m_cb = tt::CBIndex::c_5;
    constexpr uint32_t ready_cb = tt::CBIndex::c_7;
    constexpr uint32_t yout_cb = tt::CBIndex::c_16;
    constexpr uint32_t cout_cb = tt::CBIndex::c_17;
    constexpr uint32_t nout_cb = tt::CBIndex::c_18;
    constexpr uint32_t mout_cb = tt::CBIndex::c_19;

    CircularBuffer cb_wx(wx_cb), cb_r(r_cb), cb_y(y_cb), cb_c(c_cb), cb_n(n_cb), cb_m(m_cb), cb_ready(ready_cb);
    CircularBuffer cb_yout(yout_cb), cb_cout(cout_cb), cb_nout(nout_cb), cb_mout(mout_cb);

    compute_kernel_hw_startup(y_cb, r_cb, m_cb);
    cb_r.wait_front(16 * num_items);
    cb_y.wait_front(8 * L);
    const uint32_t u0 = first_item / 4;
    constexpr uint32_t m_slot = 4, c_slot = 5;
    if constexpr (kCabl & 4) {
        reconfig_data_format(r_cb, y_cb);
        matmul_init(y_cb, r_cb);
        MATH((llk_math_slstm_gates_init()));
    }

    for (uint32_t t = 0; t < T; ++t) {
        const bool first = t == 0;
        const bool emit = record || t + 1 == T;
        // Profiler zones (no-ops unless TT_METAL_DEVICE_PROFILER=1) on steps 100-103 only (marker buffer size).
        const bool prof = t >= 100 && t < 104;
        for (uint32_t e = 0; e < num_items; ++e) {
          auto item = [&]() {
            const uint32_t i = first_item + e;
            const uint32_t lu = i / 4 - u0;
            const uint32_t j = i % 4;
            if (!first && (e == 0 || j == 0)) {
                if (prof) {
                    DeviceZoneScopedN("C-READY");
                    cb_ready.wait_front(1);
                } else {
                    cb_ready.wait_front(1);
                }
                cb_ready.pop_front(1);
            }
            cb_wx.wait_front(4);
            tile_regs_acquire();
            if constexpr (!(kCabl & 4)) {
                reconfig_data_format_srca(wx_cb);
                copy_init(wx_cb);
            }
            if constexpr (!(kCabl & 1)) {
                for (uint32_t g = 0; g < 4; ++g) {
                    copy_tile(wx_cb, g, g);
                }
            }
            if (!first && !(ablate & 1)) {
                const uint32_t yb = (lu * 2 + (t - 1) % 2) * 4;
                if constexpr (!(kCabl & 4)) {
                    reconfig_data_format(r_cb, y_cb);
                    matmul_init(y_cb, r_cb);
                }
                for (uint32_t g = 0; g < 4; ++g) {
                    for (uint32_t k = 0; k < 4; ++k) {
                        matmul_tiles(y_cb, r_cb, yb + k, e * 16 + k * 4 + g, g);
                    }
                }
            }
            if (!first && st) {
                cb_m.wait_front(1);
                cb_c.wait_front(1);
                cb_n.wait_front(1);
                if constexpr (!(kCabl & 4)) {
                    reconfig_data_format_srca(m_cb);
                    copy_init(m_cb);
                }
                copy_tile(m_cb, 0, 4);
                copy_tile(c_cb, 0, 5);
                copy_tile(n_cb, 0, 6);
            }
            // Whole gate chain in one SFPU pass: y -> 3, m -> 4, c -> 5, n -> 6.
            if (!(ablate & 2)) {
                if constexpr (!(kCabl & 4)) {
                    MATH((llk_math_slstm_gates_init()));
                }
                if (first) {
                    MATH((llk_math_slstm_gates<true>()));
                } else {
                    MATH((llk_math_slstm_gates<false>()));
                }
            }
            tile_regs_commit();
            cb_wx.pop_front(4);
            if (!first && st) {
                cb_m.pop_front(1);
                cb_c.pop_front(1);
                cb_n.pop_front(1);
            }
            if (st) {
                cb_m.reserve_back(1);
                cb_c.reserve_back(1);
                cb_n.reserve_back(1);
            }
            cb_yout.reserve_back(1);
            tile_regs_wait();
            if constexpr (!(kCabl & 2)) {
                if constexpr (!(kCabl & 4)) {
                    pack_reconfig_data_format(y_cb);
                }
                pack_tile<true>(3, y_cb, (lu * 2 + t % 2) * 4 + j);
                if constexpr (!(kCabl & 4)) {
                    pack_reconfig_data_format(yout_cb);
                }
                pack_tile(3, yout_cb);
            }
            if (st) {
                pack_tile(m_slot, m_cb);
                pack_tile(c_slot, c_cb);
                pack_tile(6, n_cb);
            }
            if (emit) {
                cb_cout.reserve_back(1);
                cb_nout.reserve_back(1);
                cb_mout.reserve_back(1);
                pack_tile(c_slot, cout_cb);
                pack_tile(6, nout_cb);
                pack_tile(m_slot, mout_cb);
                cb_cout.push_back(1);
                cb_nout.push_back(1);
                cb_mout.push_back(1);
            }
            tile_regs_release();
            cb_yout.push_back(1);
            if (st) {
                cb_m.push_back(1);
                cb_c.push_back(1);
                cb_n.push_back(1);
            }
          };
          if (prof) {
              DeviceZoneScopedN("C-ITEM");
              item();
          } else {
              item();
          }
        }
    }
}
