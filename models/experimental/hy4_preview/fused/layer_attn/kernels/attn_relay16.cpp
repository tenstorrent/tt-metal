// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The relay on a merge core in program P2 at 16 token rows (attn_relay.cpp for two sub-passes with a row mask;
// attn_rows.hpp). Merge core i is the landing core of the group's queries (X3, once a pass: 16-row tiles) and of one
// column partner's and one row partner's partials (X4), which land once per sub-pass that has a live row, each in
// the sub-pass's own slot of the landing.
//   X3          A_QARR reached `q_need`: A_QGO + 1 on the 8 attention cores of group i
//   X4 column   the n-th arrival counted in A_COLARR is sub-pass p = arrival_subpass(mask, n): bit 8 p + i of
//               A_COLMASK on the merge cores whose row is live in sub-pass p; arrivals(mask) arrivals in all
//   X4 row      the same with A_ROWARR and A_ROWMASK
// The wait is bounded as in attn_relay.cpp.
//
// Runtime args: q_need, expect column arrivals (0 / 1), expect row arrivals (0 / 1), this core's index, then the 8
// attention cores of group i (NoC x | y << 16), then the 8 merge cores.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#include "attn_rows_dm.hpp"

void kernel_main() {
    const uint32_t q_need = get_arg_val<uint32_t>(0);
    const uint32_t index = get_arg_val<uint32_t>(3);
    constexpr uint32_t ATT = 4, MERGE = 12, N = 8;
    const uint32_t mask = hy4a::row_mask();
    const uint32_t col_all = get_arg_val<uint32_t>(1) != 0 ? hy4r::arrivals(mask) : 0;
    const uint32_t row_all = get_arg_val<uint32_t>(2) != 0 ? hy4r::arrivals(mask) : 0;
    uint32_t col = 0, row = 0;
    bool q = true;

    // the n-th arrival of a landing: its sub-pass's bit to the merge cores that wait for it
    auto announce = [&](uint32_t n, uint32_t word) {
        const uint32_t p = hy4r::arrival_subpass(mask, n);
        hy4r::for_each_live(mask, p, [&](uint32_t i, uint32_t) {
            const uint32_t core = get_arg_val<uint32_t>(MERGE + i);
            noc_semaphore_inc(
                get_noc_addr(core & 0xFFFF, core >> 16, hy4a::word_addr(word)), hy4r::arrival_bit(p, index));
        });
    };

    uint32_t polls = 0;
    while (q || col < col_all || row < row_all) {
        invalidate_l1_cache();
        bool progress = false;
        if (q && hy4a::word(hy4a::A_QARR)[0] >= q_need) {
            if (q_need != 0) {
                hy4a::take_word(hy4a::A_QARR, q_need);
            }
            for (uint32_t i = 0; i < N; ++i) {
                const uint32_t core = get_arg_val<uint32_t>(ATT + i);
                noc_semaphore_inc(get_noc_addr(core & 0xFFFF, core >> 16, hy4a::word_addr(hy4a::A_QGO)), 1);
            }
            q = false;
            progress = true;
        }
        if (col < col_all && hy4a::word(hy4a::A_COLARR)[0] >= 1) {
            hy4a::take_word(hy4a::A_COLARR, 1);
            announce(col, hy4a::A_COLMASK);
            ++col;
            progress = true;
        }
        if (row < row_all && hy4a::word(hy4a::A_ROWARR)[0] >= 1) {
            hy4a::take_word(hy4a::A_ROWARR, 1);
            announce(row, hy4a::A_ROWMASK);
            ++row;
            progress = true;
        }
        if (progress) {
            polls = 0;
        } else if (++polls >= HY4L_WAIT_LIMIT) {
            hy4a::fail(hy4a::ST_QARR, (q ? 1u : 0u) | ((col_all - col) << 4) | ((row_all - row) << 8));
            break;
        }
    }
    noc_async_atomic_barrier();
}
