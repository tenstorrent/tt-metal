// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The relay on a merge core in program P2 of the attention half (fused/layer_attn/program.py; PLAN.md section 5):
// merge core i is the landing core of three cross-chip payloads, whose arrival only this core can see (a sender's
// last packet counts a word of the core it lands on). The relay turns each arrival into a go for the cores that
// read the landing:
//   X3          A_QARR reached `q_need` (the chips of head group i that take part): A_QGO + 1 on the 8 attention
//               cores of group i (one per S-block), which then assemble their query pages from this core's landing
//   X4 column   A_COLARR counted (the chip of mesh row i in this chip's column has landed its partials of this
//               chip's head group): bit i of A_COLMASK on the 8 merge cores
//   X4 row      A_ROWARR counted (the chip of mesh column i in this chip's row): bit i of A_ROWMASK on the 8 merge
//               cores
// The three are polled in one loop. The wait is bounded (HY4L_WAIT_LIMIT polls without progress): a relay that
// gives up leaves its code and what was still missing in this core's status word and ends; the readers run into
// their own bounded waits.
//
// Runtime args: q_need, expect a column arrival (0 / 1), expect a row arrival (0 / 1), this core's index, then the 8
// attention cores of group i (NoC x | y << 16), then the 8 merge cores.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#include "attn_common.hpp"

void kernel_main() {
    const uint32_t q_need = get_arg_val<uint32_t>(0);
    bool col = get_arg_val<uint32_t>(1) != 0;
    bool row = get_arg_val<uint32_t>(2) != 0;
    const uint32_t index = get_arg_val<uint32_t>(3);
    constexpr uint32_t ATT = 4, MERGE = 12, N = 8;
    bool q = true;

    auto to_all = [&](uint32_t first, uint32_t word, uint32_t value) {
        for (uint32_t i = 0; i < N; ++i) {
            const uint32_t core = get_arg_val<uint32_t>(first + i);
            noc_semaphore_inc(get_noc_addr(core & 0xFFFF, core >> 16, hy4a::word_addr(word)), value);
        }
    };

    uint32_t polls = 0;
    while (q || col || row) {
        invalidate_l1_cache();
        bool progress = false;
        if (q && hy4a::word(hy4a::A_QARR)[0] >= q_need) {
            if (q_need != 0) {
                hy4a::take_word(hy4a::A_QARR, q_need);
            }
            to_all(ATT, hy4a::A_QGO, 1);
            q = false;
            progress = true;
        }
        if (col && hy4a::word(hy4a::A_COLARR)[0] >= 1) {
            hy4a::take_word(hy4a::A_COLARR, 1);
            to_all(MERGE, hy4a::A_COLMASK, 1u << index);
            col = false;
            progress = true;
        }
        if (row && hy4a::word(hy4a::A_ROWARR)[0] >= 1) {
            hy4a::take_word(hy4a::A_ROWARR, 1);
            to_all(MERGE, hy4a::A_ROWMASK, 1u << index);
            row = false;
            progress = true;
        }
        if (progress) {
            polls = 0;
        } else if (++polls >= HY4L_WAIT_LIMIT) {
            hy4a::fail(hy4a::ST_QARR, (q ? 1u : 0u) | (col ? 2u : 0u) | (row ? 4u : 0u));
            break;
        }
    }
    noc_async_atomic_barrier();
}
