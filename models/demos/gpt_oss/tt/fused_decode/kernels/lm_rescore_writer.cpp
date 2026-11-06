// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Two-stage DFlash LM head, stage 2 with tile de-duplication, writer (BRISC) (dflash/lm_rescore.py): waits
// for the table (program semaphore 1); if this core's entry owns its column tile n (no earlier entry with the same
// tile): tells compute (mailbox 1, else 0), takes the [rows, 32] logits tile of n and writes [key, column] for every
// entry e' (row r', candidate j') whose column is in n to list j', row r' of the stage-2 lists.
//
// runtime args: [lists2_addr, r, j]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"

void kernel_main() {
    const uint32_t lists_addr = get_arg_val<uint32_t>(0);
    const uint32_t r = get_arg_val<uint32_t>(1);
    const uint32_t j = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_tab = get_compile_time_arg_val(1);
    constexpr uint32_t rows = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);
    constexpr uint32_t lists_bytes = get_compile_time_arg_val(4);
    constexpr auto l_args = TensorAccessorArgs<5>();
    const auto s_l = TensorAccessor(l_args, lists_addr, lists_bytes);
    constexpr uint32_t n_ent = rows * K;
    const uint32_t e = r * K + j;

    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1)), 1);
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* t_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_tab));
    const uint32_t n = t_[e] / 32;
    bool owner = true;
    for (uint32_t i = 0; i < e; ++i) {
        owner &= (t_[i] / 32) != n;
    }
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, owner ? 1 : 0);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, owner ? 1 : 0);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, owner ? 1 : 0);
    if (!owner) {
        return;
    }
    cb_wait_front(cb_out, 1);
    volatile tt_l1_ptr uint16_t* o = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_out));
    const uint64_t l0 = s_l.get_noc_addr(0);
    for (uint32_t i = e; i < n_ent; ++i) {
        const uint32_t c = t_[i];
        if (c / 32 != n) {
            continue;
        }
        const uint32_t ri = i / K, ji = i % K, lane = c % 32;
        // [rows, 32] tile: face (lane / 16) of rows x 16 values, row ri
        int32_t s = static_cast<int32_t>(static_cast<int16_t>(o[(lane / 16) * 16 * rows + ri * 16 + lane % 16]));
        s ^= (s >> 31) & 0x7FFF;
        const uint64_t dst = l0 + 8 * (rows * ji + ri);
        noc_inline_dw_write(dst, static_cast<uint32_t>(s));
        noc_inline_dw_write(dst + 4, c);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
