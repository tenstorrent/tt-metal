// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of the V4.1 attention-output head layout (tt/v41/head_layout.py o_heads): per unit (tile row r, head h),
// tilize the head's no-rope rows (NT tiles, straight to the writer) and its rope-tail rows (RT tiles), then the
// inverse RoPE of the tail (heads_rope.hpp with the caller's -sin) for the writer.
//
// compile_time_args = [NT, RT, H]
// runtime args      = [unit_start, unit_count]  (unit u = (r = u / H, h = u % H))

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "heads_rope.hpp"

namespace tcfg = compute_kernel_lib::tilize_config;

void kernel_main() {
    const uint32_t unit_start = get_arg_val<uint32_t>(0);
    const uint32_t unit_count = get_arg_val<uint32_t>(1);
    constexpr uint32_t NT = get_compile_time_arg_val(0);
    constexpr uint32_t RT = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t cb_rm_nope = 0, cb_rm_tail = 1, cb_cos = 2, cb_sin = 3, cb_trans = 4, cb_rot = 5, cb_sini = 6,
                       cb_cosi = 7, cb_tail = 8, cb_nope = 16, cb_roped = 17;
    if (unit_count == 0) {
        return;
    }

    DataflowBuffer cos(cb_cos);
    DataflowBuffer sin(cb_sin);
    DataflowBuffer trans(cb_trans);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_tail, cb_trans, cb_rot);
    trans.wait_front(1);
    uint32_t row = 0xFFFFFFFFu;
    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        const uint32_t r = u / H;
        if (r != row) {
            if (row != 0xFFFFFFFFu) {
                cos.pop_front(RT);
                sin.pop_front(RT);
            }
            row = r;
            cos.wait_front(RT);
            sin.wait_front(RT);
        }
        compute_kernel_lib::tilize<
            NT,
            cb_rm_nope,
            cb_nope,
            tcfg::InitUninitMode::InitAndUninit,
            tcfg::WaitMode::WaitBlock,
            tcfg::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
        compute_kernel_lib::tilize<
            RT,
            cb_rm_tail,
            cb_tail,
            tcfg::InitUninitMode::InitAndUninit,
            tcfg::WaitMode::WaitBlock,
            tcfg::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
        rope_tail<RT, cb_tail, cb_cos, cb_sin, cb_trans, cb_rot, cb_sini, cb_cosi, cb_roped>();
    }
    cos.pop_front(RT);
    sin.pop_front(RT);
    trans.pop_front(1);
}
