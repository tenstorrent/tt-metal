// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of the V4.1 query head layout (tt/v41/head_layout.py q_heads): per unit (tile row r, head h), RoPE on the
// head's RT rope-tail tiles (heads_rope.hpp), then untilize the NT no-rope tiles and the rotated tail into two
// row-major blocks that the writer places side by side in the head's rows.
//
// compile_time_args = [NT, RT, H]
// runtime args      = [unit_start, unit_count]  (unit u = (r = u / H, h = u % H))

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "heads_rope.hpp"

namespace ucfg = compute_kernel_lib::untilize_config;

void kernel_main() {
    const uint32_t unit_start = get_arg_val<uint32_t>(0);
    const uint32_t unit_count = get_arg_val<uint32_t>(1);
    constexpr uint32_t NT = get_compile_time_arg_val(0);
    constexpr uint32_t RT = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t cb_nope = 0, cb_tail = 1, cb_cos = 2, cb_sin = 3, cb_trans = 4, cb_rot = 5, cb_sini = 6,
                       cb_cosi = 7, cb_roped = 8, cb_rm_nope = 16, cb_rm_tail = 17;
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
        rope_tail<RT, cb_tail, cb_cos, cb_sin, cb_trans, cb_rot, cb_sini, cb_cosi, cb_roped>();
        compute_kernel_lib::untilize<
            NT,
            cb_nope,
            cb_rm_nope,
            ucfg::InitUninitMode::InitAndUninit,
            ucfg::WaitMode::WaitBlock,
            ucfg::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
        compute_kernel_lib::untilize<
            RT,
            cb_roped,
            cb_rm_tail,
            ucfg::InitUninitMode::InitAndUninit,
            ucfg::WaitMode::WaitBlock,
            ucfg::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
    }
    cos.pop_front(RT);
    sin.pop_front(RT);
    trans.pop_front(1);
}
