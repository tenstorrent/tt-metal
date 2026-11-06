// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// The GDN step's input prefetch as an op of its own (gdn_step.py prefetch_op): for the first GDN layer of a decode
// step, which no earlier linear hosts. Common args: pf, nv, group, qt, wab, consts, conv state, recurrent state,
// grid_x.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "../../ternary/kernels/core_span.hpp"
#include "step_prefetch.hpp"

void kernel_main() {
    constexpr auto pw = TensorAccessorArgs<0>();
    constexpr auto pc = TensorAccessorArgs<pw.next_compile_time_args_offset()>();
    constexpr auto pv = TensorAccessorArgs<pc.next_compile_time_args_offset()>();
    constexpr auto ps = TensorAccessorArgs<pv.next_compile_time_args_offset()>();
    const uint32_t nv = get_common_arg_val<uint32_t>(1);
    const uint32_t j = core_index(get_common_arg_val<uint32_t>(8)) - STEP_FIRST;
    step_pf::prefetch(
        get_common_arg_val<uint32_t>(0),
        j,
        nv,
        get_common_arg_val<uint32_t>(2),
        get_common_arg_val<uint32_t>(3),
        TensorAccessor(pw, get_common_arg_val<uint32_t>(4), 2048),
        TensorAccessor(pc, get_common_arg_val<uint32_t>(5), 2048),
        TensorAccessor(pv, get_common_arg_val<uint32_t>(6), 2048),
        TensorAccessor(ps, get_common_arg_val<uint32_t>(7), 2048));
    noc_async_read_barrier();
}
