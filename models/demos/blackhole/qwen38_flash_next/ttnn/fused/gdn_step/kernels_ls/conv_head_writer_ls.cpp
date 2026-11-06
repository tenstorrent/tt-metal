// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Writer of a conv core that also takes head items (QWEN38_GDN_LS_CONVHEAD=1, option B of plan_nov05/g1_gdnlane): the
// conv writer phase (conv_writer.cpp's code, in namespace convphase), then the head writer of writer_vb.cpp (included
// as it is, its kernel_main renamed).  The conv phase sends the conv tile, then the semaphore increments, and waits for
// the write and atomic barriers: it ends only when every tile and every increment of this core has been delivered,
// before the head phase starts.  The conv tile lives in CB 28 (CONVOUT), NOT in CB 11: CB 11 is this core's head-side
// receive area (slots written by the other conv cores), so the conv result must not sit in it.  The destination address
// is the base of CB 11 (one descriptor over all cores: the same base everywhere) + slot * 2048. Runtime args: 0-2 +
// items * (3 + 2 * VB) the head writer's (writer_vb.cpp); then the conv writer's: destination count, per destination
// (noc_x, noc_y, slot, semaphore id).  Compile-time args: the head writer's (writer_vb.cpp).
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

#define kernel_main ls_head_writer_main
#include "../kernels/writer_vb.cpp"
#undef kernel_main

namespace convphase {
constexpr uint32_t CB_QKV = 11, CB_CONVOUT = 28;
constexpr uint32_t BF16_TILE = 2048;

FORCE_INLINE void run() {
    constexpr uint32_t VBC = get_compile_time_arg_val(3);
    const uint32_t items = get_arg_val<uint32_t>(2);
    const uint32_t a0 = 3 + items * (3 + 2 * VBC);  // first conv writer arg
    const uint32_t count = get_arg_val<uint32_t>(a0);
    FUSED_ZONE("fz_gls_cw");
    const uint32_t base = get_write_ptr(CB_QKV);  // CB 11 base: nothing of this core has touched CB 11 yet
    cb_wait_front(CB_CONVOUT, 1);
    const uint32_t src = get_read_ptr(CB_CONVOUT);
    for (uint32_t d = 0; d < count; ++d) {
        const uint32_t x = get_arg_val<uint32_t>(a0 + 1 + 4 * d);
        const uint32_t y = get_arg_val<uint32_t>(a0 + 2 + 4 * d);
        const uint32_t slot = get_arg_val<uint32_t>(a0 + 3 + 4 * d);
        noc_async_write(src, get_noc_addr(x, y, base + slot * BF16_TILE), BF16_TILE);
    }
    noc_async_write_barrier();
    for (uint32_t d = 0; d < count; ++d) {
        const uint32_t x = get_arg_val<uint32_t>(a0 + 1 + 4 * d);
        const uint32_t y = get_arg_val<uint32_t>(a0 + 2 + 4 * d);
        const uint32_t sem = get_arg_val<uint32_t>(a0 + 4 + 4 * d);
        noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(sem)), 1);
    }
    noc_async_atomic_barrier();
    cb_pop_front(CB_CONVOUT, 1);
}
}  // namespace convphase

void kernel_main() {
    convphase::run();  // all conv tiles and increments delivered (both barriers) before the first head item
    ls_head_writer_main();
}
