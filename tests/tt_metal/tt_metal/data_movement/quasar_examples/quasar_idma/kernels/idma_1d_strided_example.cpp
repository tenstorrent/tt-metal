// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// IDMA 1D strided read: reads num_elements elements from src with a stride,
// writing them linearly to dst. One IDMA transaction per element.
//
// src inner loop: stride = src_stride (every other element by default)
// dst inner loop: stride = elem_size (linear)
//
// Mirrors the im2col.cpp pattern — addrgen inner loop drives src addressing,
// cmdbuf_0 issues one transaction per push.
//
// Sequence:
//   1. Configure src inner loop (strided) and dst inner loop (linear)
//   2. Configure cmdbuf_0 for IDMA copy, len = elem_size
//   3. Loop: push_both + issue for each element
//   4. Wait for IDMA ack

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"
#include <cstdint>

using namespace overlay;

constexpr uint32_t num_elements = 10;
constexpr uint32_t elem_size = 8;
constexpr uint32_t src_stride = 2 * elem_size;  // 16 — read every other element

void kernel_main() {
    const uint32_t src_base = get_compile_time_arg_val(0);
    const uint32_t dst_base = get_compile_time_arg_val(1);

    reset_cmdbuf_0();

    /* Address generator setup */
    reset_addrgen_0();
    // Src: strided inner loop
    setup_src_base_start_addrgen_0(src_base);
    setup_src_inner_loop_addrgen_0(src_stride, (uint64_t)num_elements * src_stride);
    // Dst: linear inner loop
    setup_dest_base_start_addrgen_0(dst_base);
    setup_dest_inner_loop_addrgen_0(elem_size, (uint64_t)num_elements * elem_size + 1);

    /* CMD Misc register, only difference to NOC*/
    idma_setup_as_copy_cmdbuf_0(false);
    /* Vcs = IDMA channel*/
    setup_vcs_cmdbuf_0(false);
    /* Trids work the same way as for NOC*/
    setup_trids_cmdbuf_0(0);
    /* Set transfer length */
    set_len_cmdbuf_0(elem_size);

    // TODO: wrap with DeviceTimestampedData profiling once Quasar perf counters are supported
    for (uint32_t i = 0; i < num_elements; ++i) {
        push_both_addrgen_0();
        issue_cmdbuf_0();
    }

    /* wait on IDMA to finish */
    while (!idma_acked_cmdbuf_0());

    DPRINT << "IDMA 1D strided done: " << DEC() << num_elements << " elements, src_stride=" << src_stride << ENDL();
}
