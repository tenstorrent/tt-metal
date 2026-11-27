// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DS matmul — activation mover (NCRISC, NoC1): the activation K blocks pulled from x (tile pages = K column tiles of
// the single row tile) into the double-buffered CB_IN0; after the compute, the NR output tiles written to y (pages n0
// .. n0 + NR - 1). Compile-time args: NB, KB, NR, then TensorAccessorArgs of x and y. Runtime args: x, y addresses, n0
// (first output column tile of this reader).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ds_mm_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"

using namespace dsmm;

void kernel_main() {
    constexpr uint32_t NB = get_compile_time_arg_val(0);
    constexpr uint32_t KB = get_compile_time_arg_val(1);
    constexpr uint32_t NR = get_compile_time_arg_val(2);
    constexpr auto x_args = TensorAccessorArgs<3>();
    constexpr auto y_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const auto y = TensorAccessor(y_args, get_arg_val<uint32_t>(1));
    const uint32_t n0 = get_arg_val<uint32_t>(2);

    MK_MARK();  // 0: start
    const uint32_t xpage = get_tile_size(CB_IN0);
#if defined(DSMM_WAIT_SEM_ID)
    // megakernel hand-off: the activation is the previous phase's output; its DSMM_WAIT_COUNT producer cores each
    // increment this reader's semaphore after their tiles landed (and reset it here for the next launch's count)
    volatile tt_l1_ptr uint32_t* ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(DSMM_WAIT_SEM_ID));
    noc_semaphore_wait(ready, DSMM_WAIT_COUNT);
    noc_semaphore_set(ready, 0);
#endif
    MK_MARK();  // 1: activation ready (hand-off received)
    for (uint32_t blk = 0; blk < NB; ++blk) {
        cb_reserve_back(CB_IN0, KB);
        const uint32_t l1 = get_write_ptr(CB_IN0);
        for (uint32_t k = 0; k < KB; ++k) {
            noc_async_read_page(blk * KB + k, x, l1 + k * xpage);
        }
        noc_async_read_barrier();
        cb_push_back(CB_IN0, KB);
    }
    MK_MARK();  // 2: last activation block pulled
    cb_wait_front(CB_OUT, NR);
    MK_MARK();  // 3: output tiles computed
    const uint32_t l1 = get_read_ptr(CB_OUT);
    const uint32_t opage = get_tile_size(CB_OUT);
    for (uint32_t j = 0; j < NR; ++j) {
        noc_async_write_page(n0 + j, y, l1 + j * opage);
    }
    noc_async_write_barrier();
    cb_pop_front(CB_OUT, NR);
    MK_MARK();  // 4: output written
#if defined(MK_PROBE_ARG)
    MK_FLUSH(get_write_ptr(MK_PROBE_CB) + 64);
#endif
}
