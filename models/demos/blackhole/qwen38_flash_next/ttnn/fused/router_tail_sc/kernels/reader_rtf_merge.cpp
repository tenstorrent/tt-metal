// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// QWEN38_TOPK_PAR=1, core B reader: the fold cores write their partial top-32 into cb_mk / cb_mi over NoC and raise
// fold_sem_id; this hands the NL pages to compute once all NL arrived, then sends the merged index tile to core A
// exactly as reader_rtf_sort.cpp does.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "../../kernels/zones.h"

void kernel_main() {
    const uint32_t a_x = get_arg_val<uint32_t>(0);
    const uint32_t a_y = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_mk = get_named_compile_time_arg_val("cb_mk");
    constexpr uint32_t cb_mi = get_named_compile_time_arg_val("cb_mi");
    constexpr uint32_t cb_idx_t = get_named_compile_time_arg_val("cb_idx_t");
    constexpr uint32_t NL = get_named_compile_time_arg_val("NL");
    constexpr uint32_t sem_id = get_named_compile_time_arg_val("sem_id");
    constexpr uint32_t fold_sem_id = get_named_compile_time_arg_val("fold_sem_id");

    DataflowBuffer mk(cb_mk);
    DataflowBuffer mi(cb_mi);
    DataflowBuffer idx_t(cb_idx_t);

    {
        FUSED_ZONE("fz_rt_r_fold_wait");
        mk.reserve_back(NL);
        mi.reserve_back(NL);
        volatile tt_l1_ptr uint32_t* fold_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(fold_sem_id));
        noc_semaphore_wait_min(fold_sem, NL);
        noc_semaphore_set(fold_sem, 0);
        mk.push_back(NL);
        mi.push_back(NL);
    }

    {
        FUSED_ZONE("fz_rt_r_send");
        idx_t.wait_front(1);
        const uint32_t src = idx_t.get_read_ptr();
        noc_async_write(src, get_noc_addr(a_x, a_y, src), 640);  // rows 0..9 of face 0: all the 10 indices of token 0
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(a_x, a_y, get_semaphore(sem_id)), 1);
        noc_async_atomic_barrier();
        idx_t.pop_front(1);
    }
}
