// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// QWEN38_TOPK_PAR=1, fold core, second data-movement RISC: reads the NT index tiles, then sends the partial top-32 (the
// key tile in DST 0 and the index tile in DST 2 as the fold leaves them) into slot `slot` of core B's cb_mk / cb_mi
// (same L1 address on every core: one CB table), then raises B's fold semaphore.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

void kernel_main() {
    const uint32_t index_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_base = get_arg_val<uint32_t>(1);
    const uint32_t b_x = get_arg_val<uint32_t>(2);
    const uint32_t b_y = get_arg_val<uint32_t>(3);
    const uint32_t slot = get_arg_val<uint32_t>(4);
    constexpr uint32_t cb_index = get_named_compile_time_arg_val("cb_index");
    constexpr uint32_t cb_ok = get_named_compile_time_arg_val("cb_vals_t");
    constexpr uint32_t cb_oi = get_named_compile_time_arg_val("cb_idx_t");
    constexpr uint32_t cb_mk = get_named_compile_time_arg_val("cb_mk");
    constexpr uint32_t cb_mi = get_named_compile_time_arg_val("cb_mi");
    constexpr uint32_t NT = get_named_compile_time_arg_val("NT");
    constexpr uint32_t fold_sem_id = get_named_compile_time_arg_val("fold_sem_id");
    constexpr auto index_args = TensorAccessorArgs<0, 0>();
    const auto index_template = TensorAccessor(index_args, index_addr);
    Noc noc;
    DataflowBuffer index(cb_index);
    DataflowBuffer ok(cb_ok);
    DataflowBuffer oi(cb_oi);
    DataflowBuffer mk(cb_mk);
    DataflowBuffer mi(cb_mi);
    {
        FUSED_ZONE("fz_rt_f_idx");
        const uint32_t tile_bytes = index.get_entry_size();
        index.reserve_back(NT);
        for (uint32_t w = 0; w < NT; ++w) {
            noc.async_read(
                index_template, index, tile_bytes, {.page_id = tile_base + w}, {.offset_bytes = w * tile_bytes});
        }
        noc.async_read_barrier();
        index.push_back(NT);
    }
    FUSED_ZONE("fz_rt_f_send");
    ok.wait_front(1);
    oi.wait_front(1);
    const uint32_t tb = ok.get_entry_size();
    noc_async_write(ok.get_read_ptr(), get_noc_addr(b_x, b_y, mk.get_write_ptr() + slot * tb), tb);
    noc_async_write(oi.get_read_ptr(), get_noc_addr(b_x, b_y, mi.get_write_ptr() + slot * tb), tb);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(b_x, b_y, get_semaphore(fold_sem_id)), 1);
    noc_async_atomic_barrier();
    ok.pop_front(1);
    oi.pop_front(1);
}
