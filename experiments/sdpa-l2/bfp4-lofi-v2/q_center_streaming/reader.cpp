// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
void kernel_main() {
    constexpr uint32_t chunks = get_compile_time_arg_val(0);
    constexpr auto qa = TensorAccessorArgs<1>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto ca = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    auto q = TensorAccessor(qa, get_arg_val<uint32_t>(0));
    auto k = TensorAccessor(ka, get_arg_val<uint32_t>(1));
    auto v = TensorAccessor(va, get_arg_val<uint32_t>(2));
    auto correction = TensorAccessor(ca, get_arg_val<uint32_t>(3));
    Noc noc;
    DataflowBuffer qcb(0), kcb(1), vcb(2), ccb(20);
    const uint32_t kb = get_tile_size(1), vb = get_tile_size(2);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        3, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    generate_bcast_col_scalar(CircularBuffer(4), 0x3f803f80);
    qcb.reserve_back(16);
    for (uint32_t j = 0; j < 16; ++j) {
        noc.async_read(q, qcb, 2048, {.page_id = j}, {.offset_bytes = j * 2048});
    }
    noc.async_read_barrier();
    qcb.push_back(16);
    for (uint32_t ki = 0; ki < chunks; ++ki) {
        kcb.reserve_back(64);
        vcb.reserve_back(64);
        ccb.reserve_back(16);
        for (uint32_t j = 0; j < 64; ++j) {
            noc.async_read(k, kcb, kb, {.page_id = ki * 64 + (j % 16) * 4 + j / 16}, {.offset_bytes = j * kb});
            noc.async_read(v, vcb, vb, {.page_id = ki * 64 + j}, {.offset_bytes = j * vb});
        }
        for (uint32_t j = 0; j < 16; ++j) {
            // One32-row correction tile per key tile, reused for all four Q rows.
            noc.async_read(correction, ccb, 4096, {.page_id = ki * 16 + j}, {.offset_bytes = j * 4096});
        }
        noc.async_read_barrier();
        ccb.push_back(16);
        kcb.push_back(64);
        vcb.push_back(64);
    }
}
