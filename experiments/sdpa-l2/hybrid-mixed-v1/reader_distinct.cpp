// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t q_repeats = get_compile_time_arg_val(0);
    constexpr uint32_t k_chunks = get_compile_time_arg_val(1);
    constexpr uint32_t kv_slots = get_compile_time_arg_val(2);
    constexpr auto qa = TensorAccessorArgs<3>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    auto q = TensorAccessor(qa, get_arg_val<uint32_t>(0));
    auto k = TensorAccessor(ka, get_arg_val<uint32_t>(1));
    auto v = TensorAccessor(va, get_arg_val<uint32_t>(2));
    Noc noc;
    DataflowBuffer qcb(0), kcb(1), vcb(2);
    // Correctness-only reader: stream distinct K/V chunks; not a no-DM benchmark.
    qcb.reserve_back(64);
    for (uint32_t i = 0; i < 64; ++i) {
        noc.async_read(q, qcb, 2048, {.page_id = i % 32}, {.offset_bytes = i * 2048});
    }
    noc.async_read_barrier();
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        3,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    generate_bcast_col_scalar(CircularBuffer(4), 0x3f803f80);
    // Publish one chunk at a time, preserving ordinary CB producer/consumer
    // semantics and ring capacities. Reserve calls cannot overwrite resident data.
    for (uint32_t qi = 0; qi < q_repeats; ++qi) {
        if (qi != 0) {
            qcb.reserve_back(32);
        }
        qcb.push_back(32);
        for (uint32_t ki = 0; ki < k_chunks; ++ki) {
            kcb.reserve_back(64);
            vcb.reserve_back(64);
            for (uint32_t i = 0; i < 64; ++i) {
                const uint32_t k_page = ki * 64 + (i % 16) * 4 + i / 16;
                noc.async_read(k, kcb, 2048, {.page_id = k_page}, {.offset_bytes = i * 2048});
                noc.async_read(v, vcb, 2048, {.page_id = ki * 64 + i}, {.offset_bytes = i * 2048});
            }
            noc.async_read_barrier();
            kcb.push_back(64);
            vcb.push_back(64);
        }
    }
}
