// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Compute-throughput benchmark reader: loads one Q block and kv_slots K/V blocks once, then
// republishes the same resident tiles for every (Q job, K chunk). No DRAM traffic in steady state.
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
    constexpr uint32_t q_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t d_tiles = get_compile_time_arg_val(5);
    constexpr auto qa = TensorAccessorArgs<6>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr uint32_t q_block = q_tiles * d_tiles;
    constexpr uint32_t kv_block = k_tiles * d_tiles;
    auto q = TensorAccessor(qa, get_arg_val<uint32_t>(0));
    auto k = TensorAccessor(ka, get_arg_val<uint32_t>(1));
    auto v = TensorAccessor(va, get_arg_val<uint32_t>(2));
    Noc noc;
    const uint32_t qbytes = get_tile_size(0);
    const uint32_t kbytes = get_tile_size(1);
    const uint32_t vbytes = get_tile_size(2);
    DataflowBuffer qcb(0), kcb(1), vcb(2);
    qcb.reserve_back(2 * q_block);
    kcb.reserve_back(kv_block * kv_slots);
    vcb.reserve_back(kv_block * kv_slots);
    for (uint32_t i = 0; i < 2 * q_block; ++i) {
        noc.async_read(q, qcb, qbytes, {.page_id = i % q_block}, {.offset_bytes = i * qbytes});
    }
    for (uint32_t i = 0; i < kv_block * kv_slots; ++i) {
        // Same layout as the recipe reader: transpose the K tile grid (not each tile).
        const uint32_t j = i % kv_block;
        const uint32_t k_page = (j % k_tiles) * d_tiles + j / k_tiles;
        noc.async_read(k, kcb, kbytes, {.page_id = k_page}, {.offset_bytes = i * kbytes});
        noc.async_read(v, vcb, vbytes, {.page_id = j}, {.offset_bytes = i * vbytes});
    }
    noc.async_read_barrier();
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        3,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    generate_bcast_col_scalar(CircularBuffer(4), 0x3f803f80);
    for (uint32_t qi = 0; qi < q_repeats; ++qi) {
        if (qi != 0) {
            qcb.reserve_back(q_block);
        }
        qcb.push_back(q_block);
        for (uint32_t ki = 0; ki < k_chunks; ++ki) {
            if (qi != 0 || ki != 0) {
                kcb.reserve_back(kv_block);
                vcb.reserve_back(kv_block);
            }
            kcb.push_back(kv_block);
            vcb.push_back(kv_block);
        }
    }
}
