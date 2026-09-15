// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t q_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t k_chunks = get_compile_time_arg_val(1);
    constexpr uint32_t queries_per_head = get_compile_time_arg_val(2);
    constexpr auto qa = TensorAccessorArgs<3>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    auto q = TensorAccessor(qa, get_arg_val<uint32_t>(0));
    auto k = TensorAccessor(ka, get_arg_val<uint32_t>(1));
    auto v = TensorAccessor(va, get_arg_val<uint32_t>(2));
    const uint32_t first_job = get_arg_val<uint32_t>(3);
    const uint32_t jobs = get_arg_val<uint32_t>(4);
    Noc noc;
    DataflowBuffer qcb(0), kcb(1), vcb(2);
    const uint32_t qbytes = get_tile_size(0), kbytes = get_tile_size(1), vbytes = get_tile_size(2);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        3,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    generate_bcast_col_scalar(CircularBuffer(4), 0x3f803f80);
    for (uint32_t job = first_job; job < first_job + jobs; ++job) {
        const uint32_t head = job / queries_per_head;
        const uint32_t qbase = job * q_tiles * 4;
        const uint32_t kvbase = head * k_chunks * 64;
        qcb.reserve_back(q_tiles * 4);
        for (uint32_t i = 0; i < q_tiles * 4; ++i) {
            noc.async_read(q, qcb, qbytes, {.page_id = qbase + i}, {.offset_bytes = i * qbytes});
        }
        noc.async_read_barrier();
        qcb.push_back(q_tiles * 4);
        for (uint32_t ki = 0; ki < k_chunks; ++ki) {
            kcb.reserve_back(64);
#ifndef SDPA_READER_SPLIT_KV
            vcb.reserve_back(64);
#endif
            for (uint32_t i = 0; i < 64; ++i) {
#ifdef SDPA_READER_LINEAR_K
                const uint32_t kpage = kvbase + ki * 64 + i;
                const uint32_t kdest = (i % 4) * 16 + i / 4;
#else
                const uint32_t kpage = kvbase + ki * 64 + (i % 16) * 4 + i / 16;
                const uint32_t kdest = i;
#endif
                noc.async_read(k, kcb, kbytes, {.page_id = kpage}, {.offset_bytes = kdest * kbytes});
#ifndef SDPA_READER_SPLIT_KV
                const uint32_t vpage = kvbase + ki * 64 + i;
                noc.async_read(v, vcb, vbytes, {.page_id = vpage}, {.offset_bytes = i * vbytes});
#endif
#if SDPA_READER_BARRIER_TILES > 0
                if ((i + 1) % SDPA_READER_BARRIER_TILES == 0) {
                    noc.async_read_barrier();
                }
#endif
            }
            noc.async_read_barrier();
            kcb.push_back(64);
#ifdef SDPA_READER_SPLIT_KV
            vcb.reserve_back(64);
            for (uint32_t i = 0; i < 64; ++i) {
                noc.async_read(v, vcb, vbytes, {.page_id = kvbase + ki * 64 + i}, {.offset_bytes = i * vbytes});
#if SDPA_READER_BARRIER_TILES > 0
                if ((i + 1) % SDPA_READER_BARRIER_TILES == 0) {
                    noc.async_read_barrier();
                }
#endif
            }
            noc.async_read_barrier();
#endif
            vcb.push_back(64);
        }
    }
}
