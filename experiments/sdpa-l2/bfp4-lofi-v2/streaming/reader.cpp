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
#ifdef SDPA_LOFI_RESIDUALS
    constexpr auto kra = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto vra = TensorAccessorArgs<kra.next_compile_time_args_offset()>();
    auto kr = TensorAccessor(kra, get_arg_val<uint32_t>(3));
    auto vr = TensorAccessor(vra, get_arg_val<uint32_t>(4));
    DataflowBuffer krcb(17), vrcb(18);
    const uint32_t krbytes = get_tile_size(17), vrbytes = get_tile_size(18);
#endif
#ifdef SDPA_LOFI_V_BIAS
#ifdef SDPA_LOFI_RESIDUALS
    constexpr auto ba = TensorAccessorArgs<vra.next_compile_time_args_offset()>();
    auto bias = TensorAccessor(ba, get_arg_val<uint32_t>(5));
#else
    constexpr auto ba = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    auto bias = TensorAccessor(ba, get_arg_val<uint32_t>(3));
#endif
    DataflowBuffer bias_cb(19);
#endif
    Noc noc;
    const uint32_t qbytes = get_tile_size(0);
    const uint32_t kbytes = get_tile_size(1);
    const uint32_t vbytes = get_tile_size(2);
    DataflowBuffer qcb(0), kcb(1), vcb(2);
    // Correctness-only reader: stream distinct K/V chunks; not a no-DM benchmark.
    qcb.reserve_back(32);
    for (uint32_t i = 0; i < 32; ++i) {
        noc.async_read(q, qcb, qbytes, {.page_id = i % 16}, {.offset_bytes = i * qbytes});
    }
    noc.async_read_barrier();
#ifdef SDPA_LOFI_V_BIAS
    bias_cb.reserve_back(4);
    for (uint32_t i = 0; i < 4; ++i) {
        noc.async_read(bias, bias_cb, 4096, {.page_id = i}, {.offset_bytes = i * 4096});
    }
    noc.async_read_barrier();
    bias_cb.push_back(4);
#endif
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
            qcb.reserve_back(16);
        }
        qcb.push_back(16);
        for (uint32_t ki = 0; ki < k_chunks; ++ki) {
            kcb.reserve_back(64);
            vcb.reserve_back(64);
#ifdef SDPA_LOFI_RESIDUALS
            krcb.reserve_back(64);
            vrcb.reserve_back(64);
#endif
            for (uint32_t i = 0; i < 64; ++i) {
                const uint32_t k_page = ki * 64 + (i % 16) * 4 + i / 16;
                noc.async_read(k, kcb, kbytes, {.page_id = k_page}, {.offset_bytes = i * kbytes});
                noc.async_read(v, vcb, vbytes, {.page_id = ki * 64 + i}, {.offset_bytes = i * vbytes});
#ifdef SDPA_LOFI_RESIDUALS
                noc.async_read(kr, krcb, krbytes, {.page_id = k_page}, {.offset_bytes = i * krbytes});
                noc.async_read(vr, vrcb, vrbytes, {.page_id = ki * 64 + i}, {.offset_bytes = i * vrbytes});
#endif
            }
            noc.async_read_barrier();
            kcb.push_back(64);
            vcb.push_back(64);
#ifdef SDPA_LOFI_RESIDUALS
            krcb.push_back(64);
            vrcb.push_back(64);
#endif
        }
    }
}
