// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#ifndef QB2_ENTRY
#define QB2_ENTRY kernel_main
#endif
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"
#include "tools/profiler/kernel_profiler.hpp"
#include "zero_l1.hpp"
#if COMPACT_NORM_OUTPUT
#include "compact_rows.hpp"
#endif
#ifdef PREFETCH_ROLE
#include "prefetch_projection.hpp"
#endif
constexpr auto input_args=TensorAccessorArgs<0>();
constexpr auto output_args=TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
uint64_t norm_address(uint32_t rank, uint32_t addr) {
    return get_noc_addr(get_arg_val<uint32_t>(4+2*rank),get_arg_val<uint32_t>(5+2*rank),addr);
}
void QB2_ENTRY() {
    const uint32_t rank=get_arg_val<uint32_t>(0);
#ifdef READER
#if defined(PREFETCH_ROLE) && PREFETCH_ROLE == 0
    prefetch_projection_weights();
#endif
#ifdef FUSE_GATHER
    {
        DeviceZoneScopedN("MLP-NORM-WAIT-GATHER");
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(12)),1);
    }
#endif
    // Reduction packing writes only the statistic column. Clear masked lanes
    // before publishing input so later full-row reduction cannot see old L1.
    if (initialize_layer_scratch(2)) {
#if TINY_NORM_M
        zero_l1<16 * 2048>(get_write_ptr(16));
#endif
#if NORM_STATS_FACE
        // For16-row tiles all row statistics, including padded rows, live in
        // face0. Other faces remain zero through this invocation.
        if (rank == 0) { zero_l1<8 * 4096>(get_write_ptr(9)); }
        zero_l1<4096>(get_write_ptr(10));
#endif
        for (uint32_t cb : {7u, 8u, 11u}) {
            zero_l1<4096>(get_write_ptr(cb));
        }
    }
    const auto input=TensorAccessor(input_args,get_arg_val<uint32_t>(1),2048);
    cb_reserve_back(0,16);
    for(uint32_t t=0;t<16;++t) { noc_async_read_page(rank*16+t,input,get_write_ptr(0)+t*2048); }
    noc_async_read_barrier();
    cb_push_back(0,16);
    cb_wait_front(7,1);
    noc_semaphore_inc(norm_address(0,get_semaphore(8)),1);
    if(rank==0) {
        DeviceZoneScopedN("MLP-NORM-REDUCE-BARRIER");
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(8)),8);
        cb_reserve_back(9,8);
        for(uint32_t src=0;src<8;++src) {
            noc_async_read(norm_address(src,get_read_ptr(7)),get_write_ptr(9)+src*4096,NORM_STATS_FACE ? 1024 : 4096);
        }
        noc_async_read_barrier();
        cb_push_back(9,8);
        cb_wait_front(11,1);
        // All receiver CB layouts are identical. Reserve occurs before use;
        // no prior program can reference this program-local storage.
        for(uint32_t dst=0;dst<8;++dst) {
            noc_async_write(get_read_ptr(11),norm_address(dst,get_write_ptr(10)),NORM_STATS_FACE ? 1024 : 4096);
        }
        noc_async_write_barrier();
        for(uint32_t dst=0;dst<8;++dst) { noc_semaphore_inc(norm_address(dst,get_semaphore(9)),1); }
        noc_async_atomic_barrier();
        cb_pop_front(11,1);
    }
    cb_reserve_back(10,1);
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(9)),1);
    cb_push_back(10,1);
    cb_pop_front(7,1);
#if defined(PREFETCH_ROLE) && PREFETCH_ROLE == 1
    prefetch_projection_weights();
#endif
#else
    if (initialize_layer_scratch(2)) {
        dataflow_kernel_lib::prepare_reduce_scaler<2,ckernel::PoolType::SUM,ckernel::ReduceDim::REDUCE_ROW>(1.0f/512.0f);
        zero_l1<2048>(get_write_ptr(3));
        DataflowBuffer epsilon(3);
        generate_bcast_col_scalar(epsilon,get_arg_val<uint32_t>(3));
        if(rank==0) {
            dataflow_kernel_lib::prepare_reduce_scaler<4,ckernel::PoolType::AVG,ckernel::ReduceDim::REDUCE_ROW>(1.0f/8.0f);
        }
    } else {
        // These one-tile rings contain layer-invariant constants and are
        // read-only to compute. Reset metadata, then republish their contents.
        cb_reserve_back(2, 1); cb_push_back(2, 1);
        cb_reserve_back(3, 1); cb_push_back(3, 1);
        if (rank == 0) { cb_reserve_back(4, 1); cb_push_back(4, 1); }
    }
    cb_wait_front(16,16);
    const auto output=TensorAccessor(output_args,get_arg_val<uint32_t>(2),2048);
#if COMPACT_NORM_OUTPUT
    compact_bf16_rows<16>(get_read_ptr(16));
    noc_async_write(get_read_ptr(16), output.get_noc_addr(rank * 16), 16 * 64);
#else
    for(uint32_t t=0;t<16;++t) { noc_async_write_page(rank*16+t,output,get_read_ptr(16)+t*2048); }
#endif
    noc_async_write_barrier();
    cb_pop_front(16,16);
#ifdef FUSE_NORM
    const uint32_t coord_x=get_arg_val<uint32_t>(20), coord_y=get_arg_val<uint32_t>(21);
    noc_semaphore_inc(get_noc_addr(coord_x,coord_y,get_semaphore(NORM_READY_SEMAPHORE)),1);
    noc_async_atomic_barrier();
#endif
#endif
}
