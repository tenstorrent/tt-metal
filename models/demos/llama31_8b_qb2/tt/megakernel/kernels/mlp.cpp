// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#ifndef QB2_ENTRY
#define QB2_ENTRY kernel_main
#endif

// One local MLP body, selected by a DRAM weight-address-table row. Projection
// and SFPU workers are disjoint; only BRISC produces the phase-release flags.
// Every flag is program-local and is reset by dispatch before trace replay.
#ifndef GU_WORKERS
#define GU_WORKERS 8
#endif
constexpr unsigned gu_width = 224 / GU_WORKERS;
#ifndef PROJECTION_WIDE
#define PROJECTION_WIDE 0
#endif

#if defined(READER) || defined(WRITER)
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr auto input_args = TensorAccessorArgs<0>();
constexpr auto gu_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
constexpr auto down_args = TensorAccessorArgs<gu_args.next_compile_time_args_offset()>();
constexpr auto packed_args = TensorAccessorArgs<down_args.next_compile_time_args_offset()>();
constexpr auto product_args = TensorAccessorArgs<packed_args.next_compile_time_args_offset()>();
constexpr auto output_args = TensorAccessorArgs<product_args.next_compile_time_args_offset()>();
constexpr auto table_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();

#ifdef FUSE_OUTPUT
constexpr auto o_weight_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
constexpr auto o_input_args = TensorAccessorArgs<o_weight_args.next_compile_time_args_offset()>();
constexpr unsigned coord_start = 8;
#else
constexpr unsigned coord_start = 7;
#endif
uint64_t worker_address(uint32_t worker, uint32_t address) {
    return get_noc_addr(get_arg_val<uint32_t>(coord_start + 2 * worker), get_arg_val<uint32_t>(coord_start + 1 + 2 * worker), address);
}

void wait_phase(uint32_t id) {
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id)), 1);
}

void notify_coordinator(uint32_t id) {
    noc_semaphore_inc(worker_address(0, get_semaphore(id)), 1);
}

void release_workers(uint32_t id, uint32_t begin, uint32_t end) {
    for (uint32_t worker = begin; worker < end; ++worker) {
        noc_semaphore_inc(worker_address(worker, get_semaphore(id)), 1);
    }
    noc_async_atomic_barrier();
}

#if SHARED_QKV && defined(PROJECTION)
#include "shared_qkv.hpp"
#endif
#if defined(PROJECTION) && defined(READER)
#include "projection_reader.hpp"
#ifdef HEAD_PREFETCH_RECEIVER
#include "prefetch_receiver.hpp"
#endif
template <uint32_t A, uint32_t B, uint32_t KBlock, uint32_t N, uint32_t K, uint32_t Workers, typename Input, typename Weight>
void stream_projection(const Input& input, const Weight& weight, uint32_t bank) {
#if PROJECTION_READER > 0
    uint64_t prefetched_base = 0;
    constexpr uint32_t prefetched_blocks = B == 1 ? GU_PREFETCH_BLOCKS : (B == 3 ? DOWN_PREFETCH_BLOCKS : 0);
#if GU_PREFETCH_BLOCKS || DOWN_PREFETCH_BLOCKS
    if constexpr (prefetched_blocks > 0) {
        static_assert(Workers == 8);
        constexpr uint32_t role = B == 1 ? 0 : 1;
        auto* state = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(LOOP_RT_OFFSET));
        noc_semaphore_wait_min(state + 300 + role, 1);
        prefetched_base = get_noc_addr(
            get_arg_val<uint32_t>(PREFETCH_COORD_OFFSET + role * 16 + 2 * bank),
            get_arg_val<uint32_t>(PREFETCH_COORD_OFFSET + role * 16 + 2 * bank + 1), state[300 + role]);
    }
#endif
#ifdef HEAD_PREFETCH_RECEIVER
    if constexpr (B == 7) {
        prefetched_base = wait_head_prefetch(bank);
        tuned_stream_projection<A, B, KBlock, N, K, Workers, 1088>(
            input, weight, bank, prefetched_base, K / KBlock);
        finish_head_prefetch(bank);
    } else
#endif
    {
        constexpr uint32_t local_prefix = (EARLY_WEIGHT_PHASES & (B == 7 ? 2 : B == 1 ? 4 : 8)) ? EARLY_WEIGHT_BLOCKS : 0;
        tuned_stream_projection<A, B, KBlock, N, K, Workers, B == 1 ? 576 : 1088>(
            input, weight, bank, prefetched_base, prefetched_blocks, local_prefix);
    }
#else
    for (uint32_t k = 0; k < K; k += KBlock) {
        cb_reserve_back(A, KBlock);
        cb_reserve_back(B, KBlock * N);
        const uint32_t a = get_write_ptr(A);
        const uint32_t b = get_write_ptr(B);
        constexpr uint32_t weight_bytes = B == 1 ? 576 : 1088;
        for (uint32_t row = 0; row < KBlock; ++row) {
            noc_async_read_page(k + row, input, a + row * 2048);
            // The selected baseline weights are width-sharded over eight DRAM
            // banks. N contiguous tiles form one bank-local K row.
            noc_async_read(weight.get_noc_addr((k + row) * (Workers * N) + bank * N), b + row * N * weight_bytes, N * weight_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(A, KBlock);
        cb_push_back(B, KBlock * N);
    }
#endif
}

void QB2_ENTRY() {
    const uint32_t bank = get_arg_val<uint32_t>(0);
    const auto table = TensorAccessor(table_args, get_arg_val<uint32_t>(5), 128);
    const uint32_t scratch = get_write_ptr(31);
    noc_async_read(table.get_noc_addr(get_arg_val<uint32_t>(6)), scratch, 128);
    noc_async_read_barrier();
    const auto* addresses = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
#if SHARED_QKV
    shared_qkv_read(bank, addresses[3]);
#endif
#ifdef FUSE_OUTPUT
    if (bank < 8) {
#if EARLY_WEIGHT_BLOCKS && (EARLY_WEIGHT_PHASES & 2)
        {
            DeviceZoneScopedN("MLP-O-LOCAL-WEIGHT-PREFETCH");
            const auto weight = TensorAccessor(o_weight_args, addresses[2], 1088);
            prefetch_local_projection_weights<7, 4, 16, 8, 1088>(weight, bank);
        }
#endif
#ifdef FUSE_ATTENTION
        {
#ifndef LOOP_RT_OFFSET
        // Keep all32 layer read/math/barrier phases within the marker buffer.
        DeviceZoneScopedN("MLP-WAIT-ATTENTION");
#endif
            wait_phase(14);
        }
#endif
#if !SHARED_QKV
        DeviceZoneScopedN("MLP-O-READ");
#endif
        const auto attn = TensorAccessor(o_input_args, get_arg_val<uint32_t>(7), 2048);
        const auto weight = TensorAccessor(o_weight_args, addresses[2], 1088);
        stream_projection<6, 7, 4, 16, 32, 8>(attn, weight, bank);
    }
#endif
#if EARLY_WEIGHT_BLOCKS && (EARLY_WEIGHT_PHASES & 4)
    wait_phase(8);  // The local O writer has finished consuming aliased weight storage.
    {
        DeviceZoneScopedN("MLP-GU-LOCAL-WEIGHT-PREFETCH");
        const auto weight = TensorAccessor(gu_args, addresses[0], 576);
        prefetch_local_projection_weights<1, 8, 28, 8, 576>(weight, bank);
    }
#endif
#ifdef FUSE_NORM
    {
#ifndef LOOP_RT_OFFSET
        // Keep all32 layer read/math/barrier phases within the marker buffer.
        DeviceZoneScopedN("MLP-WAIT-NORM");
#endif
        if (bank == 0) {
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(6)), 8);
            release_workers(7, 0, GU_WORKERS);
        }
        wait_phase(7);
    }
#endif
    const auto input = TensorAccessor(input_args, get_arg_val<uint32_t>(1), 2048);
    const auto gu = TensorAccessor(gu_args, addresses[0], 576);
    const auto down = TensorAccessor(down_args, addresses[1], 1088);
    const auto product = TensorAccessor(product_args, get_arg_val<uint32_t>(3), 2048);
    {
        DeviceZoneScopedN("MLP-GU-READ");
        const uint32_t gu_column = DRAM_NEAR_PROJECTION && GU_WORKERS == 16 ? 2 * (bank % 8) + bank / 8 : bank;
        stream_projection<0, 1, 8, gu_width, 128, GU_WORKERS>(input, gu, gu_column);
    }
#if EARLY_WEIGHT_BLOCKS && (EARLY_WEIGHT_PHASES & 8)
    wait_phase(9);  // The local GU writer has finished consuming aliased weight storage.
    {
        DeviceZoneScopedN("MLP-DOWN-LOCAL-WEIGHT-PREFETCH");
        prefetch_local_projection_weights<3, 7, 16, 8, 1088>(down, bank);
    }
#endif
    {
#ifndef LOOP_RT_OFFSET
        // Keep all32 layer read/math/barrier phases within the marker buffer.
        DeviceZoneScopedN("MLP-WAIT-PRODUCT");
#endif
        wait_phase(3);
    }
    if (bank >= 8) { return; }
    {
        DeviceZoneScopedN("MLP-DOWN-READ");
        stream_projection<4, 3, 7, 16, 112, 8>(product, down, bank);
    }
}
#elif defined(PROJECTION) && defined(WRITER)
void QB2_ENTRY() {
#if SHARED_QKV
    shared_qkv_write(get_arg_val<uint32_t>(0));
#endif
    const uint32_t bank = get_arg_val<uint32_t>(0);
#ifdef FUSE_OUTPUT
    if (bank < 8) {
#ifndef LOOP_RT_OFFSET
        // Keep all32 layer read/math/barrier phases within the marker buffer.
        DeviceZoneScopedN("MLP-O-WRITE");
#endif
        cb_wait_front(17, 16);
        const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(4), 2048);
        for (uint32_t tile = 0; tile < 16; ++tile) {
            noc_async_write_page(bank * 16 + tile, output, get_read_ptr(17) + tile * 2048);
        }
        noc_async_write_barrier();
        cb_pop_front(17, 16);
#if EARLY_WEIGHT_BLOCKS && (EARLY_WEIGHT_PHASES & 4)
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(8)), 1);
#endif
        notify_coordinator(10);
        if (bank == 0) {
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(10)), 8);
            release_workers(11, GU_WORKERS + 16, GU_WORKERS + 18);
        }
        noc_async_atomic_barrier();
    }
#endif
    cb_wait_front(16, gu_width);
#if DRAM_NEAR_PROJECTION
    const auto packed_output = TensorAccessor(packed_args, get_arg_val<uint32_t>(2), 2048);
    const uint32_t gu_column = GU_WORKERS == 16 ? 2 * (bank % 8) + bank / 8 : bank;
    for (uint32_t tile = 0; tile < gu_width; ++tile) {
        noc_async_write_page(gu_column * gu_width + tile, packed_output, get_read_ptr(16) + tile * 2048);
    }
    noc_async_write_barrier();
#endif
#if EARLY_WEIGHT_BLOCKS && (EARLY_WEIGHT_PHASES & 8)
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(9)), 1);
#endif
    notify_coordinator(0);
    if (bank == 0) {
        {
    #if !SHARED_QKV
        DeviceZoneScopedN("MLP-GU-BARRIER");
#endif
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), GU_WORKERS);
        }
        release_workers(1, GU_WORKERS, GU_WORKERS + 16);
        {
            DeviceZoneScopedN("MLP-PRODUCT-BARRIER");
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2)), 16);
        }
        release_workers(3, 0, GU_WORKERS);
    }
    // The packed tensor remains live while the SFPU readers consume it.
    wait_phase(3);
    cb_pop_front(16, gu_width);
    if (bank >= 8) { return; }
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(4), 2048);
    cb_wait_front(17, 16);
    {
#ifndef LOOP_RT_OFFSET
        // Keep all32 layer read/math/barrier phases within the marker buffer.
        DeviceZoneScopedN("MLP-OUTPUT-WRITE");
#endif
        for (uint32_t tile = 0; tile < 16; ++tile) {
            noc_async_write_page(bank * 16 + tile, output, get_read_ptr(17) + tile * 2048);
        }
        noc_async_write_barrier();
    }
    cb_pop_front(17, 16);
#ifdef FUSE_REDUCE
    // Every down shard must be globally visible before collective readers
    // consume slices spanning projection workers. The write barrier above
    // orders data before these release atomics.
    notify_coordinator(4);
    if (bank == 0) {
        DeviceZoneScopedN("MLP-DOWN-BARRIER");
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4)), 8);
        release_workers(5, GU_WORKERS + 16, GU_WORKERS + 18);
    }
    noc_async_atomic_barrier();
#endif
}
#elif defined(SWIGLU) && defined(READER)
void QB2_ENTRY() {
    {
        DeviceZoneScopedN("MLP-SWIGLU-WAIT");
        wait_phase(1);
    }
    const uint32_t first_tile = get_arg_val<uint32_t>(0) * 7;
    const auto packed = TensorAccessor(packed_args, get_arg_val<uint32_t>(2), 2048);
#if BATCH_SWIGLU
    cb_reserve_back(0, 7);
    cb_reserve_back(1, 7);
    for (uint32_t tile = 0; tile < 7; ++tile) {
        noc_async_read_page(first_tile + tile, packed, get_write_ptr(0) + tile * 2048);
        noc_async_read_page(112 + first_tile + tile, packed, get_write_ptr(1) + tile * 2048);
    }
    noc_async_read_barrier();
    cb_push_back(0, 7);
    cb_push_back(1, 7);
#else
    for (uint32_t tile = 0; tile < 7; ++tile) {
        cb_reserve_back(0, 1);
        cb_reserve_back(1, 1);
        noc_async_read_page(first_tile + tile, packed, get_write_ptr(0));
        noc_async_read_page(112 + first_tile + tile, packed, get_write_ptr(1));
        noc_async_read_barrier();
        cb_push_back(0, 1);
        cb_push_back(1, 1);
    }
#endif
}
#elif defined(SWIGLU) && defined(WRITER)
void QB2_ENTRY() {
    cb_wait_front(18, 7);
    notify_coordinator(2);
    noc_async_atomic_barrier();
    cb_pop_front(18, 7);
}
#endif

#elif defined(COMPUTE)
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "tools/profiler/kernel_profiler.hpp"

using namespace ckernel;

#if defined(PROJECTION)
#include "api/compute/matmul.h"

#include "projection.hpp"
#if SHARED_QKV
#include "shared_qkv.hpp"
#endif

void QB2_ENTRY() {
#if SHARED_QKV
    shared_qkv_compute();
#endif
#ifdef FUSE_OUTPUT
    if (get_arg_val<uint32_t>(0) < 8) {
#if !SHARED_QKV
        DeviceZoneScopedN("MLP-O-MATH");
#endif
        compute_kernel_hw_startup<SrcOrder::Reverse>(6, 7, 25);
        projection<6, 7, 17, 25, 4, 16, 32, PROJECTION_WIDE ? 8 : 4>();
    }
#endif
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 24);
    {
        DeviceZoneScopedN("MLP-GU-MATH");
        projection<0, 1, 16, 24, 8, gu_width, 128, PROJECTION_WIDE ? 7 : (GU_WORKERS == 16 ? 2 : 4)>();
    }
    if (get_arg_val<uint32_t>(0) >= 8) { return; }
    reconfig_data_format(1, 3, 0, 4);
    {
        DeviceZoneScopedN("MLP-DOWN-MATH");
        projection<4, 3, 17, 25, 7, 16, 112, PROJECTION_WIDE ? 8 : 4>();
    }
}
#elif defined(SWIGLU)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"

void QB2_ENTRY() {
    compute_kernel_hw_startup(0, 18);
    DeviceZoneScopedN("MLP-SWIGLU-MATH");
#if BATCH_SWIGLU
    // Preserve the native BF16 SiLU intermediate before the multiply. Seven
    // outputs plus one input scratch tile fit the eight-tile half-DST bank.
    cb_wait_front(0, 7);
    cb_reserve_back(2, 7);
    copy_init(0);
    tile_regs_acquire();
    for (uint32_t tile = 0; tile < 7; ++tile) { copy_tile(0, tile, tile); }
    silu_tile_init();
    for (uint32_t tile = 0; tile < 7; ++tile) { silu_tile(tile); }
    tile_regs_commit();
    tile_regs_wait();
    pack_block(0, 2, 7);
    tile_regs_release();
    cb_pop_front(0, 7);
    cb_push_back(2, 7);
    cb_wait_front(2, 7);
    cb_wait_front(1, 7);
    cb_reserve_back(18, 7);
    tile_regs_acquire();
    copy_init(2);
    for (uint32_t tile = 0; tile < 7; ++tile) { copy_tile(2, tile, tile); }
    for (uint32_t tile = 0; tile < 7; ++tile) {
        copy_init(1);
        copy_tile(1, tile, 7);
        mul_binary_tile_init();
        mul_binary_tile(tile, 7, tile);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_block(0, 18, 7);
    tile_regs_release();
    cb_pop_front(2, 7);
    cb_pop_front(1, 7);
    cb_push_back(18, 7);
#else
    for (uint32_t tile = 0; tile < 7; ++tile) {
        cb_wait_front(0, 1);
        cb_reserve_back(2, 1);
        copy_init(0);
        tile_regs_acquire();
        copy_tile(0, 0, 0);
        silu_tile_init();
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, 2);
        tile_regs_release();
        cb_pop_front(0, 1);
        cb_push_back(2, 1);
        cb_wait_front(2, 1);
        cb_wait_front(1, 1);
        cb_reserve_back(18, 1);
        tile_regs_acquire();
        copy_init(2);
        copy_tile(2, 0, 0);
        copy_init(1);
        copy_tile(1, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, 18);
        tile_regs_release();
        cb_pop_front(2, 1);
        cb_pop_front(1, 1);
        cb_push_back(18, 1);
    }
#endif
}
#endif
#endif
