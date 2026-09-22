// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#ifndef QB2_ENTRY
#define QB2_ENTRY kernel_main
#endif
#define kernel_main native_reduce_reader_main
#include "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_direct/device/kernels/reduce_scatter_minimal_direct_reader.cpp"
#undef kernel_main
#include "tools/profiler/kernel_profiler.hpp"
#include "zero_l1.hpp"
#include "read_alignment.hpp"

void QB2_ENTRY() {
#ifdef FUSE_OUTPUT
    constexpr uint32_t phases = 2;
#else
    constexpr uint32_t phases = 1;
#endif
    for (uint32_t phase = 0; phase < phases; ++phase) {
#ifdef FUSE_OUTPUT
        if (phase == 1) {
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(13)), 1);
        }
#endif
#ifdef FUSE_RESIDUAL
    constexpr auto residual_args = TensorAccessorArgs<RESIDUAL_CT_OFFSET>();
    const auto residual = TensorAccessor(residual_args, get_arg_val<uint32_t>(14 + phase), 2048);
    cb_reserve_back(2, 16);
#ifdef FUSE_EMBEDDING
    if (phase == 0 && get_arg_val<uint32_t>(EMBED_RT_OFFSET + 2)) {
        DeviceZoneScopedN("DECODER-EMBEDDING");
        constexpr auto embedding_args = TensorAccessorArgs<EMBED_CT_OFFSET>();
        constexpr auto token_args = TensorAccessorArgs<embedding_args.next_compile_time_args_offset()>();
        const auto embedding = TensorAccessor(embedding_args, get_arg_val<uint32_t>(EMBED_RT_OFFSET), 2048);
        const auto token = TensorAccessor(token_args, get_arg_val<uint32_t>(EMBED_RT_OFFSET + 1));
        const uint32_t scratch = get_write_ptr(31);
        const uint32_t token_id = read_scalar_u32(token.get_noc_addr(0), scratch);
        const uint64_t embedding_row = embedding.get_noc_addr(token_id) + get_arg_val<uint32_t>(5) * 64;
        const uint32_t row_scratch = aligned_read_destination(scratch + 128, embedding_row);
        noc_async_read(embedding_row, row_scratch, 1024);
        noc_async_read_barrier();
        auto* target = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(2));
        const auto* source = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row_scratch);
        zero_l1<16 * 2048>(get_write_ptr(2));
        for (uint32_t t = 0; t < 16; ++t) {
            for (uint32_t word = 0; word < 8; ++word) {
                target[t * 512 + word] = source[t * 16 + word];
                target[t * 512 + 128 + word] = source[t * 16 + 8 + word];
            }
        }
    } else
#endif
    {
        for (uint32_t t = 0; t < 16; ++t) {
            noc_async_read_page(get_arg_val<uint32_t>(5) + t, residual, get_write_ptr(2) + t * 2048);
        }
        noc_async_read_barrier();
    }
    cb_push_back(2, 16);
#endif

    {
        DeviceZoneScopedN("MLP-REDUCE-WAIT-DOWN");
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(phases == 2 && phase == 0 ? 11 : 5)), 1);
    }
    native_reduce_reader_main();
    }
}
