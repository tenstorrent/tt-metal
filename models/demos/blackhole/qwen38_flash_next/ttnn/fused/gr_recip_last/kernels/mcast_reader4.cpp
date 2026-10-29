// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// mcast_reader2.cpp for the GR front v2 (QWEN38_GR_FRONT_V2=1): the u row arrives PACKED (the valid-row chunks of the
// 40 tiles, [tile][chunk]) in a landing buffer and is unpacked into the u CB's tiles once the senders signalled.  Extra
// compile-time args after the two accessor sets: 0 landing cb, 1 chunk bytes, 2 bf16 face bytes, 3 tiles per sender
// block.  Otherwise gr_read/kernels/mcast_reader.cpp with a SECOND multicast receive: the down worker of
// ttnn/fused/gr_recip_last reserves the u-row CB and the recip CB, optionally generates a zero tile, streams up to two
// TILE tensors (its weight column), then pushes the u row once `senders` producers signalled and the recip tiles once
// `senders2` did. Compile-time args: 0 recv cb, 1 recv tiles, 2 senders, 3 semaphore id, 4 recv2 cb, 5 recv2 tiles, 6
// senders2,
//   7 semaphore2 id, 8 num_streams (0..2), 9-10 stream cbs, 11 zero-tile cb (0xFF none), 12.. two TensorAccessorArgs
//   sets (unused slots repeat a used tensor's).
// Runtime args: per stream 7 (addr, outer, inner, first, inner_stride, outer_stride, batch).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "../../kernels/zones.h"

constexpr uint32_t RECV_CB = get_compile_time_arg_val(0);
constexpr uint32_t RECV_TILES = get_compile_time_arg_val(1);
constexpr uint32_t SENDERS = get_compile_time_arg_val(2);
constexpr uint32_t SEM_ID = get_compile_time_arg_val(3);
constexpr uint32_t RECV2_CB = get_compile_time_arg_val(4);
constexpr uint32_t RECV2_TILES = get_compile_time_arg_val(5);
constexpr uint32_t SENDERS2 = get_compile_time_arg_val(6);
constexpr uint32_t SEM2_ID = get_compile_time_arg_val(7);
constexpr uint32_t NUM_STREAMS = get_compile_time_arg_val(8);
constexpr uint32_t ZERO_CB = get_compile_time_arg_val(11);
constexpr uint32_t ACCESSOR_BASE = 12;
constexpr uint32_t STREAM_RT_ARGS = 7;

template <typename Args>
FORCE_INLINE void read_stream(const Args& args, uint32_t cb, uint32_t rt) {
    const uint32_t addr = get_arg_val<uint32_t>(rt);
    const uint32_t outer = get_arg_val<uint32_t>(rt + 1);
    const uint32_t inner = get_arg_val<uint32_t>(rt + 2);
    const uint32_t first = get_arg_val<uint32_t>(rt + 3);
    const uint32_t inner_stride = get_arg_val<uint32_t>(rt + 4);
    const uint32_t outer_stride = get_arg_val<uint32_t>(rt + 5);
    const uint32_t batch = get_arg_val<uint32_t>(rt + 6);
    const auto accessor = TensorAccessor(args, addr);
    const uint32_t tile_bytes = get_tile_size(cb);
    Noc noc;
    DataflowBuffer dfb(cb);
    uint32_t pending = 0;
    for (uint32_t o = 0; o < outer; ++o) {
        for (uint32_t i = 0; i < inner; ++i) {
            if (pending == 0) {
                dfb.reserve_back(batch);
            }
            noc.async_read(
                accessor,
                dfb,
                tile_bytes,
                {.page_id = first + o * outer_stride + i * inner_stride},
                {.offset_bytes = pending * tile_bytes});
            if (++pending == batch) {
                noc.async_read_barrier();
                dfb.push_back(batch);
                pending = 0;
            }
        }
    }
}

void kernel_main() {
    DataflowBuffer recv(RECV_CB);
    DataflowBuffer recv2(RECV2_CB);
    Semaphore<> sem(SEM_ID);
    Semaphore<> sem2(SEM2_ID);
    recv.reserve_back(RECV_TILES);
    recv2.reserve_back(RECV2_TILES);
    if constexpr (ZERO_CB != 0xFF) {
        dataflow_kernel_lib::prepare_zero_tile<ZERO_CB>();
    }
    {
        FUSED_ZONE("fz_gl_r2_streams");
        constexpr auto args0 = TensorAccessorArgs<ACCESSOR_BASE>();
        constexpr auto args1 = TensorAccessorArgs<args0.next_compile_time_args_offset()>();
        if constexpr (NUM_STREAMS > 0) {
            read_stream(args0, get_compile_time_arg_val(9), 0);
        }
        if constexpr (NUM_STREAMS > 1) {
            read_stream(args1, get_compile_time_arg_val(10), STREAM_RT_ARGS);
        }
    }
    {
        FUSED_ZONE("fz_gl_r2_wait");
        sem.wait(SENDERS);
        constexpr auto a0 = TensorAccessorArgs<ACCESSOR_BASE>();
        constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
        constexpr uint32_t EXTRA = a1.next_compile_time_args_offset();
        constexpr uint32_t CHUNK = get_compile_time_arg_val(EXTRA + 1);
        constexpr uint32_t FACE = get_compile_time_arg_val(EXTRA + 2);
        DataflowBuffer land(get_compile_time_arg_val(EXTRA + 0));
        const uint32_t land_addr = land.get_write_ptr();
        const uint32_t in0_addr = recv.get_write_ptr();
        const uint32_t in0_tile = get_tile_size(RECV_CB);
        for (uint32_t t = 0; t < RECV_TILES; ++t) {
            for (uint32_t c = 0; c < 2; ++c) {
                volatile tt_l1_ptr uint32_t* from =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(land_addr + (t * 2 + c) * CHUNK);
                volatile tt_l1_ptr uint32_t* to =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_addr + t * in0_tile + c * FACE);
                for (uint32_t w = 0; w < CHUNK / 4;
                     w += 8) {  // 8 independent loads then 8 stores: a chunk is a multiple of 32 B
                    const uint32_t a0 = from[w], a1 = from[w + 1], a2 = from[w + 2], a3 = from[w + 3];
                    const uint32_t a4 = from[w + 4], a5 = from[w + 5], a6 = from[w + 6], a7 = from[w + 7];
                    to[w] = a0;
                    to[w + 1] = a1;
                    to[w + 2] = a2;
                    to[w + 3] = a3;
                    to[w + 4] = a4;
                    to[w + 5] = a5;
                    to[w + 6] = a6;
                    to[w + 7] = a7;
                }
            }
        }
        recv.push_back(RECV_TILES);
    }
    {
        FUSED_ZONE("fz_gl_r2_wait2");
        sem2.wait(SENDERS2);
        recv2.push_back(RECV2_TILES);
    }
}
