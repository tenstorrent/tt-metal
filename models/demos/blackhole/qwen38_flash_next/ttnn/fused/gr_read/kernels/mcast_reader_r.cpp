// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GR gate v3 rows-fast variant (QWEN38_GR_ROWS_FAST=1, rows 2..8) of mcast_reader_t.cpp: the receive CB and stream 0's
// CB hold 8x32 BF16 tiles (512 B, faces of 256 B); stream 0's tile is rows 0..R-1 of face 0 and of face 1 of the DRAM
// page (GT_CHUNK bytes each, face 1 at page byte GT_SRC_FACE), read straight to the CB faces (no scratch, no copy).
// Stream 0's tile is the two valid-row chunks of the DRAM page (face 0 row 0 at page byte 0, face 1 row 0 at page byte
// GT_SRC_FACE): the first 64 B of the page land at the tile (face 0 row 0 + row 1), the 64 B at GT_SRC_FACE land in the
// scratch CB GT_SCRATCH_CB (a DRAM read wants source and destination at the same offset modulo 64), and the RISC copies
// the 32 B of face 1 row 0 into the tile's second half.  Stream 1 (the up weight) streams whole tiles as before.
// Otherwise mcast_reader.cpp. A consumer core's reader: reserves the multicast CB, waits until `senders` producers have
// raised the semaphore (their tiles have landed), pushes the block, optionally generates a zero tile, then streams up
// to two TILE tensors (first + o * outer_stride + i * inner_stride, `batch` at a time) as reader.cpp does. Compile-time
// args: 0 recv cb, 1 recv tiles, 2 senders, 3 semaphore id, 4 num_streams (0..2), 5-6 stream cbs,
//   7 zero-tile cb (0xFF none), 8.. two TensorAccessorArgs sets (unused slots repeat a used tensor's).
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
constexpr uint32_t NUM_STREAMS = get_compile_time_arg_val(4);
constexpr uint32_t ZERO_CB = get_compile_time_arg_val(7);
constexpr uint32_t ACCESSOR_BASE = 8;
constexpr uint32_t STREAM_RT_ARGS = 7;

template <typename Args, bool CHUNKED>
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
    DataflowBuffer dfb(cb);
    uint32_t pending = 0;
    uint32_t base = 0;
    for (uint32_t o = 0; o < outer; ++o) {
        for (uint32_t i = 0; i < inner; ++i) {
            if (pending == 0) {
                dfb.reserve_back(batch);
                base = dfb.get_write_ptr();
            }
            const uint32_t page = first + o * outer_stride + i * inner_stride;
            if constexpr (CHUNKED) {
                noc_async_read(accessor.get_noc_addr(page, 0), base + pending * tile_bytes, GT_CHUNK);
                noc_async_read(
                    accessor.get_noc_addr(page, GT_SRC_FACE), base + pending * tile_bytes + tile_bytes / 2, GT_CHUNK);
            } else {
                noc_async_read(accessor.get_noc_addr(page, 0), base + pending * tile_bytes, tile_bytes);
            }
            if (++pending == batch) {
                noc_async_read_barrier();
                dfb.push_back(batch);
                pending = 0;
            }
        }
    }
}

void kernel_main() {
    DataflowBuffer recv(RECV_CB);
    Semaphore<> sem(SEM_ID);
    recv.reserve_back(RECV_TILES);
    if constexpr (ZERO_CB != 0xFF) {
        dataflow_kernel_lib::prepare_zero_tile<ZERO_CB>();
    }
    // The streams' CBs hold their whole stream, so they prefetch while the producers still compute.
    {
        FUSED_ZONE("fz_gr_mcr_streams");
        constexpr auto args0 = TensorAccessorArgs<ACCESSOR_BASE>();
        constexpr auto args1 = TensorAccessorArgs<args0.next_compile_time_args_offset()>();
        if constexpr (NUM_STREAMS > 0) {
            read_stream<decltype(args0), true>(args0, get_compile_time_arg_val(5), 0);
        }
        if constexpr (NUM_STREAMS > 1) {
            read_stream<decltype(args1), false>(args1, get_compile_time_arg_val(6), STREAM_RT_ARGS);
        }
    }
    {
        FUSED_ZONE("fz_gr_mcr_wait");
        sem.wait(SENDERS);
        recv.push_back(RECV_TILES);
    }
}
