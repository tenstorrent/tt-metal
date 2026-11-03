// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GR gate v3 rows-fast variant (QWEN38_GR_ROWS_FAST=1, rows 2..8) of reader_t.cpp: the streams' CBs hold 8x32 tiles
// (two 8-row faces), so a stream tile is the two valid-row chunks of the DRAM page (rows 0..R-1 of face 0 and of face
// 1, GT_CHUNK bytes each) at the faces' offsets of the CB tile. GT_CHUNK = bytes of one chunk (a half of the CB tile),
// GT_SRC_FACE = byte offset of face 1 in the DRAM page.  Both reads are 64 B aligned in DRAM and land at the same
// offset modulo 64 in L1.  Otherwise reader.cpp.
//
// Streams whole tiles of up to four TILE tensors into one CB each, plus up to three generated constant tiles.
// Stream s reads pages first + o * outer_stride + i * inner_stride for o < outer, i < inner, pushed `batch` at a time.
// Compile-time args: 0 num_streams, 1-4 cb per stream, 5 num_consts, 6-11 (kind, cb) per const, 12 gated stream
//   (0xFF none), 13 gate semaphore id, 14 gate count (the stream is read once the program-local semaphore reached
//   the count; then it is reset: a producer of the same program signals that the stream's pages are complete),
//   15.. four TensorAccessorArgs sets, chained (unused slots repeat a used tensor's).  Const kinds: 1 reduce scaler
//   (row 0 of every face, the CB's format), 2 broadcast column scalar (column 0, bf16 upper half of the bits), 3 zero
//   tile. Tiles are streamed as they are: a one-row operand of a row-broadcast op is stored row-repeated on the host
//   side (copying tile row 0 into 31 rows on the RISC cost 175 us for the 20 gamma tiles of one read).
// Runtime args: per stream 7 (addr, outer, inner, first, inner_stride, outer_stride, batch), then per const 1 (bits).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "../../kernels/zones.h"

constexpr uint32_t NUM_STREAMS = get_compile_time_arg_val(0);
constexpr uint32_t NUM_CONSTS = get_compile_time_arg_val(5);
constexpr uint32_t GATE_STREAM = get_compile_time_arg_val(12);
constexpr uint32_t GATE_SEM = get_compile_time_arg_val(13);
constexpr uint32_t GATE_COUNT = get_compile_time_arg_val(14);
constexpr uint32_t ACCESSOR_BASE = 15;
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
    const uint32_t tile_bytes = get_tile_size(cb);  // the 8x32 tile
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
            noc_async_read(accessor.get_noc_addr(page, 0), base + pending * tile_bytes, GT_CHUNK);
            noc_async_read(
                accessor.get_noc_addr(page, GT_SRC_FACE), base + pending * tile_bytes + tile_bytes / 2, GT_CHUNK);
            if (++pending == batch) {
                noc_async_read_barrier();
                dfb.push_back(batch);
                pending = 0;
            }
        }
    }
}

template <uint32_t kind, uint32_t cb>
FORCE_INLINE void make_const(uint32_t bits) {
    if constexpr (kind == 1) {
        dataflow_kernel_lib::prepare_reduce_scaler<cb, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            __builtin_bit_cast(float, bits));
    } else if constexpr (kind == 2) {
        generate_bcast_col_scalar(CircularBuffer(cb), bits);
    } else if constexpr (kind == 3) {
        dataflow_kernel_lib::prepare_zero_tile<cb>();
    }
}

void kernel_main() {
    {
        FUSED_ZONE("fz_gr_rd_consts");
        constexpr uint32_t const_rt = NUM_STREAMS * STREAM_RT_ARGS;
        if constexpr (NUM_CONSTS > 0) {
            make_const<get_compile_time_arg_val(6), get_compile_time_arg_val(7)>(get_arg_val<uint32_t>(const_rt));
        }
        if constexpr (NUM_CONSTS > 1) {
            make_const<get_compile_time_arg_val(8), get_compile_time_arg_val(9)>(get_arg_val<uint32_t>(const_rt + 1));
        }
        if constexpr (NUM_CONSTS > 2) {
            make_const<get_compile_time_arg_val(10), get_compile_time_arg_val(11)>(get_arg_val<uint32_t>(const_rt + 2));
        }
    }
    // Four accessor arg sets are always present (unused slots repeat a used tensor's): a discarded `if constexpr`
    // branch of a non-template function is still compiled.
    constexpr auto args0 = TensorAccessorArgs<ACCESSOR_BASE>();
    constexpr auto args1 = TensorAccessorArgs<args0.next_compile_time_args_offset()>();
    constexpr auto args2 = TensorAccessorArgs<args1.next_compile_time_args_offset()>();
    constexpr auto args3 = TensorAccessorArgs<args2.next_compile_time_args_offset()>();
    auto gate = [](uint32_t stream) {
        if constexpr (GATE_STREAM != 0xFF) {
            if (stream == GATE_STREAM) {
                Semaphore<> ready(GATE_SEM);
                ready.wait(GATE_COUNT);
                ready.set(0);
            }
        }
    };
    {
        FUSED_ZONE("fz_gr_rd_streams");
        gate(0);
        read_stream(args0, get_compile_time_arg_val(1), 0);
        if constexpr (NUM_STREAMS > 1) {
            gate(1);
            read_stream(args1, get_compile_time_arg_val(2), STREAM_RT_ARGS);
        }
        if constexpr (NUM_STREAMS > 2) {
            gate(2);
            read_stream(args2, get_compile_time_arg_val(3), 2 * STREAM_RT_ARGS);
        }
        if constexpr (NUM_STREAMS > 3) {
            gate(3);
            read_stream(args3, get_compile_time_arg_val(4), 3 * STREAM_RT_ARGS);
        }
    }
}
