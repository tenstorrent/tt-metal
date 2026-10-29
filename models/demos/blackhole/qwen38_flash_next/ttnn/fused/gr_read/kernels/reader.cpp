// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Streams whole tiles of up to four TILE tensors into one CB each, plus up to three generated constant tiles.
// Stream s reads pages first + o * outer_stride + i * inner_stride for o < outer, i < inner, pushed `batch` at a time.
// Compile-time args: 0 num_streams, 1-4 cb per stream, 5 num_consts, 6-11 (kind, cb) per const, 12 gated stream
//   (0xFF none), 13 gate semaphore id, 14 gate count (the stream is read once the program-local semaphore reached
//   the count; then it is reset: a producer of the same program signals that the stream's pages are complete),
//   15.. four TensorAccessorArgs sets, chained (unused slots repeat a used tensor's).  Const kinds: 1 reduce scaler
//   (row 0 of every face, the CB's format), 2 broadcast column scalar (column 0, bf16 upper half of the bits), 3 zero
//   tile. Tiles are streamed as they are: a one-row operand of a row-broadcast op is stored row-repeated on the host.
// Runtime args: per stream 7 (addr, outer, inner, first, inner_stride, outer_stride, batch), then per const 1 (bits).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/circular_buffer.h"
#include "api/semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"

constexpr uint32_t num_streams = get_compile_time_arg_val(0);
constexpr uint32_t num_consts = get_compile_time_arg_val(5);
constexpr uint32_t gated_stream = get_compile_time_arg_val(12);
constexpr uint32_t gate_semaphore_id = get_compile_time_arg_val(13);
constexpr uint32_t gate_count = get_compile_time_arg_val(14);
constexpr uint32_t accessor_args_base = 15;
constexpr uint32_t stream_runtime_args = 7;

template <typename Args>
FORCE_INLINE void read_stream(const Args& args, uint32_t cb_id, uint32_t first_arg) {
    const uint32_t addr = get_arg_val<uint32_t>(first_arg);
    const uint32_t outer = get_arg_val<uint32_t>(first_arg + 1);
    const uint32_t inner = get_arg_val<uint32_t>(first_arg + 2);
    const uint32_t first = get_arg_val<uint32_t>(first_arg + 3);
    const uint32_t inner_stride = get_arg_val<uint32_t>(first_arg + 4);
    const uint32_t outer_stride = get_arg_val<uint32_t>(first_arg + 5);
    const uint32_t batch = get_arg_val<uint32_t>(first_arg + 6);
    const auto accessor = TensorAccessor(args, addr);
    const uint32_t tile_bytes = get_tile_size(cb_id);
    const Noc noc;
    DataflowBuffer stream_cb(cb_id);
    uint32_t pending = 0;
    for (uint32_t o = 0; o < outer; ++o) {
        for (uint32_t i = 0; i < inner; ++i) {
            if (pending == 0) {
                stream_cb.reserve_back(batch);
            }
            noc.async_read(
                accessor,
                stream_cb,
                tile_bytes,
                {.page_id = first + o * outer_stride + i * inner_stride},
                {.offset_bytes = pending * tile_bytes});
            if (++pending == batch) {
                noc.async_read_barrier();
                stream_cb.push_back(batch);
                pending = 0;
            }
        }
    }
}

template <uint32_t kind, uint32_t cb_id>
FORCE_INLINE void make_const(uint32_t bits) {
    if constexpr (kind == 1) {
        dataflow_kernel_lib::prepare_reduce_scaler<cb_id, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            __builtin_bit_cast(float, bits));
    } else if constexpr (kind == 2) {
        generate_bcast_col_scalar(CircularBuffer(cb_id), bits);
    } else if constexpr (kind == 3) {
        dataflow_kernel_lib::prepare_zero_tile<cb_id>();
    }
}

// Before the gated stream is read: wait until the program's producer raised the gate semaphore to gate_count, then
// reset it.  Other streams pass straight through.
FORCE_INLINE void wait_for_gate(uint32_t stream) {
    if constexpr (gated_stream != 0xFF) {
        if (stream == gated_stream) {
            Semaphore<> ready(gate_semaphore_id);
            ready.wait(gate_count);
            ready.set(0);
        }
    }
}

void kernel_main() {
    {
        constexpr uint32_t first_const_arg = num_streams * stream_runtime_args;
        if constexpr (num_consts > 0) {
            make_const<get_compile_time_arg_val(6), get_compile_time_arg_val(7)>(
                get_arg_val<uint32_t>(first_const_arg));
        }
        if constexpr (num_consts > 1) {
            make_const<get_compile_time_arg_val(8), get_compile_time_arg_val(9)>(
                get_arg_val<uint32_t>(first_const_arg + 1));
        }
        if constexpr (num_consts > 2) {
            make_const<get_compile_time_arg_val(10), get_compile_time_arg_val(11)>(
                get_arg_val<uint32_t>(first_const_arg + 2));
        }
    }
    // Four accessor arg sets are always present (unused slots repeat a used tensor's): a discarded `if constexpr`
    // branch of a non-template function is still compiled.
    constexpr auto args0 = TensorAccessorArgs<accessor_args_base>();
    constexpr auto args1 = TensorAccessorArgs<args0.next_compile_time_args_offset()>();
    constexpr auto args2 = TensorAccessorArgs<args1.next_compile_time_args_offset()>();
    constexpr auto args3 = TensorAccessorArgs<args2.next_compile_time_args_offset()>();
    {
        wait_for_gate(0);
        read_stream(args0, get_compile_time_arg_val(1), 0);
        if constexpr (num_streams > 1) {
            wait_for_gate(1);
            read_stream(args1, get_compile_time_arg_val(2), stream_runtime_args);
        }
        if constexpr (num_streams > 2) {
            wait_for_gate(2);
            read_stream(args2, get_compile_time_arg_val(3), 2 * stream_runtime_args);
        }
        if constexpr (num_streams > 3) {
            wait_for_gate(3);
            read_stream(args3, get_compile_time_arg_val(4), 3 * stream_runtime_args);
        }
    }
}
