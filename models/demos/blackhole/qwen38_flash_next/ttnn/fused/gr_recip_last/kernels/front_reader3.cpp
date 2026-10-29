// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// gr_read/kernels/reader.cpp for the norm core of the GR front v2 (QWEN38_GR_FRONT_V2=1): the same streams and constant
// tiles, but the gated stream (stream 2, the gathered stats) is not read from DRAM pages: once the stats transport
// raised the gate, the core pulls the 8 devices' packed stats row chunks (two chunks of ROWS rows of 32 B per source,
// face 0 and face 1) from the transport core's L1 into the stats CB's tiles (the landing slots of the other devices,
// the staging slot for this device's own chunk).  Nothing else reads those pages, so the DRAM unpack, its write barrier
// and this read go away. Extra compile-time args after the four accessor sets: 0 pack CB (transport cores' raw L1
// buffer, also declared here), 1 slot bytes, 2 packed tile bytes, 3 chunk bytes, 4 RING, 5 face bytes.  Extra runtime
// args after the constants: 0 rank, 1-2 the transport core's NoC x, y, 3 this core's tile slot on it.

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

// the reads of one stream (outer = 1, inner tiles of stride inner_stride) without the barrier or the push: the caller's
template <typename Args>
FORCE_INLINE void issue_stream(const Args& args, uint32_t cb, uint32_t rt) {
    const uint32_t addr = get_arg_val<uint32_t>(rt);
    const uint32_t inner = get_arg_val<uint32_t>(rt + 2);
    const uint32_t first = get_arg_val<uint32_t>(rt + 3);
    const uint32_t inner_stride = get_arg_val<uint32_t>(rt + 4);
    const auto accessor = TensorAccessor(args, addr);
    const uint32_t tile_bytes = get_tile_size(cb);
    Noc noc;
    DataflowBuffer dfb(cb);
    dfb.reserve_back(inner);
    for (uint32_t i = 0; i < inner; ++i) {
        noc.async_read(
            accessor, dfb, tile_bytes, {.page_id = first + i * inner_stride}, {.offset_bytes = i * tile_bytes});
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
        // front v3: the residual and gamma tiles of the whole row are in flight BEFORE the constant tiles are generated
        // (the constants overlap the DRAM latency) and each stream is one batch (one barrier); the same tiles in the
        // same CBs.
        issue_stream(args0, get_compile_time_arg_val(1), 0);
        issue_stream(args1, get_compile_time_arg_val(2), STREAM_RT_ARGS);
        {
            FUSED_ZONE("fz_gr_rd_consts");
            constexpr uint32_t const_rt = NUM_STREAMS * STREAM_RT_ARGS;
            if constexpr (NUM_CONSTS > 0) {
                make_const<get_compile_time_arg_val(6), get_compile_time_arg_val(7)>(get_arg_val<uint32_t>(const_rt));
            }
            if constexpr (NUM_CONSTS > 1) {
                make_const<get_compile_time_arg_val(8), get_compile_time_arg_val(9)>(
                    get_arg_val<uint32_t>(const_rt + 1));
            }
            if constexpr (NUM_CONSTS > 2) {
                make_const<get_compile_time_arg_val(10), get_compile_time_arg_val(11)>(
                    get_arg_val<uint32_t>(const_rt + 2));
            }
        }
        noc_async_read_barrier();
        DataflowBuffer(get_compile_time_arg_val(1)).push_back(get_arg_val<uint32_t>(2));
        DataflowBuffer(get_compile_time_arg_val(2)).push_back(get_arg_val<uint32_t>(STREAM_RT_ARGS + 2));
        if constexpr (NUM_STREAMS > 2) {
            gate(2);
            FUSED_ZONE("fz_gr2_rd_stats");
            constexpr uint32_t EXTRA = args3.next_compile_time_args_offset();
            constexpr uint32_t PACK_CB = get_compile_time_arg_val(EXTRA + 0);
            constexpr uint32_t SLOT = get_compile_time_arg_val(EXTRA + 1);
            constexpr uint32_t PACK_TILE = get_compile_time_arg_val(EXTRA + 2);
            constexpr uint32_t CHUNK = get_compile_time_arg_val(EXTRA + 3);
            constexpr uint32_t RING = get_compile_time_arg_val(EXTRA + 4);
            constexpr uint32_t FACE = get_compile_time_arg_val(EXTRA + 5);
            constexpr uint32_t extra_rt = NUM_STREAMS * STREAM_RT_ARGS + NUM_CONSTS;
            const uint32_t rank = get_arg_val<uint32_t>(extra_rt + 0);
            const uint32_t tx = get_arg_val<uint32_t>(extra_rt + 1);
            const uint32_t ty = get_arg_val<uint32_t>(extra_rt + 2);
            const uint32_t k = get_arg_val<uint32_t>(extra_rt + 3);
            DataflowBuffer stats(get_compile_time_arg_val(3));
            DataflowBuffer pack(PACK_CB);
            const uint32_t tile_bytes = get_tile_size(get_compile_time_arg_val(3));
            const uint32_t pack_base = pack.get_write_ptr();
            stats.reserve_back(RING);
            const uint32_t dst = stats.get_write_ptr();
            for (uint32_t r = 0; r < RING; ++r) {
                // the other devices' chunks landed in slot r; this device's own are still in the (BRISC) staging slot
                const uint32_t slot = (r == rank) ? pack_base + RING * SLOT : pack_base + r * SLOT;
                for (uint32_t c = 0; c < 2; ++c) {
                    noc_async_read(
                        get_noc_addr(tx, ty, slot + k * PACK_TILE + c * CHUNK), dst + r * tile_bytes + c * FACE, CHUNK);
                }
            }
            noc_async_read_barrier();
            stats.push_back(RING);
        }
        if constexpr (NUM_STREAMS > 3) {
            gate(3);
            read_stream(args3, get_compile_time_arg_val(4), 3 * STREAM_RT_ARGS);
        }
    }
}
