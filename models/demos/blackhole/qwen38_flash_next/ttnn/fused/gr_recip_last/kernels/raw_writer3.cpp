// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Down worker's writer of the GR front v2 (QWEN38_GR_FRONT_V2=1): the device's partial tile (CB_OUT, fp32, ROWS valid
// rows) goes to the transport core as the two row chunks the packed exchange carries (face 0 and face 1, ROWS rows of
// 64 B each) straight into that core's staging buffer (the scratch CB, one address on every core that declares it) at
// the packed offset of this worker's slot, then the core's scratch semaphore steps; only after that the same two chunks
// go to the device's own page of the gathered partials (nothing waits on that write but the end of the program).  The
// packed bytes are exactly the bytes the default path's transport copies out of the full scratch tile, so the exchange
// is unchanged downstream. Compile-time args: 0 CB_OUT, 1 scratch CB (on the transport core), 2 ROWS (<= 16), 3 scratch
// semaphore id; then the TensorAccessorArgs of the gathered partials.  Runtime args: 0 transport NoC x, 1 y, 2 slot
// (this worker's index on the transport core), 3 own page, 4 gathered partials address.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

void kernel_main() {
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(0);
    constexpr uint32_t SCRATCH_CB = get_compile_time_arg_val(1);
    constexpr uint32_t ROWS = get_compile_time_arg_val(2);
    constexpr uint32_t SEM_SCRATCH = get_compile_time_arg_val(3);
    constexpr auto out_args = TensorAccessorArgs<4>();
    constexpr uint32_t FACE_BYTES = 1024, ROW_BYTES = 64;  // fp32 face and one face row
    constexpr uint32_t CHUNK = ROWS * ROW_BYTES;
    const uint32_t tx = get_arg_val<uint32_t>(0);
    const uint32_t ty = get_arg_val<uint32_t>(1);
    const uint32_t slot = get_arg_val<uint32_t>(2);
    const uint32_t page = get_arg_val<uint32_t>(3);
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(4));
    DataflowBuffer src_cb(CB_OUT);
    DataflowBuffer scratch(SCRATCH_CB);
#ifdef QWEN38_FRONT_UNPACK_SPLIT
    {
        // rows >= 2: the second half of the u row's unpack (the reader, NCRISC, does the first
        // QWEN38_FRONT_UNPACK_SPLIT tiles).  Extra compile-time args after the accessor set: 0 landing cb, 1 u row cb,
        // 2 tiles, 3 u semaphore id, 4 senders, 5 chunk bytes, 6 bf16 face bytes
        constexpr uint32_t X = out_args.next_compile_time_args_offset();
        constexpr uint32_t UNPACK_TILES = get_compile_time_arg_val(X + 2);
        constexpr uint32_t UCHUNK = get_compile_time_arg_val(X + 5);
        constexpr uint32_t UFACE = get_compile_time_arg_val(X + 6);
        DataflowBuffer land(get_compile_time_arg_val(X + 0));
        DataflowBuffer in0(get_compile_time_arg_val(X + 1));
        Semaphore<> usem(get_compile_time_arg_val(X + 3));
        Semaphore<> unpacked(QWEN38_FRONT_UNPACK_SEM);
        usem.wait(get_compile_time_arg_val(X + 4));
        const uint32_t land_addr = land.get_write_ptr();
        const uint32_t in0_addr = in0.get_write_ptr();
        const uint32_t in0_tile = get_tile_size(get_compile_time_arg_val(X + 1));
        for (uint32_t t = QWEN38_FRONT_UNPACK_SPLIT; t < UNPACK_TILES; ++t) {
            for (uint32_t c = 0; c < 2; ++c) {
                volatile tt_l1_ptr uint32_t* from =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(land_addr + (t * 2 + c) * UCHUNK);
                volatile tt_l1_ptr uint32_t* to =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_addr + t * in0_tile + c * UFACE);
                for (uint32_t w = 0; w < UCHUNK / 4; w += 8) {
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
        unpacked.up(1);
    }
#endif
    FUSED_ZONE("fz_gr2_w_partial");
    src_cb.wait_front(1);
    const uint32_t src = src_cb.get_read_ptr();
    const uint32_t dst = scratch.get_write_ptr() + slot * 2 * CHUNK;
    noc_async_write(src, get_noc_addr(tx, ty, dst), CHUNK);
    noc_async_write(src + FACE_BYTES, get_noc_addr(tx, ty, dst + CHUNK), CHUNK);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(tx, ty, get_semaphore(SEM_SCRATCH)), 1);
    noc_async_write(src, out.get_noc_addr(page, 0), CHUNK);
    noc_async_write(src + FACE_BYTES, out.get_noc_addr(page, FACE_BYTES), CHUNK);
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    src_cb.pop_front(1);
}
