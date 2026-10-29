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
