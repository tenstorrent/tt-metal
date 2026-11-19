// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Copy three compact BF16 history rows into one request row of a TILE pool.
// Each core owns128 channels.32-byte writes touch only this request's two
// tile-face rows; no read/modify/write of another request's state is needed.
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(src_args, get_arg_val<uint32_t>(0), 4096);
    const auto dst = TensorAccessor(dst_args, get_arg_val<uint32_t>(1), 2048);
    const uint32_t row = get_arg_val<uint32_t>(2);
    const uint32_t channel = get_arg_val<uint32_t>(3);
    const uint32_t scratch = get_write_ptr(0);
    const uint32_t face_row = (row / 16) * 1024 + (row % 16) * 32;
    for (uint32_t tap = 0; tap < 3; ++tap) {
        noc_async_read(src.get_noc_addr(tap) + channel * 2, scratch, 256);
        noc_async_read_barrier();
        for (uint32_t t = 0; t < 4; ++t) {
            const uint64_t target = dst.get_noc_addr(tap * 64 + channel / 32 + t);
            noc_async_write(scratch + t * 64, target + face_row, 32);
            noc_async_write(scratch + t * 64 + 32, target + face_row + 512, 32);
        }
        noc_async_write_barrier();
    }
}
