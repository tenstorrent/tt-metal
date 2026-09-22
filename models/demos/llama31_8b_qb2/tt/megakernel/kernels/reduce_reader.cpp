// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#define kernel_main native_reduce_reader_main
#include "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_direct/device/kernels/reduce_scatter_minimal_direct_reader.cpp"
#undef kernel_main
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
#ifdef FUSE_RESIDUAL
    constexpr auto residual_args = TensorAccessorArgs<RESIDUAL_CT_OFFSET>();
    const auto residual = TensorAccessor(residual_args, get_arg_val<uint32_t>(14), 2048);
    cb_reserve_back(2, 16);
    for (uint32_t t = 0; t < 16; ++t) {
        noc_async_read_page(get_arg_val<uint32_t>(5) + t, residual, get_write_ptr(2) + t * 2048);
    }
    noc_async_read_barrier();
    cb_push_back(2, 16);
#endif

    {
        DeviceZoneScopedN("MLP-REDUCE-WAIT-DOWN");
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(5)), 1);
    }
    native_reduce_reader_main();
}
