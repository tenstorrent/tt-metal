// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#define kernel_main native_reduce_reader_main
#include "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_direct/device/kernels/reduce_scatter_minimal_direct_reader.cpp"
#undef kernel_main
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    {
        DeviceZoneScopedN("MLP-REDUCE-WAIT-DOWN");
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(5)), 1);
    }
    native_reduce_reader_main();
}
