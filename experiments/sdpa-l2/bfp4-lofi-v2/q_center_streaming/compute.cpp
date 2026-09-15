// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/compute_kernel_hw_startup.h"
#include "experiments/sdpa-l2/hybrid-mixed-v1/candidate/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#include "add.hpp"
#include "compute_streaming.hpp"
void kernel_main() {
    constexpr uint32_t chunks = get_compile_time_arg_val(0);
    constexpr uint32_t scale = get_compile_time_arg_val(1);
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 16);
    matmul_init(0, 1);
    cb_wait_front(0, 16);
    cb_wait_front(3, 1);
    cb_wait_front(4, 1);
    DeviceZoneScopedN("Q_CENTER_STREAMING");
    sdpa_standard_v2<4, 16, 16 * chunks, 4, 4, scale, 1, 4, 1, 4, false, 0, 1, 2, 6, 3, 14, 4, 5, 16, 15>(
        1, chunks, 8, 9, 10, 11, 12, 13);
}
