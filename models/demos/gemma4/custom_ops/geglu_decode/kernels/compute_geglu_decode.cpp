// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Compute for GegluDecode: per unit the cb_in pair [up, gate] -> out = up * gelu(gate), exact (erf) GELU on the
// SFPU over the fp32 dest copy of the gate tile, the product on dest (binary SFPU), one bf16 pack.
// compile-time args: [cb_in, cb_out]; runtime args: [unit count]
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t CB_IN = get_compile_time_arg_val(0);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(1);

void kernel_main() {
    compute_kernel_hw_startup(CB_IN, CB_OUT);
    const uint32_t count = get_arg_val<uint32_t>(0);
    CircularBuffer cbi(CB_IN), cbo(CB_OUT);
    copy_init(CB_IN);
    for (uint32_t t = 0; t < count; ++t) {
        cbi.wait_front(2);
        tile_regs_acquire();
        copy_tile(CB_IN, 1, 0);  // gate
        copy_tile(CB_IN, 0, 1);  // up
        gelu_tile_init<false>();
        gelu_tile<false>(0);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        cbo.reserve_back(1);
        pack_tile(0, CB_OUT, 0);
        cbo.push_back(1);
        tile_regs_release();
        cbi.pop_front(2);
    }
}
