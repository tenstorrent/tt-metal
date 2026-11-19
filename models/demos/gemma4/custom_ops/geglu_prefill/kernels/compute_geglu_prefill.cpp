// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Compute for GegluPrefill, the arithmetic of ttnn.mul(gate, up, input_tensor_a_activations=[GELU]) (binary_ng's
// SFPU multiply with an input-a activation) step for step. Per unit of B tiles, on a 16-bit dest:
//   1. gelu(gate): copy each gate tile to dest, exact (erf) GELU on the SFPU, pack bf16 -> cb_mid
//      (binary_ng preprocess_sfpu_impl);
//   2. gelu(gate) * up: copy the B gelu tiles to dest slots 2i and the B up tiles to 2i + 1, SFPU multiply into 2i,
//      pack bf16 -> cb_out (binary_ng process_sfpu_tiles).
// compile-time args: [cb_in, cb_mid, cb_out, B]; runtime args: [unit count]
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t CB_IN = get_compile_time_arg_val(0);
constexpr uint32_t CB_MID = get_compile_time_arg_val(1);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(2);
constexpr uint32_t B = get_compile_time_arg_val(3);

void kernel_main() {
    compute_kernel_hw_startup(CB_MID, CB_OUT);
    const uint32_t count = get_arg_val<uint32_t>(0);
    CircularBuffer cbi(CB_IN), cbm(CB_MID), cbo(CB_OUT);
    for (uint32_t u = 0; u < count; ++u) {
        cbi.wait_front(2 * B);
        // 1. gelu(gate) -> cb_mid
        cbm.reserve_back(B);
        tile_regs_acquire();
        for (uint32_t i = 0; i < B; ++i) {
            copy_init(CB_IN);
            copy_tile(CB_IN, B + i, i);
            gelu_tile_init<false>();
            gelu_tile<false>(i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < B; ++i) {
            pack_tile(i, CB_MID);
        }
        tile_regs_release();
        cbm.push_back(B);
        // 2. gelu(gate) * up -> cb_out
        cbm.wait_front(B);
        cbo.reserve_back(B);
        mul_binary_tile_init();
        tile_regs_acquire();
        copy_init(CB_MID);
        for (uint32_t i = 0; i < B; ++i) {
            copy_tile(CB_MID, i, i * 2);
        }
        copy_init(CB_IN);
        for (uint32_t i = 0; i < B; ++i) {
            copy_tile(CB_IN, i, i * 2 + 1);
            mul_binary_tile(i * 2, i * 2 + 1, i * 2);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < B; ++i) {
            pack_tile(i * 2, CB_OUT);
        }
        tile_regs_release();
        cbo.push_back(B);
        cbm.pop_front(B);
        cbi.pop_front(2 * B);
    }
}
