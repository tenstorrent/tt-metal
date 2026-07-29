// SPDX-License-Identifier: Apache-2.0
//
// Fused SwiGLU compute WITH bias-fold: adds per-expert gate/up bias inside the
// kernel (only for the active experts streamed by the reader), removing the wide
// [1,E,1,2I] bias-add op. Computes per tile:
//     out = silu(clamp(gate + gbias, max=cap)) * (clamp(up + ubias, -lim, lim) + 1)
// gate is alpha-scaled at build (#66); gate bias is alpha-scaled too (weights.py).

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/activations.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/circular_buffer.h"

using namespace ckernel;

constexpr uint32_t n_tiles = get_compile_time_arg_val(0);
constexpr uint32_t cb_gate = get_compile_time_arg_val(1);
constexpr uint32_t cb_up = get_compile_time_arg_val(2);
constexpr uint32_t cb_out = get_compile_time_arg_val(3);
constexpr uint32_t clamp_max_u = get_compile_time_arg_val(4);
constexpr uint32_t up_min_u = get_compile_time_arg_val(5);
constexpr uint32_t up_max_u = get_compile_time_arg_val(6);
constexpr uint32_t cb_gbias = get_compile_time_arg_val(7);
constexpr uint32_t cb_ubias = get_compile_time_arg_val(8);
constexpr uint32_t one_u = 0x3F800000u;
constexpr uint32_t clamp_min_u = 0xF149F2CAu;

void kernel_main() {
    CircularBuffer cbg(cb_gate);
    CircularBuffer cbu(cb_up);
    CircularBuffer cbo(cb_out);
    CircularBuffer cbgb(cb_gbias);
    CircularBuffer cbub(cb_ubias);

    unary_op_init_common(cb_gate, cb_out);

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cbo.reserve_back(1);
        cbg.wait_front(1);
        cbu.wait_front(1);
        cbgb.wait_front(1);
        cbub.wait_front(1);

        tile_regs_acquire();

        // dst0 = gate + gbias  (FPU binary add of two CBs)
        add_tiles_init(cb_gate, cb_gbias);
        add_tiles(cb_gate, cb_gbias, 0, 0, 0);
        clamp_tile_init();
        clamp_tile(0, clamp_min_u, clamp_max_u);  // dst0 = clamp(gate+gbias, max=cap)
        silu_tile_init();
        silu_tile(0);  // dst0 = silu(...)

        // dst1 = up + ubias
        add_tiles_init(cb_up, cb_ubias);
        add_tiles(cb_up, cb_ubias, 0, 0, 1);
        clamp_tile_init();
        clamp_tile(1, up_min_u, up_max_u);  // dst1 = clamp(up+ubias, -lim, lim)
        binop_with_scalar_tile_init();
        add_unary_tile(1, one_u);  // dst1 = (up+ubias) + 1
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);  // dst0 = silu(gate) * (up+1)

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cbg.pop_front(1);
        cbu.pop_front(1);
        cbgb.pop_front(1);
        cbub.pop_front(1);
        cbo.push_back(1);
    }
}
