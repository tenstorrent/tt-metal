// SPDX-License-Identifier: Apache-2.0
//
// gpt-oss custom fused SwiGLU compute kernel (decode).
//
// Computes, per tile, the gpt-oss SwiGLU inner math AFTER the gate/up biases
// have been folded/added:
//     out = silu(clamp(gate, max=cap)) * up
// where `gate` is already alpha-scaled at build time (#66 alpha-fold), so
// silu(gate) == alpha*gate_raw*sigmoid(alpha*gate_raw), and `up` already
// includes the (+1) folded into up_proj_bias. Fuses the SwiGLU op chain
//   clamp(gate) -> silu(gate) -> mul(up, glu)
// (3 unary/binary launches) into ONE generic_op compute kernel.
//
// CBs: cb_gate (in0), cb_up (in1), cb_out (out). Streams n_tiles tiles per core.

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

constexpr uint32_t n_tiles = get_compile_time_arg_val(0);  // tiles this core processes
constexpr uint32_t cb_gate = get_compile_time_arg_val(1);
constexpr uint32_t cb_up = get_compile_time_arg_val(2);
constexpr uint32_t cb_out = get_compile_time_arg_val(3);
constexpr uint32_t clamp_max_u = get_compile_time_arg_val(4);  // FP32 bits of gate cap (alpha*swiglu_limit)
constexpr uint32_t up_min_u = get_compile_time_arg_val(5);     // FP32 bits of -swiglu_limit
constexpr uint32_t up_max_u = get_compile_time_arg_val(6);     // FP32 bits of +swiglu_limit
constexpr uint32_t one_u = 0x3F800000u;                        // FP32 bits of 1.0f

// clamp_tile takes std::bit_cast<uint32_t>(float) params (matches ttnn). Gate min
// bound = -1e30f so it only caps the max side. FP32 bits of -1e30 == 0xF149F2CA.
constexpr uint32_t clamp_min_u = 0xF149F2CAu;

void kernel_main() {
    CircularBuffer cbg(cb_gate);
    CircularBuffer cbu(cb_up);
    CircularBuffer cbo(cb_out);

    unary_op_init_common(cb_gate, cb_out);

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cbo.reserve_back(1);
        cbg.wait_front(1);
        cbu.wait_front(1);

        tile_regs_acquire();

        // Re-init each SFPU op immediately before use (matches ttnn logit_kernel):
        // clamp and silu both program SFPU state, so silu must re-init after clamp.
        copy_tile_init(cb_gate);
        copy_tile(cb_gate, 0, 0);  // dst0 = gate (alpha-scaled)
        clamp_tile_init();
        clamp_tile(0, clamp_min_u, clamp_max_u);  // dst0 = clamp(gate, max=cap)
        silu_tile_init();
        silu_tile(0);  // dst0 = silu(clamp(gate,max=cap))
        copy_tile_init(cb_up);
        copy_tile(cb_up, 0, 1);  // dst1 = up
        clamp_tile_init();
        clamp_tile(1, up_min_u, up_max_u);  // dst1 = clamp(up, -limit, +limit)
        binop_with_scalar_tile_init();
        add_unary_tile(1, one_u);  // dst1 = up + 1
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);  // dst0 = silu(gate) * (up+1)

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cbg.pop_front(1);
        cbu.pop_front(1);
        cbo.push_back(1);
    }
}
