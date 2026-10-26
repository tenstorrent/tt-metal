// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Exhaustive bit-identity check of a production SFPU kernel against main's single-row form (sfpu_reference.h).
//
// Per tile t of this core's range, the SFPU writes the input bit patterns tile_bits(t) .. +1023 (shifted into the
// top half for a BF16 destination) into DEST tiles 0 and 1, with no unpacker or packer conversion in between. The
// reference kernel runs on tile 0 and the production kernel on tile 1 (each after its own init), and tile 2 gets,
// per element, 0 where the two results are equal bit for bit and otherwise a marker encoding the element's input:
// the BF16-exact float with bits (0x3F81 + offset) << 16. The packer writes tile 2 to the writer, which counts the
// markers. op 0 checks the generator instead: tile 0 is reloaded and compared with the pattern it should hold.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/compute/pack.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "tests/ttnn/unit_tests/operations/eltwise/sfpu_bit_identity/kernels/sfpu_reference.h"
#endif

using namespace ckernel;

constexpr uint32_t op = get_compile_time_arg_val(0);
// 1: run the BF16-destination kernel (is_fp32_dest_acc_en = false) in this FP32 destination, on the BF16 patterns
// (shifted into the top half): it sees exactly the lreg values a BF16 destination would load, subnormals included.
constexpr uint32_t bf16_lreg = get_compile_time_arg_val(1);
constexpr bool kernel_fp32 = DST_ACCUM_MODE && !bf16_lreg;
constexpr uint32_t cb_flags = 16;

enum : uint32_t {
    OP_GENERATOR = 0,
    OP_SIGMOID = 1,     // sigmoid_tile, VectorMode::RC, 8 iterations
    OP_SIGMOID_R2 = 2,  // sigmoid_tile<VectorMode::R, ..., 2> (prepare_chunk_recurrence's beta)
    OP_SILU = 3,        // silu_tile
    OP_EXP = 4,         // exp_tile, VectorMode::RC
    OP_EXP_R2 = 5,      // exp_tile<..., 2>(VectorMode::R) (prepare_chunk_recurrence's anchor)
};

#ifdef TRISC_MATH
namespace ckernel::sfpu {

constexpr uint32_t dst_tile_size_sfpi = 32;

// Element offset of each lane in iteration d of a face: face * 256 + d * 32 + lane (LReg[15] holds 2 * lane).
sfpi_inline sfpi::vInt element_offset(uint32_t face, int d) {
    return (sfpi::as<sfpi::vInt>(sfpi::vConstTileId) >> 1) + static_cast<int>(face * 256 + d * 32);
}

sfpi_inline sfpi::vInt input_bits(uint32_t tile_bits, uint32_t face, int d) {
    sfpi::vInt bits = element_offset(face, d) + sfpi::vInt(tile_bits);
    if constexpr (!kernel_fp32) {
        bits = bits << 16;
    }
    return bits;
}

inline void generate_inputs(uint32_t tile_bits, uint32_t* face) {
#pragma GCC unroll 1
    for (int d = 0; d < 8; d++) {
        // Stored as integers into an FP32 destination: an FP32-format SFPSTORE would flush subnormals, which DEST
        // written by the unpacker or the FPU can hold. A BF16 destination gets the BF16-format store, which flushes
        // them; the bf16_lreg cases cover those inputs.
        sfpi::vInt v = input_bits(tile_bits, *face, d);
        if constexpr (DST_ACCUM_MODE) {
            sfpi::dst_reg[0] = v;
            sfpi::dst_reg[dst_tile_size_sfpi] = v;
        } else {
            sfpi::dst_reg[0] = sfpi::as<sfpi::vFloat>(v);
            sfpi::dst_reg[dst_tile_size_sfpi] = sfpi::as<sfpi::vFloat>(v);
        }
        sfpi::dst_reg++;
    }
    ++*face;
}

// Tile 2 = marker where tiles 0 and 1 differ (op > 0), or where tile 0 differs from its generated pattern (op 0).
inline void compare(uint32_t, uint32_t, uint32_t, uint32_t tile_bits, uint32_t* face) {
#pragma GCC unroll 1
    for (int d = 0; d < 8; d++) {
        sfpi::vInt a = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[0]));
        sfpi::vInt b;
        if constexpr (op == OP_GENERATOR) {
            b = input_bits(tile_bits, *face, d);
        } else {
            b = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[dst_tile_size_sfpi]));
        }
        sfpi::vInt marker = (element_offset(*face, d) + 0x3F81) << 16;
        sfpi::vInt flag = 0;
        v_if((a ^ b) != 0) { flag = marker; }
        v_endif;
        sfpi::dst_reg[2 * dst_tile_size_sfpi] = sfpi::as<sfpi::vFloat>(flag);
        sfpi::dst_reg++;
    }
    ++*face;
}

}  // namespace ckernel::sfpu

template <bool production>
inline void init_op() {
    if constexpr (op == OP_SIGMOID || op == OP_SIGMOID_R2) {
        if constexpr (production) {
            llk_math_eltwise_unary_sfpu_init<SfpuType::sigmoid>(sfpu::sigmoid_init<false>);
        } else {
            llk_math_eltwise_unary_sfpu_init<SfpuType::sigmoid>(sfpu::ref::sigmoid_init);
        }
    } else if constexpr (op == OP_SILU) {
        if constexpr (production) {
            llk_math_eltwise_unary_sfpu_init<SfpuType::silu>(sfpu::silu_init<false>);
        } else {
            llk_math_eltwise_unary_sfpu_init<SfpuType::silu>(sfpu::ref::sigmoid_init);
        }
    } else if constexpr (op == OP_EXP || op == OP_EXP_R2) {
        static_assert(DST_ACCUM_MODE, "the reference exp is the FP32 path");
        // exp_init is main's for both.
        llk_math_eltwise_unary_sfpu_init<SfpuType::exponential>(
            sfpu::exp_init<false, 0x3F800000, true, DST_ACCUM_MODE>);
    }
}

// The production calls are those of the compute API (sigmoid_tile, silu_tile, exp_tile) the KDA kernels make.
template <bool production>
inline void run_op(uint32_t dst) {
    if constexpr (op == OP_SIGMOID) {
        if constexpr (production) {
            _llk_math_eltwise_unary_sfpu_params_(sfpu::calculate_sigmoid<false, kernel_fp32, 8>, dst, VectorMode::RC);
        } else {
            _llk_math_eltwise_unary_sfpu_params_(sfpu::ref::calculate_sigmoid<kernel_fp32, 8>, dst, VectorMode::RC);
        }
    } else if constexpr (op == OP_SIGMOID_R2) {
        if constexpr (production) {
            _llk_math_eltwise_unary_sfpu_params_(sfpu::calculate_sigmoid<false, kernel_fp32, 2>, dst, VectorMode::R);
        } else {
            _llk_math_eltwise_unary_sfpu_params_(sfpu::ref::calculate_sigmoid<kernel_fp32, 2>, dst, VectorMode::R);
        }
    } else if constexpr (op == OP_SILU) {
        if constexpr (production) {
            _llk_math_eltwise_unary_sfpu_params_(sfpu::calculate_silu<kernel_fp32, 8>, dst, VectorMode::RC);
        } else {
            _llk_math_eltwise_unary_sfpu_params_(sfpu::ref::calculate_silu<kernel_fp32, 8>, dst, VectorMode::RC);
        }
    } else if constexpr (op == OP_EXP) {
        if constexpr (production) {
            _llk_math_eltwise_unary_sfpu_params_(
                sfpu::calculate_exponential<false, DST_ACCUM_MODE, false, 8, true>,
                dst,
                VectorMode::RC,
                p_sfpu::kCONST_1_FP16B);
        } else {
            _llk_math_eltwise_unary_sfpu_params_(sfpu::ref::calculate_exponential_fp32<8>, dst, VectorMode::RC);
        }
    } else if constexpr (op == OP_EXP_R2) {
        if constexpr (production) {
            _llk_math_eltwise_unary_sfpu_params_(
                sfpu::calculate_exponential<false, DST_ACCUM_MODE, false, 2, true>,
                dst,
                VectorMode::R,
                p_sfpu::kCONST_1_FP16B);
        } else {
            _llk_math_eltwise_unary_sfpu_params_(sfpu::ref::calculate_exponential_fp32<2>, dst, VectorMode::R);
        }
    }
}
#endif

void kernel_main() {
    const uint32_t first_tile = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    init_sfpu(cb_flags, cb_flags);

    for (uint32_t t = first_tile; t < first_tile + num_tiles; t++) {
        cb_reserve_back(cb_flags, 1);
        tile_regs_acquire();
#ifdef TRISC_MATH
        const uint32_t tile_bits = t * 1024;
        uint32_t face = 0;
        _llk_math_eltwise_unary_sfpu_params_(sfpu::generate_inputs, 0, VectorMode::RC, tile_bits, &face);
        if constexpr (op != OP_GENERATOR) {
            init_op<false>();
            run_op<false>(0);
            init_op<true>();
            run_op<true>(1);
        }
        face = 0;
        _llk_math_eltwise_binary_sfpu_params_(sfpu::compare, 0, 1, 2, VectorMode::RC, tile_bits, &face);
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(2, cb_flags);
        tile_regs_release();
        cb_push_back(cb_flags, 1);
    }
}
