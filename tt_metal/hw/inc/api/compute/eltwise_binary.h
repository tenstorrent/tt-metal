// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"
#ifdef TRISC_MATH
#include "llk_math_binary_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_api.h"
#include "llk_unpack_A_api.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs the hardware and software initialization shared by every element-wise binary op (add, sub,
 * mul). Configures the unpacker, math (FPU), and packer for the given input/output circular buffers.
 * It must be followed by the op-specific init (binary_tiles_init, or one of add_tiles_init /
 * sub_tiles_init / mul_tiles_init) before the matching *_tiles call.
 *
 * NOTE: This function currently performs full hardware bring-up (unpack/math/pack hardware configure
 * plus pack init) that duplicates the one-time hardware start programmed by compute_kernel_hw_startup -
 * the initialization our programming model expects at the very top of a compute kernel, and which
 * reduce and matmul already rely on. The intended end state is for this function to become
 * functionally equivalent to a per-op binary_tiles_init (software-only init), with the hardware
 * bring-up owned solely by compute_kernel_hw_startup. That convergence is deferred to a follow-up
 * because a large number of existing callers still depend on this function for hardware configuration.
 *
 * Return value: None
 *
 * | Argument       | Description                                                   | Type     | Valid Range                | Required |
 * |----------------|---------------------------------------------------------------|----------|----------------------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31                    | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31                    | True     |
 * | ocb            | The identifier of the circular buffer (CB) containing output  | uint32_t | 0 to 31, defaults to CB 16 | True     |
 */
// clang-format on
ALWI void binary_op_init_common(uint32_t icb0, uint32_t icb1, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
#ifndef ARCH_QUASAR
    state_configure(icb0, icb1, ocb, call_line);

    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(icb0, icb1)));
    UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1)));

    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb0, icb1)));

    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, PackMode::Default>()));
#else
    UNPACK((llk_unpack_hw_configure(icb0, icb1)));
    UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1)));

    MATH((llk_math_pack_sync_init()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb0, icb1)));

    PACK((llk_pack_hw_configure(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init()));
#endif
}

// clang-format off
/**
 * Initializes an element-wise binary operation (add, sub, mul), including the optional dest-reuse
 * variant. This is the software init that pairs with mul_tiles / add_tiles / sub_tiles, or - when a
 * dest-reuse mode is selected - with binary_dest_reuse_tiles. Call binary_op_init_common (or
 * compute_kernel_hw_startup) once before this to perform the hardware configuration.
 *
 * Template parameters:
 * full_init:           if true, both the unpacker and math are initialized; if false, only the math
 *                      init is performed (use when the unpacker is already configured for these CBs).
 * eltwise_binary_type: the binary operation type, values = ELWADD / ELWSUB / ELWMUL.
 * binary_reuse_dest:   dest-reuse mode, values = NONE / DEST_TO_SRCA / DEST_TO_SRCB. NONE performs a
 *                      standard two-operand unpack init. DEST_TO_SRCA / DEST_TO_SRCB configure the
 *                      single-operand unpack path that loads one source register from the DST register
 *                      (see binary_dest_reuse_tiles); this is the path used by
 *                      binary_dest_reuse_tiles_init.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                                       | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------------------------------------------|----------|-------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A                                           | uint32_t | 0 to 31     | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B                                           | uint32_t | 0 to 31     | True     |
 * | acc_to_dest    | If true, operation = A [+,-,x] B + dst_tile_idx of *_tiles, depending on the eltwise_binary_type  | bool     | 0,1         | False    |
 */
// clang-format on
template <
    bool full_init,
    EltwiseBinaryType eltwise_binary_type,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
ALWI void binary_tiles_init(
    uint32_t icb0, uint32_t icb1, bool acc_to_dest = false, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, call_line);

    MATH((llk_math_eltwise_binary_init<eltwise_binary_type, BroadcastType::NONE, MATH_FIDELITY, binary_reuse_dest>(
        icb0, icb1, acc_to_dest)));

    if constexpr (full_init) {
        if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::NONE) {
            UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1, Transpose::None)));
        } else {
#ifndef ARCH_QUASAR
            UNPACK(constexpr bool acc_to_dest_reuse = true);
#else
            UNPACK(constexpr bool acc_to_dest_reuse = false);
#endif
            UNPACK((llk_unpack_A_init<BroadcastType::NONE, acc_to_dest_reuse, binary_reuse_dest>(false, false, icb0)));
        }
    }
}

// clang-format off
/**
 * Short init for element-wise multiply; pairs with mul_tiles. Configures the unpacker and math for the
 * two input CBs. Call binary_op_init_common (or compute_kernel_hw_startup) once before this. For
 * back-compatibility with Quasar this variant always accumulates into DST (acc_to_dest = true); use the
 * three-argument overload for explicit control over accumulation.
 *
 * Return value: None
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31     | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void mul_tiles_init(uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    // acc_to_dest is unused for WH/BH and accumulation is default behaviour.
    // For back compatibility with Quasar, acc_to_dest=true in this API for all ops.
    // More control is provided with 3-arg version of init API.
    binary_tiles_init<true /* full_init */, EltwiseBinaryType::ELWMUL>(icb0, icb1, true /* acc_to_dest */, call_line);
}

// clang-format off
/**
 * Short init for element-wise multiply with explicit accumulation control; pairs with mul_tiles.
 * Configures the unpacker and math for the two input CBs. Call binary_op_init_common (or
 * compute_kernel_hw_startup) once before this.
 *
 * Return value: None
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31     | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31     | True     |
 * | acc_to_dest    | If true, operation = A * B + dst_tile_idx of mul_tiles        | uint32_t | 0,1         | True     |
 */
// clang-format on
ALWI void mul_tiles_init(uint32_t icb0, uint32_t icb1, uint32_t acc_to_dest, uint32_t call_line = __builtin_LINE()) {
    binary_tiles_init<true /* full_init */, EltwiseBinaryType::ELWMUL>(
        icb0, icb1, acc_to_dest /* acc_to_dest */, call_line);
}

// clang-format off
/**
 * Short init for element-wise addition; pairs with add_tiles. Configures the unpacker and math for the
 * two input CBs. Call binary_op_init_common (or compute_kernel_hw_startup) once before this.
 *
 * Return value: None
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31     | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31     | True     |
 * | acc_to_dest    | If true, operation = A + B + dst_tile_idx of add_tiles        | bool     | 0,1         | False    |
 */
// clang-format on
ALWI void add_tiles_init(
    uint32_t icb0, uint32_t icb1, bool acc_to_dest = false, uint32_t call_line = __builtin_LINE()) {
    binary_tiles_init<true /* full_init */, EltwiseBinaryType::ELWADD>(
        icb0, icb1, acc_to_dest /* acc_to_dest */, call_line);
}

// clang-format off
/**
 * Short init for element-wise subtraction; pairs with sub_tiles. Configures the unpacker and math for
 * the two input CBs. Call binary_op_init_common (or compute_kernel_hw_startup) once before this.
 *
 * Return value: None
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31     | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31     | True     |
 * | acc_to_dest    | If true, operation = A - B + dst_tile_idx of sub_tiles        | bool     | 0,1         | False    |
 */
// clang-format on
ALWI void sub_tiles_init(
    uint32_t icb0, uint32_t icb1, bool acc_to_dest = false, uint32_t call_line = __builtin_LINE()) {
    binary_tiles_init<true /* full_init */, EltwiseBinaryType::ELWSUB>(
        icb0, icb1, acc_to_dest /* acc_to_dest */, call_line);
}

// clang-format off
/**
 * Performs element-wise multiplication C=A*B of tiles in two CBs at given
 * indices and writes the result to the DST register at index dst_tile_index.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                              | Type     | Valid Range                                    | Required |
 * |----------------|----------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the circular buffer (CB) containing A  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the circular buffer (CB) containing B  | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of tile A within the first CB                  | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of tile B within the second CB                 | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result C        | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
ALWI void mul_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    // static bool first = true; // TODO(AP): static initializer causes a hang, possibly investigate
    // if (first)
    //  one possible solution is to add a local context in the kernel, pass it around and store init flags in it
    //  this way the compiler should be able to perform loop hoisting optimization
    //  - might need to add __attribute__((pure)) to init calls for this to work
    //  Also pass -fmove-loop-invariants to g++
    // mul_tiles_initf();
    // first = false;

    UNPACK((llk_unpack_AB(icb0, icb1, itile0, itile1)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true /* clear_fp32_dst_acc */)));
}

// clang-format off
/**
 * Performs element-wise addition C=A+B of tiles in two CBs at given indices
 * and writes the result to the DST register at index dst_tile_index. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call
 * is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                              | Type     | Valid Range                                    | Required |
 * |----------------|----------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the circular buffer (CB) containing A  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the circular buffer (CB) containing B  | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of tile A within the first CB                  | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of tile B within the second CB                 | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result C        | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
ALWI void add_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    UNPACK((llk_unpack_AB(icb0, icb1, itile0, itile1)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWADD,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          MathFidelity::LoFi,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true /* clear_fp32_dst_acc */)));
}

// clang-format off
/**
 * Performs element-wise subtraction C=A-B of tiles in two CBs at given indices
 * and writes the result to the DST register at index dst_tile_index. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call
 * is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                              | Type     | Valid Range                                    | Required |
 * |----------------|----------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the circular buffer (CB) containing A  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the circular buffer (CB) containing B  | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of tile A within the first CB                  | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of tile B within the second CB                 | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result C        | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
ALWI void sub_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    UNPACK((llk_unpack_AB(icb0, icb1, itile0, itile1)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWSUB,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          MathFidelity::LoFi,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true /* clear_fp32_dst_acc */)));
}

// clang-format off
/**
 * Init for the dest-reuse element-wise binary path; pairs with binary_dest_reuse_tiles. In this path
 * one source register is loaded from the DST register (selected by binary_reuse_dest) rather than from
 * a second CB, so only a single input CB is configured. Call binary_op_init_common (or
 * compute_kernel_hw_startup) once before this.
 *
 * This is a thin wrapper that forwards to binary_tiles_init<true, eltwise_binary_type,
 * binary_reuse_dest> with both operands set to icb0; it is retained as a backward-compatible alias so
 * existing callers keep compiling. The dest-reuse init logic now lives in binary_tiles_init.
 *
 * Template parameters:
 * eltwise_binary_type: the binary operation type, values = ELWADD / ELWSUB / ELWMUL.
 * binary_reuse_dest:   which source register is loaded from DST, values = NONE / DEST_TO_SRCA / DEST_TO_SRCB.
 *
 * Return value: None
 *
 * | Argument       | Description                                             | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------|----------|-------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A | uint32_t | 0 to 31     | True     |
 */
// clang-format on
template <
    EltwiseBinaryType eltwise_binary_type = EltwiseBinaryType::ELWADD,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
ALWI void binary_dest_reuse_tiles_init(uint32_t icb0, uint32_t call_line = __builtin_LINE()) {
    binary_tiles_init<true /* full_init */, eltwise_binary_type, binary_reuse_dest>(
        icb0, icb0, false /* acc_to_dest */, call_line);
}

// clang-format off
/**
 * Performs element-wise binary operations, such as multiply, add, or sub of tiles.
 * If binary_reuse_dest = EltwiseBinaryReuseDestType::DEST_TO_SRCA, then the tile specified by idst will be loaded from
 * the DST register buffer into SRCA. The binary operation will operate on SRCA & SRCB inputs, and the result will be
 * written back to the DST register buffer specified by idst. Similar to DEST_TO_SRCA, if binary_reuse_dest =
 * EltwiseBinaryReuseDestType::DEST_TO_SRCB, then tile specified by idst will be loaded from the DST into SRCB register
 * buffer.
 *
 * EltwiseBinaryReuseDestType::DEST_TO_SRCA and EltwiseBinaryReuseDestType::DEST_TO_SRCB assume that another operation has
 * populated the dest register, otherwise dest will contain zeroes.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                                               | Type     | Valid Range                                    | Required |
 * |----------------|-----------------------------------------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in_cb_id       | The identifier of the circular buffer (CB) containing A                                                   | uint32_t | 0 to 31                                        | True     |
 * | in_tile_index  | The index of tile A within the first CB                                                                   | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of tile B that will be moved to Src reg, and the index of the tile in DST REG for the result C  | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
template <
    EltwiseBinaryType eltwise_binary_type = EltwiseBinaryType::ELWADD,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
ALWI void binary_dest_reuse_tiles(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
#ifndef ARCH_QUASAR
    UNPACK(constexpr bool acc_to_dest = true);
#else
    UNPACK(constexpr bool acc_to_dest = false);
#endif
    UNPACK((llk_unpack_A<BroadcastType::NONE, acc_to_dest, binary_reuse_dest>(in_cb_id, in_tile_index)));
    MATH((llk_math_eltwise_binary<
          eltwise_binary_type,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          binary_reuse_dest>(in_cb_id, in_cb_id, dst_tile_index, true /* clear_fp32_dst_acc */)));
}

}  // namespace ckernel
