// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"
#ifdef TRISC_MATH
#include "llk_math_binary_api.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_api.h"
#include "llk_unpack_A_api.h"
#endif

namespace ckernel {

// clang-format off
 /**
 * Template for initializing element-wise binary operations.
 * Template parameters:
 * full_init: if true, the full init is performed (unpack+math), otherwise only math init is performed
 * eltwise_binary_type: the binary operation type
 *
 * Function
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31     | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31     | True     |
 * | acc_to_dest    | If true, operation = A [+,-,x] B + dst_tile_idx of *_tiles, depending on the eltwise_binary_type | bool | 0,1  | False |
 */
// clang-format on
template <bool full_init, EltwiseBinaryType eltwise_binary_type>
ALWI void binary_tiles_init(
    uint32_t icb0, uint32_t icb1, bool acc_to_dest = false, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, call_line);

    MATH((llk_math_eltwise_binary_init<eltwise_binary_type, BroadcastType::NONE, MATH_FIDELITY>(
        icb0, icb1, acc_to_dest)));

    if constexpr (full_init) {
        UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1, Transpose::None)));
    }
}

namespace detail {
// Single source of truth for the dest-reuse init path. One source operand is taken from DST, so only
// icb0 is unpacked (into SrcA/SrcB per reuse_dest). This is a single-operand (SrcA-only) reconfigure.
// Preserves the historic divergence: WH/BH accumulate the unpacked operand into DST (acc_to_dest=true
// at the unpacker), Quasar does not. The public {add,sub,mul}_reuse_dest_init wrappers and the
// deprecated binary_dest_reuse_tiles_init shim forward here.
template <EltwiseBinaryType eltwise_binary_type, EltwiseBinaryReuseDestType reuse_dest>
ALWI void binary_reuse_dest_init(uint32_t icb0, uint32_t call_line) {
    state_configure(icb0, call_line);
#ifndef ARCH_QUASAR
    UNPACK(constexpr bool acc_to_dest = true);
#else
    UNPACK(constexpr bool acc_to_dest = false);
#endif
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, acc_to_dest, reuse_dest>(false, false, icb0)));
    MATH((llk_math_eltwise_binary_init<eltwise_binary_type, BroadcastType::NONE, MATH_FIDELITY, reuse_dest>(
        icb0, icb0, false /* acc_to_dest */)));
}
}  // namespace detail

// Per-op inits. The two-operand forms ({add,sub,mul}_init(a, b)) are the standard op; the dest-reuse
// op (one operand taken from DST) has its own explicit name: {add,sub,mul}_reuse_dest_init<reuse_dest>.
// There is deliberately no generic public binary_init (mirrors the matmul.h precedent).

// clang-format off
/**
 * Paired init for two-operand element-wise addition (add_tiles). Configures the unpacker and math
 * pipeline so that SrcA <- icb0 and SrcB <- icb1. Call before add_tiles. The one-time hardware
 * configuration must already have been performed via compute_kernel_hw_startup(icb0, icb1, ocb) at the
 * start of MAIN. For general information on init functions refer to any_init. For the dest-reuse variant
 * (one operand from DST) use add_reuse_dest_init<reuse_dest> below.
 *
 * | Argument    | Description                                              | Type     | Valid Range | Required |
 * |-------------|----------------------------------------------------------|----------|-------------|----------|
 * | icb0        | CB whose tile is unpacked into SrcA (operand A)          | uint32_t | 0 to 31     | True     |
 * | icb1        | CB whose tile is unpacked into SrcB (operand B)          | uint32_t | 0 to 31     | True     |
 * | acc_to_dest | If true, operation = A + B + dst_tile_idx of add_tiles   | bool     | 0,1         | False    |
 */
// clang-format on
ALWI void add_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest = false, uint32_t call_line = __builtin_LINE()) {
    binary_tiles_init<true /* full_init */, EltwiseBinaryType::ELWADD>(icb0, icb1, acc_to_dest, call_line);
}

// clang-format off
/**
 * Paired init for dest-reuse element-wise addition (add_reuse_dest_tiles<reuse_dest>). One addend is
 * taken from the DST register, so only a single CB is unpacked; which source register the DST tile is
 * loaded into is selected by reuse_dest:
 *   - DEST_TO_SRCA: DST -> SrcA, icb -> SrcB   (result = DST + icb)
 *   - DEST_TO_SRCB: DST -> SrcB, icb -> SrcA   (result = icb + DST)
 * Call before add_reuse_dest_tiles. compute_kernel_hw_startup must already have run at the start of
 * MAIN. For general information on init functions refer to any_init.
 *
 * | Param Type | Name       | Description                                                        | Type                       | Valid Range | Required |
 * |------------|------------|--------------------------------------------------------------------|----------------------------|-------------|----------|
 * | Template   | reuse_dest | Which source register the DST operand is loaded into (non-NONE)    | EltwiseBinaryReuseDestType | N/A         | True     |
 * | Function   | icb        | CB whose tile is unpacked into the source register not fed by DST  | uint32_t                   | 0 to 31     | True     |
 */
// clang-format on
template <EltwiseBinaryReuseDestType reuse_dest>
ALWI void add_reuse_dest_init(uint32_t icb, uint32_t call_line = __builtin_LINE()) {
    static_assert(
        reuse_dest != EltwiseBinaryReuseDestType::NONE,
        "reuse_dest must be DEST_TO_SRCA or DEST_TO_SRCB; for the two-operand op call add_init(icb0, icb1).");
    detail::binary_reuse_dest_init<EltwiseBinaryType::ELWADD, reuse_dest>(icb, call_line);
}

// clang-format off
/**
 * Paired init for two-operand element-wise subtraction (sub_tiles). Configures the unpacker and math
 * pipeline so that SrcA <- icb0 and SrcB <- icb1. Call before sub_tiles. The one-time hardware
 * configuration must already have been performed via compute_kernel_hw_startup(icb0, icb1, ocb) at the
 * start of MAIN. For general information on init functions refer to any_init. For the dest-reuse variant
 * (one operand from DST) use sub_reuse_dest_init<reuse_dest> below.
 *
 * | Argument    | Description                                              | Type     | Valid Range | Required |
 * |-------------|----------------------------------------------------------|----------|-------------|----------|
 * | icb0        | CB whose tile is unpacked into SrcA (operand A)          | uint32_t | 0 to 31     | True     |
 * | icb1        | CB whose tile is unpacked into SrcB (operand B)          | uint32_t | 0 to 31     | True     |
 * | acc_to_dest | If true, operation = A - B + dst_tile_idx of sub_tiles   | bool     | 0,1         | False    |
 */
// clang-format on
ALWI void sub_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest = false, uint32_t call_line = __builtin_LINE()) {
    binary_tiles_init<true /* full_init */, EltwiseBinaryType::ELWSUB>(icb0, icb1, acc_to_dest, call_line);
}

// clang-format off
/**
 * Paired init for dest-reuse element-wise subtraction (sub_reuse_dest_tiles<reuse_dest>). One operand
 * is taken from the DST register, so only a single CB is unpacked; which source register the DST tile
 * is loaded into is selected by reuse_dest:
 *   - DEST_TO_SRCA: DST -> SrcA, icb -> SrcB   (result = DST - icb)
 *   - DEST_TO_SRCB: DST -> SrcB, icb -> SrcA   (result = icb - DST)
 * Call before sub_reuse_dest_tiles. compute_kernel_hw_startup must already have run at the start of
 * MAIN. For general information on init functions refer to any_init.
 *
 * | Param Type | Name       | Description                                                        | Type                       | Valid Range | Required |
 * |------------|------------|--------------------------------------------------------------------|----------------------------|-------------|----------|
 * | Template   | reuse_dest | Which source register the DST operand is loaded into (non-NONE)    | EltwiseBinaryReuseDestType | N/A         | True     |
 * | Function   | icb        | CB whose tile is unpacked into the source register not fed by DST  | uint32_t                   | 0 to 31     | True     |
 */
// clang-format on
template <EltwiseBinaryReuseDestType reuse_dest>
ALWI void sub_reuse_dest_init(uint32_t icb, uint32_t call_line = __builtin_LINE()) {
    static_assert(
        reuse_dest != EltwiseBinaryReuseDestType::NONE,
        "reuse_dest must be DEST_TO_SRCA or DEST_TO_SRCB; for the two-operand op call sub_init(icb0, icb1).");
    detail::binary_reuse_dest_init<EltwiseBinaryType::ELWSUB, reuse_dest>(icb, call_line);
}

// clang-format off
/**
 * Paired init for two-operand element-wise multiplication (mul_tiles). Configures the unpacker and math
 * pipeline so that SrcA <- icb0 and SrcB <- icb1. Call before mul_tiles. The one-time hardware
 * configuration must already have been performed via compute_kernel_hw_startup(icb0, icb1, ocb) at the
 * start of MAIN. For general information on init functions refer to any_init. For the dest-reuse variant
 * (one operand from DST) use mul_reuse_dest_init<reuse_dest> below.
 *
 * acc_to_dest defaults to true here for backwards compatibility with Quasar (where it selects
 * accumulate-into-DST); it is unused on WH/BH, where accumulation is the default behaviour. Pass the
 * three-argument form for explicit control.
 *
 * | Argument    | Description                                              | Type     | Valid Range | Required |
 * |-------------|----------------------------------------------------------|----------|-------------|----------|
 * | icb0        | CB whose tile is unpacked into SrcA (operand A)          | uint32_t | 0 to 31     | True     |
 * | icb1        | CB whose tile is unpacked into SrcB (operand B)          | uint32_t | 0 to 31     | True     |
 * | acc_to_dest | If true, operation = A * B + dst_tile_idx of mul_tiles   | bool     | 0,1         | False    |
 */
// clang-format on
ALWI void mul_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest = true, uint32_t call_line = __builtin_LINE()) {
    binary_tiles_init<true /* full_init */, EltwiseBinaryType::ELWMUL>(icb0, icb1, acc_to_dest, call_line);
}

// clang-format off
/**
 * Paired init for dest-reuse element-wise multiplication (mul_reuse_dest_tiles<reuse_dest>). One operand
 * is taken from the DST register, so only a single CB is unpacked; which source register the DST tile
 * is loaded into is selected by reuse_dest:
 *   - DEST_TO_SRCA: DST -> SrcA, icb -> SrcB   (result = DST * icb)
 *   - DEST_TO_SRCB: DST -> SrcB, icb -> SrcA   (result = icb * DST)
 * Call before mul_reuse_dest_tiles. compute_kernel_hw_startup must already have run at the start of
 * MAIN. For general information on init functions refer to any_init.
 *
 * | Param Type | Name       | Description                                                        | Type                       | Valid Range | Required |
 * |------------|------------|--------------------------------------------------------------------|----------------------------|-------------|----------|
 * | Template   | reuse_dest | Which source register the DST operand is loaded into (non-NONE)    | EltwiseBinaryReuseDestType | N/A         | True     |
 * | Function   | icb        | CB whose tile is unpacked into the source register not fed by DST  | uint32_t                   | 0 to 31     | True     |
 */
// clang-format on
template <EltwiseBinaryReuseDestType reuse_dest>
ALWI void mul_reuse_dest_init(uint32_t icb, uint32_t call_line = __builtin_LINE()) {
    static_assert(
        reuse_dest != EltwiseBinaryReuseDestType::NONE,
        "reuse_dest must be DEST_TO_SRCA or DEST_TO_SRCB; for the two-operand op call mul_init(icb0, icb1).");
    detail::binary_reuse_dest_init<EltwiseBinaryType::ELWMUL, reuse_dest>(icb, call_line);
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
 * | icb0           | The identifier of the circular buffer (CB) containing A  | uint32_t | 0 to 31                                        | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B  | uint32_t | 0 to 31                                        | True     |
 * | itile0         | The index of tile A within the first CB                  | uint32_t | Must be less than the size of the CB           | True     |
 * | itile1         | The index of tile B within the second CB                  | uint32_t | Must be less than the size of the CB           | True     |
 * | idst           | The index of the tile in DST REG for the result C        | uint32_t | Must be less than the acquired size of DST REG | True     |
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
 * | icb0           | The identifier of the circular buffer (CB) containing A  | uint32_t | 0 to 31                                        | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B  | uint32_t | 0 to 31                                        | True     |
 * | itile0         | The index of tile A within the first CB                  | uint32_t | Must be less than the size of the CB           | True     |
 * | itile1         | The index of tile B within the second CB                  | uint32_t | Must be less than the size of the CB           | True     |
 * | idst           | The index of the tile in DST REG for the result C        | uint32_t | Must be less than the acquired size of DST REG | True     |
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
 * | icb0           | The identifier of the circular buffer (CB) containing A  | uint32_t | 0 to 31                                        | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B  | uint32_t | 0 to 31                                        | True     |
 * | itile0         | The index of tile A within the first CB                  | uint32_t | Must be less than the size of the CB           | True     |
 * | itile1         | The index of tile B within the second CB                  | uint32_t | Must be less than the size of the CB           | True     |
 * | idst           | The index of the tile in DST REG for the result C        | uint32_t | Must be less than the acquired size of DST REG | True     |
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
 * Performs element-wise multiplication C=A*B on `ntiles` consecutive tile pairs from two CBs, writing each result
 * to a consecutive DST register slot. This is the uniform block entry point for the multiply op: its body is a
 * simple loop over `mul_tiles`, so it inherits `mul_tiles`'s semantics and requires the same initialization
 * (`mul_init`) to have been called first. The DST register buffer must be in acquired state via
 * *acquire_dst* call. This call is blocking and is only available on the compute engine.
 *
 * NOTE: The loop implementation is transitional. In the future this for-loop must be folded into a
 * hardware MOP / REPLAY buffer (as is being done for Quasar) so the whole block issues as a single
 * packed op; the blocking then lives in llk-lib without changing this signature. Tracked under the
 * Compute API Split effort (tt-metal#35739); the per-op push-down lands in tt-metal#47482.
 *
 * Return value: None
 *
 * | Argument        | Description                                              | Type     | Valid Range                                    | Required |
 * |-----------------|----------------------------------------------------------|----------|------------------------------------------------|----------|
 * | icb0            | The identifier of the circular buffer (CB) containing A  | uint32_t | 0 to 31                                        | True     |
 * | icb1            | The identifier of the circular buffer (CB) containing B  | uint32_t | 0 to 31                                        | True     |
 * | start_itile0    | The index of the first tile A within the first CB        | uint32_t | Must be less than the size of the CB           | True     |
 * | start_itile1    | The index of the first tile B within the second CB       | uint32_t | Must be less than the size of the CB           | True     |
 * | start_idst      | The index of the first tile in DST REG for the result C  | uint32_t | Must be less than the acquired size of DST REG | True     |
 * | ntiles          | The number of consecutive tile pairs to multiply         | uint32_t | start_idst + ntiles <= acquired DST REG size   | True     |
 */
// clang-format on
ALWI void mul_block(
    uint32_t icb0, uint32_t icb1, uint32_t start_itile0, uint32_t start_itile1, uint32_t start_idst, uint32_t ntiles) {
    for (uint32_t i = 0; i < ntiles; ++i) {
        mul_tiles(icb0, icb1, start_itile0 + i, start_itile1 + i, start_idst + i);
    }
}

// clang-format off
/**
 * Performs element-wise addition C=A+B on `ntiles` consecutive tile pairs from two CBs, writing each result to a
 * consecutive DST register slot. This is the uniform block entry point for the add op: its body is a simple loop
 * over `add_tiles`, so it inherits `add_tiles`'s semantics and requires the same initialization (`add_init`)
 * to have been called first. The DST register buffer must be in acquired state via *acquire_dst* call. This call
 * is blocking and is only available on the compute engine.
 *
 * NOTE: The loop implementation is transitional. In the future this for-loop must be folded into a
 * hardware MOP / REPLAY buffer (as is being done for Quasar) so the whole block issues as a single
 * packed op; the blocking then lives in llk-lib without changing this signature. Tracked under the
 * Compute API Split effort (tt-metal#35739); the per-op push-down lands in tt-metal#47482.
 *
 * Return value: None
 *
 * | Argument        | Description                                              | Type     | Valid Range                                    | Required |
 * |-----------------|----------------------------------------------------------|----------|------------------------------------------------|----------|
 * | icb0            | The identifier of the circular buffer (CB) containing A  | uint32_t | 0 to 31                                        | True     |
 * | icb1            | The identifier of the circular buffer (CB) containing B  | uint32_t | 0 to 31                                        | True     |
 * | start_itile0    | The index of the first tile A within the first CB        | uint32_t | Must be less than the size of the CB           | True     |
 * | start_itile1    | The index of the first tile B within the second CB       | uint32_t | Must be less than the size of the CB           | True     |
 * | start_idst      | The index of the first tile in DST REG for the result C  | uint32_t | Must be less than the acquired size of DST REG | True     |
 * | ntiles          | The number of consecutive tile pairs to add              | uint32_t | start_idst + ntiles <= acquired DST REG size   | True     |
 */
// clang-format on
ALWI void add_block(
    uint32_t icb0, uint32_t icb1, uint32_t start_itile0, uint32_t start_itile1, uint32_t start_idst, uint32_t ntiles) {
    for (uint32_t i = 0; i < ntiles; ++i) {
        add_tiles(icb0, icb1, start_itile0 + i, start_itile1 + i, start_idst + i);
    }
}

// clang-format off
/**
 * Performs element-wise subtraction C=A-B on `ntiles` consecutive tile pairs from two CBs, writing each result to
 * a consecutive DST register slot. This is the uniform block entry point for the subtract op: its body is a simple
 * loop over `sub_tiles`, so it inherits `sub_tiles`'s semantics and requires the same initialization
 * (`sub_init`) to have been called first. The DST register buffer must be in acquired state via
 * *acquire_dst* call. This call is blocking and is only available on the compute engine.
 *
 * NOTE: The loop implementation is transitional. In the future this for-loop must be folded into a
 * hardware MOP / REPLAY buffer (as is being done for Quasar) so the whole block issues as a single
 * packed op; the blocking then lives in llk-lib without changing this signature. Tracked under the
 * Compute API Split effort (tt-metal#35739); the per-op push-down lands in tt-metal#47482.
 *
 * Return value: None
 *
 * | Argument        | Description                                              | Type     | Valid Range                                    | Required |
 * |-----------------|----------------------------------------------------------|----------|------------------------------------------------|----------|
 * | icb0            | The identifier of the circular buffer (CB) containing A  | uint32_t | 0 to 31                                        | True     |
 * | icb1            | The identifier of the circular buffer (CB) containing B  | uint32_t | 0 to 31                                        | True     |
 * | start_itile0    | The index of the first tile A within the first CB        | uint32_t | Must be less than the size of the CB           | True     |
 * | start_itile1    | The index of the first tile B within the second CB       | uint32_t | Must be less than the size of the CB           | True     |
 * | start_idst      | The index of the first tile in DST REG for the result C  | uint32_t | Must be less than the acquired size of DST REG | True     |
 * | ntiles          | The number of consecutive tile pairs to subtract         | uint32_t | start_idst + ntiles <= acquired DST REG size   | True     |
 */
// clang-format on
ALWI void sub_block(
    uint32_t icb0, uint32_t icb1, uint32_t start_itile0, uint32_t start_itile1, uint32_t start_idst, uint32_t ntiles) {
    for (uint32_t i = 0; i < ntiles; ++i) {
        sub_tiles(icb0, icb1, start_itile0 + i, start_itile1 + i, start_idst + i);
    }
}

namespace detail {
// Single source of truth for the dest-reuse execute. The idst tile is loaded from DST into SrcA
// (DEST_TO_SRCA) or SrcB (DEST_TO_SRCB); the op runs on SrcA & SrcB and writes back to DST[idst].
// The public {add,sub,mul}_reuse_dest_tiles wrappers and the deprecated binary_dest_reuse_tiles shim
// forward here. Assumes a prior op populated DST[idst], else it reads zeroes.
template <EltwiseBinaryType eltwise_binary_type, EltwiseBinaryReuseDestType reuse_dest>
ALWI void binary_reuse_dest_tiles(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
#ifndef ARCH_QUASAR
    UNPACK(constexpr bool acc_to_dest = true);
#else
    UNPACK(constexpr bool acc_to_dest = false);
#endif
    UNPACK((llk_unpack_A<BroadcastType::NONE, acc_to_dest, reuse_dest>(in_cb_id, in_tile_index)));
    MATH((llk_math_eltwise_binary<
          eltwise_binary_type,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          reuse_dest>(in_cb_id, in_cb_id, dst_tile_index, true /* clear_fp32_dst_acc */)));
}
}  // namespace detail

// clang-format off
/**
 * Dest-reuse element-wise add: C = A [op] B where one operand is the tile already in DST[dst_tile_index]
 * and the other is unpacked from in_cb_id. reuse_dest selects which source register the DST tile is
 * loaded into (DEST_TO_SRCA: DST->SrcA, cb->SrcB; DEST_TO_SRCB: DST->SrcB, cb->SrcA). Assumes a prior op
 * populated DST[dst_tile_index], else it reads zeroes. Pair with add_reuse_dest_init<reuse_dest>. The DST
 * register buffer must be in acquired state via *acquire_dst*. Blocking; compute engine only.
 *
 * | Param Type | Name           | Description                                                          | Type                       | Valid Range | Required |
 * |------------|----------------|----------------------------------------------------------------------|----------------------------|-------------|----------|
 * | Template   | reuse_dest     | Which source register the DST operand is loaded into (non-NONE)      | EltwiseBinaryReuseDestType | N/A         | True     |
 * | Function   | in_cb_id       | CB whose tile is unpacked into the source register not fed by DST    | uint32_t                   | 0 to 31     | True     |
 * | Function   | in_tile_index  | Index of the tile within in_cb_id                                    | uint32_t                   | < CB size   | True     |
 * | Function   | dst_tile_index | Index of the DST tile used as the other operand and as the result    | uint32_t                   | < DST size  | True     |
 */
// clang-format on
template <EltwiseBinaryReuseDestType reuse_dest>
ALWI void add_reuse_dest_tiles(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
    detail::binary_reuse_dest_tiles<EltwiseBinaryType::ELWADD, reuse_dest>(in_cb_id, in_tile_index, dst_tile_index);
}

// clang-format off
/** Dest-reuse element-wise subtract. See add_reuse_dest_tiles; pair with sub_reuse_dest_init<reuse_dest>. */
// clang-format on
template <EltwiseBinaryReuseDestType reuse_dest>
ALWI void sub_reuse_dest_tiles(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
    detail::binary_reuse_dest_tiles<EltwiseBinaryType::ELWSUB, reuse_dest>(in_cb_id, in_tile_index, dst_tile_index);
}

// clang-format off
/** Dest-reuse element-wise multiply. See add_reuse_dest_tiles; pair with mul_reuse_dest_init<reuse_dest>. */
// clang-format on
template <EltwiseBinaryReuseDestType reuse_dest>
ALWI void mul_reuse_dest_tiles(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
    detail::binary_reuse_dest_tiles<EltwiseBinaryType::ELWMUL, reuse_dest>(in_cb_id, in_tile_index, dst_tile_index);
}

// =====================================================================================================================
// Deprecated API
//
// The functions below implement the old eltwise-binary programming model. The new model is:
//   compute_kernel_hw_startup(icb0, icb1, ocb);   // once at the start of MAIN
//   add_init(icb0, icb1);   // (or sub_init / mul_init) before add_tiles / sub_tiles / mul_tiles
// The dest-reuse op (one operand from DST) uses the per-op reuse_dest init, e.g.
//   add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(in_cb);
// Generic data-format reconfiguration is done via reconfig_data_format_srca / reconfig_data_format
// (from reconfig_data_format.h).
// =====================================================================================================================

// clang-format off
/**
 * Init function for all binary ops. Performs the one-time hardware configuration of the
 * unpacker/math/packer. Body kept verbatim for backwards compatibility (it also seeds the sentinel
 * reconfig tracker via state_configure).
 *
 * | Argument       | Description                                                   | Type     | Valid Range                | Required |
 * |----------------|---------------------------------------------------------------|----------|----------------------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31                    | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31                    | True     |
 * | ocb            | The identifier of the circular buffer (CB) containing output  | uint32_t | 0 to 31, defaults to CB 16 | True     |
 */
// clang-format on
[[deprecated(
    "Use compute_kernel_hw_startup(icb0, icb1, ocb) once at kernel start, then add_init/sub_init/mul_init(icb0, "
    "icb1). This will be removed after September 15th, 2026.")]] ALWI void
binary_op_init_common(uint32_t icb0, uint32_t icb1, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
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

    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init()));
#endif
}

// clang-format off
/**
 * Short init function for mul_tiles.
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31     | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31     | True     |
 */
// clang-format on
[[deprecated("Renamed to mul_init(). This will be removed after September 15th, 2026.")]] ALWI void mul_tiles_init(
    uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    // acc_to_dest is unused for WH/BH and accumulation is default behaviour.
    // For back compatibility with Quasar, acc_to_dest=true in this API for all ops.
    // More control is provided with 3-arg version of init API.
    mul_init(icb0, icb1, true /* acc_to_dest */, call_line);
}

// clang-format off
/**
 * Short init function for mul_tiles, with explicit acc_to_dest control.
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31     | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31     | True     |
 */
// clang-format on
[[deprecated("Renamed to mul_init(). This will be removed after September 15th, 2026.")]] ALWI void mul_tiles_init(
    uint32_t icb0, uint32_t icb1, uint32_t acc_to_dest, uint32_t call_line = __builtin_LINE()) {
    mul_init(icb0, icb1, acc_to_dest /* acc_to_dest */, call_line);
}

// clang-format off
/**
 * Short init function for add_tiles.
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31     | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31     | True     |
 * | acc_to_dest    | If true, operation = A + B + dst_tile_idx of add_tiles        | bool     | 0,1         | False    |
 */
// clang-format on
[[deprecated("Renamed to add_init(). This will be removed after September 15th, 2026.")]] ALWI void add_tiles_init(
    uint32_t icb0, uint32_t icb1, bool acc_to_dest = false, uint32_t call_line = __builtin_LINE()) {
    add_init(icb0, icb1, acc_to_dest /* acc_to_dest */, call_line);
}

// clang-format off
/**
 * Short init function for sub_tiles.
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31     | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31     | True     |
 * | acc_to_dest    | If true, operation = A - B + dst_tile_idx of sub_tiles        | bool     | 0,1         | False    |
 */
// clang-format on
[[deprecated("Renamed to sub_init(). This will be removed after September 15th, 2026.")]] ALWI void sub_tiles_init(
    uint32_t icb0, uint32_t icb1, bool acc_to_dest = false, uint32_t call_line = __builtin_LINE()) {
    sub_init(icb0, icb1, acc_to_dest /* acc_to_dest */, call_line);
}

// clang-format off
/**
 * Init function for the dest-reuse binary op. Replaced by the per-op
 * {add,sub,mul}_reuse_dest_init<reuse_dest> functions.
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31     | True     |
 */
// clang-format on
template <
    EltwiseBinaryType eltwise_binary_type = EltwiseBinaryType::ELWADD,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
[[deprecated(
    "Renamed to add_reuse_dest_init / sub_reuse_dest_init / mul_reuse_dest_init<reuse_dest>, e.g. "
    "add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(in_cb). This will be removed after September 15th, 2026.")]] ALWI void
binary_dest_reuse_tiles_init(uint32_t icb0, uint32_t call_line = __builtin_LINE()) {
    // Single-operand dest-reuse init path. Kept as a shim so existing callers (and the degenerate
    // binary_reuse_dest == NONE case, e.g. the sentinel test) retain the exact reconfigure behaviour.
    detail::binary_reuse_dest_init<eltwise_binary_type, binary_reuse_dest>(icb0, call_line);
}

// clang-format off
/**
 * Dest-reuse binary execute. Renamed to the per-op {add,sub,mul}_reuse_dest_tiles<reuse_dest>.
 * See the paired init docs for operand/register semantics.
 *
 * | Argument       | Description                                                                                              | Type     | Valid Range | Required |
 * |----------------|----------------------------------------------------------------------------------------------------------|----------|-------------|----------|
 * | in_cb_id       | The identifier of the circular buffer (CB) containing A                                                  | uint32_t | 0 to 31     | True     |
 * | in_tile_index  | The index of tile A within the first CB                                                                  | uint32_t | < CB size   | True     |
 * | dst_tile_index | The index of tile B moved to Src reg, and the index of the DST tile for the result C                     | uint32_t | < DST size  | True     |
 */
// clang-format on
template <
    EltwiseBinaryType eltwise_binary_type = EltwiseBinaryType::ELWADD,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
[[deprecated(
    "Renamed to add_reuse_dest_tiles / sub_reuse_dest_tiles / mul_reuse_dest_tiles<reuse_dest>, e.g. "
    "add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(in_cb, it, idst). This will be removed after September 15th, 2026.")]] ALWI void
binary_dest_reuse_tiles(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
    detail::binary_reuse_dest_tiles<eltwise_binary_type, binary_reuse_dest>(in_cb_id, in_tile_index, dst_tile_index);
}

}  // namespace ckernel
