// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
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
 * Init function for all binary ops
 * Followed by the specific init required with an opcode (binrary_op_specific_init)
 *
 * | Argument       | Description                                                   | Type     | Valid Range                | Required |
 * |----------------|---------------------------------------------------------------|----------|----------------------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31                    | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31                    | True     |
 * | ocb            | The identifier of the circular buffer (CB) containing output  | uint32_t | 0 to 31, defaults to CB 16 | True     |
 */
// clang-format on
ALWI void binary_op_init_common(
    std::uint32_t icb0, std::uint32_t icb1, std::uint32_t ocb, std::uint32_t call_line = __builtin_LINE()) {
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
    std::uint32_t icb0, std::uint32_t icb1, bool acc_to_dest = false, std::uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, call_line);

    MATH((llk_math_eltwise_binary_init<eltwise_binary_type, BroadcastType::NONE, MATH_FIDELITY>(
        icb0, icb1, acc_to_dest)));

    if constexpr (full_init) {
        UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1, Transpose::None)));
    }
}

// clang-format off
/**
 * Short init function
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31     | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void mul_tiles_init(std::uint32_t icb0, std::uint32_t icb1, std::uint32_t call_line = __builtin_LINE()) {
    // acc_to_dest is unused for WH/BH and accumulation is default behaviour.
    // For back compatibility with Quasar, acc_to_dest=true in this API for all ops.
    // More control is provided with 3-arg version of init API.
    binary_tiles_init<true /* full_init */, EltwiseBinaryType::ELWMUL>(icb0, icb1, true /* acc_to_dest */, call_line);
}

// clang-format off
/**
 * Short init function
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31     | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void mul_tiles_init(
    std::uint32_t icb0, std::uint32_t icb1, std::uint32_t acc_to_dest, std::uint32_t call_line = __builtin_LINE()) {
    binary_tiles_init<true /* full_init */, EltwiseBinaryType::ELWMUL>(
        icb0, icb1, acc_to_dest /* acc_to_dest */, call_line);
}

// clang-format off
/**
 * Short init function
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31     | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31     | True     |
 * | acc_to_dest    | If true, operation = A + B + dst_tile_idx of add_tiles        | bool     | 0,1         | False    |
 */
// clang-format on
ALWI void add_tiles_init(
    std::uint32_t icb0, std::uint32_t icb1, bool acc_to_dest = false, std::uint32_t call_line = __builtin_LINE()) {
    binary_tiles_init<true /* full_init */, EltwiseBinaryType::ELWADD>(
        icb0, icb1, acc_to_dest /* acc_to_dest */, call_line);
}

// clang-format off
/**
 * Short init function
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31     | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31     | True     |
 * | acc_to_dest    | If true, operation = A - B + dst_tile_idx of sub_tiles        | bool     | 0,1         | False    |
 */
// clang-format on
ALWI void sub_tiles_init(
    std::uint32_t icb0, std::uint32_t icb1, bool acc_to_dest = false, std::uint32_t call_line = __builtin_LINE()) {
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
ALWI void mul_tiles(
    std::uint32_t icb0, std::uint32_t icb1, std::uint32_t itile0, std::uint32_t itile1, std::uint32_t idst) {
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
ALWI void add_tiles(
    std::uint32_t icb0, std::uint32_t icb1, std::uint32_t itile0, std::uint32_t itile1, std::uint32_t idst) {
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
ALWI void sub_tiles(
    std::uint32_t icb0, std::uint32_t icb1, std::uint32_t itile0, std::uint32_t itile1, std::uint32_t idst) {
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
 * (`mul_tiles_init`) to have been called first. The DST register buffer must be in acquired state via
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
    std::uint32_t icb0,
    std::uint32_t icb1,
    std::uint32_t start_itile0,
    std::uint32_t start_itile1,
    std::uint32_t start_idst,
    std::uint32_t ntiles) {
    for (std::uint32_t i = 0; i < ntiles; ++i) {
        mul_tiles(icb0, icb1, start_itile0 + i, start_itile1 + i, start_idst + i);
    }
}

// clang-format off
/**
 * Performs element-wise addition C=A+B on `ntiles` consecutive tile pairs from two CBs, writing each result to a
 * consecutive DST register slot. This is the uniform block entry point for the add op: its body is a simple loop
 * over `add_tiles`, so it inherits `add_tiles`'s semantics and requires the same initialization (`add_tiles_init`)
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
    std::uint32_t icb0,
    std::uint32_t icb1,
    std::uint32_t start_itile0,
    std::uint32_t start_itile1,
    std::uint32_t start_idst,
    std::uint32_t ntiles) {
    for (std::uint32_t i = 0; i < ntiles; ++i) {
        add_tiles(icb0, icb1, start_itile0 + i, start_itile1 + i, start_idst + i);
    }
}

// clang-format off
/**
 * Performs element-wise subtraction C=A-B on `ntiles` consecutive tile pairs from two CBs, writing each result to
 * a consecutive DST register slot. This is the uniform block entry point for the subtract op: its body is a simple
 * loop over `sub_tiles`, so it inherits `sub_tiles`'s semantics and requires the same initialization
 * (`sub_tiles_init`) to have been called first. The DST register buffer must be in acquired state via
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
    std::uint32_t icb0,
    std::uint32_t icb1,
    std::uint32_t start_itile0,
    std::uint32_t start_itile1,
    std::uint32_t start_idst,
    std::uint32_t ntiles) {
    for (std::uint32_t i = 0; i < ntiles; ++i) {
        sub_tiles(icb0, icb1, start_itile0 + i, start_itile1 + i, start_idst + i);
    }
}

namespace detail {

// Compile-time legality fence for the (broadcast x dest-reuse) matrix accepted by
// binary_dest_reuse_tiles_init and binary_dest_reuse_tiles. Shared by both on purpose: the init is
// what programs the unpacker MOP and the FPU broadcast mode, so a broadcast accepted by one and not
// the other is a silent wrong-data bug rather than a build failure. Nothing else links an init to
// its execute, so both must be instantiated with identical template arguments.
template <BroadcastType src_b_bcast_type, EltwiseBinaryReuseDestType binary_reuse_dest>
ALWI void assert_dest_reuse_broadcast_supported() {
    static_assert(
        src_b_bcast_type != BroadcastType::SCALAR,
        "binary_dest_reuse_tiles: BroadcastType::SCALAR cannot be combined with dest reuse. The unpacker "
        "rejects it outright (_llk_unpack_A_mop_config_ asserts !acc_to_dest for SCALAR), and its "
        "acc_to_dest=false MOP raises only 1 of the 4 SrcA dvalids the FPU consumes, so the op would also "
        "hang. Fix: materialize the scalar into a full tile first (mul_tiles_bcast<BroadcastType::SCALAR>), "
        "then run binary_dest_reuse_tiles with BroadcastType::NONE.");

    static_assert(
        !(src_b_bcast_type != BroadcastType::NONE && binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB),
        "binary_dest_reuse_tiles: a broadcast operand is unpacked into SrcB, so SrcB is not available to "
        "receive DST; the unpacker rejects this pairing (_llk_unpack_A_mop_config_). Fix: use "
        "EltwiseBinaryReuseDestType::DEST_TO_SRCA with the broadcast, or keep BroadcastType::NONE if the "
        "op really needs DEST_TO_SRCB.");

    static_assert(
        !(src_b_bcast_type != BroadcastType::NONE && binary_reuse_dest == EltwiseBinaryReuseDestType::NONE),
        "binary_dest_reuse_tiles: broadcast is only supported on the dest-reuse path. With "
        "EltwiseBinaryReuseDestType::NONE nothing loads SrcA from DST, so the FPU would consume whatever "
        "the dvalid-only unpack path leaves in SrcA and produce wrong results. Fix: pass "
        "EltwiseBinaryReuseDestType::DEST_TO_SRCA, or use the add_tiles_bcast / sub_tiles_bcast / "
        "mul_tiles_bcast family for a plain two-CB broadcast.");

#ifndef ARCH_BLACKHOLE
    static_assert(
        src_b_bcast_type == BroadcastType::NONE,
        "binary_dest_reuse_tiles: broadcast combined with dest reuse is implemented on Blackhole only. On "
        "Wormhole the unpacker MOP raises just 1 of the 4 SrcA dvalids the FPU consumes for ROW and "
        "ignores acc_to_dest entirely for COL, so the kernel hangs; on Quasar llk_unpack_A_init "
        "hard-asserts BroadcastType::NONE for dest reuse. Fix: pass BroadcastType::NONE here and do the "
        "broadcast as a separate step (add_tiles_bcast / mul_tiles_bcast), or gate the broadcasting call "
        "on #ifdef ARCH_BLACKHOLE.");
#endif
}

}  // namespace detail

// clang-format off
/**
 * Init for *binary_dest_reuse_tiles*. Programs the unpacker MOP and the FPU (op, broadcast mode, dest-reuse mode and
 * math fidelity) for the *binary_dest_reuse_tiles* calls that follow it.
 *
 * Must be instantiated with the SAME template arguments as those calls. The broadcast mode lives in the unpacker MOP
 * and the FPU config programmed here, not at the execute call site, so an init/execute template mismatch yields
 * silently wrong data rather than a compile error - nothing links the two at compile time.
 *
 * NOTE: any *src_b_bcast_type* other than BroadcastType::NONE is Blackhole-only and requires
 * *binary_reuse_dest* == EltwiseBinaryReuseDestType::DEST_TO_SRCA. Illegal combinations are rejected by static_assert.
 *
 * NOTE: BroadcastType::ROW and BroadcastType::COL additionally require a full 4-face (32x32) tile. The unpacker MOP
 * for those modes is hardwired to 4 faces, so narrower tiles are silently mis-unpacked. This is a runtime property of
 * the CB's tile shape and is therefore not caught by static_assert - the caller must guarantee it.
 *
 * Return value: None
 *
 * | Param Type | Name                | Description                                                                                                        | Type                       | Valid Range                                                                                       | Required |
 * |------------|---------------------|--------------------------------------------------------------------------------------------------------------------|----------------------------|---------------------------------------------------------------------------------------------------|----------|
 * | Template   | eltwise_binary_type | The binary operation the FPU is programmed for                                                                     | EltwiseBinaryType          | ELWADD, ELWSUB, ELWMUL. Defaults to ELWADD                                                        | False    |
 * | Template   | binary_reuse_dest   | Which source register is loaded from DST instead of from a CB                                                      | EltwiseBinaryReuseDestType | NONE, DEST_TO_SRCA, DEST_TO_SRCB. Defaults to NONE                                                | False    |
 * | Template   | src_b_bcast_type    | Broadcast applied to the CB operand as it is unpacked into SrcB                                                     | BroadcastType              | NONE everywhere. ROW/COL only on Blackhole, only with DEST_TO_SRCA, only on 4-face (32x32) tiles. SCALAR is rejected. Defaults to NONE | False    |
 * | Function   | icb0                | The identifier of the circular buffer (CB) containing A                                                            | uint32_t                   | 0 to 31                                                                                           | True     |
 */
// clang-format on
template <
    EltwiseBinaryType eltwise_binary_type = EltwiseBinaryType::ELWADD,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    BroadcastType src_b_bcast_type = BroadcastType::NONE>
ALWI void binary_dest_reuse_tiles_init(std::uint32_t icb0, std::uint32_t call_line = __builtin_LINE()) {
    detail::assert_dest_reuse_broadcast_supported<src_b_bcast_type, binary_reuse_dest>();
    state_configure(icb0, call_line);
    // acc_to_dest drives the unpacker MOP that raises all 4 SrcA dvalids for the dest-reuse and
    // ROW/COL-broadcast paths; Quasar's llk_unpack_A_init requires it to be false.
#ifndef ARCH_QUASAR
    UNPACK(constexpr bool acc_to_dest = true);
#else
    UNPACK(constexpr bool acc_to_dest = false);
#endif
    // Tile shape is runtime CB state, so this cannot be a static_assert. The COL broadcast MOP is
    // already guarded inside the LLK (llk_unpack_A.h, "Unary Broadcast Column requires num_faces == 4"),
    // but the ROW branch hardcodes outerloop/innerloop to 2x2 with no assert, so it silently desyncs
    // from the math thread's total_num_faces() on a partial-face tile. Guard both here.
    if constexpr (src_b_bcast_type != BroadcastType::NONE) {
        UNPACK((LLK_ASSERT(
            get_operand_num_faces(get_operand_id(icb0)) == 4,
            "binary_dest_reuse_tiles: ROW/COL broadcast requires a 4-face (32x32) tile - the unpacker "
            "broadcast MOP is hardwired to 4 faces")));
    }
    UNPACK((llk_unpack_A_init<src_b_bcast_type, acc_to_dest, binary_reuse_dest>(false, false, icb0)));
    MATH((llk_math_eltwise_binary_init<eltwise_binary_type, src_b_bcast_type, MATH_FIDELITY, binary_reuse_dest>(
        icb0, icb0, false /* acc_to_dest */)));
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
 * If src_b_bcast_type is not BroadcastType::NONE, the tile read from in_cb_id is broadcast as it is unpacked into
 * SRCB: BroadcastType::ROW replicates row 0 of the tile down every row, BroadcastType::COL replicates column 0 across
 * every column. This is only available together with EltwiseBinaryReuseDestType::DEST_TO_SRCA, since the broadcast
 * operand must land in SRCB.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Must be preceded by *binary_dest_reuse_tiles_init* instantiated with the SAME template arguments. The broadcast and
 * dest-reuse modes are programmed by that init, so a mismatch produces silently wrong data rather than a compile error.
 *
 * NOTE: any *src_b_bcast_type* other than BroadcastType::NONE is Blackhole-only and requires
 * *binary_reuse_dest* == EltwiseBinaryReuseDestType::DEST_TO_SRCA. Illegal combinations are rejected by static_assert.
 *
 * NOTE: BroadcastType::ROW and BroadcastType::COL additionally require a full 4-face (32x32) tile. The unpacker MOP
 * for those modes is hardwired to 4 faces, so narrower tiles are silently mis-unpacked. This is a runtime property of
 * the CB's tile shape and is therefore not caught by static_assert - the caller must guarantee it.
 *
 * Return value: None
 *
 * | Param Type | Name                | Description                                                                                               | Type                       | Valid Range                                                                                       | Required |
 * |------------|---------------------|-----------------------------------------------------------------------------------------------------------|----------------------------|---------------------------------------------------------------------------------------------------|----------|
 * | Template   | eltwise_binary_type | The binary operation performed by the FPU                                                                 | EltwiseBinaryType          | ELWADD, ELWSUB, ELWMUL. Defaults to ELWADD                                                        | False    |
 * | Template   | binary_reuse_dest   | Which source register is loaded from DST instead of from a CB                                             | EltwiseBinaryReuseDestType | NONE, DEST_TO_SRCA, DEST_TO_SRCB. Defaults to NONE                                                | False    |
 * | Template   | src_b_bcast_type    | Broadcast applied to the CB operand as it is unpacked into SRCB                                           | BroadcastType              | NONE everywhere. ROW/COL only on Blackhole, only with DEST_TO_SRCA, only on 4-face (32x32) tiles. SCALAR is rejected. Defaults to NONE | False    |
 * | Function   | in_cb_id            | The identifier of the circular buffer (CB) containing A                                                   | uint32_t                   | 0 to 31                                                                                           | True     |
 * | Function   | in_tile_index       | The index of tile A within the first CB                                                                   | uint32_t                   | Must be less than the size of the CB                                                              | True     |
 * | Function   | dst_tile_index      | The index of tile B that will be moved to Src reg, and the index of the tile in DST REG for the result C  | uint32_t                   | Must be less than the acquired size of DST REG                                                    | True     |
 */
// clang-format on
template <
    EltwiseBinaryType eltwise_binary_type = EltwiseBinaryType::ELWADD,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    BroadcastType src_b_bcast_type = BroadcastType::NONE>
ALWI void binary_dest_reuse_tiles(std::uint32_t in_cb_id, std::uint32_t in_tile_index, std::uint32_t dst_tile_index) {
    detail::assert_dest_reuse_broadcast_supported<src_b_bcast_type, binary_reuse_dest>();
    // Must match the acc_to_dest derivation in binary_dest_reuse_tiles_init - it selects the same
    // unpacker MOP branch, and the two are only consistent if derived identically.
#ifndef ARCH_QUASAR
    UNPACK(constexpr bool acc_to_dest = true);
#else
    UNPACK(constexpr bool acc_to_dest = false);
#endif
    UNPACK((llk_unpack_A<src_b_bcast_type, acc_to_dest, binary_reuse_dest>(in_cb_id, in_tile_index)));
    MATH((llk_math_eltwise_binary<
          eltwise_binary_type,
          src_b_bcast_type,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          binary_reuse_dest>(in_cb_id, in_cb_id, dst_tile_index, true /* clear_fp32_dst_acc */)));
}

}  // namespace ckernel
