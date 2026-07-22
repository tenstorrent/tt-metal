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

// Internal - single source of truth for the dest-reuse init path. One source operand is taken from
// DST, so only icb0 is unpacked (into SrcA/SrcB per binary_reuse_dest). This is a single-operand
// (SrcA-only) reconfigure. Preserves the historic divergence: WH/BH accumulate the unpacked operand
// into DST (acc_to_dest=true at the unpacker), Quasar does not. Used both by the per-op inits (when
// binary_reuse_dest != NONE) and by the deprecated binary_dest_reuse_tiles_init shim.
template <EltwiseBinaryType eltwise_binary_type, EltwiseBinaryReuseDestType binary_reuse_dest>
ALWI void binary_dest_reuse_init(uint32_t icb0, uint32_t call_line) {
    state_configure(icb0, call_line);
#ifndef ARCH_QUASAR
    UNPACK(constexpr bool acc_to_dest = true);
#else
    UNPACK(constexpr bool acc_to_dest = false);
#endif
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, acc_to_dest, binary_reuse_dest>(false, false, icb0)));
    MATH((llk_math_eltwise_binary_init<eltwise_binary_type, BroadcastType::NONE, MATH_FIDELITY, binary_reuse_dest>(
        icb0, icb0, false /* acc_to_dest */)));
}

// Internal implementation shared by the per-op short inits (add_init / sub_init / mul_init).
// It is intentionally not the public surface: there is deliberately no generic public `binary_init`;
// the per-op `*_init` functions are the API the kernels call (mirrors the matmul.h precedent).
//
// binary_reuse_dest == NONE: reconfigures the unpacker + math pipeline for the two-operand op
//   (SrcA <- icb0, SrcB <- icb1) via binary_tiles_init<true, TYPE>.
// binary_reuse_dest != NONE: dest-reuse path via binary_dest_reuse_init; icb1 and acc_to_dest are
//   unused on that path (the second source operand is taken from DST).
template <EltwiseBinaryType eltwise_binary_type, EltwiseBinaryReuseDestType binary_reuse_dest>
ALWI void binary_init_impl(uint32_t icb0, uint32_t icb1, bool acc_to_dest, uint32_t call_line) {
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::NONE) {
        binary_tiles_init<true /* full_init */, eltwise_binary_type>(icb0, icb1, acc_to_dest, call_line);
    } else {
        binary_dest_reuse_init<eltwise_binary_type, binary_reuse_dest>(icb0, call_line);
    }
}

// clang-format off
/**
 * Paired init function for add_tiles / binary_dest_reuse_tiles<ELWADD>. Configures the unpacker and
 * math pipeline for element-wise addition. Call before add_tiles (or the dest-reuse op). The one-time
 * hardware configuration must already have been performed via compute_kernel_hw_startup(icb0, icb1, ocb)
 * at the start of MAIN. For general information on init functions refer to any_init.
 *
 * When binary_reuse_dest != NONE the dest-reuse variant is configured: only icb0 is unpacked and the
 * second source operand is taken from DST (see binary_dest_reuse_tiles); icb1 is unused in that mode.
 *
 * | Param Type | Name            | Description                                                     | Type                       | Valid Range | Required |
 * |------------|-----------------|-----------------------------------------------------------------|----------------------------|-------------|----------|
 * | Template   | binary_reuse_dest | Selects the two-operand (NONE) or dest-reuse init path        | EltwiseBinaryReuseDestType | N/A         | False    |
 * | Function   | icb0            | The identifier of the circular buffer (CB) containing A         | uint32_t                   | 0 to 31     | True     |
 * | Function   | icb1            | The identifier of the circular buffer (CB) containing B         | uint32_t                   | 0 to 31     | True     |
 * | Function   | acc_to_dest     | If true, operation = A + B + dst_tile_idx of add_tiles          | bool                       | 0,1         | False    |
 */
// clang-format on
template <EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
ALWI void add_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest = false, uint32_t call_line = __builtin_LINE()) {
    binary_init_impl<EltwiseBinaryType::ELWADD, binary_reuse_dest>(icb0, icb1, acc_to_dest, call_line);
}

// clang-format off
/**
 * Paired init function for sub_tiles / binary_dest_reuse_tiles<ELWSUB>. Configures the unpacker and
 * math pipeline for element-wise subtraction. Call before sub_tiles (or the dest-reuse op). The one-time
 * hardware configuration must already have been performed via compute_kernel_hw_startup(icb0, icb1, ocb)
 * at the start of MAIN. For general information on init functions refer to any_init.
 *
 * When binary_reuse_dest != NONE the dest-reuse variant is configured: only icb0 is unpacked and the
 * second source operand is taken from DST (see binary_dest_reuse_tiles); icb1 is unused in that mode.
 *
 * | Param Type | Name            | Description                                                     | Type                       | Valid Range | Required |
 * |------------|-----------------|-----------------------------------------------------------------|----------------------------|-------------|----------|
 * | Template   | binary_reuse_dest | Selects the two-operand (NONE) or dest-reuse init path        | EltwiseBinaryReuseDestType | N/A         | False    |
 * | Function   | icb0            | The identifier of the circular buffer (CB) containing A         | uint32_t                   | 0 to 31     | True     |
 * | Function   | icb1            | The identifier of the circular buffer (CB) containing B         | uint32_t                   | 0 to 31     | True     |
 * | Function   | acc_to_dest     | If true, operation = A - B + dst_tile_idx of sub_tiles          | bool                       | 0,1         | False    |
 */
// clang-format on
template <EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
ALWI void sub_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest = false, uint32_t call_line = __builtin_LINE()) {
    binary_init_impl<EltwiseBinaryType::ELWSUB, binary_reuse_dest>(icb0, icb1, acc_to_dest, call_line);
}

// clang-format off
/**
 * Paired init function for mul_tiles / binary_dest_reuse_tiles<ELWMUL>. Configures the unpacker and
 * math pipeline for element-wise multiplication. Call before mul_tiles (or the dest-reuse op). The
 * one-time hardware configuration must already have been performed via compute_kernel_hw_startup(icb0,
 * icb1, ocb) at the start of MAIN. For general information on init functions refer to any_init.
 *
 * acc_to_dest defaults to true here for backwards compatibility with Quasar (where it selects
 * accumulate-into-DST); it is unused on WH/BH, where accumulation is the default behaviour. Pass the
 * three-argument form for explicit control.
 *
 * When binary_reuse_dest != NONE the dest-reuse variant is configured: only icb0 is unpacked and the
 * second source operand is taken from DST (see binary_dest_reuse_tiles); icb1 is unused in that mode.
 *
 * | Param Type | Name            | Description                                                     | Type                       | Valid Range | Required |
 * |------------|-----------------|-----------------------------------------------------------------|----------------------------|-------------|----------|
 * | Template   | binary_reuse_dest | Selects the two-operand (NONE) or dest-reuse init path        | EltwiseBinaryReuseDestType | N/A         | False    |
 * | Function   | icb0            | The identifier of the circular buffer (CB) containing A         | uint32_t                   | 0 to 31     | True     |
 * | Function   | icb1            | The identifier of the circular buffer (CB) containing B         | uint32_t                   | 0 to 31     | True     |
 * | Function   | acc_to_dest     | If true, operation = A * B + dst_tile_idx of mul_tiles          | bool                       | 0,1         | False    |
 */
// clang-format on
template <EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
ALWI void mul_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest = true, uint32_t call_line = __builtin_LINE()) {
    binary_init_impl<EltwiseBinaryType::ELWMUL, binary_reuse_dest>(icb0, icb1, acc_to_dest, call_line);
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

// =====================================================================================================================
// Deprecated API
//
// The functions below implement the old eltwise-binary programming model. The new model is:
//   compute_kernel_hw_startup(icb0, icb1, ocb);   // once at the start of MAIN
//   add_init(icb0, icb1);   // (or sub_init / mul_init) before add_tiles / sub_tiles / mul_tiles
// The dest-reuse init is folded into the per-op inits via the binary_reuse_dest template param, e.g.
//   add_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(in_cb, in_cb);
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
    "icb1). This will be removed after 31-08-2026.")]] ALWI void
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

    PACK((llk_pack_hw_configure(ocb)));
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
[[deprecated("Renamed to mul_init(). This will be removed after 31-08-2026.")]] ALWI void mul_tiles_init(
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
[[deprecated("Renamed to mul_init(). This will be removed after 31-08-2026.")]] ALWI void mul_tiles_init(
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
[[deprecated("Renamed to add_init(). This will be removed after 31-08-2026.")]] ALWI void add_tiles_init(
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
[[deprecated("Renamed to sub_init(). This will be removed after 31-08-2026.")]] ALWI void sub_tiles_init(
    uint32_t icb0, uint32_t icb1, bool acc_to_dest = false, uint32_t call_line = __builtin_LINE()) {
    sub_init(icb0, icb1, acc_to_dest /* acc_to_dest */, call_line);
}

// clang-format off
/**
 * Init function for the dest-reuse binary op. Folded into the per-op inits: use
 * add_init/sub_init/mul_init with the binary_reuse_dest template param instead.
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
    "Use add_init/sub_init/mul_init with the binary_reuse_dest template param, e.g. "
    "add_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(in_cb, in_cb). This will be removed after 31-08-2026.")]] ALWI void
binary_dest_reuse_tiles_init(uint32_t icb0, uint32_t call_line = __builtin_LINE()) {
    // This is the single-operand dest-reuse init path that the per-op *_init<binary_reuse_dest != NONE>
    // functions now fold in. Kept as a shim so existing callers (and the degenerate binary_reuse_dest
    // == NONE case, e.g. the sentinel test) retain the exact single-operand reconfigure behaviour.
    binary_dest_reuse_init<eltwise_binary_type, binary_reuse_dest>(icb0, call_line);
}

}  // namespace ckernel
