// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"
#include "llk_assert.h"
#include "sanitizer/api.h"
#ifdef TRISC_MATH
#include "llk_math_matmul_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_matmul_api.h"
#include "llk_unpack_common_api.h"
#endif
// defines the default throttle level for matmul kernels (default 0)
#ifndef MM_THROTTLE
#define MM_THROTTLE 0
#endif
namespace ckernel {

#ifdef ARCH_BLACKHOLE
// defines the FW-controlled throttle level for block matmul kernels on Blackhole
#define MM_THROTTLE_MAX 5
// 4-byte word at MEM_L1_ARC_FW_SCRATCH written by FW - even means no throttle, odd means throttle
volatile tt_l1_ptr uint32_t* throttle_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MEM_L1_ARC_FW_SCRATCH);
// tracks the state of the currently programmed matmul MOP (0: default throttle level, 1: max throttle level)
static uint32_t throttled_mop_status = 0;

// clang-format off
/**
 * Performs matmul block operation with dynamic throttling.
 * This function is only available on Blackhole architecture and implements
 * firmware-controlled dynamic throttling for block matmul operations.
 * The throttle level is controlled by firmware via MEM_L1_ARC_FW_SCRATCH.
 *
 * Return value: None
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                    | Required |
 * |----------------|-------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)                  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)                 | uint32_t | 0 to 31                                        | True     |
 * | idst           | The index of the tile in DST REG to which the result C will be written. | uint32_t | Must be less than the acquired size of DST REG | True     |
 * | transpose      | The transpose flag for performing transpose operation on tiles in B.    | bool     | Must be true or false                          | True     |
 * | ct_dim         | The column dimension for the output block.                              | uint32_t | Must be equal to block B column dimension      | True     |
 * | rt_dim         | The row dimension for the output block.                                 | uint32_t | Must be equal to block A row dimension         | True     |
 * | kt_dim         | The inner dimension.                                                    | uint32_t | Must be equal to block A column dimension      | True     |
 */
// clang-format on
ALWI void matmul_block_math_dynamic_throttle(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t idst, const uint32_t transpose, uint32_t ct_dim, uint32_t rt_dim) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    // Dynamic throttling is only available on Blackhole architecture
    // Check firmware-controlled throttle enable flag (even = no throttle, odd = throttle)
    volatile uint32_t mm_throttle_en = *(throttle_ptr) % 2;
    if (mm_throttle_en) {
        if (throttled_mop_status != 1) {
            MATH((
                llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE_MAX>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
            throttled_mop_status = 1;
        }
        MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE_MAX>(idst, ct_dim, rt_dim)));
    } else {
        if (throttled_mop_status != 0) {
            MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
            throttled_mop_status = 0;
        }
        MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst, ct_dim, rt_dim)));
    }
#endif
}
#endif

// clang-format off
/**
 * Short init for matmul_tiles. Configures the unpacker and math engine to matmul mode.
 *
 * Must be called before matmul_tiles. The one-time HW configuration must already have been
 * performed via compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out) at the start of MAIN.
 * Matmul maps in0 -> SrcB and in1 -> SrcA (the reverse of other ops), which is why
 * compute_kernel_hw_startup must use SrcOrder::Reverse.
 *
 * NOTE (known gap, #46769): if a preceding op left SrcA/SrcB with asymmetric tile sizes (i.e. different
 * data formats per source) and the following matmul uses the same formats, matmul_init cannot fix the
 * per-source tile sizes on its own. It does not re-program the tile descriptor, and a reconfig_data_format
 * is inappropriate when the data formats did not change. No current kernel hits this; tracked in #46769.
 *
 * Return value: None
 *
 * | Argument       | Description                                                   | Type     | Valid Range                                       | Required |
 * |----------------|---------------------------------------------------------------|----------|---------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)        | uint32_t | 0 to 31                                           | True     |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)       | uint32_t | 0 to 31                                           | True     |
 * | transpose      | The transpose flag for performing transpose operation on B    | uint32_t | Any positive value will indicate transpose is set | False    |
 */
// clang-format on
ALWI void matmul_init(
    uint32_t in0_cb_id, uint32_t in1_cb_id, const uint32_t transpose = 0, uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    state_configure(in1_cb_id, in0_cb_id, call_line);
    MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose)));
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose)));
#else
    LLK_ASSERT(transpose == 0, "non-default transpose not supported on Quasar");
    UNPACK((llk_unpack_AB_matmul_init<false /*transpose*/>(in0_cb_id, in1_cb_id)));
    MATH((llk_math_matmul_init<MATH_FIDELITY>(in0_cb_id, in1_cb_id)));
#endif
}

// clang-format off
/**
 * (Quasar) Undo the automatic MxFp4 -> MxFp4_2x_B src-format selection applied by matmul_init.
 *
 * matmul_init overrides an MxFp4 operand's unpacker OUT_DATA_FORMAT and ALU format to the 2x-packed
 * MxFp4_2x_B, diverging from the op-agnostic unpack_dst_format[] table. That override PERSISTS: the
 * non-matmul unpack inits never reprogram OUT_DATA_FORMAT, and reconfig_data_format is silently
 * skipped for a same-format operand. So a kernel that feeds the SAME MxFp4 buffer to matmul and then
 * to a non-matmul op (datacopy/SFPU/eltwise) MUST call mm_uninit(in0, in1) after the matmuls and
 * before the next op, or that op will keep unpacking the buffer as MxFp4_2x_B and produce garbage.
 * A no-op on non-MxFp4 operands and on non-Quasar architectures.
 *
 * | Argument  | Description                                            | Type     | Valid Range | Required |
 * |-----------|--------------------------------------------------------|----------|-------------|----------|
 * | in0_cb_id | First input CB used in the matmul (same as matmul_init)| uint32_t | 0 to 31     | True     |
 * | in1_cb_id | Second input CB used in the matmul                     | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void mm_uninit(uint32_t in0_cb_id, uint32_t in1_cb_id) {
#ifdef ARCH_QUASAR
    UNPACK((llk_unpack_AB_matmul_uninit(in0_cb_id, in1_cb_id)));
    MATH((llk_math_matmul_uninit(in0_cb_id, in1_cb_id)));
#endif
}

// clang-format off
/**
 * Performs tile-sized matrix multiplication *C=A\*B* between the tiles in two
 * specified input CBs and accumulates the result to DST (DST += C). The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking and
 * is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                    | Required |
 * |----------------|-------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)                  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)                 | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of the tile A from the first input CB                         | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of the tile B from the second input CB                        | uint32_t | Must be less than the size of the CB           | True     |
 * | idst           | The index of the tile in DST REG to which the result C will be written. | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
ALWI void matmul_tiles(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t in0_tile_index, uint32_t in1_tile_index, uint32_t idst) {
    LLK_SAN_FUNCTION();
    UNPACK((llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index)));
#ifndef ARCH_QUASAR
    MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst)));
#else
    MATH((llk_math_matmul_tile(idst)));
#endif
}

// clang-format off
/**
 * Short init for matmul_block. Configures the unpacker and math engine to matmul mode.
 *
 * Must be called before matmul_block. The one-time HW configuration must already have been
 * performed via compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out) at the start of MAIN.
 * Matmul maps in0 -> SrcB and in1 -> SrcA (the reverse of other ops), which is why
 * compute_kernel_hw_startup must use SrcOrder::Reverse.
 *
 * NOTE (known gap, #46769): if a preceding op left SrcA/SrcB with asymmetric tile sizes (i.e. different
 * data formats per source) and the following matmul uses the same formats, matmul_block_init cannot fix
 * the per-source tile sizes on its own. It does not re-program the tile descriptor, and a
 * reconfig_data_format is inappropriate when the data formats did not change. No current kernel hits this;
 * tracked in #46769.
 *
 * Return value: None
 *
 * | Argument  | Description                                                | Type     | Valid Range                                                                                    | Required |
 * |-----------|------------------------------------------------------------|----------|------------------------------------------------------------------------------------------------|----------|
 * | in0_cb_id | The identifier of the first input circular buffer (CB)     | uint32_t | 0 to 31                                                                                        | True     |
 * | in1_cb_id | The identifier of the second input circular buffer (CB)    | uint32_t | 0 to 31                                                                                        | True     |
 * | transpose | The transpose flag for performing transpose operation on B | uint32_t | Any positive value will indicate transpose is set                                              | False    |
 * | ct_dim    | The column dimension for the output block.                 | uint32_t | Must be equal to block B column dimension; 1 to 8 in half-sync mode, 1 to 16 in full-sync mode | False    |
 * | rt_dim    | The row dimension for the output block.                    | uint32_t | Must be equal to block A row dimension; 1 to 8 in half-sync mode, 1 to 16 in full-sync mode    | False    |
 * | kt_dim    | The inner dimension.                                       | uint32_t | Must be equal to block A column dimension                                                      | False    |
 */
// clang-format on
ALWI void matmul_block_init(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    const uint32_t transpose = 0,
    uint32_t ct_dim = 1,
    uint32_t rt_dim = 1,
    uint32_t kt_dim = 1,
    uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    state_configure(in1_cb_id, in0_cb_id, call_line);
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
#ifdef ARCH_BLACKHOLE
    // Dynamic throttling is only available on Blackhole architecture
    MATH((throttled_mop_status = 0));
#endif
#else
    LLK_ASSERT(transpose == 0, "non-default transpose not supported on Quasar");
    UNPACK((llk_unpack_AB_matmul_init<false /*transpose*/>(in0_cb_id, in1_cb_id, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_init<MATH_FIDELITY>(in0_cb_id, in1_cb_id, ct_dim, rt_dim)));
#endif
}

// clang-format off
/**
 * Performs block-sized matrix multiplication *C=A\*B* between the blocks in two
 * different input CBs and accumulates the result to DST (DST += C). The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking and
 * is only available on the compute engine.
 *
 * A block is a rectangle of tiles: A is rt_dim x kt_dim tiles, B is kt_dim x ct_dim tiles, and the
 * output C is rt_dim x ct_dim tiles. So a block is just ct_dim * rt_dim output tiles produced in one
 * call (with kt_dim tiles along the shared inner dimension). The output must fit in DST, so the block
 * size is limited by DST size and sync mode (see matmul_block_init for the valid ct_dim/rt_dim ranges).
 *
 * Return value: None
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                    | Required |
 * |----------------|-------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)                  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)                 | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of the tile in block A from the first input CB                | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of the tile in block B from the second input CB               | uint32_t | Must be less than the size of the CB           | True     |
 * | idst           | The index of the tile in DST REG to which the result C will be written. | uint32_t | Must be less than the acquired size of DST REG | True     |
 * | transpose      | The transpose flag for performing transpose operation on tiles in B.    | bool     | Must be true or false                          | True     |
 * | ct_dim         | The column dimension for the output block.                              | uint32_t | Must be equal to block B column dimension      | True     |
 * | rt_dim         | The row dimension for the output block.                                 | uint32_t | Must be equal to block A row dimension         | True     |
 * | kt_dim         | The inner dimension.                                                    | uint32_t | Must be equal to block A column dimension      | True     |
 */
// clang-format on
ALWI void matmul_block(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t in0_tile_index,
    uint32_t in1_tile_index,
    uint32_t idst,
    const uint32_t transpose,
    uint32_t ct_dim,
    uint32_t rt_dim,
    uint32_t kt_dim,
    uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    state_configure(in1_cb_id, in0_cb_id, call_line);
    UNPACK((llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, ct_dim, rt_dim, kt_dim)));
#ifdef ARCH_BLACKHOLE
    // Dynamic throttling is only available on Blackhole architecture
    MATH((matmul_block_math_dynamic_throttle(in0_cb_id, in1_cb_id, idst, transpose, ct_dim, rt_dim)));
#else
    MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst, ct_dim, rt_dim)));
#endif
#else
    LLK_ASSERT(transpose == 0, "non-default transpose not supported on Quasar");
    LLK_ASSERT(idst == 0, "non-default idst not supported on Quasar");
    UNPACK((llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_block(ct_dim, rt_dim)));
#endif
}

// clang-format off
/**
 * Same as matmul_block_init, plus the SrcA tile stride that matmul_block_in1_at needs.
 *
 * matmul_block_init leaves the unpacker stepping SrcA by cb_in1's circular-buffer page size, which
 * the matmul MOP uses to walk the ct_dim tiles of one call. When that page is a whole K-block --
 * PrefetcherPipe delivery -- the MOP would stride a block per tile, so the true tile size is
 * restated here. Call this instead of matmul_block_init everywhere a block-paged cb_in1 re-enters
 * matmul mode: any data-format reconfig that names cb_in1 re-derives the stride from its page.
 *
 * in1_tile_size is in the units of LocalCBInterface, i.e. 16-byte words on TRISC.
 *
 * Asserts on Quasar; see matmul_block_in1_at.
 *
 * Return value: None
 *
 * | Argument      | Description                                                | Type     | Valid Range                                                                                    | Required |
 * |---------------|------------------------------------------------------------|----------|------------------------------------------------------------------------------------------------|----------|
 * | in0_cb_id     | The identifier of the first input circular buffer (CB)     | uint32_t | 0 to 31                                                                                        | True     |
 * | in1_cb_id     | The identifier of the second input circular buffer (CB)    | uint32_t | 0 to 31                                                                                        | True     |
 * | in1_tile_size | Size of one in1 tile, in LocalCBInterface units            | uint32_t | Non-zero                                                                                       | True     |
 * | transpose     | The transpose flag for performing transpose operation on B | uint32_t | Any positive value will indicate transpose is set                                              | False    |
 * | ct_dim        | The column dimension for the output block.                 | uint32_t | Must be equal to block B column dimension; 1 to 8 in half-sync mode, 1 to 16 in full-sync mode | False    |
 * | rt_dim        | The row dimension for the output block.                    | uint32_t | Must be equal to block A row dimension; 1 to 8 in half-sync mode, 1 to 16 in full-sync mode    | False    |
 * | kt_dim        | The inner dimension.                                       | uint32_t | Must be equal to block A column dimension                                                      | False    |
 */
// clang-format on
ALWI void matmul_block_init_in1_at(
    [[maybe_unused]] uint32_t in0_cb_id,
    [[maybe_unused]] uint32_t in1_cb_id,
    [[maybe_unused]] uint32_t in1_tile_size,
    [[maybe_unused]] const uint32_t transpose = 0,
    [[maybe_unused]] uint32_t ct_dim = 1,
    [[maybe_unused]] uint32_t rt_dim = 1,
    [[maybe_unused]] uint32_t kt_dim = 1,
    [[maybe_unused]] uint32_t call_line = __builtin_LINE()) {
#ifndef ARCH_QUASAR
    matmul_block_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim, call_line);
    UNPACK((llk_unpack_AB_matmul_set_operand_b_tile_size(in1_cb_id, in1_tile_size)));
#else
    LLK_ASSERT(false, "matmul_block_init_in1_at is not supported on Quasar");
#endif
}

// clang-format off
/**
 * Same as matmul_block, except that the in1 block's tiles are addressed from an explicit read
 * pointer and tile size rather than from cb_in1's page stride. Use it when cb_in1 is paged coarser
 * than a tile -- one whole K-block per page, which is what a PrefetcherPipe delivers -- so that
 * `in1_tile_index` still strides by tiles inside the block.
 *
 * in1_read_ptr and in1_tile_size are in the units of LocalCBInterface, i.e. 16-byte words on TRISC:
 * pass get_local_cb_interface(in1_cb_id).fifo_rd_ptr after the block's wait_front, and the tile size
 * in the same units. in1_cb_id is still required, and still names the block's buffer: the data
 * format, face geometry and throttle state all stay circular-buffer derived.
 *
 * Pair it with matmul_block_init_in1_at, which restates the same tile size for the unpacker.
 *
 * Asserts on Quasar, whose unpacker API has no address-taking matmul form.
 *
 * Return value: None
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                    | Required |
 * |----------------|-------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)                  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)                 | uint32_t | 0 to 31                                        | True     |
 * | in1_read_ptr   | Read pointer of the in1 block, in LocalCBInterface units                | uint32_t | Inside the in1 CB                              | True     |
 * | in1_tile_size  | Size of one in1 tile, in LocalCBInterface units                         | uint32_t | Non-zero                                       | True     |
 * | in0_tile_index | The index of the tile in block A from the first input CB                | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of the tile in block B, counted from in1_read_ptr             | uint32_t | Must be less than the block's tile count       | True     |
 * | idst           | The index of the tile in DST REG to which the result C will be written. | uint32_t | Must be less than the acquired size of DST REG | True     |
 * | transpose      | The transpose flag for performing transpose operation on tiles in B.    | bool     | Must be true or false                          | True     |
 * | ct_dim         | The column dimension for the output block.                              | uint32_t | Must be equal to block B column dimension      | True     |
 * | rt_dim         | The row dimension for the output block.                                 | uint32_t | Must be equal to block A row dimension         | True     |
 * | kt_dim         | The inner dimension.                                                    | uint32_t | Must be equal to block A column dimension      | True     |
 */
// clang-format on
ALWI void matmul_block_in1_at(
    [[maybe_unused]] uint32_t in0_cb_id,
    [[maybe_unused]] uint32_t in1_cb_id,
    [[maybe_unused]] uint32_t in1_read_ptr,
    [[maybe_unused]] uint32_t in1_tile_size,
    [[maybe_unused]] uint32_t in0_tile_index,
    [[maybe_unused]] uint32_t in1_tile_index,
    [[maybe_unused]] uint32_t idst,
    [[maybe_unused]] const uint32_t transpose,
    [[maybe_unused]] uint32_t ct_dim,
    [[maybe_unused]] uint32_t rt_dim,
    [[maybe_unused]] uint32_t kt_dim,
    [[maybe_unused]] uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    state_configure(in1_cb_id, in0_cb_id, call_line);
    UNPACK((llk_unpack_AB_matmul_at(
        in0_cb_id, in1_cb_id, in1_read_ptr, in1_tile_size, in0_tile_index, in1_tile_index, ct_dim, rt_dim, kt_dim)));
#ifdef ARCH_BLACKHOLE
    // Dynamic throttling is only available on Blackhole architecture
    MATH((matmul_block_math_dynamic_throttle(in0_cb_id, in1_cb_id, idst, transpose, ct_dim, rt_dim)));
#else
    MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst, ct_dim, rt_dim)));
#endif
#else
    LLK_ASSERT(false, "matmul_block_in1_at is not supported on Quasar");
#endif
}

}  // namespace ckernel
