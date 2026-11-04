// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common.h"
// Blackhole-only, like the LLKs below: the layout enum the templates take on every thread.
#if defined(ARCH_BLACKHOLE)
#include "experimental/llk_custom_mm_common.h"
#endif
// Blackhole-only: the compressed_custom_mm LLKs live only in the Blackhole llk_api / llk_lib trees.
#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_math_compressed_custom_mm_api.h"
#endif
#if defined(TRISC_UNPACK) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_unpack_AB_compressed_custom_mm_api.h"
#endif
#if defined(TRISC_PACK) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_pack_custom_mm.h"
#endif
namespace ckernel {

#if defined(ARCH_BLACKHOLE)

// clang-format off
/**
 * Full initialization for compressed_custom_mm_block operation. Must be called before compressed_custom_mm_block and only once at the beginning of the kernel.
 * For initializing compressed_custom_mm_block in the middle of the kernel, please use compressed_custom_mm_block_init_short.
 *
 * Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
 * in0 tile shape: [{1, 2, 4, 8}, 32]
 * in1 tile shape: [32, 32]
 * rt_dim: 1
 * ct_dim: any integer from 1 to 64, as long as the output tiles fit in DEST (see CustomMmDestLayout)
 * kt_dim: even number from 2 to 256 (inclusive)
 * fidelity: LoFi only
 * throttle: not supported
 *
 * Return value: None
 *
 * | Argument         | Description                                                                               | Type               | Valid Range                                             | Required                |
 * |------------------|-------------------------------------------------------------------------------------------|--------------------|---------------------------------------------------------|-------------------------|
 * | transpose        | The transpose flag for performing transpose operation on in1                              | bool               | true/false                                              | False (default false)   |
 * | layout           | Where the output tiles sit in DEST, see CustomMmDestLayout; execute and uninit must match | CustomMmDestLayout | Default, Dense, Compact                                 | False (default Default) |
 * | fp32_dest_acc_en | Whether dest accumulates in fp32; defaults to the kernel's DST_ACCUM_MODE                 | bool               | true/false                                              | False                   |
 * | clear_src        | Whether to clear SrcB at init (saves power as only 1/8 FPU rows are used)                 | bool               | true/false                                              | False (default true)    |
 * | in0_cb_id        | The identifier of the first input circular buffer (CB)                                    | uint32_t           | 0 to 31                                                 | True                    |
 * | in1_cb_id        | The identifier of the second input circular buffer (CB)                                   | uint32_t           | 0 to 31                                                 | True                    |
 * | out_cb_id        | The identifier of the output circular buffer (CB)                                         | uint32_t           | 0 to 31                                                 | True                    |
 */
// clang-format on
template <
    bool transpose = false,
    CustomMmDestLayout layout = CustomMmDestLayout::Default,
    bool fp32_dest_acc_en = DST_ACCUM_MODE,
    bool clear_src = true>
ALWI void compressed_custom_mm_block_init(
    const std::uint32_t in0_cb_id, const std::uint32_t in1_cb_id, const std::uint32_t out_cb_id) {
    // Intentionally swap in0 and in1 as operation specific hw_configures are deprecated
    UNPACK((llk_unpack_hw_configure<fp32_dest_acc_en>(in1_cb_id, in0_cb_id)));
    UNPACK((llk_unpack_AB_compressed_custom_mm_init<transpose, clear_src>(in0_cb_id, in1_cb_id)));

    MATH((llk_math_pack_sync_init<fp32_dest_acc_en>()));
    MATH((llk_math_hw_configure<fp32_dest_acc_en>(in0_cb_id, in1_cb_id)));
    MATH((llk_math_compressed_custom_mm_init<transpose, layout>(in0_cb_id, in1_cb_id)));

    PACK((llk_pack_dest_init<fp32_dest_acc_en, PackMode::Default>(out_cb_id)));
    PACK((llk_pack_hw_configure<fp32_dest_acc_en>(out_cb_id)));
    PACK((llk_pack_init<PackMode::Default, false /* zero_output */>(out_cb_id)));
    PACK((_llk_pack_custom_mm_init_<layout>(pack_src_format[get_output_id(out_cb_id)])));
}

// clang-format off
/**
 * Short initialization for compressed_custom_mm_block operation. Must be called before compressed_custom_mm_block and is safe to call at any point in the kernel.
 * For initializing compressed_custom_mm_block at the beginning of the kernel, please use compressed_custom_mm_block_init.
 *
 * Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
 * in0 tile shape: [{1, 2, 4, 8}, 32]
 * in1 tile shape: [32, 32]
 * rt_dim: 1
 * ct_dim: any integer from 1 to 64, as long as the output tiles fit in DEST (see CustomMmDestLayout)
 * kt_dim: even number from 2 to 256 (inclusive)
 * fidelity: LoFi only
 * throttle: not supported
 *
 * Return value: None
 *
 * | Argument  | Description                                                                               | Type               | Valid Range                                             | Required                |
 * |-----------|-------------------------------------------------------------------------------------------|--------------------|---------------------------------------------------------|-------------------------|
 * | transpose | The transpose flag for performing transpose operation on in1                              | bool               | true/false                                              | False (default false)   |
 * | layout    | Where the output tiles sit in DEST, see CustomMmDestLayout; execute and uninit must match | CustomMmDestLayout | Default, Dense, Compact                                 | False (default Default) |
 * | clear_src | Whether to clear SrcB at init (saves power as only 1/8 FPU rows are used)                 | bool               | true/false                                              | False (default true)    |
 * | in0_cb_id | The identifier of the first input circular buffer (CB)                                    | uint32_t           | 0 to 31                                                 | True                    |
 * | in1_cb_id | The identifier of the second input circular buffer (CB)                                   | uint32_t           | 0 to 31                                                 | True                    |
 * | out_cb_id | The identifier of the output circular buffer (CB)                                         | uint32_t           | 0 to 31                                                 | True                    |
 */
// clang-format on
template <bool transpose = false, CustomMmDestLayout layout = CustomMmDestLayout::Default, bool clear_src = true>
ALWI void compressed_custom_mm_block_init_short(
    const std::uint32_t in0_cb_id, const std::uint32_t in1_cb_id, const std::uint32_t out_cb_id) {
    UNPACK((llk_unpack_AB_compressed_custom_mm_init<transpose, clear_src>(in0_cb_id, in1_cb_id)));

    MATH((llk_math_compressed_custom_mm_init<transpose, layout>(in0_cb_id, in1_cb_id)));

    PACK((_llk_pack_custom_mm_init_<layout>(pack_src_format[get_output_id(out_cb_id)])));
}

// clang-format off
/**
 * Performs block-sized matrix multiplication *C=A\*B* between the blocks in two
 * different input CBs and writes the result to DST. The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking and
 * is only available on the compute engine.
 *
 * Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
 * in0 tile shape: [{1, 2, 4, 8}, 32]
 * in1 tile shape: [32, 32]
 * rt_dim: 1
 * ct_dim: any integer from 1 to 64, as long as the output tiles fit in DEST (see CustomMmDestLayout)
 * kt_dim: even number from 2 to 256 (inclusive)
 * fidelity: LoFi only
 * throttle: not supported
 *
 * Return value: None
 *
 * | Argument          | Description                                                              | Type               | Valid Range                                             | Required                |
 * |-------------------|--------------------------------------------------------------------------|--------------------|---------------------------------------------------------|-------------------------|
 * | layout            | Must match the init; dst_index counts its custom_mm_tile_rows            | CustomMmDestLayout | Default, Dense, Compact                                 | False (default Default) |
 * | in0_cb_id         | The identifier of the first input circular buffer (CB)                   | uint32_t           | 0 to 31                                                 | True                    |
 * | in1_cb_id         | The identifier of the second input circular buffer (CB)                  | uint32_t           | 0 to 31                                                 | True                    |
 * | base_address_meta | L1 byte address of the per-tile format metadata                          | uint32_t           | L1 address                                              | True                    |
 * | dst_index         | The first DST output tile, in tiles of the layout's custom_mm_tile_rows  | uint32_t           | Output tiles must fit the acquired DST REG              | True                    |
 * | kt_dim            | The inner dimension in tiles                                             | uint32_t           | Must be an even number from 2 to 256 (inclusive)        | True                    |
 * | ct_dim            | The width of the output matrix in tiles                                  | uint32_t           | 1 to 64                                                 | False (default 1)       |
 */
// clang-format on
template <CustomMmDestLayout layout = CustomMmDestLayout::Default>
ALWI void compressed_custom_mm_block(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t base_address_meta,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    UNPACK((llk_unpack_AB_compressed_custom_mm(in0_cb_id, in1_cb_id, base_address_meta, kt_dim, ct_dim)));
    MATH((llk_math_compressed_custom_mm<layout>(in0_cb_id, in1_cb_id, base_address_meta, dst_index, kt_dim, ct_dim)));
}

// clang-format off
/**
 * Performs unpack part of block-sized matrix multiplication *C=A\*B* between the blocks in two
 * different input CBs and writes the result to DST. The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking and
 * is only available on the compute engine.
 *
 * Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
 * in0 tile shape: [{1, 2, 4, 8}, 32]
 * in1 tile shape: [32, 32]
 * rt_dim: 1
 * ct_dim: any integer from 1 to 64, as long as the output tiles fit in DEST (see CustomMmDestLayout)
 * kt_dim: even number from 2 to 256 (inclusive)
 * fidelity: LoFi only
 * throttle: not supported
 *
 * Return value: None
 *
 * | Argument          | Description                                             | Type     | Valid Range                                      | Required          |
 * |-------------------|---------------------------------------------------------|----------|--------------------------------------------------|-------------------|
 * | in0_cb_id         | The identifier of the first input circular buffer (CB)  | uint32_t | 0 to 31                                          | True              |
 * | in1_cb_id         | The identifier of the second input circular buffer (CB) | uint32_t | 0 to 31                                          | True              |
 * | base_address_meta | L1 byte address of the per-tile format metadata         | uint32_t | L1 address                                       | True              |
 * | kt_dim            | The inner dimension in tiles                            | uint32_t | Must be an even number from 2 to 256 (inclusive) | True              |
 * | ct_dim            | The width of the output matrix in tiles                 | uint32_t | 1 to 64                                          | False (default 1) |
 */
// clang-format on
ALWI void compressed_custom_mm_block_unpack(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t base_address_meta,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    UNPACK((llk_unpack_AB_compressed_custom_mm(in0_cb_id, in1_cb_id, base_address_meta, kt_dim, ct_dim)));
}

// clang-format off
/**
 * Performs math part of block-sized matrix multiplication *C=A\*B* between the blocks in two
 * different input CBs and writes the result to DST. The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking and
 * is only available on the compute engine.
 *
 * Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
 * in0 tile shape: [{1, 2, 4, 8}, 32]
 * in1 tile shape: [32, 32]
 * rt_dim: 1
 * ct_dim: any integer from 1 to 64, as long as the output tiles fit in DEST (see CustomMmDestLayout)
 * kt_dim: even number from 2 to 256 (inclusive)
 * fidelity: LoFi only
 * throttle: not supported
 *
 * Return value: None
 *
 * | Argument          | Description                                                              | Type               | Valid Range                                             | Required                |
 * |-------------------|--------------------------------------------------------------------------|--------------------|---------------------------------------------------------|-------------------------|
 * | layout            | Must match the init; dst_index counts its custom_mm_tile_rows            | CustomMmDestLayout | Default, Dense, Compact                                 | False (default Default) |
 * | in0_cb_id         | The identifier of the first input circular buffer (CB)                   | uint32_t           | 0 to 31                                                 | True                    |
 * | in1_cb_id         | The identifier of the second input circular buffer (CB)                  | uint32_t           | 0 to 31                                                 | True                    |
 * | base_address_meta | L1 byte address of the per-tile format metadata                          | uint32_t           | L1 address                                              | True                    |
 * | dst_index         | The first DST output tile, in tiles of the layout's custom_mm_tile_rows  | uint32_t           | Output tiles must fit the acquired DST REG              | True                    |
 * | kt_dim            | The inner dimension in tiles                                             | uint32_t           | Must be an even number from 2 to 256 (inclusive)        | True                    |
 * | ct_dim            | The width of the output matrix in tiles                                  | uint32_t           | 1 to 64                                                 | False (default 1)       |
 */
// clang-format on
template <CustomMmDestLayout layout = CustomMmDestLayout::Default>
ALWI void compressed_custom_mm_block_math(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t base_address_meta,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    MATH((llk_math_compressed_custom_mm<layout>(in0_cb_id, in1_cb_id, base_address_meta, dst_index, kt_dim, ct_dim)));
}

// clang-format off
/**
 * Uninitializes the compressed_custom_mm_block operation, must be called after the final compressed_custom_mm_block call in a sequence and before initializing another operation.
 *
 *
 * Return value: None
 *
 * | Argument  | Description                                        | Type               | Valid Range                                             | Required                |
 * |-----------|----------------------------------------------------|--------------------|---------------------------------------------------------|-------------------------|
 * | layout    | Must match the init                                | CustomMmDestLayout | Default, Dense, Compact                                 | False (default Default) |
 * | out_cb_id | The output circular buffer (CB) the init was given | uint32_t           | 0 to 31                                                 | True                    |
 */
// clang-format on
template <CustomMmDestLayout layout = CustomMmDestLayout::Default>
ALWI void compressed_custom_mm_block_uninit(const std::uint32_t out_cb_id) {
    PACK((_llk_pack_custom_mm_uninit_<layout>(pack_src_format[get_output_id(out_cb_id)])));
}

#endif  // ARCH_BLACKHOLE

}  // namespace ckernel
