/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "api/compute/common_globals.h"
#include "llk_assert.h"
// Every TRISC: the mask_tile signature names llk::sfpu::MaskFill.
#include "sfpu/macro/mask.h"

namespace ckernel {

ALWI void mask_tile_init() { MATH((llk::sfpu::Mask::init())); }

// clang-format off
/**
 * Performs element-wise computation of mask on the region *shape* of the tiles in DST register:
 * output = (mask == 0) ? fill : input. Out of place; pass the same index for output and input to mask in place.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument     | Description                                                                | Type               | Valid Range                                    | Required |
 * |--------------|----------------------------------------------------------------------------|--------------------|------------------------------------------------|----------|
 * | shape        | Region of the tiles to process (template argument)                         | const TensorShape& | At most 2x2 faces; must have static storage    | True     |
 * | fill         | Value written where the mask is zero (template argument)                   | llk::sfpu::MaskFill| FloatPosInf needs Float16_b or Float16         | False    |
 * | format       | The format of the data and mask (template argument)                        | DataFormat         | Float16_b, Float16 or Int32                    | False    |
 * | output       | The index of the tile in DST REG the result is written to                  | uint32_t           | Must be less than the acquired size of DST REG | True     |
 * | input        | The index of the tile in DST REG for the data                              | uint32_t           | Must be less than the acquired size of DST REG | True     |
 * | mask         | The index of the tile in DST REG for the mask                              | uint32_t           | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
template <const TensorShape& shape, llk::sfpu::MaskFill fill, DataFormat format = DataFormat::Float16_b>
ALWI void mask_tile(uint32_t output, uint32_t input, uint32_t mask) {
    MATH((llk::sfpu::Mask::execute<shape, fill, format>(output, input, mask)));
}

// clang-format off
/**
 * @deprecated Use `mask_tile<shape, fill, format>(output, input, mask)`, which names the input and mask tiles
 * explicitly and takes the fill and format at compile time. These forwarding shims are retained only for backwards
 * compatibility.
 *
 * Performs element-wise computation of mask on each element of a tile
 * in data and mask DST register. *mask_tile* will mask each element with 0,
 * *mask_posinf_tile* will mask each element with *float(inf)*.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * idst2_mask is not used: the mask is read from (idst_data + 1), which callers rely on.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type       | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|------------|-------------------------------------------------------|----------|
 * | dst_data_index | The index of the tile in DST REG for the data and result                   | uint32_t   | Must be less than the acquired size of DST REG        | True     |
 * | dst_mask_index | Unused: the mask is read from dst_data_index + 1                           | uint32_t   | Must be less than the acquired size of DST REG        | True     |
 * | data_format    | The format of the data and mask (supports Float16, Float16_b, and Int32)   | DataFormat | Must be a valid data format                           | False    |
 */
// clang-format on
[[deprecated("Use mask_tile<shape, MaskFill::Zero, format>(output, input, mask) instead.")]] ALWI void mask_tile(
    uint32_t idst_data, [[maybe_unused]] uint32_t idst2_mask, DataFormat data_format = DataFormat::Float16_b) {
    switch (data_format) {
        case DataFormat::Float16_b:
            mask_tile<DEFAULT_TENSOR_SHAPE, llk::sfpu::MaskFill::Zero, DataFormat::Float16_b>(
                idst_data, idst_data, idst_data + 1);
            break;
        case DataFormat::Float16:
            mask_tile<DEFAULT_TENSOR_SHAPE, llk::sfpu::MaskFill::Zero, DataFormat::Float16>(
                idst_data, idst_data, idst_data + 1);
            break;
        case DataFormat::Int32:
            mask_tile<DEFAULT_TENSOR_SHAPE, llk::sfpu::MaskFill::Zero, DataFormat::Int32>(
                idst_data, idst_data, idst_data + 1);
            break;
        default: LLK_ASSERT(false, "mask_tile: supports Float16_b, Float16 and Int32");
    }
}

[[deprecated("Use mask_tile<shape, MaskFill::FloatPosInf, format>(output, input, mask) instead.")]] ALWI void
mask_posinf_tile(uint32_t idst_data, [[maybe_unused]] uint32_t idst2_mask) {
    mask_tile<DEFAULT_TENSOR_SHAPE, llk::sfpu::MaskFill::FloatPosInf, DataFormat::Float16_b>(
        idst_data, idst_data, idst_data + 1);
}

}  // namespace ckernel
