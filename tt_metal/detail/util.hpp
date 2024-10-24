// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "tt_metal/common/math.hpp"
#include "tt_metal/common/tt_backend_api_types.hpp"
#include "tt_metal/hostdevcommon/common_values.hpp"
#include "tt_metal/impl/kernels/data_types.hpp"
#include "llrt/hal.hpp"

namespace tt::tt_metal::detail{

    /**
     * Returns tile size of given data format in bytes
     *
     * Return value: uint32_t
     *
     * | Argument    | Description    | Type                | Valid Range | Required |
     * |-------------|----------------|---------------------|-------------|----------|
     * | data_format | Format of data | tt::DataFormat enum |             | Yes      |
     */
    inline uint32_t TileSize(const DataFormat &data_format)
    {
        return tt::tile_size(data_format);
    }

    inline DeviceAddr SizeBytesPerBank(DeviceAddr size_bytes, DeviceAddr page_size_bytes, uint32_t num_banks, uint32_t alignment_bytes) {
        TT_ASSERT(page_size_bytes == 0 ? size_bytes == 0 : size_bytes % page_size_bytes == 0, "Page size {} should be divisible by buffer size {}", page_size_bytes, size_bytes);
        DeviceAddr num_pages = page_size_bytes == 0 ? 0 : size_bytes / page_size_bytes;
        DeviceAddr num_equally_distributed_pages = num_pages == 0 ? 0 : 1 + ((num_pages - 1) / num_banks);
        return num_equally_distributed_pages * round_up(page_size_bytes, alignment_bytes);
    }

    inline NOC GetPreferredNOCForDRAMRead(ARCH arch) {
            switch (arch) {
                case ARCH::GRAYSKULL:
                    return NOC::NOC_1;
                case ARCH::WORMHOLE_B0:
                default:
                    return NOC::NOC_0;

            }
        }

        inline NOC GetPreferredNOCForDRAMWrite(ARCH arch) {
            switch (arch) {
                case ARCH::GRAYSKULL:
                    return NOC::NOC_0;
                case ARCH::WORMHOLE_B0:
                default:
                    return NOC::NOC_1;

            }
        }

}
