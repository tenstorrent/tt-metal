#pragma once
#include "tt_metal/common/tt_backend_api_types.hpp"

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
}
