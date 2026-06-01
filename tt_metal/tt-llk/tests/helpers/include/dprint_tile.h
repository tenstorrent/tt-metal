// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// LLK-side companion to tt_metal/hw/inc/api/debug/dprint_tile.h.
// Metal's TileSlice constructor reads tile data from a CB; LLK doesn't have
// CBs, so this header adds a factory that fills a Metal TileSlice<MAX_BYTES>
// from a raw L1 byte address instead. The wire payload (and therefore the
// host-side decoder in helpers/device_print.py) is shared with Metal.
//
// Assumes a standard 32x32 / 4-face tile layout in L1. Bfp formats are
// rejected — extend later if LLK tests need them.

#pragma once

#include "dprint.h"

#if defined(DEBUG_PRINT_ENABLED) && !defined(COVERAGE)

#include <cstdint>

#include "api/debug/dprint_tile.h"
#include "tensix_types.h"

namespace llk_dprint
{

template <int MAX_BYTES = 64>
inline TileSlice<MAX_BYTES> tile_slice(std::uint32_t l1_addr, DataFormat fmt, const SliceRange& sr, bool endl_rows = true)
{
    TileSlice<MAX_BYTES> ts {};
    ts.slice_range = sr;
    ts.cb_ptr      = l1_addr;
    ts.data_format = static_cast<std::uint8_t>(fmt);
    ts.endl_rows   = endl_rows;
    ts.return_code = DPrintOK;

    if (!is_supported_format(static_cast<CommonDataFormat>(fmt)) || is_bfp(static_cast<CommonDataFormat>(fmt)))
    {
        ts.return_code = DPrintErrorUnsupportedFormat;
        return ts;
    }

    tile_info_t info {};
    info.tile_dim_r                       = 32;
    info.tile_dim_c                       = 32;
    info.face_dim_r                       = 16;
    info.face_dim_c                       = 16;
    info.num_faces                        = 4;
    const std::uint32_t bpd               = dprint_datum_size(static_cast<CommonDataFormat>(fmt));
    volatile tt_l1_ptr std::uint8_t* base = reinterpret_cast<volatile tt_l1_ptr std::uint8_t*>(l1_addr);

    std::uint32_t byte_idx = 0;
    for (std::uint32_t h = sr.h0; h < sr.h1; h += sr.hs)
    {
        for (std::uint32_t w = sr.w0; w < sr.w1; w += sr.ws)
        {
            const std::uint32_t i = TileSlice<MAX_BYTES>::get_data_index(info, h, w, /*untilize=*/true);
            for (std::uint32_t b = 0; b < bpd; ++b)
            {
                if (byte_idx >= MAX_BYTES)
                {
                    return ts;
                }
                ts.data[byte_idx++] = base[i * bpd + b];
            }
            ++ts.data_count;
        }
    }
    return ts;
}

} // namespace llk_dprint

#endif // defined(DEBUG_PRINT_ENABLED) && !defined(COVERAGE)
