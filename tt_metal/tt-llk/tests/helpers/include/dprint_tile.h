// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Wrapper for tt_metal/hw/inc/api/debug/dprint_tile.h.
// Metal's TileSlice constructor reads tile data from a CB; we avoid that
// by making a function that fills a TileSlice directly from an L1 address.

#pragma once

#include "dprint.h"

#if defined(DEBUG_PRINT_ENABLED) && !defined(COVERAGE)

#include <cstdint>

#include "api/debug/dprint_tile.h"
#include "ckernel_defs.h"
#include "tensix_types.h"

template <int MAX_BYTES = 64>
inline TileSlice<MAX_BYTES> tile_slice(
    std::uint32_t l1_addr,
    DataFormat fmt,
    const SliceRange& sr,
    bool endl_rows           = true,
    std::uint32_t tile_dim_r = ckernel::TILE_R_DIM,
    std::uint32_t tile_dim_c = ckernel::TILE_C_DIM,
    std::uint32_t face_dim_r = ckernel::FACE_R_DIM,
    std::uint32_t face_dim_c = ckernel::FACE_C_DIM,
    std::uint32_t num_faces  = ckernel::TILE_NUM_FACES)
{
    TileSlice<MAX_BYTES> ts {};
    ts.slice_range = sr;
    ts.cb_ptr      = l1_addr;
    ts.data_format = static_cast<std::uint8_t>(fmt);
    ts.endl_rows   = endl_rows;
    ts.return_code = DPrintOK;

    if (!is_supported_format(static_cast<CommonDataFormat>(fmt)))
    {
        ts.return_code = DPrintErrorUnsupportedFormat;
        return ts;
    }

    tile_info_t info {};
    info.tile_dim_r = tile_dim_r;
    info.tile_dim_c = tile_dim_c;
    info.face_dim_r = face_dim_r;
    info.face_dim_c = face_dim_c;
    info.num_faces  = num_faces;

    volatile std::uint8_t* base = reinterpret_cast<volatile std::uint8_t*>(l1_addr);
    const bool bfp              = is_bfp(static_cast<CommonDataFormat>(fmt));
    // Bfp L1 layout: face_dim_r * num_faces shared-exponent bytes at offset 0
    // (one per (face, row-within-face)), then mantissa bytes packed per
    // dprint_datum_size(fmt) (Bfp8: 1 elt per byte, Bfp4: 2 elts/byte). We
    // emit one (exp, mantissa) byte pair per element. Other formats walk
    // dprint_datum_size(fmt) raw bytes per element.
    const std::uint32_t bytes_per_datum = bfp ? 2 : dprint_datum_size(static_cast<CommonDataFormat>(fmt));
    const std::uint32_t mantissa_offset = info.face_dim_r * info.num_faces;

    std::uint32_t byte_idx = 0;
    for (std::uint32_t h = sr.h0; h < sr.h1; h += sr.hs)
    {
        for (std::uint32_t w = sr.w0; w < sr.w1; w += sr.ws)
        {
            if (byte_idx + bytes_per_datum > MAX_BYTES)
            {
                return ts;
            }
            if (bfp)
            {
                ts.data[byte_idx++] = base[TileSlice<MAX_BYTES>::get_exponent_index(info, h, w, /*untilize=*/true)];
                ts.data[byte_idx++] = get_datum(fmt, base + mantissa_offset, TileSlice<MAX_BYTES>::get_data_index(info, h, w, /*untilize=*/true));
            }
            else
            {
                const std::uint32_t i = TileSlice<MAX_BYTES>::get_data_index(info, h, w, /*untilize=*/true);
                for (std::uint32_t b = 0; b < bytes_per_datum; ++b)
                {
                    ts.data[byte_idx++] = base[i * bytes_per_datum + b];
                }
            }
            ++ts.data_count;
        }
    }
    return ts;
}

#endif // defined(DEBUG_PRINT_ENABLED) && !defined(COVERAGE)
