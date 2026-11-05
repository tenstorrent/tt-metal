// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "perf.h"

static constexpr std::uint32_t COMPRESSED_META_SLOTS_PER_WORD = 10;
static constexpr std::uint32_t COMPRESSED_META_BITS_PER_SLOT  = 3;
static constexpr std::uint32_t COMPRESSED_META_USE_B_OFFSET   = 2;
static constexpr std::uint32_t COMPRESSED_META_FORMAT_OFFSET  = 3;
static constexpr std::uint32_t COMPRESSED_META_FORMAT_MASK    = 0x3;
static constexpr std::uint32_t COMPRESSED_META_BFP8_CODE      = 3;
static constexpr std::uint32_t COMPRESSED_META_BFP4_CODE      = 2;
static constexpr std::uint32_t COMPRESSED_META_BFP2_CODE      = 1;

// Mirror _llk_math_compressed_custom_mm_'s metadata walk and dvalid sequence.
template <bool post>
inline void compressed_mm_isolate_dvalids(
    const std::uint32_t loop_factor,
    const std::uint32_t num_calls,
    const std::uint32_t kt_per_call,
    const std::uint32_t ct_dim,
    const std::uint32_t meta_base,
    const std::uint32_t meta_bytes_per_call)
{
    for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
    {
        for (std::uint32_t call = 0; call < num_calls; ++call)
        {
            const std::uint32_t* meta_ptr = reinterpret_cast<const std::uint32_t*>(meta_base + call * meta_bytes_per_call);
            std::uint32_t index           = 0;
            std::uint32_t meta_index      = 0;
            std::uint32_t meta            = meta_ptr[0] >> COMPRESSED_META_FORMAT_OFFSET;
            auto advance                  = [&]()
            {
                ++index;
                if (index == COMPRESSED_META_SLOTS_PER_WORD)
                {
                    index = 0;
                    ++meta_index;
                    meta = meta_ptr[meta_index] >> COMPRESSED_META_FORMAT_OFFSET;
                }
                else
                {
                    meta >>= COMPRESSED_META_BITS_PER_SLOT;
                }
            };

            for (std::uint32_t k = 0; k < kt_per_call; ++k)
            {
                if constexpr (post)
                {
                    _perf_unpack_loop_set_valid<false /* set_a */, true /* set_b */>(1 /* iterations */);
                }
                for (std::uint32_t j = 0; j + 1 < ct_dim; ++j)
                {
                    if ((meta & COMPRESSED_META_FORMAT_MASK) != 0)
                    {
                        if constexpr (post)
                        {
                            _perf_unpack_loop_set_valid<true /* set_a */, false /* set_b */>(1 /* iterations */);
                        }
                        else
                        {
                            _perf_math_loop_clear_valid<true /* clear_a */, false /* clear_b */>(1 /* iterations */);
                        }
                    }
                    advance();
                }
                if ((meta & COMPRESSED_META_FORMAT_MASK) != 0)
                {
                    if constexpr (post)
                    {
                        _perf_unpack_loop_set_valid<true /* set_a */, false /* set_b */>(1 /* iterations */);
                    }
                    else
                    {
                        _perf_math_loop_clear_valid<true /* clear_a */, true /* clear_b */>(1 /* iterations */);
                    }
                }
                else if constexpr (!post)
                {
                    _perf_math_loop_clear_valid<false /* clear_a */, true /* clear_b */>(1 /* iterations */);
                }
                advance();
            }
        }
    }
}
