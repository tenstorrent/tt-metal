// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Small wrappers over the SrcS auto-loop config helpers in llk_srcs_tdma.h,
// used by tests that stream a full 32x32 tile through SrcS. They dispatch on
// a runtime srcs_32bit_mode flag and pick INSTRN_LOOP_COUNT = srcs_dims::slice_count(mode)
// so a single _llk_{unpack,pack}_srcs_ call covers one tile.

#pragma once

#include <cstdint>

#include "llk_srcs_tdma.h"

template <std::uint8_t INSTRN_COUNT>
inline void _llk_unpack_srcs_config_for_tile_(const bool srcs_32bit_mode)
{
    if (srcs_32bit_mode)
    {
        _llk_unpack_srcs_config_<INSTRN_COUNT, srcs_dims::slice_count(true)>();
    }
    else
    {
        _llk_unpack_srcs_config_<INSTRN_COUNT, srcs_dims::slice_count(false)>();
    }
}

template <std::uint8_t INSTRN_COUNT>
inline void _llk_pack_srcs_config_for_tile_(const bool srcs_32bit_mode)
{
    if (srcs_32bit_mode)
    {
        _llk_pack_srcs_config_<INSTRN_COUNT, srcs_dims::slice_count(true)>();
    }
    else
    {
        _llk_pack_srcs_config_<INSTRN_COUNT, srcs_dims::slice_count(false)>();
    }
}
