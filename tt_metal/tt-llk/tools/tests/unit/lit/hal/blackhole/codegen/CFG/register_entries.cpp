// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %s -o %t.o
// RUN: %{blackhole_objdump} -drz %t.o > %t.dump
// RUN: %{blackhole_compare_codegen} %t.dump

// Compare each HAL function with its adjacent reference_ function. Disable
// identical-code folding so both bodies are emitted independently.
// Setting every entry of a register with compile-time set() operands in one write produces the
// immediate RMWCIBs of the packed word. The cases are the TILE_FACE_SET_MAPPING_0 words of
// _llk_pack_reduce_mask_config_: entry S selects row-set table 1 or 0 for face S.

#include <cstddef>
#include <cstdint>
#include <utility>

#include "hal/cfg.h"

namespace cfg = hal::cfg;

// Row-set table 1 applies the reduction mask; table 0 masks the whole face.
inline constexpr std::uint32_t ReductionMaskTable = 1;
inline constexpr std::uint32_t MaskedFaceTable    = 0;

// Of every Period faces, the first ReducedFaces use the reduction mask and the rest are masked.
template <std::uint32_t Period, std::uint32_t ReducedFaces>
constexpr std::uint32_t row_set_table(const std::size_t face)
{
    if (face % Period < ReducedFaces)
    {
        return ReductionMaskTable;
    }
    return MaskedFaceTable;
}

// One write with a set() per entry: the HAL groups the sixteen entries into one update per byte.
template <std::uint32_t Period, std::uint32_t ReducedFaces, std::size_t... Face>
inline __attribute__((always_inline)) void select_row_set_tables(std::index_sequence<Face...>)
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::TileFaceSetMapping[0][Face], cfg::Sec::S0, row_set_table<Period, ReducedFaces>(Face)>()...);
}

template <std::uint32_t Period, std::uint32_t ReducedFaces>
inline __attribute__((always_inline)) void select_row_set_tables()
{
    select_row_set_tables<Period, ReducedFaces>(std::make_index_sequence<16> {});
}

#define REFERENCE_WORD(BYTE)                                                                      \
    TTI_RMWCIB0(0xff /*Mask*/, BYTE /*Data*/, TILE_FACE_SET_MAPPING_0_face_set_mapping_0_ADDR32); \
    TTI_RMWCIB1(0xff /*Mask*/, BYTE /*Data*/, TILE_FACE_SET_MAPPING_0_face_set_mapping_0_ADDR32); \
    TTI_RMWCIB2(0xff /*Mask*/, BYTE /*Data*/, TILE_FACE_SET_MAPPING_0_face_set_mapping_0_ADDR32); \
    TTI_RMWCIB3(0xff /*Mask*/, BYTE /*Data*/, TILE_FACE_SET_MAPPING_0_face_set_mapping_0_ADDR32)

// [1]: every face uses table 1 (0x55555555).
extern "C" __attribute__((noinline, used)) void every_face()
{
    select_row_set_tables<1 /*Period*/, 1 /*ReducedFaces*/>();
}

extern "C" __attribute__((noinline, used)) void reference_every_face()
{
    REFERENCE_WORD(0x55);
}

// [1,0] (0x11111111).
extern "C" __attribute__((noinline, used)) void every_second_face()
{
    select_row_set_tables<2 /*Period*/, 1 /*ReducedFaces*/>();
}

extern "C" __attribute__((noinline, used)) void reference_every_second_face()
{
    REFERENCE_WORD(0x11);
}

// [1,1,0,0] (0x05050505).
extern "C" __attribute__((noinline, used)) void first_two_of_four()
{
    select_row_set_tables<4 /*Period*/, 2 /*ReducedFaces*/>();
}

extern "C" __attribute__((noinline, used)) void reference_first_two_of_four()
{
    REFERENCE_WORD(0x05);
}

// [1,0,0,0] (0x01010101).
extern "C" __attribute__((noinline, used)) void first_of_four()
{
    select_row_set_tables<4 /*Period*/, 1 /*ReducedFaces*/>();
}

extern "C" __attribute__((noinline, used)) void reference_first_of_four()
{
    REFERENCE_WORD(0x01);
}
