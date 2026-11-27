// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Wormhole readback test: the math thread's ALU source-format writers must mask each DataFormat to
// its 4-bit field, or bit 4 of SrcA (UInt8 = 30, UInt32 = 24) spills onto the LSB of SrcB (srcB | 1).
// Output: magic, record count, words per record, then per record {writer, srcA, srcB, config word}.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"

namespace
{

using namespace ckernel;

constexpr std::uint32_t RESULT_MAGIC     = 0x414C5546; // "ALUF"
constexpr std::uint32_t WORDS_PER_RECORD = 4;
constexpr std::uint32_t HEADER_WORDS     = 3;

enum class Writer : std::uint32_t
{
    HwConfigure      = 0, // _llk_math_hw_configure_
    Reconfig         = 1, // _llk_math_reconfig_data_format_ (re-derives INT8 math enable)
    ReconfigSkipInt8 = 2, // _llk_math_reconfig_data_format_<..., skip_int8 = true>
};
constexpr std::uint32_t NUM_WRITERS = 3;

struct FormatPair
{
    DataFormat srca;
    DataFormat srcb;
};

// Spill pairs first (SrcA bit 4 set, SrcB bit 0 clear, so a spill reads as srcB | 1); {UInt32, Tf32}
// is the only spill pair where neither operand enables INT8 math. The last two fit the field (controls).
constexpr FormatPair PAIRS[] = {
    {DataFormat::UInt8, DataFormat::Int8},
    {DataFormat::UInt8, DataFormat::Int32},
    {DataFormat::UInt8, DataFormat::Tf32},
    {DataFormat::UInt32, DataFormat::Int8},
    {DataFormat::UInt32, DataFormat::Int32},
    {DataFormat::UInt32, DataFormat::Tf32},
    {DataFormat::Int8, DataFormat::Int8},
    {DataFormat::Float16_b, DataFormat::Float16_b},
};
constexpr std::uint32_t NUM_PAIRS   = sizeof(PAIRS) / sizeof(PAIRS[0]);
constexpr std::uint32_t NUM_RECORDS = NUM_WRITERS * NUM_PAIRS;

inline std::uint32_t read_alu_format_word()
{
    tensix_sync();
    return get_cfg_pointer()[ALU_FORMAT_SPEC_REG0_SrcA_ADDR32];
}

// Seed so a stale read cannot match a pair: the skip_int8 writer must leave INT8 math enable as it
// found it (seeded set); the other writers re-derive it (seeded clear).
inline void seed(const Writer writer)
{
    if (writer == Writer::ReconfigSkipInt8)
    {
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(to_underlying(DataFormat::Int8), to_underlying(DataFormat::Float16_b));
    }
    else
    {
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(to_underlying(DataFormat::Float32), to_underlying(DataFormat::Float32));
    }
}

inline void program(const Writer writer, const std::uint32_t srca, const std::uint32_t srcb)
{
    switch (writer)
    {
        case Writer::HwConfigure:
            _llk_math_hw_configure_<is_fp32_dest_acc_en>(srca, srcb);
            break;
        case Writer::Reconfig:
            _llk_math_reconfig_data_format_<is_fp32_dest_acc_en>(srca, srcb);
            break;
        case Writer::ReconfigSkipInt8:
            _llk_math_reconfig_data_format_<is_fp32_dest_acc_en, true /*skip_int8*/>(srca, srcb);
            break;
    }
}

} // namespace

void run_kernel(RUNTIME_PARAMETERS params)
{
    volatile std::uint32_t* out = reinterpret_cast<volatile std::uint32_t*>(params.buffer_Res[0]);

    out[0] = RESULT_MAGIC;
    out[1] = NUM_RECORDS;
    out[2] = WORDS_PER_RECORD;

    std::uint32_t k = HEADER_WORDS;
    for (std::uint32_t w = 0; w < NUM_WRITERS; w++)
    {
        for (std::uint32_t p = 0; p < NUM_PAIRS; p++)
        {
            const std::uint32_t srca = to_underlying(PAIRS[p].srca);
            const std::uint32_t srcb = to_underlying(PAIRS[p].srcb);

            const Writer writer = static_cast<Writer>(w);
            seed(writer);
            program(writer, srca, srcb);

            out[k++] = w;
            out[k++] = srca;
            out[k++] = srcb;
            out[k++] = read_alu_format_word();
        }
    }
}

#else

void run_kernel(RUNTIME_PARAMETERS)
{
}

#endif
