// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Readback test for the math thread's ALU source-format programming on Wormhole.
//
// ALU_FORMAT_SPEC_REG0_SrcA and ALU_FORMAT_SPEC_REG1_SrcB are adjacent 4-bit fields of one config
// word (SrcA bits 17..20, SrcB bits 21..24). The host passes some DataFormat values with bits above
// the field width set (UInt8 = 30, UInt32 = 24) and expects the register-write site to mask them.
// A write that shifts the raw value in spills bit 4 of SrcA onto the LSB of SrcB, and the combined
// RMW mask of the two-field writers carries that bit into the register: SrcB is programmed as
// (srcb | 1), e.g. Int8 (14) becomes Bfp2_b (15), Int32 (8) becomes UInt16 (9).
//
// MATH seeds both fields with Float32 (0) through the shipped configurator, then programs each
// (srcA, srcB) pair through each writer under test and reads the word back. Only MATH runs; UNPACK
// and PACK do nothing. The variant is built with FP32 dest accumulation (is_fp32_dest_acc_en), which
// the Int8/UInt8/Int32 formats require.
//
// Result words:
//   0        magic
//   1        number of records
//   2        words per record
//   3 + 4r   record r: writer, srcA (raw), srcB (raw), config word read back

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

// The first five pairs carry a SrcA value with bit 4 set (UInt8 = 30, UInt32 = 24) and a SrcB value
// with bit 0 clear, so a spill from SrcA is visible as SrcB | 1. The last two pairs have no spill
// (Int8 = 14 and Float16_b = 5 fit the field) and only show that the readback tracks the write.
constexpr FormatPair PAIRS[] = {
    {DataFormat::UInt8, DataFormat::Int8},
    {DataFormat::UInt8, DataFormat::Int32},
    {DataFormat::UInt8, DataFormat::Tf32},
    {DataFormat::UInt32, DataFormat::Int8},
    {DataFormat::UInt32, DataFormat::Int32},
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

inline void seed_float32()
{
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(to_underlying(DataFormat::Float32), to_underlying(DataFormat::Float32));
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

            // Seed the word so a stale read cannot match the expectation.
            seed_float32();
            program(static_cast<Writer>(w), srca, srcb);

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
