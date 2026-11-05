// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Format rules every Blackhole OpApi shares: which register format the unpacker writes for an L1 format, which
// formats are integers, and which have a 5-bit (A) or 8-bit (B) exponent. Each op's options() builds its OpConfig from these.

#pragma once

#include "tensor/tensor.h"

namespace llk2
{

/// The register format the unpacker writes to SrcA/SrcB for an L1 format, or DataFormat::Invalid if it cannot.
/// Source: tt-isa-documentation, WormholeB0/TensixTile/TensixCoprocessor/Unpackers/FormatConversion.md, "To SrcA or
/// SrcB" (the page also covers Blackhole's FP8 e4m3; there is no separate Blackhole page). The format ids are what the
/// unpacker's InDataFormat / OutDataFormat fields take, so most formats keep their id and the unpacker converts.
constexpr DataFormat src_register_format(DataFormat l1, bool dest_fp32)
{
    switch (l1)
    {
        // FP32 is cut to 19 bits in Src. With a 32-bit Dest, keep all 19 (TF32). With a 16-bit Dest, BF16.
        // Open question: a 16-bit Dest also accepts 19-bit datums (they are truncated on the way in), so TF32 would
        // work here too; BF16 matches what the test harness infers today.
        case DataFormat::Float32:
        case DataFormat::Tf32:
            return dest_fp32 ? DataFormat::Tf32 : DataFormat::Float16_b;
        // 8-bit-exponent families: BF16 and the BFP_b block formats go to TF32 or BF16 under their own id.
        case DataFormat::Float16_b:
        case DataFormat::Bfp8_b:
        case DataFormat::Bfp4_b:
        case DataFormat::Bfp2_b:
        // 5-bit-exponent families go to FP16 under their own id.
        case DataFormat::Float16:
        case DataFormat::Bfp8:
        case DataFormat::Bfp4:
        case DataFormat::Bfp2:
        case DataFormat::Lf8:
        case DataFormat::Fp8_e4m3:
        // Integers: INT16 is opaque 16-bit data; INT8 and UINT8 become Integer "8" (UINT8 also needs SrcAUnsigned /
        // SrcBUnsigned, which the unpack configure sets).
        case DataFormat::UInt16:
        case DataFormat::Int8:
        case DataFormat::UInt8:
            return l1;
        // INT32 and UINT32 cannot be unpacked to Src (only to Dest). INT8 -> BF16 through a forced shared exponent
        // is possible but not supported here.
        default:
            return DataFormat::Invalid;
    }
}

/// Integer formats: INT8, UINT8, UINT16, INT32, UINT32.
constexpr bool is_integer_format(DataFormat f)
{
    return f == DataFormat::Int8 || f == DataFormat::UInt8 || f == DataFormat::UInt16 || f == DataFormat::Int32 || f == DataFormat::UInt32;
}

/// The formats with a 5-bit exponent ("A" formats), which the FPU reads as FP16: it writes them to a 16-bit Dest as
/// FP16, the rest as BF16. Source: tt-isa-documentation, WormholeB0/TensixTile/TensixCoprocessor/ELWADD.md, functional model (SrcAStyle).
constexpr bool has_5bit_exponent(DataFormat f)
{
    return f == DataFormat::Float16 || f == DataFormat::Bfp8 || f == DataFormat::Bfp4 || f == DataFormat::Bfp2 || f == DataFormat::Lf8 ||
           f == DataFormat::Fp8_e4m3;
}

/// The formats with an 8-bit exponent ("B" formats): FP32, TF32, BF16 and the BFP-B family.
constexpr bool is_b_format(DataFormat f)
{
    return f == DataFormat::Float32 || f == DataFormat::Tf32 || f == DataFormat::Float16_b || f == DataFormat::Bfp8_b || f == DataFormat::Bfp4_b ||
           f == DataFormat::Bfp2_b;
}

} // namespace llk2
