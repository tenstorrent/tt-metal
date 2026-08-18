// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "tensix_types.h"  // Blackhole DataFormat enum

// =====================================================================================================
// data_format_derive.h (BLACKHOLE) -- compile-time derivation of the *register* data format from a
// buffer's *L1* data format. This is a HARDWARE/ARCH concern (it encodes Blackhole's register format
// tables), so it lives at the LLK API level and is consumed only inside the LLK spec wrappers -- the
// register format is NEVER exposed above the LLK boundary. Compute APIs pass only L1 metadata.
//
// This is the BLACKHOLE copy. Wormhole gets its own copy under
// ckernels/wormhole_b0/metal/llk_api/data_format_derive.h; Quasar is a completely separate arch.
//
// Mirrors the SCALAR CORES of jit_build/data_format.cpp:
//   * infer_unpack_dst_format  <- get_single_unpack_dst_format  (data_format.cpp:141-170)
//   * infer_pack_src_format    <- get_single_pack_src_format    (data_format.cpp:226-360)
// scoped to DATACOPY (single operand), with the op-wide knobs fixed to their datacopy defaults:
// bfp8_pack_precise=false, int_fpu_en=false, enable_2x_src_format=false, arch=BLACKHOLE. The one live
// input is fp32_dest_acc; unpack_conditional_dst_format is the standard datacopy conditional
// (Float32 with dest-acc, else Float16_b).
//
// KEY SIMPLIFICATION vs the host cores: get_single_pack_src_format computes input_exp_width and
// output_exp_width BOTH from the single data_format, so they are ALWAYS equal and the host's
// "different exponent width" else-branch (the only user of CONVERT_EXP_WIDTH) is UNREACHABLE for a
// single-operand op -- no CONVERT_EXP_WIDTH map on device. MX formats are absent from the Blackhole
// DataFormat enum, so is_mx handling collapses too. Divergent-format ops (eltwise binary) would grow
// the two-operand inputs when added.
// =====================================================================================================

namespace ckernel {

// ---- format-class predicates (device subset of data_format.cpp; MX formats are not in the BH enum) --

constexpr bool df_is_exp_b_format(DataFormat f) {
    return f == DataFormat::Tf32 || f == DataFormat::Float16_b || f == DataFormat::Bfp8_b || f == DataFormat::Bfp4_b ||
           f == DataFormat::Bfp2_b;
}

constexpr bool df_is_bfp_format(DataFormat f) {
    return f == DataFormat::Bfp8_b || f == DataFormat::Bfp8 || f == DataFormat::Bfp4_b || f == DataFormat::Bfp4 ||
           f == DataFormat::Bfp2_b || f == DataFormat::Bfp2;
}

constexpr bool df_is_integer_format(DataFormat f) {
    return f == DataFormat::Int8 || f == DataFormat::UInt8 || f == DataFormat::UInt16 || f == DataFormat::Int32 ||
           f == DataFormat::UInt32;
}

// ---- unpack SrcA register format  (mirrors get_single_unpack_dst_format, datacopy scope) -------------
// Non-fp32 buffer formats unpack into src registers unchanged. Float32 in L1 unpacks to the op's
// conditional dst format: Float32 when accumulating in fp32 dest, else Float16_b.
constexpr DataFormat infer_unpack_dst_format(DataFormat l1_format, bool fp32_dest_acc) {
    if (l1_format == DataFormat::Float32) {
        return fp32_dest_acc ? DataFormat::Float32 : DataFormat::Float16_b;
    }
    return l1_format;
}

// ---- pack Dest register format  (mirrors get_single_pack_src_format, datacopy scope) -----------------
constexpr DataFormat infer_pack_src_format(DataFormat l1_format, bool fp32_dest_acc) {
    const DataFormat cond = fp32_dest_acc ? DataFormat::Float32 : DataFormat::Float16_b;

    if (l1_format == DataFormat::UInt16) {
        return DataFormat::UInt16;
    }
    if (l1_format == DataFormat::Invalid) {
        return DataFormat::Invalid;
    }
    if (l1_format == DataFormat::Fp8_e4m3) {
        return df_is_exp_b_format(cond) ? DataFormat::Float16_b : DataFormat::Float16;
    }

    if (fp32_dest_acc) {
        if (df_is_bfp_format(l1_format)) {
            // bfp8_pack_precise == false
            return df_is_exp_b_format(l1_format) ? DataFormat::Bfp8_b : DataFormat::Bfp8;
        }
        if (df_is_exp_b_format(l1_format) || l1_format == DataFormat::Float32) {
            return l1_format;
        }
        if (l1_format == DataFormat::Float16) {
            return DataFormat::Float16_b;
        }
        if (l1_format == DataFormat::UInt32) {
            return DataFormat::UInt32;
        }
        if (l1_format == DataFormat::Int32) {
            return DataFormat::Int32;
        }
        if (l1_format == DataFormat::UInt8) {
            return DataFormat::UInt8;
        }
        if (l1_format == DataFormat::Int8) {
            return DataFormat::Int8;
        }
        return l1_format;  // fp32-dest fallthrough (host TT_THROWs; keep format for device)
    }

    if (df_is_integer_format(l1_format)) {
        return l1_format;
    }

    // Single-operand exp-width-match branch (host :297): input_exp_width == output_exp_width always.
    if (l1_format == DataFormat::Float32) {
        // fp32 buffer, no dest-acc: pack src is the conditional fp16a/b (here Float16_b).
        return cond;
    }
    if (df_is_bfp_format(l1_format)) {
        // bfp8_pack_precise == false
        return df_is_exp_b_format(l1_format) ? DataFormat::Bfp8_b : DataFormat::Bfp8;
    }
    return l1_format;
}

}  // namespace ckernel
