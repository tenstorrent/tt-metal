// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "tensix_types.h"  // Blackhole DataFormat enum
#include "ckernel_defs.h"  // IS_BFP_FORMAT (canonical device bfp-format list)

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
// NOTE: df_is_exp_b_format / df_is_integer_format duplicate host logic (is_exp_b_format @ jit_build/
// data_format.cpp:54; tt::is_integer_format @ common/tt_backend_api_types.cpp:59) that is NOT includable from
// a device kernel, so a device-side copy is required. Keep the BH subset here in sync with those host sources.

// Keep in sync with is_exp_b_format (jit_build/data_format.cpp:54) -- BH-enum subset (no MX formats on BH).
constexpr bool df_is_exp_b_format(DataFormat f) {
    return f == DataFormat::Tf32 || f == DataFormat::Float16_b || f == DataFormat::Bfp8_b || f == DataFormat::Bfp4_b ||
           f == DataFormat::Bfp2_b;
}

// Delegates to the canonical device bfp-format list (ckernel_defs.h) so the set is defined once. masked_data_format
// (applied inside IS_BFP_FORMAT) is a no-op for plain L1 formats -- same value as an explicit enum compare.
constexpr bool df_is_bfp_format(DataFormat f) { return ckernel::IS_BFP_FORMAT(static_cast<std::uint32_t>(f)); }

// Keep in sync with tt::is_integer_format (common/tt_backend_api_types.cpp:59) -- BH-enum subset. NOTE the device
// ckernel_defs.h is_int8_or_int32_format is a NARROWER set (int8/int32 only), so it is not a substitute here.
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

// ---- register-format helpers (uint8 form) ------------------------------------------------------------
// The LLK cores take the register format as a raw uint8. These wrap infer_unpack_dst_format / infer_pack_src_format
// with the static_cast so the 2.0 LLK overloads call one helper instead of repeating the cast at every site.
constexpr std::uint8_t infer_unpack_reg_fmt(DataFormat l1_format, bool fp32_dest_acc) {
    return static_cast<std::uint8_t>(infer_unpack_dst_format(l1_format, fp32_dest_acc));
}
constexpr std::uint8_t infer_pack_reg_fmt(DataFormat l1_format, bool fp32_dest_acc) {
    return static_cast<std::uint8_t>(infer_pack_src_format(l1_format, fp32_dest_acc));
}

// =====================================================================================================
// TWO-OPERAND exponent-width reconciliation (C1). The HW does NOT support mixing exponent widths across the
// two source registers: both must be the 5-bit "a" family (fp16 / Bfp8 / Bfp4 / ...) or both the 8-bit "b"
// family (bf16 / Bfp8_b / Tf32 / ...). Float32 is rebiased to whichever family the OTHER operand uses, and
// UInt32, UInt16, UInt8 and Int32 carry no exponent family at all, so none of those five constrains the pairing
// -- they are the formats the host skips when it picks the family (check_consistent_format_across_buffers in
// jit_build/data_format.cpp). Int8 and Fp8_e4m3 are NOT among them (see df_is_exp_family_wildcard). Any other
// cross-family pairing is a hardware error, caught here as a hard compile failure. Formats are passed as
// template args so both the assert and the derivation fold at compile time.
// =====================================================================================================

// Formats with no exponent family of their own in the source register: they adopt the other operand's. This is
// the skip list of check_consistent_format_across_buffers (jit_build/data_format.cpp) minus three deliberate
// exclusions, so it is NOT a plain copy of that list:
//   * RawUInt32 / RawUInt16 / RawUInt8: not in the Blackhole DataFormat enum.
//   * Int8: the host does not skip it either; it stays a 5-bit "a" family operand.
//   * Fp8_e4m3: the host skips it, but only so that it never REJECTS a pairing; its source-register datum is
//     Float16 (jit_build/genfiles.cpp, generate_pack_data_formats: "Fp8_e4m3 is always unpacked to Float16
//     (A-family)", which is why the host rewrites the pack-src formats of its neighbours to the "a" family). So
//     here it is a concrete "a" family operand: a Float32 partner rebiases to Float16 and a b-family partner is
//     rejected (the unpacker has no bf16 -> fp16 conversion). Making it a wildcard would put a Float32 partner in
//     Float16_b next to an fp16 datum -- exactly the mixed-width pairing this block forbids.
constexpr bool df_is_exp_family_wildcard(DataFormat f) {
    return f == DataFormat::Float32 || f == DataFormat::UInt32 || f == DataFormat::UInt16 || f == DataFormat::UInt8 ||
           f == DataFormat::Int32;
}

// The common exponent-width family for two operand formats (true = 8-bit "b" family). A wildcard
// (df_is_exp_family_wildcard) adopts the other operand's family; two wildcards default to "b" (the host's
// get_data_exp_precision default, so a Float32 operand unpacks to Float16_b and not to the 5-bit-exponent
// Float16); two concrete formats of different families are rejected.
template <DataFormat A, DataFormat B>
constexpr bool common_exp_is_b() {
    constexpr bool a_wild = df_is_exp_family_wildcard(A);
    constexpr bool b_wild = df_is_exp_family_wildcard(B);
    if constexpr (a_wild && b_wild) {
        return true;  // neither operand has a family: the host default is B
    } else if constexpr (a_wild) {
        return df_is_exp_b_format(B);  // A adopts B's family
    } else if constexpr (b_wild) {
        return df_is_exp_b_format(A);  // B adopts A's family
    } else {
        static_assert(
            df_is_exp_b_format(A) == df_is_exp_b_format(B),
            "HW does not support mixed exponent-width operands (only Float32, UInt32, UInt16, UInt8 and Int32 adopt "
            "the other's width).");
        return df_is_exp_b_format(A);
    }
}

// Unpack-dst (register) format for the SELF operand of a two-operand op, honoring the common exp-width family
// so both source registers agree. Non-Float32 formats are unchanged (identical to the single-operand infer);
// a Float32 operand adopts the common family: Float32 under fp32-dest-acc, else Float16_b (b) / Float16 (a).
// For matching formats this is exactly infer_unpack_dst_format(Self, ...), so it changes nothing on that path
// -- its job is the Float32-rebias case and the compile-time mixed-width guard.
template <DataFormat Self, DataFormat Other>
constexpr DataFormat infer_unpack_dst_format_2op(bool fp32_dest_acc) {
    static_cast<void>(common_exp_is_b<Self, Other>());  // fires the mixed-width static_assert for any pairing
    if constexpr (Self == DataFormat::Float32) {
        if (fp32_dest_acc) {
            return DataFormat::Float32;
        }
        return common_exp_is_b<Self, Other>() ? DataFormat::Float16_b : DataFormat::Float16;
    } else {
        return infer_unpack_dst_format(Self, fp32_dest_acc);
    }
}

}  // namespace ckernel
