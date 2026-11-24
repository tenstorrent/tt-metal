// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Id-free (2.0) Quasar LLK API: copy (FPU datacopy and unpack-to-dest), eltwise binary, matmul, pack.
// New functions only -- the legacy DFB-id LLK API (llk_*_api.h) is untouched. Every function takes the operand's
// LLKMemDescriptor (L1 format + tile shape) as an NTTP and, where the op reads/writes L1, a buffer-descriptor base
// (16B units) at init and L1 tile indices relative to it at execute. No DFB id, no host format tables.
//
// Init functions are noinline (one copy per descriptor set) and program ALL state their op reads: buffer descriptor
// (fixed id per TDMA engine, ckernel_buf_desc.h), register formats, MOP. So no reconfig and no uninit:
// on Quasar every op is re-initialised before it runs anyway. Execute functions and the once-per-kernel hw_configure
// setups (24-52 B) are plain inline.
//
// Register formats are the L1 format (no Quasar format-derivation table exists on device yet), so the compute layer
// restricts 2.0 operands to 16-bit float formats, plus Float32 where it never enters SrcA/SrcB: unpack-to-dest copy and
// pack, at a 32-bit Dest. With a 32-bit Dest the packer reads Float32.

#include <cstdint>

#include "api/compute/experimental/2_0/internal/llk_descriptor.h"

#ifdef TRISC_UNPACK
#include "llk_unpack_common.h"
#endif
#ifdef TRISC_MATH
#include "llk_math_common.h"
#endif
#ifdef TRISC_PACK
#include "llk_pack_common.h"
#endif

namespace llk_2_0 {

using ckernel::experimental::LLKMemDescriptor;

// Register format == L1 format holds for the 16-bit float formats only (see the header comment).
constexpr bool is_supported_format(DataFormat f) { return f == DataFormat::Float16_b || f == DataFormat::Float16; }

// Copy (unpack-to-dest) and pack operands also take Float32 at a 32-bit Dest: it never enters SrcA/SrcB.
constexpr bool is_supported_format(DataFormat f, bool en_32bit_dest) {
    return is_supported_format(f) || (en_32bit_dest && f == DataFormat::Float32);
}

#ifdef TRISC_UNPACK
/**
 * @brief Kernel-wide unpack setup: SrcA/SrcB register formats and the unpack-to-dest bank offset reset.
 * @tparam A, B: Descriptors of the operands unpacked to SrcA / SrcB.
 */
template <LLKMemDescriptor A, LLKMemDescriptor B>
inline void unpack_hw_configure() {
    _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(A.format, B.format);
    _llk_unpack_dest_init_();
}
#endif

#ifdef TRISC_MATH
/**
 * @brief Kernel-wide math setup: Dest mode (16/32-bit), ALU formats, and the legacy MATH/PACK Dest semaphores
 * (llk_math_pack_sync_init).
 * @tparam A, B: Descriptors of the SrcA / SrcB operands.
 * @tparam EN_32BIT_DEST: 32-bit Dest mode.
 */
template <LLKMemDescriptor A, LLKMemDescriptor B, bool EN_32BIT_DEST>
inline void math_hw_configure() {  // once per kernel: inline
    _llk_math_srcAB_hw_configure_<EN_32BIT_DEST>(A.format, B.format);
    _llk_math_pack_sync_init_<DST_SYNC_MODE>();
}
#endif

#ifdef TRISC_PACK
/** @brief Packer input (Dest) format: Float32 with a 32-bit Dest, else the L1 format. */
template <LLKMemDescriptor D, bool EN_32BIT_DEST>
constexpr DataFormat pack_src_format() {
    return EN_32BIT_DEST ? DataFormat::Float32 : D.format;
}

/**
 * @brief Kernel-wide pack setup: packer input format and Dest bank 0 (llk_pack_dest_init).
 */
template <LLKMemDescriptor D, bool EN_32BIT_DEST>
inline void pack_hw_configure() {
    _llk_pack_hw_configure_<p_pacr::PACK0, EN_32BIT_DEST>(
        pack_src_format<D, EN_32BIT_DEST>(), ckernel::ReluConfig::none());
    _llk_pack_dest_init_<p_pacr::PACK0, DST_SYNC_MODE>();
}
#endif

}  // namespace llk_2_0
