
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Regression test for the ch1 stride-staleness fix (tt-llk#1161 follow-up):
// The ch1 (register-side) Z/Y strides are FORMAT-derived (datum size). They are now re-committed on
// EVERY format reconfig (in _llk_unpack_reconfig_data_format_srca/srcb_impl_), independent of
// dim_stride_target -- so a format-only reconfig (IGNORE) that changes datum size no longer leaves them
// stale for a following partial-face unpack (e.g. partial-face matmul, which reads them without setting
// them). Both runs assert the srcA/srcB ch1 Z-stride equals the NEW format's canonical stride:
//   run_idx 0 (control): configure DIRECTLY to fp32.
//   run_idx 1 (the fix): configure fp16, then a format-only (IGNORE) reconfig to fp32.
// Before the fix, run_idx 1 fired the assert (stale 512 != expected 1024); it now passes.
// FormatConfig: unpack_A slots = prev (fp16), pack slots = next (fp32).

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t prev_src = (std::uint32_t)params.formats.unpack_A_src; // fp16
    const std::uint32_t prev_dst = (std::uint32_t)params.formats.unpack_A_dst; // fp16
    const std::uint32_t next_src = (std::uint32_t)params.formats.pack_src;     // fp32
    const std::uint32_t next_dst = (std::uint32_t)params.formats.pack_dst;     // fp32

    constexpr std::uint32_t SIZE      = 16 * 16 * 4;
    constexpr std::uint32_t num_faces = 4;

    if (params.CONFIGURE_TEST_RUN_IDX == 0)
    {
        // CONTROL: configure DIRECTLY to next (fp32). ch1 Z-stride committed for fp32 -> asserts must pass.
        // Proves the fp32 path + the stride asserts are sound (isolates any failure to the reconfig path).
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            next_src, next_src, next_dst, next_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces, SIZE, SIZE);
    }
    else
    {
        // BUGGY PATH: configure to prev (fp16), then a format-only reconfig to next (fp32) with geometry OFF
        // -- exactly what reconfig_data_format<...>() does. The ch1 Z-stride is NOT re-committed.
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            prev_src, prev_src, prev_dst, prev_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces, SIZE, SIZE);

        _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false>(
            next_src, next_dst, SIZE, FACE_R_DIM, num_faces);
        _llk_unpack_reconfig_data_format_srcb_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false>(
            next_src, next_dst, SIZE, FACE_R_DIM, num_faces);
    }

    // Read back the ch1 Z-stride registers and compare against the NEW format's canonical value.
    tensix_sync();
    for (std::uint32_t i = 0; i < 10; i++)
    {
        asm volatile("nop");
    }
    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer();

    const std::uint32_t actual_a_z   = cfg[UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32];
    const std::uint32_t expected_a_z = canonical_unpA_z_stride(next_dst) << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT;
    LLK_ASSERT(
        actual_a_z == expected_a_z,
        "BUG: srcA ch1 Z-stride is stale after a format-only reconfig (fp16->fp32); strides were not re-committed.");

    const std::uint32_t actual_b_z   = cfg[UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32];
    const std::uint32_t expected_b_z = (datum_size_in_bytes(next_dst) * FACE_C_DIM * FACE_R_DIM)
                                       << UNP1_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT;
    LLK_ASSERT(
        actual_b_z == expected_b_z,
        "BUG: srcB ch1 Z-stride is stale after a format-only reconfig (fp16->fp32); strides were not re-committed.");
}

#endif

#ifdef LLK_TRISC_MATH

void run_kernel(RUNTIME_PARAMETERS params)
{
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
{
}

#endif
