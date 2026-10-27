// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// PACK_ISOLATE benchmark for row packing to an L1 address. All TRISCs enter the same
// profiler zones; only PACK does work. No unpack/math handshake is timed.
#if defined(ARCH_QUASAR) && defined(SPEED_OF_LIGHT)
// The standalone harness emits its logical FP8 name in constant-format build.h;
// Quasar calls the same E4M3 hardware format Fp8P. Keep this alias local to the
// benchmark driver; runtime-format builds pass Fp8P's value (16) from
// QUASAR_DATA_FORMAT_ENUM_VALUES.
#define Fp8_e4m3 Fp8P
#endif

#include <cstdint>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "perf.h"
#include "profiler.h"

#ifndef ARCH_QUASAR
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;
#endif

#ifdef LLK_TRISC_PACK
#include "llk_pack_common.h"
#include "llk_pack_rows.h"
#ifdef ARCH_QUASAR
#include "llk_bfd_alloc.h"
#include "quasar_test_common.h"
#else
#include "llk_lib_pack_wrappers.h"
#endif
#endif

#include "params.h"

using ckernel::tensix_sync;

void run_kernel(RUNTIME_PARAMETERS params)
{
    (void)params;
#if defined(LLK_TRISC_PACK)
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR      = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT         = params.TILE_CNT;
    const std::uint32_t NUM_ROWS_TO_PACK = params.NUM_ROWS_TO_PACK;
#endif
#ifdef ARCH_QUASAR
    std::uint8_t bfd_pack = 0;
#endif
#endif

    {
        START_PERF_MEASURE("INIT")
#ifdef LLK_TRISC_PACK
#ifdef ARCH_QUASAR
        // Match the existing Quasar pack-isolate test: disable the DEST wait mask.
        set_up_zero_dest_dvalid_handshake_for_pack();
        bfd_pack = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(
            ckernel::DEFAULT_TENSOR_SHAPE, 0 /*L1 base, set per call*/, formats.pack_dst);
        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
        _llk_pack_dest_init_<p_pacr::PACK0, dest_sync>();
        _llk_pack_rows_init_(bfd_pack, NUM_ROWS_TO_PACK);
#else
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(
            formats.pack_src, formats.pack_dst, 1024 /*datums per tile*/, FACE_R_DIM, TILE_C_DIM, 4 /*num_faces*/);
        _llk_pack_init_wrapper_<ckernel::PackMode::Default, false /*zero_output*/>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, 4 /*num_faces*/);
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_pack_rows_init_(NUM_ROWS_TO_PACK);
#endif
#endif
        PROFILER_SYNC();
    }

#ifdef LLK_TRISC_PACK
    // Warm every destination slot outside the measured loop. As in existing
    // PACK_ISOLATE tests, DEST contents are irrelevant; this measures execution.
    for (std::uint32_t tile = 0; tile < TILE_CNT; tile++)
    {
        const std::uint32_t address_16b = (PERF_OUTPUT + tile * 4096) / 16;
#ifdef ARCH_QUASAR
        _llk_pack_rows_(bfd_pack, 0 /*DEST tile*/, address_16b);
#else
        _llk_pack_rows_(0 /*DEST tile*/, address_16b - 1);
#endif
    }
    PROFILER_SYNC();
#endif

    {
        START_PERF_MEASURE("TILE_LOOP")
#ifdef LLK_TRISC_PACK
        // Identical 16-slot ring and loop geometry on both architectures. Each
        // slot is 4096 bytes, enough for 64 rows even with Float32 output.
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            for (std::uint32_t tile = 0; tile < TILE_CNT; tile++)
            {
                const std::uint32_t address_16b = (PERF_OUTPUT + tile * 4096) / 16;
#ifdef ARCH_QUASAR
                _llk_pack_rows_(bfd_pack, 0 /*DEST tile*/, address_16b);
#else
                _llk_pack_rows_(0 /*DEST tile*/, address_16b - 1);
#endif
            }
        }
#endif
        // Drain issued work before the profiler records the zone end.
        PROFILER_SYNC();
    }
}
