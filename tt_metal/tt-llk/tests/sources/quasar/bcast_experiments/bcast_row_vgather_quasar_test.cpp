// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "ckernel_trisc_common.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"
#include "quasar_test_common.h"
#include "tensix.h"
#include "tensix_types.h"

constexpr std::uint32_t GATHER_ELEM_CNT = 32; // smallest gather the unpacker / math path supports
constexpr std::uint32_t FACE_ROW_ELEMS  = 16;

using namespace ckernel;
using ckernel::trisc::bd_table;

constexpr std::uint32_t SRC_BD_SLOT = 17;

inline void program_bd_slot(std::uint32_t slot, std::uint32_t l1_addr_16B, std::uint32_t format, std::uint32_t x_dim, std::uint32_t y_dim)
{
    buffer_descriptor_u bd_val;
    bd_val.words[0] = bd_val.words[1] = bd_val.words[2] = bd_val.words[3] = 0;

    bd_val.f.l1_addr_16B = l1_addr_16B;
    bd_val.f.format      = static_cast<std::uint8_t>(format);
    bd_val.f.x_dim       = x_dim;
    bd_val.f.y_dim       = y_dim;
    bd_val.f.z_dim       = 1;

    bd_table[slot].words[0] = bd_val.words[0];
    bd_table[slot].words[1] = bd_val.words[1];
    bd_table[slot].words[2] = bd_val.words[2];
}

#ifdef LLK_TRISC_UNPACK

#include "ckernel_vector.h"
#include "llk_unpack_gather_compress.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_A         = params.buffer_A;   // scalar: first u32 holds the double-packed 16-bit value
    const Operand& buffer_Res       = params.buffer_Res; // bcast tile: element 0 is the gather source
#endif
    auto unpack_arg_mbox = (std::uint8_t*)UNPACKER_ARG_MAILBOX_BASE1;
    std::uint16_t scalar_val;
    volatile std::uint16_t* tile;

    {
        ZONE_SCOPED("INIT")
        scalar_val = static_cast<std::uint16_t>(*reinterpret_cast<volatile const std::uint32_t*>(buffer_A[0]));
        tile       = reinterpret_cast<volatile std::uint16_t*>(buffer_Res[0]);
        program_bd_slot(SRC_BD_SLOT, L1_ADDRESS(buffer_Res[0]), formats.unpack_B_src, FACE_ROW_ELEMS, 1 /*y_dim*/);
        _llk_unpack_gather_compress_init_<GatherCompressModeSelect::GATHER>(SRC_BD_SLOT, formats.unpack_B_dst);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // One SrcB dvalid per gather invocation (UNPACR1_STRIDE with SetDatValid).
            _perf_unpack_loop_set_valid<false /*set_a*/, true /*set_b*/>(LOOP_FACTOR);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                // All-zero gather indices: every output lane reads source element 0.
                (void)vsetvl<E8, M2, GATHER_ELEM_CNT>();
                asm volatile("vmv.v.i v0, 0\n");
                vector_store<V0>(unpack_arg_mbox);

                // Write the scalar to element 0 (row 0, face 0) and retire it before the unpacker reads L1.
                // The previous iteration's pack writes the same value here, so the overlap is benign.
                tile[0] = scalar_val;
                asm volatile("fence" ::: "memory");

                _llk_unpack_gather_compress_();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_gather_compress.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif

    {
        ZONE_SCOPED("INIT")
        // Only end-to-end and math-isolate runs use the FPU->PACK dest-dvalid handshake.
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
        }
        _llk_math_gather_compress_init_(formats.unpack_B_dst);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            // Consume the one SrcB dvalid each gather unpack produces.
            _perf_math_loop_clear_valid<false /*clear_a*/, true /*clear_b*/>(LOOP_FACTOR);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_math_gather_compress_();
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_math_gather_compress_();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_bfd_alloc.h"
#include "llk_pack_common.h"
#include "llk_pack_gather_compress.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_Res       = params.buffer_Res; // bcast tile: 32x32 16-bit, only row 0 is written
#endif

    constexpr ckernel::TensorShape res_shape   = ckernel::make_tensor_shape(1, FACE_ROW_ELEMS, 1, 1);
    constexpr std::uint32_t FACE0_ROW0_L1_TILE = 0;
    constexpr std::uint32_t FACE1_ROW0_L1_TILE = FACE_ROW_ELEMS;

    {
        ZONE_SCOPED("INIT")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            set_up_zero_dest_dvalid_handshake_for_pack();
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();
        }

        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(res_shape, L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
        _llk_pack_gather_compress_init_(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>());
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            // No dest-dvalid section_done: pack runs without a math handshake.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_pack_gather_compress_(FACE0_ROW0_L1_TILE);
                _llk_pack_gather_compress_(FACE1_ROW0_L1_TILE);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_pack_gather_compress_(FACE0_ROW0_L1_TILE);
                _llk_pack_gather_compress_(FACE1_ROW0_L1_TILE);
                _llk_pack_dest_dvalid_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_ISOLATE_SFPU

void run_kernel(RUNTIME_PARAMETERS params)
{
    (void)params;
}

#endif
