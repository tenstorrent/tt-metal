// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "ckernel_trisc_common.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "pad_row_section.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"
#include "quasar_test_common.h"
#include "tensix.h"
#include "tensix_types.h"

constexpr std::uint32_t GATHER_ELEM_CNT = 32; // smallest gather the unpacker / math path supports

constexpr std::uint32_t face_row0_l1_row_16(const std::uint32_t face)
{
    return face * FACE_ELEM_CNT / FACE_COLS;
}

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

// Merge index vector for face FACE, kept in V0 (face 0) / V2 (face 1) at E8 M2.
template <std::uint32_t FACE>
constexpr vreg MERGE_IDX_VREG = FACE == 0 ? V0 : V2;

// Build the merge indices in L1 scratch and load them into MERGE_IDX_VREG<FACE>. Lanes 16-31 land in a
// dest row that is never packed; they read the scalar so every index stays inside the face row.
template <std::uint32_t FACE>
inline void load_merge_idx(volatile std::uint8_t* scratch)
{
    if constexpr (face_padded(FACE))
    {
        for (std::uint32_t j = 0; j < GATHER_ELEM_CNT; ++j)
        {
            const bool keep = j < seg_start(FACE) || (j >= seg_end(FACE) && j < FACE_COLS);
            scratch[j]      = static_cast<std::uint8_t>(keep ? j : seg_start(FACE));
        }
        asm volatile("fence" ::: "memory");
        vector_load<MERGE_IDX_VREG<FACE>>(const_cast<const std::uint8_t*>(scratch));
    }
}

template <std::uint32_t FACE>
inline void gather_pad_face_row(std::uint8_t* unpack_arg_mbox)
{
    if constexpr (face_padded(FACE))
    {
        vector_store<MERGE_IDX_VREG<FACE>>(unpack_arg_mbox);
        _llk_unpack_gather_compress_(face_row0_l1_row_16(FACE));
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_A         = params.buffer_A;   // scalar: low 16 bits of the first u32
    const Operand& buffer_B         = params.buffer_B;   // unused input, reused as L1 scratch for the merge indices
    const Operand& buffer_Res       = params.buffer_Res; // pad tile: gather source and pack destination
#endif
    auto unpack_arg_mbox = (std::uint8_t*)UNPACKER_ARG_MAILBOX_BASE1;
    std::uint16_t scalar_val;
    volatile std::uint16_t* tile;

    {
        ZONE_SCOPED("INIT")
        scalar_val = static_cast<std::uint16_t>(*reinterpret_cast<volatile const std::uint32_t*>(buffer_A[0]));
        tile       = reinterpret_cast<volatile std::uint16_t*>(buffer_Res[0]);
        program_bd_slot(SRC_BD_SLOT, L1_ADDRESS(buffer_Res[0]), formats.unpack_B_src, FACE_COLS, 1 /*y_dim*/);
        _llk_unpack_gather_compress_init_<GatherCompressModeSelect::GATHER>(SRC_BD_SLOT, formats.unpack_B_dst);

        auto idx_scratch = reinterpret_cast<volatile std::uint8_t*>(buffer_B[0]);
        (void)vsetvl<E8, M2, GATHER_ELEM_CNT>();
        load_merge_idx<0>(idx_scratch);
        load_merge_idx<1>(idx_scratch + GATHER_ELEM_CNT);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // One SrcB dvalid per gather invocation (UNPACR1_STRIDE with SetDatValid), one per padded face.
            _perf_unpack_loop_set_valid<false /*set_a*/, true /*set_b*/>(LOOP_FACTOR * NUM_PADDED_FACES);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                (void)vsetvl<E8, M2, GATHER_ELEM_CNT>();

                // Write the scalar to the first pad element of each padded face row and retire it before
                // the unpacker reads L1. The previous iteration's pack writes the same values to these rows
                // (scalar at pad lanes, unchanged data elsewhere), so the overlap is benign.
                if constexpr (face_padded(0))
                {
                    tile[seg_elem_offset(0)] = scalar_val;
                }
                if constexpr (face_padded(1))
                {
                    tile[seg_elem_offset(1)] = scalar_val;
                }
                asm volatile("fence" ::: "memory");

                gather_pad_face_row<0>(unpack_arg_mbox);
                gather_pad_face_row<1>(unpack_arg_mbox);
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
            // Consume the one SrcB dvalid each gather unpack produces, one per padded face.
            _perf_math_loop_clear_valid<false /*clear_a*/, true /*clear_b*/>(LOOP_FACTOR * NUM_PADDED_FACES);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR * NUM_PADDED_FACES; loop++)
            {
                _llk_math_gather_compress_();
            }
        }
        else
        {
            // One dest section per padded face: the packer drains each merged row before the next gather.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR * NUM_PADDED_FACES; loop++)
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

// Pack one 16-element row per call. The L1 BD is then 1x16, so L1 tile index i lands at element 16 * i:
// row 0 of face f is L1 tile index f * FACE_COLS (FACE_ELEM_CNT / 16 elements per face).
constexpr std::uint32_t face_row0_l1_tile(const std::uint32_t face)
{
    return face * FACE_ELEM_CNT / FACE_COLS;
}

// In end-to-end runs the pack LLK releases dest back to math after each face row.
template <std::uint32_t FACE>
inline void pack_pad_face_row()
{
    if constexpr (face_padded(FACE))
    {
        _llk_pack_gather_compress_(face_row0_l1_tile(FACE));
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_Res       = params.buffer_Res; // pad tile: 32x32 16-bit, only padded face rows of row 0 are rewritten
#endif

    // Dest row 0 holds the merged face row (gather lanes 0-15).
    constexpr ckernel::TensorShape res_shape = ckernel::make_tensor_shape(1, FACE_COLS, 1, 1);

    {
        ZONE_SCOPED("INIT")
        // PACK_ISOLATE and L1_CONGESTION have no dest producer.
        // Explicitly clear wait_mask because CFG persists across run types.
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
                pack_pad_face_row<0>();
                pack_pad_face_row<1>();
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                pack_pad_face_row<0>();
                pack_pad_face_row<1>();
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
