// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Math-thread cycles per tile of the GGUF decoders, the DEST-fed matmul and the unpack-to-DEST page copy (Blackhole,
// 32-bit DEST, MATH_ISOLATE).

#include <cstdint>

#include "build.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

using namespace ckernel;

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static_assert(PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE, "gguf_decode_mm_perf: MATH_ISOLATE only");
static_assert(is_fp32_dest_acc_en, "gguf_decode_mm_perf: 32-bit DEST");

// The math thread's TILE_LOOP runs LOOP_FACTOR * TILE_CNT steps of GGUF_STEP:
//   0 decode, 1 params, 2 body: sfpu_gguf_decode_tile<GGUF_FMT> with both phases, phase 1 or phase 2, over the tile
//     indices 0..TILE_CNT-1 of a group (the timing does not depend on the page's bits); decode and params include
//     sfpu_gguf_decode_group once per group;
//   3 matmul: _llk_math_dest_srca_mm_ with GGUF_IN0_ROWS rows at MATH_FIDELITY, SrcA and SrcB set valid without data;
//   4 copy: the unpack-to-DEST copy of a GGUF_PAGE_FACES-face 32-bit page, from the post to the end of the wait.
constexpr bool STEP_DECODE = GGUF_STEP <= 2;
constexpr bool STEP_MATMUL = GGUF_STEP == 3;
constexpr bool STEP_COPY   = GGUF_STEP == 4;
constexpr std::uint32_t STEPS = LOOP_FACTOR * TILE_CNT;

constexpr std::uint32_t PAGE_FORMAT = ckernel::to_underlying(DataFormat::UInt32);
constexpr std::uint32_t IN0_FORMAT  = ckernel::to_underlying(DataFormat::Float16_b);
constexpr std::uint32_t OUT_FORMAT  = ckernel::to_underlying(DataFormat::Float32);
constexpr std::uint32_t IN0_FACES   = GGUF_IN0_ROWS <= 16 ? 2 : 4;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            PAGE_FORMAT, IN0_FORMAT, PAGE_FORMAT, IN0_FORMAT, FACE_R_DIM, FACE_R_DIM, GGUF_PAGE_FACES, IN0_FACES);
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, true>(
            0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, GGUF_PAGE_FACES), PAGE_FORMAT, PAGE_FORMAT);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (STEP_MATMUL)
        {
            _perf_unpack_loop_set_valid<true, true>(STEPS);
        }
        else if constexpr (STEP_COPY)
        {
            for (std::uint32_t i = 0; i < STEPS; i++)
            {
                _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, true>(
                    PERF_ADDRESS(PERF_INPUT_A, i), PAGE_FORMAT, PAGE_FORMAT);
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "experimental/ckernel_gguf_page.h"
#include "experimental/llk_math_dest_srca_mm.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu/experimental/ckernel_sfpu_gguf_decode.h"

// A template, so that the format's layout is only instantiated for the decode steps.
template <std::uint32_t FMT>
inline void decode_steps()
{
    constexpr std::uint32_t group_tiles = gguf::page_layout<static_cast<gguf::Format>(FMT)>().group_tiles;
    for (std::uint32_t i = 0; i < STEPS; i++)
    {
        const std::uint32_t ti = i % TILE_CNT % group_tiles;
        if (GGUF_STEP != 2 && ti == 0)
        {
            sfpu::sfpu_gguf_decode_group<FMT>(0);
        }
        sfpu::sfpu_gguf_decode_tile<FMT, GGUF_STEP != 2, GGUF_STEP != 1>(0, ti);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncFull, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(OUT_FORMAT, IN0_FORMAT);
        if constexpr (STEP_DECODE)
        {
            _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
            sfpu::sfpu_gguf_decode_init<GGUF_FMT>();
        }
        else
        {
            _llk_math_dest_srca_mm_init_<MATH_FIDELITY, GGUF_IN0_ROWS>();
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (STEP_DECODE)
        {
            decode_steps<GGUF_FMT>();
        }
        else if constexpr (STEP_MATMUL)
        {
            for (std::uint32_t i = 0; i < STEPS; i++)
            {
                _llk_math_dest_srca_mm_<MATH_FIDELITY, GGUF_IN0_ROWS>(1, 0);
            }
        }
        else if constexpr (STEP_COPY)
        {
            for (std::uint32_t i = 0; i < STEPS; i++)
            {
                const std::uint32_t page = i & 1 ? 5 : 2;
                _llk_math_dest_srca_mm_unpack_post_(page);
                _llk_math_dest_srca_mm_unpack_wait_(page, GGUF_PAGE_FACES);
            }
        }
        PROFILER_SYNC();
    }
    if constexpr (!STEP_DECODE)
    {
        _llk_math_dest_srca_mm_uninit_();
    }
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        START_PERF_MEASURE("INIT")
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        PROFILER_SYNC();
    }
}

#endif
