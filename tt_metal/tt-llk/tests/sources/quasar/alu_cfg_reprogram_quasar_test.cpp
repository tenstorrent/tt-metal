// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Microbenchmark: cost of one ALU format (re)configure call on the math thread.
//
// ALU_CFG_MODE selects the implementation:
//   0 = tracked:       _configure_alu_cfg_ (early return when already programmed, writes only changed words)
//   1 = unconditional: build the words and store all NUM_WORDS_ALU_FORMAT of them on every call
//   2 = tracked_no_diff: same early return as tracked; on a miss store all words (no per-word diff, no word cache)
// ALU_CFG_PATTERN selects what each call requests:
//   0 = same:          every call requests the state already programmed
//   1 = alt_formats:   DEFAULT, SrcA/SrcB alternate between formats.math and ALU_CFG_ALT_FORMAT
//   2 = alt_type:      formats.math, config type alternates DEFAULT / MOV_OPS_EXPLICIT_FORMAT
//
// TILE_LOOP runs LOOP_FACTOR calls with TILE_CNT = 1, so the post-processed TILE_LOOP figure is cycles per call.
// Unpack and pack do no work; only the math thread is measured.

#include <cstdint>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "perf.h"
#include "profiler.h"
#include "sfpu_stub.h"

using namespace ckernel;

#ifdef LLK_TRISC_UNPACK

#include "params.h"

void run_kernel([[maybe_unused]] RUNTIME_PARAMETERS params)
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

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "params.h"

namespace
{

// Baseline: what _configure_alu_cfg_ did before tracking - rebuild and store every word, keep no state.
template <AluCfgType TYPE, bool EN_32BIT_DEST>
inline void configure_alu_cfg_unconditional(const DataFormat srca, const DataFormat srcb)
{
    alu_config_u target;
    _build_alu_words_<TYPE, EN_32BIT_DEST>(srca, srcb, target);
    for (std::uint32_t i = 0; i < NUM_WORDS_ALU_FORMAT; i++)
    {
        cfg[ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32 + i] = target.val[i];
    }
}

// Candidate: _configure_alu_cfg_ without the per-word diff. Keeps the state early return; a miss stores every word
// and records only the requested state, so alu_cfg.words is neither read nor written.
template <AluCfgType TYPE, bool EN_32BIT_DEST>
inline void configure_alu_cfg_tracked_no_diff(const DataFormat srca, const DataFormat srcb)
{
    const AluCfgState wanted {TYPE, srca, srcb};
    if (alu_cfg.programmed == wanted)
    {
        return;
    }
    configure_alu_cfg_unconditional<TYPE, EN_32BIT_DEST>(srca, srcb);
    alu_cfg.programmed = wanted;
}

template <AluCfgType TYPE>
inline void configure_alu_cfg(const DataFormat srca, const DataFormat srcb)
{
    if constexpr (ALU_CFG_MODE == 0)
    {
        _configure_alu_cfg_<TYPE, is_fp32_dest_acc_en>(srca, srcb);
    }
    else if constexpr (ALU_CFG_MODE == 1)
    {
        configure_alu_cfg_unconditional<TYPE, is_fp32_dest_acc_en>(srca, srcb);
    }
    else
    {
        configure_alu_cfg_tracked_no_diff<TYPE, is_fp32_dest_acc_en>(srca, srcb);
    }
}

} // namespace

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif

    // Read through volatile so the compiler cannot prove the requested state is loop-invariant and fold the
    // tracker's compare (or the unconditional stores' word build) out of the loop.
    volatile DataFormat fmt_slots[2] = {static_cast<DataFormat>(formats.math), static_cast<DataFormat>(ALU_CFG_ALT_FORMAT)};

    {
        START_PERF_MEASURE("INIT")
        _llk_math_srcAB_hw_configure_<is_fp32_dest_acc_en>(fmt_slots[0], fmt_slots[0]);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        for (std::uint32_t i = 0; i < LOOP_FACTOR; i++)
        {
            const std::uint32_t odd = i & 1;
            if constexpr (ALU_CFG_PATTERN == 0)
            {
                const DataFormat fmt = fmt_slots[0];
                configure_alu_cfg<AluCfgType::DEFAULT>(fmt, fmt);
            }
            else if constexpr (ALU_CFG_PATTERN == 1)
            {
                const DataFormat fmt = fmt_slots[odd];
                configure_alu_cfg<AluCfgType::DEFAULT>(fmt, fmt);
            }
            else
            {
                const DataFormat fmt = fmt_slots[0];
                if (odd)
                {
                    configure_alu_cfg<AluCfgType::MOV_OPS_EXPLICIT_FORMAT>(fmt, fmt);
                }
                else
                {
                    configure_alu_cfg<AluCfgType::DEFAULT>(fmt, fmt);
                }
            }
            asm volatile("" ::: "memory");
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "params.h"

void run_kernel([[maybe_unused]] RUNTIME_PARAMETERS params)
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
