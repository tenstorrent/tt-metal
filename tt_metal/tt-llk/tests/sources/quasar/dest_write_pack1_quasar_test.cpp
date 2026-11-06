// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Test-only alternative output path, not a Dest preload replacement.
#include "ckernel.h"
#include "counters.h"
#include "dest_write_experiments_quasar.h"
#include "llk_memory_checks.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"
#include "quasar_test_common.h"
#ifdef LLK_TRISC_ISOLATE_SFPU
#include "llk_sfpu_srcs_api.h"
using Math = dwp::AffineChain<DWP_MATH_DEPTH>;
using Op   = ckernel::sfpu::SrcsUnary<Math, sfpi::DataLayout::F32, ckernel::sfpu::SfpuImpl::Sfpi>;

void run_kernel(RUNTIME_PARAMETERS params)
{
    const auto& formats      = params.formats;
    constexpr unsigned BANKS = DWP_NUM_BLOCKS * DWP_LOOP_FACTOR;
    static_assert(DWP_PACK1 && DWP_KERNEL_OP == dwp::OP_CHAIN);
    {
        START_PERF_MEASURE("INIT")
        Op::init(
            L1_ADDRESS(params.buffer_A[0]),
            static_cast<DataFormat>(formats.unpack_A_src),
            L1_ADDRESS(params.buffer_Res[0]),
            static_cast<DataFormat>(formats.pack_dst));
        dwp::snapshot_hash(params.buffer_C[0]);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        dwp::BankStamps<DWP_STAMP_SYNC, DWP_STAMP_EVERY> stamps(params.buffer_C[0], BANKS);
        stamps.start();
        const auto unp  = ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp2_Slice0>();
        const auto pack = ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack1>();
        for (unsigned bank = 0; bank < BANKS; ++bank)
        {
            for (unsigned tile = 0; tile < DWP_TILES_PER_BANK; ++tile)
            {
                const auto offset = ((bank % DWP_NUM_BLOCKS) * DWP_TILES_PER_BANK + tile) * dwp::SRCS_SLICES_PER_TILE;
                _llk_unpack_srcs_<1>(unp, offset);
                // Queue pack first to avoid blocking behind a full SFPU instruction buffer.
                _llk_pack_srcs_<1>(pack, offset);
                for (unsigned slice = 0; slice < dwp::SRCS_SLICES_PER_TILE; ++slice)
                {
                    Op::calculate();
                    _llk_math_eltwise_sfpu_srcs_clear_vlds_<true, true>();
                }
            }
            stamps.bank_done(bank);
        }
        wait_sfpu_idle();
        wait_unpack_idle();
        stamps.end();
        PROFILER_SYNC();
    }
}
#else
void run_kernel(RUNTIME_PARAMETERS params)
{
    (void)params;
    {
        START_PERF_MEASURE("INIT")
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
    }
}
#endif
