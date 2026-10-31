// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compile-time contract of the HAL thread-CFG write paths. Every public spelling of a thread-CFG
// write must accept an ordinary thread register and must refuse the CFG state ID: the HAL emits a
// bare SETC16, and a state-ID write needs the padding and the software bank-tracker update that
// only ckernel::flip_cfg_state_id provides (see SETC16.md, instruction scheduling).
//
// build.h selects the register (HAL_CFG_PROBE_FIELD) and the spelling (HAL_CFG_PROBE_<SPELLING>).
// The test compiles this kernel and asserts on the compiler's verdict; it never runs on a device.

#include <cstdint>

#include "cfg.h"
#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

void run_kernel(RUNTIME_PARAMETERS params)
{
    using hal::cfg::Access;
    using hal::cfg::Sec;

    // A value the compiler cannot fold, so the runtime spellings stay runtime.
    [[maybe_unused]] const std::uint32_t value = *reinterpret_cast<volatile std::uint32_t*>(params.buffer_A[0]) & 1u;

#if defined(HAL_CFG_PROBE_CONSTANT)
    hal::cfg::write<Access::TensixCfgUnit, HAL_CFG_PROBE_FIELD, Sec::S0, 1>();
#elif defined(HAL_CFG_PROBE_RUNTIME)
    hal::cfg::write<Access::TensixCfgUnit, HAL_CFG_PROBE_FIELD, Sec::S0>(value);
#elif defined(HAL_CFG_PROBE_PREPACKED)
    hal::cfg::write<Access::TensixCfgUnit, HAL_CFG_PROBE_FIELD>(0 /*section*/, 1 /*value*/);
#elif defined(HAL_CFG_PROBE_GROUPED)
    hal::cfg::write<Access::TensixCfgUnit>(hal::cfg::set<HAL_CFG_PROBE_FIELD, Sec::S0>(value));
#else
#error "build.h must select one HAL_CFG_PROBE_<SPELLING>"
#endif
}
