// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Build-only: an unqualified `cfg_state_id` must resolve from a global-scope function that has
// `using namespace ckernel;` in scope -- the shape of every llk_lib header.
//
// The variable is ckernel::cfg_state_id (declared in ckernel.h, defined in the firmware and in the
// harness's ckernel_helper.h). A second, global-scope `extern cfg_state_id` with no definition
// makes that unqualified name ambiguous, so no llk_lib function could ever read the bank it is
// about to configure. Both architectures share the same headers; test_cfg_state_id_scope.py
// builds this driver for each and never runs it.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_globals.h"
#include "llk_defs.h"
#include "params.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

using namespace ckernel;

// The probe: same scope and lookup as an llk_lib function.
inline std::uint32_t current_cfg_state_id()
{
    return cfg_state_id;
}

#ifdef LLK_TRISC_UNPACK

void run_kernel([[maybe_unused]] RUNTIME_PARAMETERS params)
{
    [[maybe_unused]] volatile std::uint32_t state = current_cfg_state_id();
}

#endif

#ifdef LLK_TRISC_MATH

void run_kernel([[maybe_unused]] RUNTIME_PARAMETERS params)
{
    [[maybe_unused]] volatile std::uint32_t state = current_cfg_state_id();
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel([[maybe_unused]] RUNTIME_PARAMETERS params)
{
    [[maybe_unused]] volatile std::uint32_t state = current_cfg_state_id();
}

#endif
