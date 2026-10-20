// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Regression test for the mid-kernel FP32 dest-acc handshake (`_llk_set_fp32_dest_acc_`): when it
// returns on UNPACK and PACK, the dest-acc config MATH just programmed must already be in effect.
//
// Built with LLK_ASSERT compiled out, as tt-metal builds kernels by default, so the handshake runs
// with its mailbox values unchecked, exactly as it does in production. The harness enables asserts
// by default, so this file switches them off itself rather than depending on how it is run.
#undef ENABLE_LLK_ASSERT

#include <cstdint>

#include "build.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_fp32_dest_acc.h"

using namespace ckernel;

// Globals the harness links against.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

namespace
{
constexpr std::uint32_t CYCLES = 256;

inline bool dest_acc_field_set(volatile std::uint32_t tt_reg_ptr* cfg)
{
    return (cfg[PCK_DEST_RD_CTRL_Read_32b_data_ADDR32] & PCK_DEST_RD_CTRL_Read_32b_data_MASK) != 0;
}

// Runs the released thread's side of every enable/disable cycle and counts the calls after which
// the field already holds the value MATH programmed.
template <ThreadId thread_id>
inline void released_side(std::uint32_t& enabled_seen, std::uint32_t& disabled_seen)
{
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    for (std::uint32_t i = 0; i < CYCLES; i++)
    {
        _llk_set_fp32_dest_acc_<thread_id>();
        enabled_seen += dest_acc_field_set(cfg) ? 1 : 0;
        _llk_set_fp32_dest_acc_<thread_id>();
        disabled_seen += dest_acc_field_set(cfg) ? 0 : 1;
    }
}
} // namespace

#ifdef LLK_TRISC_UNPACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    std::uint32_t enabled_seen  = 0;
    std::uint32_t disabled_seen = 0;
    released_side<ThreadId::UnpackThreadId>(enabled_seen, disabled_seen);

    auto* res = reinterpret_cast<std::uint32_t*>(params.buffer_Res[0]);
    res[3]    = enabled_seen;
    res[4]    = disabled_seen;
}

#endif

#ifdef LLK_TRISC_MATH

void run_kernel(RUNTIME_PARAMETERS)
{
    for (std::uint32_t i = 0; i < CYCLES; i++)
    {
        _llk_set_fp32_dest_acc_<ThreadId::MathThreadId>(true /*enable*/);
        _llk_set_fp32_dest_acc_<ThreadId::MathThreadId>(false /*enable*/);
    }

    // Leave the dest-acc fields as the variant was built.
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(is_fp32_dest_acc_en);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW>(is_fp32_dest_acc_en);
    cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Read_32b_data_RMW>(is_fp32_dest_acc_en);
    tensix_sync();
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    std::uint32_t enabled_seen  = 0;
    std::uint32_t disabled_seen = 0;
    released_side<ThreadId::PackThreadId>(enabled_seen, disabled_seen);

    auto* res = reinterpret_cast<std::uint32_t*>(params.buffer_Res[0]);
    res[0]    = CYCLES;
    res[1]    = enabled_seen;
    res[2]    = disabled_seen;
}

#endif
