// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Regression test for `mailbox_read`: it returns only once the message has arrived, even when the
// caller never uses the value. Code after the call may then rely on everything the sender did
// before its `mailbox_write`.
//
// Each cycle MATH programs a config field, waits for the write to complete, and then releases
// UNPACK and PACK through their mailboxes. The released threads discard the message and read the
// field back, which must already hold the value MATH programmed in that cycle.

#include <cstdint>

#include "build.h"
#include "ckernel.h"
#include "ckernel_defs.h"

using namespace ckernel;

// Globals the harness links against.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

namespace
{
constexpr std::uint32_t CYCLES = 256;
constexpr std::uint32_t READY  = 0x1;

constexpr std::uint32_t expected_field(const std::uint32_t cycle)
{
    return cycle & 0x1;
}

inline std::uint32_t read_field(volatile std::uint32_t tt_reg_ptr* cfg)
{
    return (cfg[PCK_DEST_RD_CTRL_Read_32b_data_ADDR32] & PCK_DEST_RD_CTRL_Read_32b_data_MASK) >> PCK_DEST_RD_CTRL_Read_32b_data_SHAMT;
}

// Runs the released thread's side of every cycle and counts the cycles in which the field already
// holds MATH's value once the release message has been read.
template <ThreadId thread_id>
inline std::uint32_t released_side()
{
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    std::uint32_t seen                     = 0;
    for (std::uint32_t i = 0; i < CYCLES; i++)
    {
        mailbox_write(ThreadId::MathThreadId, READY);
        [[maybe_unused]] const std::uint32_t release = mailbox_read(ThreadId::MathThreadId);
        seen += read_field(cfg) == expected_field(i) ? 1 : 0;
    }
    return seen;
}
} // namespace

#ifdef LLK_TRISC_UNPACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t seen = released_side<ThreadId::UnpackThreadId>();

    auto* res = reinterpret_cast<std::uint32_t*>(params.buffer_Res[0]);
    res[2]    = seen;
}

#endif

#ifdef LLK_TRISC_MATH

void run_kernel(RUNTIME_PARAMETERS)
{
    for (std::uint32_t i = 0; i < CYCLES; i++)
    {
        // Wait for both released threads with blocking loads, so this side's ordering does not
        // depend on the function under test.
        load_blocking(&mailbox_base[ThreadId::UnpackThreadId][0]);
        load_blocking(&mailbox_base[ThreadId::PackThreadId][0]);

        cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Read_32b_data_RMW>(expected_field(i));
        tensix_sync();

        mailbox_write(ThreadId::UnpackThreadId, i);
        mailbox_write(ThreadId::PackThreadId, i);
    }

    // Leave the field as the variant was built.
    cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Read_32b_data_RMW>(is_fp32_dest_acc_en);
    tensix_sync();
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t seen = released_side<ThreadId::PackThreadId>();

    auto* res = reinterpret_cast<std::uint32_t*>(params.buffer_Res[0]);
    res[0]    = CYCLES;
    res[1]    = seen;
}

#endif
