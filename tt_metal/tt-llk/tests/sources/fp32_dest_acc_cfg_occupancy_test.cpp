// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Ordering test for the mid-kernel FP32 dest-acc handshake (`_llk_set_fp32_dest_acc_`).
//
// The invariant under test: when MATH releases UNPACK/PACK, the three dest-acc config writes it
// just issued must already be visible to them.
//
// Two sections, because they answer different questions.
//
// SECTION 1 -- the real function. All three threads call their specialization (enable, then
// disable) and PACK reads the field back after each. This exercises the shipped code path end to
// end; it does not attempt to break it.
//
// SECTION 2 -- the ordering mechanism, on a replica of MATH's sequence. The replica exists because
// the invariant cannot be stressed through the real function: its own preamble (`tensix_sync()`,
// then two blocking `mailbox_read`s) leaves MATH's Tensix pipe empty, which hands the config writes
// a head start nothing outside the function can close. The replica reproduces the sequence and adds
// the one thing that closes it -- issue occupancy in front of the writes.
//
// Why occupancy is the lever, and why it must be a MOP:
//   * `cfg_reg_rmw_tensix` expands to `RMWCIB`, a Tensix instruction. The RISC pushes it and moves
//     on; it does not wait for the write to issue, let alone land.
//   * A MOP is ONE RISC push that expands into many issued slots, so the RISC reaches the release
//     immediately while the writes are still queued behind the expansion.
//   * Pushing extra `RMWCIB`s instead does NOT work -- measured flat at 0/4096 up to depth 128.
//     The FIFO drains about as fast as the RISC can push, so extra pushes delay the release as much
//     as they delay the writes.
//   * Depths >= 512 hang the board. 128 already saturates, so the sweep stops there.
//
// The arms, and every one of them is load-bearing:
//   Shipped      -- replica with a `STALLWAIT` on the per-thread condition. Fails under occupancy.
//   DrainBefore  -- replica with the RISC-blocking drain before the release. Must hold at zero.
//   DrainAfter   -- POSITION CONTROL: the same drain, after the release. Must fail like Shipped.
//                   Without it, DrainBefore's zero is indistinguishable from "any added delay
//                   helps" rather than "the ordering is what helps".
//   Plumbing     -- never writes the field, so PACK must read stale every trial. Guards against a
//                   blind detector making every other zero unfalsifiable.
//   Direction    -- RISC work between the writes and the release. This is the PROTECTIVE direction
//                   (the writes are already pushed), so it must never create the race, and it is
//                   expected to rescue the occupied case.

#include <cstdint>

#include "build.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "llk_fp32_dest_acc.h"

using namespace ckernel;

// Globals the harness links against.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

namespace
{
constexpr std::uint32_t TRIALS      = 4096; // per mechanism arm
constexpr std::uint32_t REAL_TRIALS = 64;   // per real-function arm

// What MATH's replica does between its config writes and its release.
enum class Mode : std::uint32_t
{
    Shipped     = 0,
    DrainBefore = 1,
    DrainAfter  = 2,
};

// Mirrors the shipped mask, so the replica measures the shipped encoding.
constexpr std::uint32_t REPLICA_STALL = p_stall::STALL_UNPACK | p_stall::STALL_PACK | p_stall::STALL_MATH | p_stall::STALL_SFPU;

constexpr std::uint32_t PACK_READY = 0x52444902;
constexpr std::uint32_t MATH_DONE  = 0x444F4E02;

constexpr std::uint32_t NUM_OCC      = 4;
constexpr std::uint32_t OCC_DEPTH[4] = {0, 8, 32, 128};

constexpr std::uint32_t NUM_DIR     = 4;
constexpr std::uint32_t DIR_NOPS[4] = {0, 8, 64, 512};
constexpr std::uint32_t DIR_OCC     = 32; // a depth measured as saturated

// Unrolled at compile time so the zero point interposes nothing of its own. A runtime loop here
// would measure its own overhead instead of the gap.
template <std::uint32_t N>
inline void risc_nops()
{
    if constexpr (N > 0)
    {
        __asm__ __volatile__("nop");
        risc_nops<N - 1>();
    }
}
} // namespace

#ifdef LLK_TRISC_UNPACK

void run_kernel(RUNTIME_PARAMETERS)
{
    // Section 1 only: UNPACK is a participant in the real handshake. It takes no part in the
    // replica, which keeps MATH's release path to a single mailbox store and so makes the window
    // under test strictly tighter than the shipped three-thread form.
    for (std::uint32_t i = 0; i < REAL_TRIALS; i++)
    {
        _llk_set_fp32_dest_acc_<ThreadId::UnpackThreadId>();
        _llk_set_fp32_dest_acc_<ThreadId::UnpackThreadId>();
    }
}

#endif

#ifdef LLK_TRISC_MATH

template <std::uint32_t OCC, Mode MODE, std::uint32_t RISC_NOPS = 0>
static void replica_arm()
{
    if constexpr (OCC > 0)
    {
        // Programmed once, outside the measured loop, so none of its own config writes land in the
        // window. Each inner iteration issues loop_op0 and loop_op1.
        ckernel_template tmpl(1, OCC, TT_OP_NOP);
        tmpl.program();
    }

    for (std::uint32_t trial = 0; trial < TRIALS; trial++)
    {
        // Arm the field to its old value and confirm it landed, so a later stale reading can only
        // mean this trial's write has not been processed.
        cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Read_32b_data_RMW>(0);
        tensix_sync();

        if (mailbox_read(ThreadId::PackThreadId) != PACK_READY)
        {
            return; // desync; the host sees a zeroed result and fails
        }

        if constexpr (OCC > 0)
        {
            ckernel_template::run(); // one push, many issued slots
        }

        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(1);
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW>(1);
        cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Read_32b_data_RMW>(1);
        TTI_STALLWAIT(REPLICA_STALL, p_stall::TRISC_CFG);

        if constexpr (MODE == Mode::DrainBefore)
        {
            tensix_sync();
        }

        risc_nops<RISC_NOPS>();

        mailbox_write(ThreadId::PackThreadId, MATH_DONE);

        if constexpr (MODE == Mode::DrainAfter)
        {
            tensix_sync();
        }
    }
}

static void plumbing_arm()
{
    for (std::uint32_t trial = 0; trial < TRIALS; trial++)
    {
        cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Read_32b_data_RMW>(0);
        tensix_sync();
        if (mailbox_read(ThreadId::PackThreadId) != PACK_READY)
        {
            return;
        }
        mailbox_write(ThreadId::PackThreadId, MATH_DONE);
    }
}

void run_kernel(RUNTIME_PARAMETERS)
{
    // ---- SECTION 1: the real function ------------------------------------------------------
    for (std::uint32_t i = 0; i < REAL_TRIALS; i++)
    {
        _llk_set_fp32_dest_acc_<ThreadId::MathThreadId>(true);
        _llk_set_fp32_dest_acc_<ThreadId::MathThreadId>(false);
    }

    // ---- SECTION 2: the ordering mechanism -------------------------------------------------
    plumbing_arm();

    replica_arm<OCC_DEPTH[0], Mode::Shipped>();
    replica_arm<OCC_DEPTH[0], Mode::DrainBefore>();
    replica_arm<OCC_DEPTH[0], Mode::DrainAfter>();
    replica_arm<OCC_DEPTH[1], Mode::Shipped>();
    replica_arm<OCC_DEPTH[1], Mode::DrainBefore>();
    replica_arm<OCC_DEPTH[1], Mode::DrainAfter>();
    replica_arm<OCC_DEPTH[2], Mode::Shipped>();
    replica_arm<OCC_DEPTH[2], Mode::DrainBefore>();
    replica_arm<OCC_DEPTH[2], Mode::DrainAfter>();
    replica_arm<OCC_DEPTH[3], Mode::Shipped>();
    replica_arm<OCC_DEPTH[3], Mode::DrainBefore>();
    replica_arm<OCC_DEPTH[3], Mode::DrainAfter>();

    // Direction: RISC work between the writes and the release, clean then occupied.
    replica_arm<0, Mode::Shipped, DIR_NOPS[0]>();
    replica_arm<0, Mode::Shipped, DIR_NOPS[1]>();
    replica_arm<0, Mode::Shipped, DIR_NOPS[2]>();
    replica_arm<0, Mode::Shipped, DIR_NOPS[3]>();
    replica_arm<DIR_OCC, Mode::Shipped, DIR_NOPS[0]>();
    replica_arm<DIR_OCC, Mode::Shipped, DIR_NOPS[1]>();
    replica_arm<DIR_OCC, Mode::Shipped, DIR_NOPS[2]>();
    replica_arm<DIR_OCC, Mode::Shipped, DIR_NOPS[3]>();
}

#endif

#ifdef LLK_TRISC_PACK

namespace
{
volatile std::uint32_t* tt_reg_ptr g_cfg = nullptr;
std::uint32_t g_desync                   = 0;

inline bool dest_acc_field_set()
{
    return (g_cfg[PCK_DEST_RD_CTRL_Read_32b_data_ADDR32] & PCK_DEST_RD_CTRL_Read_32b_data_MASK) != 0;
}

// One replica measurement loop: count the trials on which the release beat the config write.
inline std::uint32_t measure()
{
    std::uint32_t stale = 0;
    for (std::uint32_t trial = 0; trial < TRIALS; trial++)
    {
        // Announce readiness, then park in the blocking read so the release is the only thing
        // between MATH's config write and this thread's sample.
        mailbox_write(ThreadId::MathThreadId, PACK_READY);
        if (mailbox_read(ThreadId::MathThreadId) != MATH_DONE)
        {
            g_desync++;
            continue;
        }
        TTI_STALLWAIT(REPLICA_STALL, p_stall::TRISC_CFG);
        if (!dest_acc_field_set())
        {
            stale++;
        }
    }
    return stale;
}
} // namespace

void run_kernel(RUNTIME_PARAMETERS params)
{
    g_cfg     = get_cfg_pointer();
    auto* res = reinterpret_cast<std::uint32_t*>(params.buffer_Res[0]);

    // ---- SECTION 1: the real function ------------------------------------------------------
    std::uint32_t real_enabled_seen  = 0;
    std::uint32_t real_disabled_seen = 0;
    for (std::uint32_t i = 0; i < REAL_TRIALS; i++)
    {
        _llk_set_fp32_dest_acc_<ThreadId::PackThreadId>();
        if (dest_acc_field_set())
        {
            real_enabled_seen++;
        }
        _llk_set_fp32_dest_acc_<ThreadId::PackThreadId>();
        if (!dest_acc_field_set())
        {
            real_disabled_seen++;
        }
    }

    // ---- SECTION 2: the ordering mechanism -------------------------------------------------
    const std::uint32_t plumbing = measure();

    std::uint32_t shipped[NUM_OCC]     = {};
    std::uint32_t drainbefore[NUM_OCC] = {};
    std::uint32_t drainafter[NUM_OCC]  = {};
    for (std::uint32_t i = 0; i < NUM_OCC; i++)
    {
        shipped[i]     = measure();
        drainbefore[i] = measure();
        drainafter[i]  = measure();
    }

    std::uint32_t dir_clean[NUM_DIR]    = {};
    std::uint32_t dir_occupied[NUM_DIR] = {};
    for (std::uint32_t i = 0; i < NUM_DIR; i++)
    {
        dir_clean[i] = measure();
    }
    for (std::uint32_t i = 0; i < NUM_DIR; i++)
    {
        dir_occupied[i] = measure();
    }

    res[0] = TRIALS;
    res[1] = g_desync;
    res[2] = plumbing;
    res[3] = NUM_OCC;
    for (std::uint32_t i = 0; i < NUM_OCC; i++)
    {
        res[4 + 4 * i] = OCC_DEPTH[i];
        res[5 + 4 * i] = shipped[i];
        res[6 + 4 * i] = drainbefore[i];
        res[7 + 4 * i] = drainafter[i];
    }

    std::uint32_t k = 4 + 4 * NUM_OCC;
    res[k++]        = REAL_TRIALS;
    res[k++]        = real_enabled_seen;
    res[k++]        = real_disabled_seen;
    res[k++]        = NUM_DIR;
    res[k++]        = DIR_OCC;
    for (std::uint32_t i = 0; i < NUM_DIR; i++)
    {
        res[k++] = DIR_NOPS[i];
        res[k++] = dir_clean[i];
        res[k++] = dir_occupied[i];
    }
}

#endif
