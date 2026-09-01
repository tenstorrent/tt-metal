// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Disabled: every case here counts reports through detail::mock_report_count, which output.h no
// longer keeps. Re-enable once the reports are observable from the host again.
// UNSUPPORTED: true
//
// REQUIRES: fmt
// DEFINE: %{cflags} = -std=c++17 -Wall -Wextra -Werror -DLLK_SAN_ENABLE -DCOMPILE_FOR_TRISC=0 -DLLK_SAN_SETTING_HOST_DEPS=1 -DDEBUG_PRINT_ENABLED %{fmt_flags}
// RUN: %clangxx %{cflags} -I %{sanitizer_include} %s -o %t
// RUN: %t

#include <cassert>
#include <cstdint>

#include "sanitizer/api.h"

using namespace llk::san;

using Tilize = OperationUnpackTilize;
using Unary  = OperationUnpackUnary;
using Unp    = Operand<Exu::Unpack>;

namespace llk::san
{

static State thread_state {};

State* const state = &thread_state;

} // namespace llk::san

// Each case starts from a thread that has run nothing.
static void reset()
{
    thread_state              = State {};
    detail::mock_report_count = 0;
}

static int reports()
{
    return static_cast<int>(detail::mock_report_count);
}

// The zones are templates, so this is the only place in the suite that compiles their bodies: the
// enabled build instantiates them nowhere else, and test_hooks.cpp reaches the macros in a disabled
// build, where they expand to nothing. Run before the first reset(), so no case can see it.
static void zones()
{
    LLK_SAN_FUNCTION();
    LLK_SAN_SILENT_ZONE();
}

int main()
{
    zones();

    // ---- the sequence a well-behaved kernel writes ----------------------------------------------
    // configure() records the operand state; the three operation hooks restate a slice of it and are
    // each satisfied by it. Without this case a hook that reported everything would pass the rest.
    reset();
    configure<Thread::TRISC0>(StateVal<Unp::InputFormatA>(1u), StateVal<Unp::FaceHeightA>(16u));
    init<Tilize, Thread::TRISC0>(StateVal<Tilize::BlockCtDim>(2u), StateVal<Tilize::NarrowTile>(false), StateVal<Unp::FaceHeightA>(16u));
    execute<Tilize, Thread::TRISC0>(StateVal<Tilize::BlockCtDim>(2u), StateVal<Unp::FaceHeightA>(16u));
    execute<Tilize, Thread::TRISC0>(StateVal<Tilize::BlockCtDim>(2u));
    uninit<Tilize, Thread::TRISC0>(StateVal<Tilize::BlockCtDim>(2u), StateVal<Tilize::NarrowTile>(false), StateVal<Unp::FaceHeightA>(16u));
    assert(reports() == 0);

    // A nullary uninit() has no argument to disagree about and must stay silent.
    reset();
    init<Tilize, Thread::TRISC0>(StateVal<Tilize::BlockCtDim>(2u));
    uninit<Tilize, Thread::TRISC0>();
    assert(reports() == 0);

    // A discarded parameter is not compared, whatever the recorded value is.
    reset();
    init<Tilize, Thread::TRISC0>(StateVal<Tilize::BlockCtDim>(2u));
    uninit<Tilize, Thread::TRISC0>(StateDiscard<std::uint32_t>(9u));
    assert(reports() == 0);

    // ---- uninit() is held to the operation state init() recorded --------------------------------
    // The check this file exists for: an operation torn down under a different parameter than it was
    // set up with is reported, exactly as execute() would report it.
    reset();
    init<Tilize, Thread::TRISC0>(StateVal<Tilize::BlockCtDim>(2u), StateVal<Tilize::NarrowTile>(false));
    uninit<Tilize, Thread::TRISC0>(StateVal<Tilize::BlockCtDim>(3u));
    assert(reports() == 1);

    reset();
    init<Tilize, Thread::TRISC0>(StateVal<Tilize::BlockCtDim>(2u), StateVal<Tilize::NarrowTile>(false));
    uninit<Tilize, Thread::TRISC0>(StateVal<Tilize::NarrowTile>(true));
    assert(reports() == 1);

    // A field init() never mentioned is unknown, not zero, so restating it is a disagreement too.
    reset();
    init<Tilize, Thread::TRISC0>(StateVal<Tilize::BlockCtDim>(2u));
    uninit<Tilize, Thread::TRISC0>(StateVal<Tilize::NarrowTile>(false));
    assert(reports() == 1);

    // Each argument is judged on its own: one wrong field among right ones is still one report.
    reset();
    init<Tilize, Thread::TRISC0>(StateVal<Tilize::BlockCtDim>(2u), StateVal<Tilize::NarrowTile>(false));
    uninit<Tilize, Thread::TRISC0>(StateVal<Tilize::BlockCtDim>(2u), StateVal<Tilize::NarrowTile>(true));
    assert(reports() == 1);

    // ---- and to the operand state ---------------------------------------------------------------
    reset();
    configure<Thread::TRISC0>(StateVal<Unp::FaceHeightA>(16u));
    init<Tilize, Thread::TRISC0>(StateVal<Unp::FaceHeightA>(16u));
    uninit<Tilize, Thread::TRISC0>(StateVal<Unp::FaceHeightA>(32u));
    assert(reports() == 1);

    // An operand value that was never configured is unknown, and restating it says so.
    reset();
    init<Tilize, Thread::TRISC0>();
    uninit<Tilize, Thread::TRISC0>(StateVal<Unp::FaceHeightA>(16u));
    assert(reports() == 1);

    // A reconfigure() inside the operation moves state it was set up for. Only the snapshot init()
    // took can see that -- the argument uninit() restates matches the new state, not the old.
    reset();
    configure<Thread::TRISC0>(StateVal<Unp::FaceHeightA>(16u));
    init<Tilize, Thread::TRISC0>(StateVal<Unp::FaceHeightA>(16u));
    reconfigure<Thread::TRISC0>(StateVal<Unp::FaceHeightA>(32u));
    uninit<Tilize, Thread::TRISC0>(StateVal<Unp::FaceHeightA>(32u));
    assert(reports() == 1);

    // The same drift, seen by a uninit() that restates nothing at all.
    reset();
    configure<Thread::TRISC0>(StateVal<Unp::FaceHeightA>(16u));
    init<Tilize, Thread::TRISC0>(StateVal<Unp::FaceHeightA>(16u));
    reconfigure<Thread::TRISC0>(StateVal<Unp::FaceHeightA>(32u));
    uninit<Tilize, Thread::TRISC0>();
    assert(reports() == 1);

    // A reconfigure() that leaves the snapshotted value where it was is not drift.
    reset();
    configure<Thread::TRISC0>(StateVal<Unp::FaceHeightA>(16u), StateVal<Unp::InputFormatA>(1u));
    init<Tilize, Thread::TRISC0>(StateVal<Unp::FaceHeightA>(16u));
    reconfigure<Thread::TRISC0>(StateVal<Unp::InputFormatA>(2u));
    uninit<Tilize, Thread::TRISC0>();
    assert(reports() == 0);

    // ---- the operation being ended has to be the one that was initialized -----------------------
    reset();
    uninit<Tilize, Thread::TRISC0>();
    assert(reports() == 1);

    reset();
    init<Unary, Thread::TRISC0>(StateVal<Unary::UnpackToDest>(0u));
    uninit<Tilize, Thread::TRISC0>();
    assert(reports() == 1);

    // Nothing survives the record being seated on another operation, so a second uninit() of the
    // right operation reports once, not twice.
    reset();
    init<Tilize, Thread::TRISC0>(StateVal<Tilize::BlockCtDim>(2u));
    init<Unary, Thread::TRISC0>(StateVal<Unary::UnpackToDest>(0u));
    uninit<Tilize, Thread::TRISC0>(StateVal<Tilize::BlockCtDim>(2u));
    assert(reports() == 1);

    // ---- execute() reads the same way, which is why they share a body ---------------------------
    reset();
    execute<Tilize, Thread::TRISC0>();
    assert(reports() == 1);

    reset();
    init<Tilize, Thread::TRISC0>(StateVal<Tilize::BlockCtDim>(2u));
    execute<Tilize, Thread::TRISC0>(StateVal<Tilize::BlockCtDim>(3u));
    assert(reports() == 1);

    // ---- init() writes, never checks --------------------------------------------------------------
    // An operand value handed to init() that differs from the configured state is snapshotted as
    // given and reported at the next hook, as drift.
    reset();
    configure<Thread::TRISC0>(StateVal<Unp::FaceHeightA>(16u));
    init<Tilize, Thread::TRISC0>(StateVal<Unp::FaceHeightA>(32u));
    assert(reports() == 0);
    uninit<Tilize, Thread::TRISC0>();
    assert(reports() == 1);

    // The same argument that is a disagreement in uninit() above is simply the recorded value here.
    reset();
    init<Tilize, Thread::TRISC0>(StateVal<Tilize::BlockCtDim>(2u));
    init<Tilize, Thread::TRISC0>(StateVal<Tilize::BlockCtDim>(3u));
    uninit<Tilize, Thread::TRISC0>(StateVal<Tilize::BlockCtDim>(3u));
    assert(reports() == 0);

    return 0;
}
