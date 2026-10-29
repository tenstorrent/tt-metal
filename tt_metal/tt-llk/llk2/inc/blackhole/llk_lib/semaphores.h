// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, Blackhole: the Tensix semaphores the threads hand work over with, in one place so that every header that
// waits on or posts one names the same semaphore, without pulling in the other thread's code. The numbers are
// ckernel::semaphore's (ckernel_structs.h), so code built on the old LLKs and on llk-lib agree.
//
//   unpack_sync           unpack thread <-> its unpackers: config contexts in flight (unpack.h)
//   unpack_to_dest_sync   unpack -> math: tiles unpacked straight to Dest (unpack.h, copy.h)
//   math_pack_sync        math -> pack: a tensor math owned, handed to the pack thread (sync.h)
//   pack_math_sync        pack -> math: a tensor the pack thread owned, handed to math (sync.h)
//   dest_ring_sync        math <-> pack: Dest ring sections in use (dest.h)
//   math_done_sync        math: its own earlier Dest writes are done, before an unpack to Dest (copy.h)
//
// pack_math_sync and dest_ring_sync reuse ckernel numbers that no LLK uses on this path: FPU_SFPU (experimental SDPA
// kernels only) and PACK_DONE (perf delay insertion only).

#pragma once

#include <cstdint>

#include "hal/sync.h"

namespace llk2::llk_lib
{

/// ckernel::semaphore::UNPACK_SYNC: config contexts handed to the unpackers and not yet finished.
inline constexpr hal::sync::Semaphore unpack_sync = hal::sync::Semaphore::S5;

/// ckernel::semaphore::UNPACK_TO_DEST: tiles unpacked to Dest, not yet taken by math.
inline constexpr hal::sync::Semaphore unpack_to_dest_sync = hal::sync::Semaphore::S2;

/// ckernel::semaphore::MATH_PACK: tensors math has handed to the pack thread and the pack thread has not taken yet.
inline constexpr hal::sync::Semaphore math_pack_sync = hal::sync::Semaphore::S1;

/// ckernel::semaphore::FPU_SFPU: tensors the pack thread has handed to math and math has not taken yet.
inline constexpr hal::sync::Semaphore pack_math_sync = hal::sync::Semaphore::S0;

/// ckernel::semaphore::PACK_DONE: Dest ring sections math has reserved and the pack thread has not released.
inline constexpr hal::sync::Semaphore dest_ring_sync = hal::sync::Semaphore::S4;

/// ckernel::semaphore::MATH_DONE: math and the SFPU are idle, before an unpack to Dest.
inline constexpr hal::sync::Semaphore math_done_sync = hal::sync::Semaphore::S7;

/// This thread: whether it has set semaphore S up in this kernel launch. One flag per semaphore, in .bss, so false at
/// every kernel launch (the harness's do_crt0 and tt-metal's do_crt1 zero .bss).
template <hal::sync::Semaphore S>
inline bool semaphore_ready = false;

/// The first time this thread uses semaphore S in a kernel launch: wait until S is 0, then set it to value 0, maximum
/// Max. Later calls do nothing. No init call: the thread that posts S calls this before its first post. The thread
/// that takes S only acts on what was posted, so it never sees S before this. The wait for 0 is there because a count
/// an earlier kernel left behind would be read as one of this kernel's.
template <hal::sync::Semaphore S, std::uint32_t Max>
inline __attribute__((always_inline)) void semaphore_setup_once()
{
    if (!semaphore_ready<S>)
    {
        while (hal::sync::semaphore::read<S>() > 0)
        {
        }
        hal::sync::semaphore::init<0 /* value */, Max /* maximum */, S>();
        semaphore_ready<S> = true;
    }
}

} // namespace llk2::llk_lib
