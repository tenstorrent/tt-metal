// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, Blackhole: the Tensix semaphores the threads hand work over with, in one place so that every header that
// waits on or posts one names the same semaphore, without pulling in the other thread's code. The numbers are
// ckernel::semaphore's (ckernel_structs.h), so code built on the old LLKs and on llk-lib agree.
//
//   unpack_sync           unpack thread <-> its unpackers: config contexts in flight (unpack.h)
//   unpack_to_dest_sync   unpack -> math: tiles unpacked straight to Dest (unpack.h, copy.h)
//   math_pack_sync        math -> pack: Dest sections handed to the packer (dest.h)
//   math_done_sync        math: its own earlier Dest writes are done, before an unpack to Dest (copy.h)

#pragma once

#include "hal/sync.h"

namespace llk2::llk_lib
{

/// ckernel::semaphore::UNPACK_SYNC: config contexts handed to the unpackers and not yet finished.
inline constexpr hal::sync::Semaphore unpack_sync = hal::sync::Semaphore::S5;

/// ckernel::semaphore::UNPACK_TO_DEST: tiles unpacked to Dest, not yet taken by math.
inline constexpr hal::sync::Semaphore unpack_to_dest_sync = hal::sync::Semaphore::S2;

/// ckernel::semaphore::MATH_PACK: Dest sections math has handed to the packer and the packer has not freed.
inline constexpr hal::sync::Semaphore math_pack_sync = hal::sync::Semaphore::S1;

/// ckernel::semaphore::MATH_DONE: math and the SFPU are idle, before an unpack to Dest.
inline constexpr hal::sync::Semaphore math_done_sync = hal::sync::Semaphore::S7;

} // namespace llk2::llk_lib
