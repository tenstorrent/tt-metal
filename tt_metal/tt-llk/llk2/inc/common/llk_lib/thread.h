// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib's thread split, the same on every arch. Every TRISC compiles the same kernel body, so every thread meets
// every Compute API call. llk-lib's per-thread calls (unpack, matmul, ...) take the thread they run on (On) as a
// template parameter: they plan on every thread, and configure and execute only where current_thread == On.
// Elsewhere that branch is discarded and never instantiated, so each TRISC's build carries only its own thread's
// code (tt-llk-lab llk-tensor, report 4, section 3:
// https://github.com/tenstorrent/tt-llk-lab/tree/lpremovic/llk-tensor/experiments/llk-tensor).
//
// The shape of an llk-lib op. An op is what ONE execution unit does for a Compute API call, on the thread that drives
// the unit: unpack (the unpackers; one op for every call that reads L1, the Plan says what it unpacks), matmul,
// eltwise_binary, reduce, copy (the FPU), vec (the SFPU), pack (the packer). A Compute API op is an unpack and one
// of the others, or one op alone (vec, pack). Every <op>.h has the same five parts, in this order, and the Compute
// API calls only the last:
//
//   struct <op>_params             everything configure and execute need to know, as compile-time values
//   <op>_params_from(Plan)         constexpr: the Resolver's Plan as <op>_params
//   <op>_configure<P>()            programs everything the op's unit needs for P, on every call, assuming nothing about
//                                  what ran before; what is already in place is the compiler's to remove. No run-time
//                                  arguments. Order: the unit's state config (Dest mode, formats: the words that need
//                                  the unit drained, with their wait and only they inside it), address modifiers,
//                                  replay or MOP, counters
//   <op>_execute<P>(addresses..., counts..., scalars...)
//                                  the whole call's work, every tile. Its arguments are what P cannot hold: the Dest
//                                  tensor's address (its absolute Dest row), L1 byte addresses, run-time scalars, and
//                                  each loop count the op takes at run time, as a ct (P's number) or an rt of at most
//                                  that number (resolver/work_counts.h)
//   <op><On, Plan>(counts, addresses..., scalars...)
//                                  the per-thread entry: static_assert that On drives the unit, params from the Plan,
//                                  and, only where runs_on<On>, configure then execute. counts is the WorkCounts of the
//                                  Plan's Work (the run-time side of its sizes), for the ops that take run-time counts;
//                                  the rest of its arguments are execute's
//
// What llk-lib is written in. HAL calls (tt_llk_blackhole/llk_lib/hal/): every instruction, config register, counter,
// semaphore, mailbox, GPR, L1 address and format id an op touches goes through the HAL. One exception, the same for
// every vector function: the arithmetic of an SFPU function (its init and its iterations, llk_lib/vec/<function>.h) is
// SFPI code. That is sfpi:: C++ where the compiler can schedule the body, and the SFPI compiler's instruction
// builtins (TTI_SFP*, with the operand names p_sfpu and InstrModLoadStore) where the body is a hand-scheduled
// sequence: the SFPLOADMACRO pipelines and the replayed bodies of the exps and the reciprocals. vec's configure and
// execute are HAL calls like every op's. llk-lib includes and calls nothing of the 1.0 LLK or its SFPU library; what
// is not the HAL's is named under "Not HAL" in each file's comment (LLK_ASSERT, the Dest size constants of
// tensix_types.h).
//
// Around the op. Taking the op's tensors over from their owners (llk_lib/sync.h, acquire) is the caller's, before the
// entry, on every thread; handing them on is implicit (the result's spec names the unit that wrote it). The Dest ring
// (llk_lib/dest.h) and sync.h are not ops: they have no unit to configure.
//
// In the file: the file comment (what the op is, the five parts with one line each, how one tile is computed,
// "Departures from the shape" with the reason for each, "Ported from", "Not HAL"), then params, params_from, the
// op's constants and helpers, configure, execute, the entry. Every name a file defines starts with the op's name
// (eltwise_binary.h: eltwise_ for all but the five parts); what two ops share is in a common file (math_common.h,
// formats.h, semaphores.h) under a name that says what it is, not whose.
// Names: a template parameter `P` is the op's params and `Pl` the Plan; as a function parameter or a local they are
// `params` and `plan`.
//
// A departure is kept only where the hardware forces it or, where noted, where removing it would change what is
// issued; each is listed in its file's comment. Today:
//   unpack          configure's order; the config context carries on from the previous op
//   copy to Dest    the hand-off with the unpackers is inside execute
//   pack            a run-time tile count's MOP is programmed in execute (the MOP's loop count is part of its
//                   program); execute writes Pack_L1_Acc for a run-time l1_acc and between the tiles of a sum
//   reduce          execute writes ALU config per output tile (the Src zero flag, and with a 32-bit Dest SrcA's
//                   format)
//   vec             the fast exp's walk reprograms the stepping address modifier in execute
//
// What is not an op's configure. Two things are set up once per kernel launch and not per call, because they are
// counts in use across calls and cannot be rewritten while one is pending. Both are stated here as the stack's
// exceptions to "assumes nothing":
//   semaphores      a semaphore is set up by its first post in a kernel launch (semaphores.h, one flag per
//                   semaphore): Blackhole's firmware sets up only four of the semaphores, with its own maxima
//   the Dest ring   every reserve returns a zeroed section because every release zeroes it; the first reserve of a
//                   section relies on Dest being zero at kernel start, as the LLK 1.0 kernels do (llk_lib/dest.h)

#pragma once

#include "tensor/tensor.h"

namespace llk2
{

/// The thread this build is for, from the TRISC define the build sets (tt-llk harness: LLK_TRISC_*; tt-metal:
/// TRISC_*). Thread::None on host builds: plan only, nothing executes.
#if defined(LLK_TRISC_UNPACK) || defined(TRISC_UNPACK)
inline constexpr Thread current_thread = Thread::Unpack;
#elif defined(LLK_TRISC_MATH) || defined(TRISC_MATH)
inline constexpr Thread current_thread = Thread::Math;
#elif defined(LLK_TRISC_PACK) || defined(TRISC_PACK)
inline constexpr Thread current_thread = Thread::Pack;
#else
inline constexpr Thread current_thread = Thread::None;
#endif

/// True when this build is for thread On: the per-thread calls configure and execute only then.
///
/// Every per-thread call has both of these, and they check different things:
///   static_assert(On == Thread::Math, ...)   the call: this unit is driven from that thread (a kernel bug otherwise)
///   if constexpr (runs_on<On>) { ... }        the build: every TRISC compiles the call, and only thread On's build
///                                            keeps the body. Elsewhere the branch is discarded and never instantiated,
///                                            so each TRISC's ELF carries only its own thread's code
template <Thread On>
inline constexpr bool runs_on = current_thread == On;

} // namespace llk2
