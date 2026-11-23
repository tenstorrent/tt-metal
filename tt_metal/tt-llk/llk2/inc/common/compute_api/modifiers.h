// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The form every op in this directory has, and its modifiers. Start here: one op read with this beside it is all of
// them.
//
// One source, three threads. Every file here is arch-independent, and every TRISC compiles and runs the same call:
// each of llk-lib's per-thread calls keeps its body only in its own thread's build (llk_lib/thread.h). An op's header
// names its threads and says on which of them its L1 addresses are read: an address needs to be valid only there
// (tensor/tensor.h).
//
// The Resolver. Every call resolves itself, a chain of one (resolver/resolver.h), through the op's OpApi <op>_op:
// accepts is checked at compile time, then the Plan drives llk-lib's per-thread calls.
//
// The form of an op (compute_api/<op>.h):
//
//   template <op_modifiers Mods = op_modifiers {}, <spec and release of each tensor, in operand order>>
//   auto op(<inputs>, <the tensor written> [, <run-time arguments>])
//   {
//       static_assert(<where the op writes>, ...);                  "<op> writes Dest"
//       constexpr const Accepts& ok = accepted_parts<op_op, in0, in1, Mods>;
//       static_assert(ok.<part>, ...);                              one per part the op's accepts has, in its order:
//                                                                    "<op> takes <what that part accepts>. Not
//                                                                    ported: <...>"
//       constexpr const Resolved& r = resolved<op_op, in0, in1, Mods>;
//       static_assert(<r.config.result against the output>, ...);   "acc must <...>"
//       llk_lib::acquire<thread, unit, spec>();                      one per tensor: the hand-off from its owner
//       LLK_ASSERT(same_runtime_count(in0, in1), ...);               two tensors that hold the same run-time size agree
//       const auto counts = work_counts<r.work>(in0, in1);           the Work's run-time counts (compute_api/counts.h)
//       llk_lib::<part><thread, r.plan>(counts, addresses...);       one per thread that works: unpack, then the op's own
//       return result_of<thread, unit, r.work>(<the tensor written>, counts);
//   }
//
//   Template arguments  the modifier struct first, so a call names only it. A, B, C are the input specs, D the Dest
//                       tensor's, O an L1 output's; RA, ... their release actions. All but Mods are deduced from the
//                       Tensors, and a deduced spec is written `const auto&`, as Tensor declares its own. A spec the
//                       caller names (wait_front<x_spec>, DestRing<acc_spec>) is written with its type.
//   Operands            inputs first (in0, in1, or named by role: src; in, scaler; seed), then the tensor the op
//                       writes (acc: a Dest tensor, written in place; dst: pack's L1 tensor), then run-time arguments
//                       (the vec_ calls' scalars, pack's accumulate).
//   Result              the tensor written, under the op's own (thread, unit): the same address and release. Where
//                       the op ran on a run-time size, the result holds the count written (result_of).
//   Checks              static_asserts in this order (where the op writes; each part of accepts; the result against
//                       the output), then one LLK_ASSERT where two tensors may both hold a run-time count. The
//                       Resolver's accepts is the one place that decides what an op takes (resolver/resolver.h,
//                       Accepts); each part has one sentence here: what the part takes, then what is not ported.
//   Calls               llk-lib only, driven by the Plan.
//   Families            ops that differ only in their OpApi or their function (add / sub / mul, the vec_ calls) share
//                       one body in llk2::detail, named after the family. The named calls are the API.
//   Header comment      the "Compute API: <op>" paragraph (threads, where addresses are read), the call forms, what
//                       the op computes and what of 1.0 it replaces, "Result and owners.", "Resolver and llk-lib.".
//
// Where an op differs, and why (each header says it again in its own words):
//
//   op                   differs in                                 because
//   copy                 no LLK_ASSERT                              one input, and it takes no run-time size
//   reduce               no LLK_ASSERT; returns as_owned            only `in` may hold a run-time size, and the result is
//                                                                   one tile per tile row whatever the row length
//   reduce, with a seed  one call made of two ops (copy, reduce)    the seed's layout in acc is reduce's own
//   pack                 two storage checks, no result check;       the Resolver's second operand is dst, the tensor
//                        returns as_owned(dst)                      pack writes: accepts checks it, and dst already
//                                                                   holds the count packed
//   the vec_ calls       one tensor; no result check, no            in place: the Resolver's second operand is acc
//                        LLK_ASSERT; return as_owned                again, and acc already holds the count
//
// dataflow.h, dest.h and sync.h are not ops: they make and hand over the Tensors the ops take.
//
// Modifiers. Every op takes one modifier struct as its first template argument. The structs are defined beside the
// layers, in resolver/modifiers.h, because the Resolver reads them too; this header brings them to every op's file.

#pragma once

#include "resolver/modifiers.h"
