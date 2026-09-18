// SPDX-FileCopyrightText: © 2026 Zane Hambly
//
// SPDX-License-Identifier: Apache-2.0

/* Copyright (c) 2026 Zane Hambly. Apache License 2.0.
 * See LICENSE for terms. */

/* ttsim_ref.h -- an independent oracle for the expander.
 *
 * tm_expand is hand-derived from the MOP/replay rules. The lesson of
 * 2026-06-10 is that a hand-derived expander can be confidently wrong in
 * a way its own tests miss by luck. So this is a second, independent
 * implementation of the same thing, ported faithfully from Tenstorrent's
 * own simulator ttsim (src/tensix.cpp, Apache 2.0, (c) Tenstorrent): the
 * mop_expander (templates 0 and 1), the replay_expander, and the
 * frontend NOP-eat in tensix_push_inst_fifo. It walks the same plan IR
 * and produces the same NOP-free backend trace.
 *
 * The fuzzer (tfuzz.c) runs random valid plans through both this and
 * tm_expand and asserts they agree word for word. ttsim is the ground
 * truth, so agreement here is correctness, not just self-consistency.
 */

#ifndef TTSIM_REF_H
#define TTSIM_REF_H

#include "ttmop.h"

/* Expand a plan the way ttsim would. Same contract as tm_expand:
 * arena-owned stream, ok=0 with err set on overflow. */
tm_stream_t ttsim_expand(const tm_planop_t *plan, uint32_t n_ops, ka_arena_t *A);

#endif /* TTSIM_REF_H */
