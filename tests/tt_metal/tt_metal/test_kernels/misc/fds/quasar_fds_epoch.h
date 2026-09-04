// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The opening protocol every kernel of the Quasar FDS test suite shares.
//
// FDS registers and the wires between them keep their values across program launches, so a kernel
// that trusted whatever it found on entry would credit the previous epoch's signal and assert
// nothing. Both sides therefore begin by dropping what they inherited and then agreeing, over the
// wires themselves, that they are both live in this epoch.
//
// The worker clears its input registers and its own done, then holds a ready token on the done wire
// until the go arrives. The token alternates between two values because a held value survives the
// engine's own input clearing only by changing: the engine clears its inputs to shed the previous
// epoch, and a lane holding one constant token would simply stay at zero afterwards, whereas the
// next flip is a change and is recaptured. Group ids 14 and 15 are reserved for the two tokens
// suite-wide, so payload group ids stay in 1..13.
//
// The engine clears its input registers, waits for the agreed number of lanes to show a ready
// token, and only then raises its go — having cleared its output first, so the go is a change on
// the wire even when the previous epoch sent the same group id.
//
// Every wait here is bounded rather than using the unbounded fds_poll: a signal that never arrives
// has to fail the test with a readable status word rather than hang the run.

#include <cstdint>

#include "overlay/fds_functions.hpp"

// A signal must be stable for a single cycle to be accepted. Zero turns de-glitching off; the
// effective threshold out of reset is an internal shadow of 8, so writing this is a deliberate
// choice, not a restatement of the reset state.
constexpr uint32_t kNoDeglitchFilter = 0;

// The two ready tokens, alternated on the done wire until the go arrives.
constexpr uint32_t kReadyTokenA = 14;
constexpr uint32_t kReadyTokenB = 15;
// Poll iterations between ready flips: long enough that each token is held well past capture, short
// enough that the dispatch engine's ready wait sees a change soon after clearing its inputs.
constexpr uint32_t kReadySpinIterations = 200;

namespace fds_epoch {

inline void clear_dispatch_inputs(uint32_t worker_mask) {
    for (uint32_t mask = worker_mask, neo = 0; mask != 0; mask >>= 1, neo++) {
        if (mask & 1u) {
            overlay::FdsDispatch::fds_clear_neo_status(neo);
        }
    }
}

inline void clear_worker_inputs(uint32_t dispatch_mask) {
    for (uint32_t mask = dispatch_mask, inst = 0; mask != 0; mask >>= 1, inst++) {
        if (mask & 1u) {
            overlay::FdsNeo::fds_clear_de_status(inst);
        }
    }
    overlay::FdsNeo::fds_clear_done();
}

// Status is not gated by the enable masks, so this needs no group configuration.
inline bool workers_ready(uint32_t worker_mask, uint32_t num_ready) {
    const uint32_t ready_lanes = (overlay::FdsDispatch::fds_read_group_status(kReadyTokenA) |
                                  overlay::FdsDispatch::fds_read_group_status(kReadyTokenB)) &
                                 worker_mask;
    return static_cast<uint32_t>(__builtin_popcount(ready_lanes)) >= num_ready;
}

inline bool wait_for_workers(uint32_t worker_mask, uint32_t num_ready, uint32_t poll_iterations) {
    clear_dispatch_inputs(worker_mask);
    for (uint32_t i = 0; i < poll_iterations; i++) {
        if (workers_ready(worker_mask, num_ready)) {
            return true;
        }
    }
    return false;
}

inline void pulse_ready(uint32_t& ready_token, uint32_t iteration) {
    if (iteration % kReadySpinIterations == 0) {
        overlay::FdsNeo::fds_done(/*ad_enable=*/false, ready_token);
        ready_token = (ready_token == kReadyTokenA) ? kReadyTokenB : kReadyTokenA;
    }
}

// go_inst reports which engine drove this worker in the current epoch.
inline bool wait_for_go(uint32_t dispatch_mask, uint32_t awaited_value, uint32_t poll_iterations, uint32_t& go_inst) {
    clear_worker_inputs(dispatch_mask);

    uint32_t ready_token = kReadyTokenA;
    for (uint32_t i = 0; i < poll_iterations; i++) {
        pulse_ready(ready_token, i);
        for (uint32_t mask = dispatch_mask, inst = 0; mask != 0; mask >>= 1, inst++) {
            if ((mask & 1u) != 0 && overlay::FdsNeo::fds_read_de_status(inst) == awaited_value) {
                go_inst = inst;
                return true;
            }
        }
    }
    return false;
}

}  // namespace fds_epoch
