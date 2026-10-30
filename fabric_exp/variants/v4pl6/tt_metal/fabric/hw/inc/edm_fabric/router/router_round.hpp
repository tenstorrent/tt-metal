// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Once-per-round and rare work, shared by every loop: termination, router commands, the heartbeat and the context
// switch. Each is a small function of its own, called out of line between rounds, so editing it never changes a loop's
// per-pass code (RESULTS §97: per-round work inlined into a loop made unrelated edits re-lay out the whole loop).
//
// Context switch. The context-switch master (the ERISC with enable_context_switch, ERISC 0) hands the core to the base
// firmware when idle, and runs router commands (PAUSE, DRAIN, RETRAIN). On a two-ERISC router the other ERISC must not
// route meanwhile, so the master first raises RETRAIN_INTENT in a shared stream scratch register and waits for the
// other's INTENT_ACK; the other parks until RETRAIN_COMPLETE, acks it, and resumes at NORMAL_EXECUTION.

#include "router_common.hpp"

namespace fabric_router {

enum class ContextSwitchState : uint32_t {
    NORMAL_EXECUTION = 0,  // set by the master at init; the default
    RETRAIN_INTENT = 1,    // master: about to switch; wait for the other ERISC
    INTENT_ACK = 2,        // other ERISC: parked
    RETRAIN_COMPLETE = 3,  // master: switch done
    COMPLETE_ACK = 4,      // other ERISC: resuming
};

FORCE_INLINE uint32_t context_switch_state() { return read_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>(); }
FORCE_INLINE void set_context_switch_state(ContextSwitchState s) {
    write_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>(static_cast<uint32_t>(s));
}

constexpr bool is_context_switch_master = enable_context_switch;

FORCE_INLINE volatile tt::tt_fabric::TerminationSignal* termination_signal() {
    return reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_addr);
}

FORCE_INLINE tt_l1_ptr RouterStateManager* router_state() {
    auto* routing_table = reinterpret_cast<tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    return const_cast<tt_l1_ptr RouterStateManager*>(&routing_table->state_manager);
}

FORCE_INLINE bool terminating() { return got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal()); }

// Wait until the context-switch register reads `s`, or termination.
FORCE_INLINE void wait_for_context_switch_state(ContextSwitchState s) {
    while (context_switch_state() != static_cast<uint32_t>(s)) {
        if (terminating()) {
            return;
        }
    }
}

// Master: park the other ERISC, run `f`, release it.
template <typename F>
FORCE_INLINE void with_other_erisc_parked(F&& f) {
    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        set_context_switch_state(ContextSwitchState::RETRAIN_INTENT);
        wait_for_context_switch_state(ContextSwitchState::INTENT_ACK);
        f();
        set_context_switch_state(ContextSwitchState::RETRAIN_COMPLETE);
        wait_for_context_switch_state(ContextSwitchState::COMPLETE_ACK);
        set_context_switch_state(ContextSwitchState::NORMAL_EXECUTION);
    } else {
        f();
    }
}

// Non-master: park while the master switches.
FORCE_INLINE void follow_context_switch() {
    if (context_switch_state() == static_cast<uint32_t>(ContextSwitchState::RETRAIN_INTENT)) {
        set_context_switch_state(ContextSwitchState::INTENT_ACK);
        wait_for_context_switch_state(ContextSwitchState::RETRAIN_COMPLETE);
        set_context_switch_state(ContextSwitchState::COMPLETE_ACK);
        wait_for_context_switch_state(ContextSwitchState::NORMAL_EXECUTION);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Rounds
// ---------------------------------------------------------------------------------------------------------------------

// Whether to run another round: no termination, and (master) no router command pending.
__attribute__((noinline)) bool next_round() {
    if constexpr (is_context_switch_master || NUM_ACTIVE_ERISCS == 1) {
        return !terminating() && !router_state()->is_non_run_command_pending();
    } else {
        return !terminating();
    }
}

// The heartbeat, then the context switch: the master hands the core to the base firmware after `idle_interval` rounds
// in a row without a send attempt (IDLE_CONTEXT_SWITCHING; otherwise every interval); the other ERISC follows.
// Per-round bookkeeping. It lives in memory (end_round takes it by reference, out of line), so the loops' registers stay
// free for per-pass state; the loops touch it once per round, and on the rare yields.
struct RoundCounters {
    uint16_t heartbeat = 0;
    uint32_t idle_rounds = 0;
    bool attempted = false;         // sender: a send yielded (TX queue busy) this round
    uint32_t sent_at_round_start = 0;  // sender: packets sent before this round
};

// Returns whether the base firmware ran on this ERISC (the master's context switch), which may have used its command
// buffers.
__attribute__((noinline)) bool end_round(RoundCounters& counters, bool attempted_send) {
    if ((++counters.heartbeat & 0x3F) == 0) {
        *reinterpret_cast<volatile uint32_t*>(heartbeat_addr) = 0xDCBA0000 | counters.heartbeat;
    }
    if constexpr (is_context_switch_master) {
        if (IDLE_CONTEXT_SWITCHING && attempted_send) {
            counters.idle_rounds = 0;
        } else if (counters.idle_rounds++ > SWITCH_INTERVAL) {
            counters.idle_rounds = 0;
            with_other_erisc_parked([] { run_routing_without_noc_sync(); });
            return true;
        }
    } else {
        follow_context_switch();
    }
    return false;
}

// Router commands (master only): PAUSE parks the router until RUN, running DRAIN and RETRAIN when asked.
void run_drain(tt_l1_ptr RouterStateManager* state) {
    state->state = RouterState::DRAINING;
    ASSERT(false);  // not implemented (as on main)
    while (state->command == RouterCommand::DRAIN && !terminating()) {
    }
}

void run_retrain(tt_l1_ptr RouterStateManager* state) {
    state->state = RouterState::RETRAINING;
    run_routing_without_noc_sync();
    while (state->command == RouterCommand::RETRAIN && !terminating()) {
    }
}

__attribute__((noinline)) void run_router_command() {
    auto* state = router_state();
    if (state->command == RouterCommand::PAUSE) {
        with_other_erisc_parked([state] {
            while (!terminating()) {
                state->state = RouterState::PAUSED;
                const RouterCommand command = state->command;
                if (command == RouterCommand::RUN) {
                    break;
                } else if (command == RouterCommand::PAUSE) {
                    run_routing_without_noc_sync();
                } else if (command == RouterCommand::DRAIN) {
                    run_drain(state);
                } else if (command == RouterCommand::RETRAIN) {
                    run_retrain(state);
                }
            }
        });
    } else {
        ASSERT(false);
    }
}

// Before a run of rounds (master): true when the router command is RUN; otherwise runs the command and returns false.
__attribute__((noinline)) bool start_running() {
    auto* state = router_state();
    if (state->command == RouterCommand::RUN) {
        state->state = RouterState::RUNNING;
        return true;
    }
    run_router_command();
    return false;
}

// An ERISC with nothing to service (channel trimming can leave one: on a two-ERISC router, every sender channel or every
// receiver channel unused in the capture): rounds without passes, so it still runs the router commands (the master),
// the heartbeat, the context switch and termination.
__attribute__((noinline)) void run_idle() {
    RoundCounters round;
    while (!terminating()) {
        if constexpr (is_context_switch_master) {
            if (!start_running()) {
                continue;
            }
        }
        while (next_round()) {
            end_round(round, false);
        }
    }
}

}  // namespace fabric_router
