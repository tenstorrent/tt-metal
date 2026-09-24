// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/debug/waypoint.h"
#include "fds_functions.hpp"
#include "interrupt_defines.h"
#include "risc_common.h"

namespace overlay::fds_signalling {

// Cycles a go/done must be stable through the deglitcher before capture.
// Dispatch (done) and worker (go) must use the same window.
inline constexpr uint32_t filter_length_cycles = 8;

// Group 0 is the idle value on the wire, so payload groups start at 1.
inline constexpr uint32_t idle_group_id = 0;

// How many dispatch instances drive each worker, and which lanes they occupy.
inline constexpr uint32_t num_dispatch_lanes = 3;
inline constexpr uint32_t dispatch_lane_mask = (uint32_t{1} << num_dispatch_lanes) - 1;

// Dispatch listens for done on every worker lane; completion is counted in software,
// so the hardware count threshold stays at 0 and no interrupt is armed. One lane per worker that
// can report done, which bounds how many workers a single sub-device's completion tracking can cover.
inline constexpr uint32_t num_worker_lanes = 32;
inline constexpr uint32_t all_worker_lanes_mask = ~uint32_t{0} >> (32 - num_worker_lanes);
inline constexpr uint32_t dispatch_done_threshold = 0;

// A worker fires on the first go from any enabled dispatch lane.
inline constexpr uint32_t worker_go_threshold = 1;

// Neither dispatch nor worker startup arms FDS interrupts: dispatch polls done counts, and the
// worker arms its go interrupt mask only after auto dispatch is enabled and groups are programmed.
inline constexpr uint32_t interrupts_disabled = 0;

// The three cycle counts below are temporary placeholders and will be updated to their actual values later.

// At init, dispatch writes idle to the go wire directly and holds it this long before enabling
// auto dispatch. Worker filters capture only when the wire value changes, so every worker has to
// capture idle first; otherwise a first go that repeats the group the previous run left on the wire
// is never seen.
inline constexpr uint32_t init_go_park_hold_cycles = 4096;

// Auto dispatch pacing on dispatch: queued gos go out one every count + 1 cycles, so each go stays on
// the wire long enough for every worker's filter to capture it before the next value replaces it.
// The same value sizes the wait for queued gos to go out before auto dispatch is disabled.
inline constexpr uint32_t dispatch_auto_dispatch_pacing_cycle_count = 512;

// Auto dispatch pacing on the worker: queued values go out one every count + 1 cycles. A short kernel
// can queue the round's idle clear and its done back to back, so the spacing keeps idle on the wire
// long enough for dispatch to capture it, and the done then arrives as a change.
inline constexpr uint32_t worker_auto_dispatch_pacing_cycle_count = 512;

inline void wait_cycles(uint32_t cycles) {
    const uint32_t start_timestamp = get_timestamp_32b();
    while (get_timestamp_32b() - start_timestamp < cycles) {
    }
}

// Sub-device i is signalled with FDS group i + 1, since group 0 is idle.
constexpr uint32_t go_group_for_sub_device(uint32_t sub_device_index) { return sub_device_index + 1; }
constexpr uint32_t sub_device_from_go_group(uint32_t group_id) { return group_id - 1; }

// One interrupt bit per payload group, shifted past the idle group 0.
constexpr uint32_t go_interrupt_mask(uint32_t num_go_groups) { return ((uint32_t{1} << num_go_groups) - 1) << 1; }

// This readback retry is valid only while auto dispatch does not intercept the target register.
template <typename WriteFunction, typename ReadFunction>
inline void write_until_readback_matches(WriteFunction write, ReadFunction read, uint32_t expected) {
    do {
        write();
    } while (read() != expected);
}

inline uint32_t dispatch_read_auto_dispatch_enable() { return FdsDispatch::fds_read_auto_dispatch_enable(); }

inline uint32_t dispatch_read_auto_dispatch_cycle_count() { return FdsDispatch::fds_read_auto_dispatch_cycle_count(); }

inline uint32_t dispatch_read_auto_dispatch_fifo_full() { return FdsDispatch::fds_read_auto_dispatch_fifo_full(); }

inline void dispatch_disable_auto_dispatch() {
    write_until_readback_matches(
        [] { FdsDispatch::fds_disable_auto_dispatch(); },
        [] { return dispatch_read_auto_dispatch_enable(); },
        uint32_t{0});
}

inline void dispatch_enable_auto_dispatch() {
    write_until_readback_matches(
        [] { FdsDispatch::fds_enable_auto_dispatch(); },
        [] { return dispatch_read_auto_dispatch_enable(); },
        uint32_t{1});
}

inline void dispatch_config_auto_dispatch_pacing(uint32_t cycle_count) {
    write_until_readback_matches(
        [=] { FdsDispatch::fds_config_auto_dispatch_pacing(cycle_count); },
        [] { return dispatch_read_auto_dispatch_cycle_count(); },
        cycle_count);
}

inline void dispatch_config_auto_dispatch_outbox(uint32_t address) {
    write_until_readback_matches(
        [=] { FdsDispatch::fds_config_auto_dispatch_outbox(address); },
        [] { return FdsDispatch::fds_read_auto_dispatch_outbox_address(); },
        address);
}

inline uint32_t dispatch_read_filter_length() { return FdsDispatch::fds_read_filter_length(); }

inline void dispatch_config_filter_length(uint32_t threshold) {
    write_until_readback_matches(
        [=] { FdsDispatch::fds_config_filter_length(threshold); },
        [] { return dispatch_read_filter_length(); },
        threshold);
}

inline void dispatch_config_interrupt_enable(uint32_t mask) { FdsDispatch::fds_config_interrupt_en(mask); }

inline void dispatch_config_group(uint32_t group_id, uint32_t lane_mask, uint32_t count_threshold) {
    FdsDispatch::fds_config_groupid(group_id, lane_mask, count_threshold);
}

// This write is intercepted by auto dispatch.
inline void dispatch_write_go(uint32_t value) { FdsDispatch::fds_go(value); }

// Direct-path go write, valid only while auto dispatch is disabled.
inline void dispatch_write_go_direct(uint32_t value) {
    write_until_readback_matches([=] { FdsDispatch::fds_go(value); }, [] { return FdsDispatch::fds_read_go(); }, value);
}

inline uint32_t dispatch_read_group_status(uint32_t group_id) { return FdsDispatch::fds_read_group_status(group_id); }

inline void dispatch_clear_worker_status(uint32_t worker_lane) { FdsDispatch::fds_clear_neo_status(worker_lane); }

inline uint32_t dispatch_read_group_count(uint32_t group_id) { return FdsDispatch::fds_read_group_count(group_id); }

// Worker PLIC delivery: FDS group g arrives at the PLIC as source plic_source_base + g.
// PLIC source 0 means "no interrupt", so the FDS threshold sources sit one above their interrupt ids.
// Only DM0 takes these interrupts; dispatch polls done counts and never touches the PLIC.
inline constexpr uint32_t plic_source_base = DM_CORE_INT_ID_FDS_THRESHOLD_INTERRUPTS_0 + 1;

// A source is delivered only while its priority strictly exceeds the context threshold, so this is
// the minimum priority that beats the allow-all threshold.
inline constexpr uint32_t plic_fds_priority = 1;

constexpr uint32_t plic_source_for_go_group(uint32_t group_id) { return plic_source_base + group_id; }
constexpr uint32_t go_group_from_plic_source(uint32_t source) { return source - plic_source_base; }

inline uint32_t worker_read_auto_dispatch_enable() { return FdsNeo::fds_read_auto_dispatch_enable(); }

inline uint32_t worker_read_auto_dispatch_cycle_count() { return FdsNeo::fds_read_auto_dispatch_cycle_count(); }

inline void worker_disable_auto_dispatch() {
    write_until_readback_matches(
        [] { FdsNeo::fds_disable_auto_dispatch(); }, [] { return worker_read_auto_dispatch_enable(); }, uint32_t{0});
}

inline void worker_enable_auto_dispatch() {
    write_until_readback_matches(
        [] { FdsNeo::fds_enable_auto_dispatch(); }, [] { return worker_read_auto_dispatch_enable(); }, uint32_t{1});
}

inline void worker_config_auto_dispatch_pacing(uint32_t cycle_count) {
    write_until_readback_matches(
        [=] { FdsNeo::fds_config_auto_dispatch_pacing(cycle_count); },
        [] { return worker_read_auto_dispatch_cycle_count(); },
        cycle_count);
}

inline void worker_config_auto_dispatch_outbox(uint32_t address) {
    write_until_readback_matches(
        [=] { FdsNeo::fds_config_auto_dispatch_outbox(address); },
        [] { return FdsNeo::fds_read_auto_dispatch_outbox_address(); },
        address);
}

inline uint32_t worker_read_filter_length() { return FdsNeo::fds_read_filter_length(); }

inline void worker_config_filter_length(uint32_t threshold) {
    write_until_readback_matches(
        [=] { FdsNeo::fds_config_filter_length(threshold); }, [] { return worker_read_filter_length(); }, threshold);
}

inline void worker_config_interrupt_enable(uint32_t mask) { FdsNeo::fds_config_interrupt_en(mask); }

inline void worker_config_group(uint32_t group_id, uint32_t lane_mask, uint32_t count_threshold) {
    FdsNeo::fds_config_groupid(group_id, lane_mask, count_threshold);
}

inline uint32_t worker_read_group_status(uint32_t group_id) { return FdsNeo::fds_read_group_status(group_id); }

inline void worker_clear_dispatch_status(uint32_t dispatch_lane) { FdsNeo::fds_clear_de_status(dispatch_lane); }

inline void worker_wait_for_auto_dispatch_queue_space() {
    WAYPOINT("FADW");
    while (FdsNeo::fds_read_auto_dispatch_fifo_full() != 0) {
    }
    WAYPOINT("FADD");
}

// Direct-path clear, valid only while auto dispatch is disabled.
inline void worker_clear_done_direct() {
    write_until_readback_matches(
        [] { FdsNeo::fds_done(idle_group_id); }, [] { return FdsNeo::fds_read_done(); }, idle_group_id);
}

inline void worker_clear_done() {
    worker_wait_for_auto_dispatch_queue_space();
    FdsNeo::fds_done(idle_group_id);
}

inline void worker_signal_done(uint32_t group_id) {
    worker_wait_for_auto_dispatch_queue_space();
    FdsNeo::fds_done(group_id);
}

}  // namespace overlay::fds_signalling
