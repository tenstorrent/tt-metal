// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "fds_functions.hpp"

namespace overlay::fds_signalling {

// Cycles a go/done must be stable through the deglitcher before capture.
// Dispatch (done) and worker (go) must use the same window.
inline constexpr uint32_t filter_length = 8;

// Group 0 is the idle value on the wire, so payload groups start at 1.
inline constexpr uint32_t idle_group_id = 0;

// How many dispatch instances drive each worker, and which lanes they occupy.
inline constexpr uint32_t num_dispatch_lanes = 3;
inline constexpr uint32_t dispatch_lane_mask = (uint32_t{1} << num_dispatch_lanes) - 1;

// Dispatch listens for done on every worker lane; completion is counted in software,
// so the hardware count threshold stays at 0 and no interrupt is armed.
inline constexpr uint32_t all_worker_lanes_mask = 0xFFFFFFFF;
inline constexpr uint32_t dispatch_done_threshold = 0;

// A worker fires on the first go from any enabled dispatch lane.
inline constexpr uint32_t worker_go_threshold = 1;

// Neither dispatch nor worker startup arms FDS interrupts: dispatch polls done counts,
// and the worker arms its go interrupt mask only after groups are programmed.
inline constexpr uint32_t interrupts_disabled = 0;

// Placeholder pending measurement: hold each strobe level until every worker captures it.
// Must stay well above filter_length (filter + clock-domain crossing + ISR latency).
inline constexpr uint32_t go_strobe_hold_cycles = 512;
inline constexpr uint32_t init_go_clear_hold_cycles = 4096;

// Sub-device i is signalled with FDS group i + 1, since group 0 is idle.
constexpr uint32_t go_group_for_sub_device(uint32_t sub_device_index) { return sub_device_index + 1; }
constexpr uint32_t sub_device_from_go_group(uint32_t group_id) { return group_id - 1; }

// One interrupt bit per payload group, shifted past the idle group 0.
constexpr uint32_t go_interrupt_mask(uint32_t num_go_groups) { return ((uint32_t{1} << num_go_groups) - 1) << 1; }

inline void compiler_memory_barrier() { asm volatile("" ::: "memory"); }

inline void dispatch_disable_auto_dispatch() {
    compiler_memory_barrier();
    FdsDispatch::fds_disable_auto_dispatch();
    compiler_memory_barrier();
}

// A crossing-FIFO write can be acknowledged and dropped when the receiver is not ready, so these
// configuration writes are guarded by a readback.
inline uint32_t dispatch_read_filter_length() {
    compiler_memory_barrier();
    uint32_t threshold = FDS_INTF_READ(TT_FDS_DISPATCH_FILTER_COUNT_THRESHOLD_REG_ADDR);
    compiler_memory_barrier();
    return threshold;
}

inline void dispatch_config_filter_length(uint32_t threshold) {
    do {
        compiler_memory_barrier();
        FdsDispatch::fds_config_filter_length(threshold);
        compiler_memory_barrier();
    } while (dispatch_read_filter_length() != threshold);
}

inline void dispatch_config_interrupt_enable(uint32_t mask) {
    compiler_memory_barrier();
    FdsDispatch::fds_config_interrupt_en(mask);
    compiler_memory_barrier();
}

inline void dispatch_config_group(uint32_t group_id, uint32_t lane_mask, uint32_t count_threshold) {
    compiler_memory_barrier();
    FdsDispatch::fds_config_groupid(group_id, lane_mask, count_threshold);
    compiler_memory_barrier();
}

inline void dispatch_write_go(uint32_t value) {
    compiler_memory_barrier();
    FdsDispatch::fds_go(false, value);
    compiler_memory_barrier();
}

inline uint32_t dispatch_read_go() {
    compiler_memory_barrier();
    uint32_t value = FDS_INTF_READ(TT_FDS_DISPATCH_DISPATCH_TO_TENSIX_REG_ADDR);
    compiler_memory_barrier();
    return value;
}

inline uint32_t dispatch_read_group_status(uint32_t group_id) {
    compiler_memory_barrier();
    uint32_t group_status = FdsDispatch::fds_read_group_status(group_id);
    compiler_memory_barrier();
    return group_status;
}

inline void dispatch_clear_worker_status(uint32_t worker_lane) {
    compiler_memory_barrier();
    FdsDispatch::fds_clear_neo_status(worker_lane);
    compiler_memory_barrier();
}

inline uint32_t dispatch_read_group_count(uint32_t group_id) {
    compiler_memory_barrier();
    uint32_t group_count = FdsDispatch::fds_read_group_count(group_id);
    compiler_memory_barrier();
    return group_count;
}

inline void worker_disable_auto_dispatch() {
    compiler_memory_barrier();
    FdsNeo::fds_disable_auto_dispatch();
    compiler_memory_barrier();
}

inline uint32_t worker_read_filter_length() {
    compiler_memory_barrier();
    uint32_t threshold = FDS_INTF_READ(TT_FDS_TENSIXNEO_FILTER_COUNT_THRESHOLD_REG_ADDR);
    compiler_memory_barrier();
    return threshold;
}

inline void worker_config_filter_length(uint32_t threshold) {
    do {
        compiler_memory_barrier();
        FdsNeo::fds_config_filter_length(threshold);
        compiler_memory_barrier();
    } while (worker_read_filter_length() != threshold);
}

inline void worker_config_interrupt_enable(uint32_t mask) {
    compiler_memory_barrier();
    FdsNeo::fds_config_interrupt_en(mask);
    compiler_memory_barrier();
}

inline void worker_config_group(uint32_t group_id, uint32_t lane_mask, uint32_t count_threshold) {
    compiler_memory_barrier();
    FdsNeo::fds_config_groupid(group_id, lane_mask, count_threshold);
    compiler_memory_barrier();
}

inline uint32_t worker_read_group_status(uint32_t group_id) {
    compiler_memory_barrier();
    uint32_t group_status = FdsNeo::fds_read_group_status(group_id);
    compiler_memory_barrier();
    return group_status;
}

inline void worker_clear_dispatch_status(uint32_t dispatch_lane) {
    compiler_memory_barrier();
    FdsNeo::fds_clear_de_status(dispatch_lane);
    compiler_memory_barrier();
}

inline uint32_t worker_read_done() {
    // The done output must be read back so a dropped crossing-FIFO write is retried.
    compiler_memory_barrier();
    uint32_t done_value = FdsNeo::fds_read_done();
    compiler_memory_barrier();
    return done_value;
}

inline void worker_clear_done() {
    do {
        compiler_memory_barrier();
        FdsNeo::fds_clear_done();
        compiler_memory_barrier();
    } while (worker_read_done() != 0);
}

inline void worker_signal_done(uint32_t group_id) {
    do {
        compiler_memory_barrier();
        FdsNeo::fds_done(false, group_id);
        compiler_memory_barrier();
    } while (worker_read_done() != group_id);
}

}  // namespace overlay::fds_signalling
