// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "fds_functions.hpp"

namespace overlay::fds_signalling {

inline void compiler_memory_barrier() { asm volatile("" ::: "memory"); }

inline void dispatch_disable_auto_dispatch() {
    compiler_memory_barrier();
    FdsDispatch::fds_disable_auto_dispatch();
    compiler_memory_barrier();
}

// A crossing-FIFO write can be acknowledged and dropped when the receiver is not ready, so these
// configuration writes are guarded by a readback.
inline std::uint32_t dispatch_read_filter_length() {
    compiler_memory_barrier();
    std::uint32_t filter_length = FDS_INTF_READ(TT_FDS_DISPATCH_FILTER_COUNT_THRESHOLD_REG_ADDR);
    compiler_memory_barrier();
    return filter_length;
}

inline void dispatch_config_filter_length(std::uint32_t threshold) {
    do {
        compiler_memory_barrier();
        FdsDispatch::fds_config_filter_length(threshold);
        compiler_memory_barrier();
    } while (dispatch_read_filter_length() != threshold);
}

inline void dispatch_config_interrupt_enable(std::uint32_t mask) {
    compiler_memory_barrier();
    FdsDispatch::fds_config_interrupt_en(mask);
    compiler_memory_barrier();
}

inline void dispatch_config_group(std::uint32_t group_id, std::uint32_t lane_mask, std::uint32_t count_threshold) {
    compiler_memory_barrier();
    FdsDispatch::fds_config_groupid(group_id, lane_mask, count_threshold);
    compiler_memory_barrier();
}

inline void dispatch_write_go(std::uint32_t value) {
    compiler_memory_barrier();
    FdsDispatch::fds_go(false, value);
    compiler_memory_barrier();
}

inline std::uint32_t dispatch_read_go() {
    compiler_memory_barrier();
    std::uint32_t value = FDS_INTF_READ(TT_FDS_DISPATCH_DISPATCH_TO_TENSIX_REG_ADDR);
    compiler_memory_barrier();
    return value;
}

inline std::uint32_t dispatch_read_group_status(std::uint32_t group_id) {
    compiler_memory_barrier();
    std::uint32_t group_status = FdsDispatch::fds_read_group_status(group_id);
    compiler_memory_barrier();
    return group_status;
}

inline void dispatch_clear_worker_status(std::uint32_t worker_lane) {
    compiler_memory_barrier();
    FdsDispatch::fds_clear_neo_status(worker_lane);
    compiler_memory_barrier();
}

inline std::uint32_t dispatch_read_group_count(std::uint32_t group_id) {
    compiler_memory_barrier();
    std::uint32_t group_count = FdsDispatch::fds_read_group_count(group_id);
    compiler_memory_barrier();
    return group_count;
}

inline void worker_disable_auto_dispatch() {
    compiler_memory_barrier();
    FdsNeo::fds_disable_auto_dispatch();
    compiler_memory_barrier();
}

inline std::uint32_t worker_read_filter_length() {
    compiler_memory_barrier();
    std::uint32_t filter_length = FDS_INTF_READ(TT_FDS_TENSIXNEO_FILTER_COUNT_THRESHOLD_REG_ADDR);
    compiler_memory_barrier();
    return filter_length;
}

inline void worker_config_filter_length(std::uint32_t threshold) {
    do {
        compiler_memory_barrier();
        FdsNeo::fds_config_filter_length(threshold);
        compiler_memory_barrier();
    } while (worker_read_filter_length() != threshold);
}

inline void worker_config_interrupt_enable(std::uint32_t mask) {
    compiler_memory_barrier();
    FdsNeo::fds_config_interrupt_en(mask);
    compiler_memory_barrier();
}

inline void worker_config_group(std::uint32_t group_id, std::uint32_t lane_mask, std::uint32_t count_threshold) {
    compiler_memory_barrier();
    FdsNeo::fds_config_groupid(group_id, lane_mask, count_threshold);
    compiler_memory_barrier();
}

inline std::uint32_t worker_read_group_status(std::uint32_t group_id) {
    compiler_memory_barrier();
    std::uint32_t group_status = FdsNeo::fds_read_group_status(group_id);
    compiler_memory_barrier();
    return group_status;
}

inline void worker_clear_dispatch_status(std::uint32_t dispatch_lane) {
    compiler_memory_barrier();
    FdsNeo::fds_clear_de_status(dispatch_lane);
    compiler_memory_barrier();
}

inline std::uint32_t worker_read_done() {
    // The done output must be read back so a dropped crossing-FIFO write is retried.
    compiler_memory_barrier();
    std::uint32_t done_value = FdsNeo::fds_read_done();
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

inline void worker_signal_done(std::uint32_t group_id) {
    do {
        compiler_memory_barrier();
        FdsNeo::fds_done(false, group_id);
        compiler_memory_barrier();
    } while (worker_read_done() != group_id);
}

}  // namespace overlay::fds_signalling
