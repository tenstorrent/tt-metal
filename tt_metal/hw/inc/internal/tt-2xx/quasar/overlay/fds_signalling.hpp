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

inline void dispatch_config_filter_length(uint32_t threshold) {
    compiler_memory_barrier();
    FdsDispatch::fds_config_filter_length(threshold);
    compiler_memory_barrier();
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

inline void worker_config_filter_length(uint32_t threshold) {
    compiler_memory_barrier();
    FdsNeo::fds_config_filter_length(threshold);
    compiler_memory_barrier();
}

inline void worker_config_interrupt_enable(uint32_t mask) {
    compiler_memory_barrier();
    FdsNeo::fds_config_interrupt_en(mask);
    compiler_memory_barrier();
}

inline uint32_t worker_read_go_status(uint32_t group_id) {
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

inline void worker_clear_done() {
    compiler_memory_barrier();
    FdsNeo::fds_clear_done();
    compiler_memory_barrier();
}

inline void worker_signal_done(uint32_t group_id) {
    compiler_memory_barrier();
    FdsNeo::fds_done(false, group_id);
    compiler_memory_barrier();
}

}  // namespace overlay::fds_signalling
