// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "fds_functions.hpp"
#include "interrupt_defines.h"

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
// so the hardware count threshold stays at 0 and no interrupt is armed. One lane per worker that
// can report done, which bounds how many workers a single completion round can cover.
inline constexpr uint32_t num_worker_lanes = 32;
inline constexpr uint32_t all_worker_lanes_mask = ~uint32_t{0} >> (32 - num_worker_lanes);
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

inline void dispatch_disable_auto_dispatch() { FdsDispatch::fds_disable_auto_dispatch(); }

// A crossing-FIFO write can be acknowledged and dropped when the receiver is not ready, so every
// write below that must land is reissued until a readback shows the expected value.
template <typename WriteFunction, typename ReadFunction>
inline void write_until_readback_matches(WriteFunction write, ReadFunction read, uint32_t expected) {
    do {
        write();
    } while (read() != expected);
}

inline uint32_t dispatch_read_filter_length() { return FDS_INTF_READ(TT_FDS_DISPATCH_FILTER_COUNT_THRESHOLD_REG_ADDR); }

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

inline void dispatch_write_go(uint32_t value) { FdsDispatch::fds_go(/*ad_enable=*/false, value); }

inline uint32_t dispatch_read_go() { return FDS_INTF_READ(TT_FDS_DISPATCH_DISPATCH_TO_TENSIX_REG_ADDR); }

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

inline void worker_disable_auto_dispatch() { FdsNeo::fds_disable_auto_dispatch(); }

inline uint32_t worker_read_filter_length() { return FDS_INTF_READ(TT_FDS_TENSIXNEO_FILTER_COUNT_THRESHOLD_REG_ADDR); }

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

inline uint32_t worker_read_done() { return FdsNeo::fds_read_done(); }

inline void worker_clear_done() {
    write_until_readback_matches([] { FdsNeo::fds_clear_done(); }, [] { return worker_read_done(); }, idle_group_id);
}

inline void worker_signal_done(uint32_t group_id) {
    write_until_readback_matches(
        [=] { FdsNeo::fds_done(/*ad_enable=*/false, group_id); }, [] { return worker_read_done(); }, group_id);
}

}  // namespace overlay::fds_signalling
