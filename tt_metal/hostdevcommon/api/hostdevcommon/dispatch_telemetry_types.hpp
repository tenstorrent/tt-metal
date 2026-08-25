// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <hostdevcommon/common_values.hpp>
namespace tt::tt_metal {
namespace detail {

constexpr uint32_t pack(const char (&s)[5]) {
    return (uint32_t(s[0]) << 24) | (uint32_t(s[1]) << 16) | (uint32_t(s[2]) << 8) | (uint32_t(s[3]));
}

}  // namespace detail

namespace dispatch_telemetry_types {
// Increment only when breaking changes occur
constexpr uint32_t DISPATCH_TELEMETRY_VERSION = 1;

/**
 * @brief Expected signature for validating that a telemetry buffer contains dispatch telemetry data.
 */
constexpr uint32_t SMC_TELEMETRY_SIGNATURE = detail::pack("SMC_");
constexpr uint32_t DISPATCH_CORE_TELEMETRY_SIGNATURE = detail::pack("DISP");
constexpr uint32_t PREFETCH_CORE_TELEMETRY_SIGNATURE = detail::pack("PREF");

// Used to invalidate the telemetry buffer
constexpr uint32_t INVALID_TELEMETRY_SIGNATURE = 0;

constexpr uint32_t MAX_SUB_DEVICES = DISPATCH_MAX_MESSAGE_ENTRIES;
constexpr uint32_t RESERVED_SUB_DEVICE_SPACE = 8;
static_assert(
    MAX_SUB_DEVICES <= RESERVED_SUB_DEVICE_SPACE,
    "If MAX_SUB_DEVICES exceeds RESERVED_SUB_DEVICE_SPACE, then the reserved space must manually be "
    "increased and the version must be incremented as this struct will no longer be backwards "
    "compatible");

struct __attribute__((packed, aligned(4))) PrefetchCoreTelemetry {
    uint32_t version = DISPATCH_TELEMETRY_VERSION;
    uint32_t signature = PREFETCH_CORE_TELEMETRY_SIGNATURE;
    uint32_t upstream_blocked_count = 0;
    uint32_t upstream_unblocked_count = 0;
    uint32_t command_count = 0;
};

struct __attribute__((packed, aligned(8))) DispatchCoreTelemetry {
    uint32_t version = DISPATCH_TELEMETRY_VERSION;
    uint32_t signature = DISPATCH_CORE_TELEMETRY_SIGNATURE;

    // dispatch_d writes
    uint32_t upstream_blocked_count = 0;
    uint32_t upstream_unblocked_count = 0;
    uint32_t program_count = 0;

    // To preserve proper alignment. Can be utilized for future use without breaking backwards
    // compatibility.
    uint32_t reserved_0 = 0;

    // dispatch_s_compute writes
    uint64_t current_timestamp = 0;

    // dispatch_s_compute writes
    // Computed average time a worker core was running, updated only on sub device count change.
    uint64_t avg_work_runtime_per_worker = 0;  //_per_worker

    // dispatch_s_compute writes
    // Amount of time the cq is running any work on any core.
    uint64_t utilization_work_runtime = 0;

    // dispatch_s_compute writes
    // Timestamp for the active utilization interval. Set only while at least one sub-device is running work.
    uint64_t work_runtime_start = 0;

    // dispatch_s writes
    uint64_t last_work_launch_timestamp[RESERVED_SUB_DEVICE_SPACE] = {0};

    // dispatch_s_compute writes
    // Cumulative current total worker runtime for each sub device. In the case of overflow, the
    // value is compressed into avg_work_runtime_per_worker and then reset to 0. Used to avoid
    // dropping work cycles if they were preemptively averaged.
    uint64_t current_sub_device_work_runtime[RESERVED_SUB_DEVICE_SPACE] = {0};

    // dispatch_s_compute writes
    uint32_t completion_count[RESERVED_SUB_DEVICE_SPACE] = {0};

    // dispatch_s writes
    uint32_t workers_per_sub_device[RESERVED_SUB_DEVICE_SPACE] = {0};
};

// Used to determine the size of the L1 buffer that dispatch_mem_map allocates
// Note: If new telemetry types are added, update this calculation
constexpr size_t DISPATCH_TELEMETRY_SIZE = std::max(sizeof(DispatchCoreTelemetry), sizeof(PrefetchCoreTelemetry));

struct __attribute__((packed, aligned(4))) DispatchTelemetryControl {
    uint32_t sub_device_worker_counts_update = 0;
    uint32_t worker_stream_reset_update = 0;
    uint32_t compute_terminate = 0;

    // dispatch_s writes, dispatch_s_compute reads.
    // Increments by 2 for every launched workload. An odd value means last_work_launch_timestamp
    // is in progress of being written.
    uint32_t launched_work_sequence_counter[RESERVED_SUB_DEVICE_SPACE] = {0};

    // dispatch_s writes, dispatch_s_compute reads.
    // Records value of the stream semaphore when launching a new workload.
    uint32_t launched_work_start_stream_sem[RESERVED_SUB_DEVICE_SPACE] = {0};
};

constexpr uint32_t INVALID_SMC_DISPATCH_CORE_COORDS = UINT32_MAX;
// Packed virtual dispatch-core coordinates. These are consumed by host telemetry readers and are not NOC coords.
struct __attribute__((packed)) SMCDispatchCoreCoords {
    uint32_t prefetch_xy = INVALID_SMC_DISPATCH_CORE_COORDS;
    uint32_t dispatch_xy = INVALID_SMC_DISPATCH_CORE_COORDS;
    uint32_t dispatch_s_xy = INVALID_SMC_DISPATCH_CORE_COORDS;
};

constexpr uint32_t smc_dispatch_core_x(uint32_t xy) { return xy >> 16; }

constexpr uint32_t smc_dispatch_core_y(uint32_t xy) { return xy & 0xFFFF; }

constexpr uint32_t pack_smc_dispatch_core_xy(uint16_t x, uint16_t y) {
    return (static_cast<uint32_t>(x) << 16) | static_cast<uint32_t>(y);
}

constexpr uint32_t MAX_DISPATCH_CORES_PER_CQ = sizeof(SMCDispatchCoreCoords) / sizeof(uint32_t);
constexpr uint32_t RESERVED_CQ_SPACE = 3;
static_assert(MAX_NUM_HW_CQS <= RESERVED_CQ_SPACE, "Max number of hardware CQs exceeds reserved space");

// flags enum
enum class SMCDispatchTelemetryFlags : uint8_t {
    NONE = 0,
    SLOW_DISPATCH_ENABLED = 1 << 0,
};

// Stored on device but data is host read/write only, so types can be any size
struct __attribute__((packed)) SMCDispatchTelemetryControl {
    uint32_t version = DISPATCH_TELEMETRY_VERSION;
    uint32_t signature = SMC_TELEMETRY_SIGNATURE;
    uint8_t flags = 0;
    uint32_t dispatch_telemetry_addr = 0;
    uint8_t num_hw_cqs = RESERVED_CQ_SPACE;
    SMCDispatchCoreCoords cq_dispatch_core_coords[RESERVED_CQ_SPACE];
    struct __attribute__((packed)) SDTelemetry {
        // Reserved for future use
    } sd_telemetry[RESERVED_CQ_SPACE];
};

}  // namespace dispatch_telemetry_types
}  // namespace tt::tt_metal
