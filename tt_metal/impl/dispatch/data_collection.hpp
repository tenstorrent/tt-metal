// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <device.hpp>
#include <host_api.hpp>
#include <stdint.h>
#include <optional>
#include <span>
#include <string_view>
#include <tt-metalium/experimental/realtime_profiler.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include "program/program_impl.hpp"

namespace tt {
namespace tt_metal {
class Program;
struct KernelGroup;
}  // namespace tt_metal

enum data_collector_t {
    DISPATCH_DATA_CB_CONFIG,
    DISPATCH_DATA_SEMAPHORE,
    DISPATCH_DATA_RTARGS,
    DISPATCH_DATA_BINARY,
};

// Aliases to the public experimental types for internal use.
using ProgramRealtimeRecord = tt::tt_metal::experimental::ProgramRealtimeRecord;
using ProgramRealtimeRecordBatch = tt::tt_metal::experimental::ProgramRealtimeRecordBatch;

/* Record a single dispatch write, to be dumped with stats on program exit. Should only be called once per transaction
 * per program (if a program is enqueued multiple times, don't call this multiple times).
 *
 * Arguments:
 *      program - program this transaction is part of.
 *      type - what type of transaction this counts as, one of data_collector_t.
 *      transaction_size - size in bytes of this transaction.
 *      processor - processor that this transaction is used for, only relevant for DISPATCH_DATA_BINARY transactions.
 */
void RecordDispatchData(
    uint64_t program_id,
    data_collector_t type,
    uint32_t transaction_size,
    std::optional<tt_metal::HalProcessorIdentifier> processor = std::nullopt);

// Record the KernelGroups present in this program (per core type). Should only be called per program created, not
// program enqueued.
void RecordKernelGroup(
    tt_metal::detail::ProgramImpl& program,
    tt_metal::HalProgrammableCoreType core_type,
    const tt_metal::KernelGroup& kernel_group);

// Update stats with an enqueue of given program.
void RecordProgramRun(uint64_t program_id);

// Record metadata used by profiler lookups for this program dispatch.
void RecordProgramMetadata(tt_metal::detail::ProgramImpl& program);

struct ProgramSubDeviceInfo {
    uint8_t sub_device_id = 0;
    uint64_t sub_device_manager_id = 0;
    // Tensix worker cores in this sub-device when recorded at dispatch; 0 means unset (use full device grid).
    uint32_t num_available_worker_cores = 0;
};

// Record which sub-device a program executes on. Should be called at dispatch time when runtime_id is set.
void RecordProgramSubDevice(
    tt::ChipId device_id,
    uint64_t sub_device_manager_id,
    uint64_t runtime_id,
    tt::tt_metal::SubDeviceId sub_device_id,
    uint32_t num_available_worker_cores = 0);

// Look up the sub-device a program was dispatched on, keyed by physical device and runtime_id.
std::optional<ProgramSubDeviceInfo> GetProgramSubDevice(tt::ChipId device_id, uint64_t runtime_id);

// Look up kernel source paths by runtime_id; empty span if the runtime_id is unknown.
// The returned span is valid until MetalContext teardown or reinitialization.
std::span<const std::string_view> GetKernelSourcesForRuntimeId(uint16_t runtime_id);

}  // end namespace tt
