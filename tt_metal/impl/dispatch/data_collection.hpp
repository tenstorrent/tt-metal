// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <tt-metalium/sub_device_types.hpp>
#include "program/program_impl.hpp"
#include "impl/context/context_types.hpp"

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
    tt::tt_metal::ContextId context_id,
    uint64_t program_id,
    data_collector_t type,
    uint32_t transaction_size,
    std::optional<tt_metal::HalProcessorIdentifier> processor = std::nullopt);

// Record the KernelGroups present in this program (per core type). Should only be called per program created, not
// program enqueued.
void RecordKernelGroup(
    tt::tt_metal::ContextId context_id,
    tt_metal::detail::ProgramImpl& program,
    tt_metal::HalProgrammableCoreType core_type,
    const tt_metal::KernelGroup& kernel_group);

// Update stats with an enqueue of given program.
void RecordProgramRun(tt::tt_metal::ContextId context_id, uint64_t program_id);

// True while a profiler can still consume what RecordProgramMetadata and RecordProgramSubDevice
// record. Both are no-ops otherwise, so a dispatch path that would call them once per program per
// device can check this once instead.
bool IsProgramMetadataRecordingEnabled(tt::tt_metal::ContextId context_id);

// Record metadata used by profiler lookups for this program dispatch.
void RecordProgramMetadata(tt::tt_metal::ContextId context_id, tt_metal::detail::ProgramImpl& program);

struct ProgramSubDeviceInfo {
    uint8_t sub_device_id = 0;
    uint64_t sub_device_manager_id = 0;
    // Tensix worker cores in this sub-device when recorded at dispatch; 0 means unset (use full device grid).
    uint32_t num_available_worker_cores = 0;
};

// Record which sub-device a program executes on. Should be called at dispatch time when runtime_id is set.
void RecordProgramSubDevice(
    tt::tt_metal::ContextId context_id,
    tt::ChipId device_id,
    uint64_t sub_device_manager_id,
    uint64_t runtime_id,
    tt::tt_metal::SubDeviceId sub_device_id,
    uint32_t num_available_worker_cores = 0);

// Look up the sub-device a program was dispatched on, keyed by physical device and runtime_id.
std::optional<ProgramSubDeviceInfo> GetProgramSubDevice(
    tt::tt_metal::ContextId context_id, tt::ChipId device_id, uint64_t runtime_id);

}  // end namespace tt
