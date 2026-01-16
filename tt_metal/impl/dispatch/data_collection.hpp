// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <device.hpp>
#include <host_api.hpp>
#include <stdint.h>
#include <functional>
#include <optional>
#include <string>
#include <vector>
#include <tt-metalium/experimental/perf_telemetry.hpp>
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
using ProgramPerfRecord = tt::tt_metal::experimental::ProgramPerfRecord;
using ProgramPerfCallback = tt::tt_metal::experimental::ProgramPerfCallback;
using ProgramPerfCallbackHandle = tt::tt_metal::experimental::ProgramPerfCallbackHandle;

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

// Record the mapping from a program's runtime_id to its kernel source paths.
// Should be called at dispatch time when runtime_id is guaranteed to be set.
void RecordKernelSourceMap(tt_metal::detail::ProgramImpl& program);

// Look up the kernel source paths for a given runtime_id.
// Returns a comma-separated string of kernel source paths, or empty string if not found.
std::string GetKernelSourcesForRuntimeId(uint64_t runtime_id);

// Look up the kernel source paths for a given runtime_id as a vector.
std::vector<std::string> GetKernelSourcesVecForRuntimeId(uint64_t runtime_id);

// Register a callback to be invoked when program perf telemetry data arrives.
// Multiple callbacks can be registered; they are called in order of registration.
// Returns a handle that can be used to unregister the callback.
ProgramPerfCallbackHandle RegisterProgramPerfCallback(ProgramPerfCallback callback);

// Unregister a previously registered callback by its handle.
void UnregisterProgramPerfCallback(ProgramPerfCallbackHandle handle);

// Invoke all registered program perf callbacks with the given record.
// Called internally by the perf telemetry receiver thread.
void InvokeProgramPerfCallbacks(const ProgramPerfRecord& record);

}  // end namespace tt
