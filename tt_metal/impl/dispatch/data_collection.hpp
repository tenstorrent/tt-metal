// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <device.hpp>
#include <host_api.hpp>
#include <stdint.h>
#include <optional>

#include "hal_types.hpp"
#include "program/program_impl.hpp"

#include <umd/device/types/core_coordinates.hpp>

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

}  // end namespace tt
