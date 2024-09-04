// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "command_queue_interface.hpp"

namespace tt {

typedef enum e_data_collector_t {
    DISPATCH_DATA_CB_CONFIG,
    DISPATCH_DATA_SEMAPHORE,
    DISPATCH_DATA_RTARGS,
    DISPATCH_DATA_BINARY,
    DISPATCH_DATA_COUNT
} data_collector_t;

/* Record a single dispatch write, to be dumped with stats on program exit. Should only be called once per transaction
 * per program (if a program is enqueued multiple times, don't call this multiple times).
 *
 * Arguments:
 *      program - program this transaction is part of.
 *      type - what type of transaction this counts as, one of data_collector_t.
 *      transaction_size - size in bytes of this transaction.
 *      riscv - riscv core that this transaction is used for, only relevant for DISPATCH_DATA_BINARY transactions.
 */
void RecordDispatchData(Program &program, data_collector_t type, uint32_t transaction_size, RISCV riscv = RISCV::MAX);

// Record the KernelGroups present in this program (per core type). Should only be called per program created, not
// program enqueued.
void RecordKernelGroups(Program &program, CoreType core_type, std::vector<KernelGroup> &kernel_groups);

// Update stats with an enqueue of given program.
void RecordProgramRun(Program &program);

} // end namespace tt
