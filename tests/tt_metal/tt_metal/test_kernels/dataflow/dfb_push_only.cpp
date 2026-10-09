// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Posts `num_entries` entries to dfb::out and exits without waiting for them to be consumed, so the
// tile counter is left holding them after the program completes.

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries = get_arg(args::num_entries);

    DataflowBuffer dfb(dfb::out);
    dfb.reserve_back(num_entries);
    dfb.push_back(num_entries);
}
