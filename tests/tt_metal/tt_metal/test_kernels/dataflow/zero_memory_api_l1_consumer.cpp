// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// No-op consumer kernel for the L1 zero-memory-api smoke test. Satisfies the
// DFB "one producer / one consumer" invariant; the producer kernel does all
// the actual zeroing and verification.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    DataflowBuffer dfb(dfb::scratch);
    dfb.wait_front(1);
    dfb.pop_front(1);
}
