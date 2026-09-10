// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    DataflowBuffer dfb(dfb::in);
    dfb.wait_front(1);
    dfb.pop_front(1);
}
